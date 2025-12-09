"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

Fixed normalization for rot6d conversion:
- Actions: denormalize 7D -> convert to rot6d -> renormalize 10D
- States: denormalize pos/quat, convert quat from wxyz to xyzw format

"""

from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random
import os
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
        convert_to_rot6d=False,
        normalization_path=None,
        denormalize_obs=False,
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path
        self.convert_to_rot6d = convert_to_rot6d
        self.denormalize_obs = denormalize_obs

        # Load normalization parameters
        self.action_min = None
        self.action_max = None
        self.obs_min = None
        self.obs_max = None
        
        if normalization_path is not None and os.path.exists(normalization_path):
            norm = np.load(normalization_path)
            self.action_min = torch.from_numpy(norm["action_min"]).float().to(device)
            self.action_max = torch.from_numpy(norm["action_max"]).float().to(device)
            self.obs_min = torch.from_numpy(norm["obs_min"]).float().to(device)
            self.obs_max = torch.from_numpy(norm["obs_max"]).float().to(device)
            log.info(f"Loaded normalization from {normalization_path}")
            log.info(f"  action_min: {self.action_min}")
            log.info(f"  action_max: {self.action_max}")

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = (
            torch.from_numpy(dataset["states"][:total_num_steps]).float().to(device)
        )
        self.actions = (
            torch.from_numpy(dataset["actions"][:total_num_steps]).float().to(device)
        )

        # Convert actions to rot6d with proper normalization handling
        if self.convert_to_rot6d:
            self.actions = self._convert_normalized_actions_to_rot6d(self.actions)

        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        if self.use_img:
            self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(
                device
            )
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")

    def _convert_normalized_actions_to_rot6d(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Properly convert normalized 7D actions to normalized 10D actions.
        
        Flow: normalized 7D -> denormalize -> convert to rot6d -> renormalize 10D
        """
        act_dim = actions.shape[1]
        
        # Already 10D, return as-is
        if act_dim == 10:
            log.info("Actions already 10D, skipping conversion")
            return actions
        
        if act_dim != 7:
            raise ValueError(f"Expected 7D actions for rot6d conversion, got {act_dim}")
        
        if self.action_min is None or self.action_max is None:
            raise ValueError(
                "normalization_path required when convert_to_rot6d=True. "
                "Need action_min/max to properly denormalize before conversion."
            )

        log.info("Converting 7D actions to 10D with proper normalization...")
        log.info(f"  Input range: [{actions.min():.3f}, {actions.max():.3f}]")

        # 1. Denormalize to raw 7D actions
        actions_raw = (actions + 1) / 2 * (self.action_max - self.action_min + 1e-6) + self.action_min
        log.info(f"  After denorm: [{actions_raw.min():.3f}, {actions_raw.max():.3f}]")

        # 2. Extract components
        pos = actions_raw[:, :3]          # xyz position
        aa = actions_raw[:, 3:6]          # axis-angle rotation
        gripper = actions_raw[:, 6:]      # gripper

        # 3. Convert axis-angle to rot6d
        rot6d = self._axis_angle_to_rot6d(aa)  # (N, 6), range approximately [-1, 1]
        log.info(f"  rot6d range: [{rot6d.min():.3f}, {rot6d.max():.3f}]")

        # 4. Concatenate to 10D
        actions_10d = torch.cat([pos, rot6d, gripper], dim=1)

        # 5. Renormalize each component
        # pos: use original min/max
        pos_norm = (pos - self.action_min[:3]) / (self.action_max[:3] - self.action_min[:3] + 1e-6) * 2 - 1
        
        # rot6d: rotation matrix elements are inherently in [-1, 1], just clip
        rot6d_norm = torch.clamp(rot6d, -1.0, 1.0)
        
        # gripper: use original min/max
        gripper_norm = (gripper - self.action_min[6:]) / (self.action_max[6:] - self.action_min[6:] + 1e-6) * 2 - 1

        actions_norm = torch.cat([pos_norm, rot6d_norm, gripper_norm], dim=1)

        log.info(f"  Final 10D action stats:")
        log.info(f"    pos_norm range: [{pos_norm.min():.3f}, {pos_norm.max():.3f}]")
        log.info(f"    rot6d_norm range: [{rot6d_norm.min():.3f}, {rot6d_norm.max():.3f}]")
        log.info(f"    gripper_norm range: [{gripper_norm.min():.3f}, {gripper_norm.max():.3f}]")
        log.info(f"    total range: [{actions_norm.min():.3f}, {actions_norm.max():.3f}]")

        return actions_norm

    @staticmethod
    def _axis_angle_to_rot6d(vec: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle to rot6d representation.
        
        Args:
            vec: (..., 3) axis-angle rotation vectors
            
        Returns:
            (..., 6) rot6d representation (first two columns of rotation matrix)
        """
        angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
        axis = vec / (angle + 1e-8)
        x, y, z = axis.unbind(-1)
        zeros = torch.zeros_like(x)
        
        # Skew-symmetric matrix K
        K = torch.stack(
            [
                torch.stack([zeros, -z, y], dim=-1),
                torch.stack([z, zeros, -x], dim=-1),
                torch.stack([-y, x, zeros], dim=-1),
            ],
            dim=-2,
        )
        
        # Identity matrix
        I = torch.eye(3, device=vec.device, dtype=vec.dtype).expand(
            vec.shape[:-1] + (3, 3)
        )
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        ang = angle.squeeze(-1)[..., None, None]
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        R = I + sin * K + (1 - cos) * (K @ K)
        
        # Handle small angles (use identity)
        R = torch.where(ang < 1e-8, I, R)
        
        # Extract first two columns and flatten
        return R[..., :3, :2].reshape(vec.shape[:-1] + (6,))

    def _denormalize(self, x_norm: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
        """
        Denormalize from [-1, 1] to original range.
        
        Args:
            x_norm: (..., D) normalized values in [-1, 1]
            x_min: (D,) minimum values
            x_max: (D,) maximum values
            
        Returns:
            (..., D) denormalized values
        """
        return (x_norm + 1) / 2 * (x_max - x_min + 1e-6) + x_min

    def _denormalize_quat(self, quat_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize quaternion, ensure it's a valid unit quaternion,
        and convert from wxyz (npz format) to xyzw (robosuite format).
        
        Args:
            quat_norm: (..., 4) normalized quaternion in wxyz format
            
        Returns:
            (..., 4) valid unit quaternion in xyzw format
        """
        if self.obs_min is None or self.obs_max is None:
            return quat_norm
        
        # Denormalize
        quat_min = self.obs_min[3:7]
        quat_max = self.obs_max[3:7]
        quat_raw = (quat_norm + 1) / 2 * (quat_max - quat_min + 1e-6) + quat_min
        
        # Normalize to unit quaternion
        quat_unit = quat_raw / (quat_raw.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Convert wxyz to xyzw (to match robosuite format and EquivariantObsEnc expectation)
        quat_xyzw = quat_unit[..., [1, 2, 3, 0]]
        
        return quat_xyzw

    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            Batch with:
                - actions: (horizon_steps, action_dim)
                - conditions: dict with 'state' and optionally 'rgb'
                
        Note: If denormalize_obs=True:
            - pos is denormalized (for equivariance)
            - quat is denormalized and converted to xyzw format (for proper rot6d conversion)
            - gripper stays normalized (trivial repr)
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        
        # Stack observation history (more recent at the end)
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )
        
        # Denormalize pos and quat for proper equivariance
        if self.denormalize_obs and self.obs_min is not None:
            states_processed = states.clone()
            
            # Denormalize pos (indices 0:3) — required for equivariance
            states_processed[..., 0:3] = self._denormalize(
                states[..., 0:3], 
                self.obs_min[:3], 
                self.obs_max[:3]
            )
            
            # Denormalize quat (indices 3:7), normalize to unit, and convert wxyz -> xyzw
            states_processed[..., 3:7] = self._denormalize_quat(states[..., 3:7])
            
            # gripper (indices 7:9) stays normalized — trivial repr, doesn't affect equivariance
            
            conditions = {"state": states_processed}
        else:
            conditions = {"state": states}
        
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        Makes indices for sampling from dataset.
        Each index maps to a datapoint, also save the number of steps before it 
        within the same trajectory.
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning.

    Do not load the last step of **truncated** episodes since we do not have 
    the correct next state for the final step of each episode.
    """

    def __init__(
        self,
        dataset_path,
        max_n_episodes=10000,
        discount_factor=1.0,
        device="cuda:0",
        get_mc_return=False,
        **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # Discount factor
        self.discount_factor = discount_factor

        # Rewards and dones (terminals)
        self.rewards = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        
        self.dones = (
            torch.from_numpy(dataset["terminals"][:total_num_steps]).to(device).float()
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            device=device,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # Compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            for i, traj_length in tqdm(
                enumerate(cumulative_traj_length), desc="Computing reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = torch.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns[-t - 1] = (
                        traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        Skip last step of truncated episodes.
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start : (start + 1)]
        dones = self.dones[start : (start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps) : start
                + 1
                + self.horizon_steps
            ]
        else:
            # Prevents indexing error, but ignored since done=True
            next_states = torch.zeros_like(states)

        # Stack obs history
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )
        next_states = torch.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )

        # Denormalize pos and quat if needed
        if self.denormalize_obs and self.obs_min is not None:
            states_processed = states.clone()
            states_processed[..., 0:3] = self._denormalize(
                states[..., 0:3], self.obs_min[:3], self.obs_max[:3]
            )
            states_processed[..., 3:7] = self._denormalize_quat(states[..., 3:7])
            
            next_states_processed = next_states.clone()
            next_states_processed[..., 0:3] = self._denormalize(
                next_states[..., 0:3], self.obs_min[:3], self.obs_max[:3]
            )
            next_states_processed[..., 3:7] = self._denormalize_quat(next_states[..., 3:7])
            
            conditions = {"state": states_processed, "next_state": next_states_processed}
        else:
            conditions = {"state": states, "next_state": next_states}

        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images

        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start : (start + 1)]
            batch = TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
        return batch