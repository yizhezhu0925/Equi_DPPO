"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

Fixed: Added normalize_gripper_only option for equivariant policy evaluation.
When normalize_obs=False and normalize_gripper_only=True:
- pos [0:3]: kept as raw values (for equivariance)
- quat [3:7]: kept as raw unit quaternion in xyzw format (for proper rot6d conversion)
- gripper [7:9]: normalized to [-1, 1] (to match training)

"""

import numpy as np
import gym
from gym import spaces
import imageio


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        normalize_obs=True,
        normalize_action=True,
        normalize_gripper_only=False,  # NEW: only normalize gripper when normalize_obs=False
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        **kwargs,
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs

        # set up normalization
        self.has_stats = normalization_path is not None
        self.normalize_obs = normalize_obs and self.has_stats
        self.normalize_action = normalize_action and self.has_stats
        self.normalize_gripper_only = normalize_gripper_only and self.has_stats  # NEW
        
        if self.has_stats:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.obs_keys = low_dim_keys + image_keys
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 1
            elif key.endswith("state"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def _normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def _normalize_gripper(self, gripper):
        """Normalize only the gripper values (indices 7:9 of state)."""
        gripper_min = self.obs_min[7:9]
        gripper_max = self.obs_max[7:9]
        gripper_norm = 2 * (
            (gripper - gripper_min) / (gripper_max - gripper_min + 1e-6) - 0.5
        )
        return gripper_norm

    def unnormalize_action(self, action):
        if not self.normalize_action:
            return action
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}
        for key in self.obs_keys:
            if key in self.image_keys:
                img = raw_obs[key]
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1))
                if obs["rgb"] is None:
                    obs["rgb"] = img
                else:
                    obs["rgb"] = np.concatenate([obs["rgb"], img], axis=0)
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key]
                else:
                    obs["state"] = np.concatenate([obs["state"], raw_obs[key]], axis=-1)
        
        if self.normalize_obs:
            obs["state"] = self._normalize_obs(obs["state"])
        elif self.normalize_gripper_only:
            state = obs["state"].copy()
            
            quat_wxyz = state[3:7].copy()
            
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]  
            state[3:7] = quat_xyzw
            
            # gripper 归一化
            state[7:9] = self._normalize_gripper(state[7:9])
            obs["state"] = state
        
        if obs["rgb"].max() <= 1.0:
            obs["rgb"] = obs["rgb"] * 255.0
        obs["rgb"] = obs["rgb"].astype(np.float32)
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        if self.normalize_action:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["robot0_eye_in_hand_image"],
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys())
    img = wrapper.render()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test.png")