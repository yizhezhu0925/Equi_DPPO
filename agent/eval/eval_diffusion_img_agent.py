"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent


class EvalImgDiffusionAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
        # record env action dim for potential post-processing
        self.env_action_shape = tuple(self.venv.single_action_space.shape)
        self.model_action_dim = cfg.action_dim
        self.action_min = None
        self.action_max = None
        norm_path = getattr(cfg, "normalization_path", None)
        if norm_path:
            try:
                norm = np.load(norm_path)
                self.action_min = torch.from_numpy(norm["action_min"]).float()
                self.action_max = torch.from_numpy(norm["action_max"]).float()
            except FileNotFoundError:
                log.warning("Normalization path %s not found; skipping action renormalization", norm_path)

    @staticmethod
    def _rot6d_to_axis_angle(rot6: torch.Tensor) -> torch.Tensor:
        """
        rot6: (...,6) representing first two columns of rotation matrix.
        Returns axis-angle (...,3).
        """
        a1 = rot6[..., [0, 2, 4]]  
        a2 = rot6[..., [1, 3, 5]]  
        b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
        # Gram-Schmidt
        a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(a2, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack([b1, b2, b3], dim=-1)  # (...,3,3)
        trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        # handle small angles
        small = sin_theta.abs() < 1e-5
        axis = torch.stack(
            [R[..., 2, 1] - R[..., 1, 2],
             R[..., 0, 2] - R[..., 2, 0],
             R[..., 1, 0] - R[..., 0, 1]],
            dim=-1
        ) / (2 * sin_theta.unsqueeze(-1) + 1e-8)
        axis = torch.where(small.unsqueeze(-1), torch.zeros_like(axis), axis)
        aa = axis * theta.unsqueeze(-1)
        return aa

    def _maybe_convert_actions(self, actions_np: np.ndarray) -> np.ndarray:
        """
        Convert model actions to env action dimension if needed.
        actions_np: (n_env, horizon, model_action_dim)
        """
        if self.model_action_dim == 10:
            act = torch.from_numpy(actions_np).float()
            pos = act[..., :3]      
            rot6 = act[..., 3:9]
            grip = act[..., 9:]     
            

            aa = self._rot6d_to_axis_angle(rot6)
            

            if self.action_min is not None and self.action_max is not None:
                amin = self.action_min[3:6].to(aa.device)
                amax = self.action_max[3:6].to(aa.device)

                aa = 2 * (aa - amin) / (amax - amin + 1e-8) - 1
                aa = torch.clamp(aa, -1.0, 1.0)
            act7 = torch.cat([pos, aa, grip], dim=-1)
            
            if act7.shape[1] > self.act_steps:
                act7 = act7[:, :self.act_steps]
            act7 = torch.clamp(act7, -1.0, 1.0)
            return act7.numpy()

    def run(self):

        # Start training loop
        timer = Timer()

        # Prepare video paths for each envs
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))

        # Collect a set of trajectories from env
        for step in range(self.n_steps):
            if step % 10 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            # DEBUG: 只在前3步打印
            debug = (step < 3)

            # Select action
            with torch.no_grad():
                cond = {
                    key: torch.from_numpy(prev_obs_venv[key]).float().to(self.device)
                    for key in self.obs_dims
                }
                
                if debug:
                    print(f"\n{'='*50}")
                    # print(f"[STEP {step}] Model input:")
                    # print(f"  state shape: {cond['state'].shape}")
                    # print(f"  state[0,0]: {cond['state'][0,0].cpu().numpy()}")
                    # print(f"  state[0,0] pos: {cond['state'][0,0,:3].cpu().numpy()}")
                    # print(f"  state[0,0] quat: {cond['state'][0,0,3:7].cpu().numpy()}")
                    # print(f"  state[0,0] gripper: {cond['state'][0,0,7:9].cpu().numpy()}")
                    # print(f"  rgb shape: {cond['rgb'].shape}")
                    # print(f"  rgb range: [{cond['rgb'].min():.1f}, {cond['rgb'].max():.1f}]")
                
                samples = self.model(cond=cond, deterministic=True)
                output_venv = samples.trajectories.cpu().numpy()
                
                if debug:
                    print(f"[STEP {step}] Model output (10D):")
                    # print(f"  shape: {output_venv.shape}")
                    # print(f"  output[0,0]: {output_venv[0,0]}")
                
                output_venv = self._maybe_convert_actions(output_venv)
                
                if debug:
                    print(f"[STEP {step}] After conversion (7D):")
                    # print(f"  shape: {output_venv.shape}")
                    # print(f"  output[0,0]: {output_venv[0,0]}")
                    # print(f"{'='*50}\n")
                    
            action_venv = output_venv[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            # update for next step
            prev_obs_venv = obs_venv

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            print(f"\n[DEBUG] Number of episodes: {num_episode_finished}")
            print(f"[DEBUG] Episode rewards (sum): {episode_reward[:10]}")
            print(f"[DEBUG] Max rewards per episode (raw): {[np.max(r) for r in reward_trajs_split[:10]]}")
            print(f"[DEBUG] act_steps: {self.act_steps}")
            print(f"[DEBUG] best_reward_threshold: {self.best_reward_threshold_for_success}")

            if (
                self.furniture_sparse_reward
            ):  # only for furniture tasks, where reward only occurs in one env step
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            log.info("[WARNING] No episode completed within the iteration!")

        # Log loss and save metrics
        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )
