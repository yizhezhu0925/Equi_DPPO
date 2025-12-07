"""
PPO Diffusion wrapper for equivariant models with rot6d action space.

Handles the conversion between rot6d (training space) and axis-angle (environment space).

Place this file at: model/diffusion/diffusion_ppo_equi.py
"""

import torch
from collections import namedtuple
from model.diffusion.diffusion_ppo import PPODiffusion
from model.common.rot6d_to_axis_angle import convert_rot6d_action_to_axis_angle

# Extended sample that includes both representations
SampleEqui = namedtuple("SampleEqui", "trajectories trajectories_rot6d chains")


## TODO: maybe this should actually be 
## TrainPPOImgDiffusionAgent
class PPODiffusionEqui(PPODiffusion):
    """
    Extends PPODiffusion to handle rot6d <-> axis-angle conversion.
    
    The equivariant UNet operates in rot6d space (10D), but the environment
    expects axis-angle actions (7D). This class:
    
    1. During forward(): Returns axis-angle trajectories for env stepping,
       but also returns rot6d chains for log probability computation
    2. During training: All computations happen in rot6d space
    
    The key insight is that:
    - Environment stepping needs 7D axis-angle actions
    - Log prob computation and policy updates need 10D rot6d actions
    - The chains (denoising trajectory) must stay in rot6d space
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass that returns both axis-angle and rot6d representations.
        
        Returns:
            SampleEqui with:
                - trajectories: (B, Ta, 7) axis-angle for environment
                - trajectories_rot6d: (B, Ta, 10) rot6d for logging/debugging
                - chains: (B, K+1, Ta, 10) rot6d denoising chains for training
        """
        # Get rot6d samples from parent
        sample = super().forward(
            cond=cond,
            deterministic=deterministic,
            return_chain=return_chain,
            use_base_policy=use_base_policy,
        )
        
        # Convert final trajectory to axis-angle for environment
        trajectories_aa = convert_rot6d_action_to_axis_angle(sample.trajectories)
        
        return SampleEqui(
            trajectories=trajectories_aa,        # 7D for env
            trajectories_rot6d=sample.trajectories,  # 10D for reference
            chains=sample.chains,                # 10D for training
        )
    
    def get_logprobs(
        self,
        cond,
        chains,
        get_ent=False,
        use_base_policy=False,
    ):
        """
        Compute log probabilities. Chains should be in rot6d space (10D).
        
        This is unchanged from parent - just documenting that chains must be rot6d.
        """
        return super().get_logprobs(
            cond=cond,
            chains=chains,  # Must be (B, K+1, Ta, 10) rot6d
            get_ent=get_ent,
            use_base_policy=use_base_policy,
        )
    
    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent=False,
    ):
        """
        Compute log probabilities for subsampled denoising steps.
        Chains should be in rot6d space (10D).
        """
        return super().get_logprobs_subsample(
            cond=cond,
            chains_prev=chains_prev,   # Must be (B, Ta, 10) rot6d
            chains_next=chains_next,   # Must be (B, Ta, 10) rot6d
            denoising_inds=denoising_inds,
            get_ent=get_ent,
        )