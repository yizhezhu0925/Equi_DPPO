"""
Wrapper for equivariant diffusion model that handles rot6d -> axis-angle conversion.
Use this for evaluation with environments expecting 7D actions.

Place this file at: model/diffusion/diffusion_equi.py
"""

import torch
from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract
from model.common.rot6d_to_axis_angle import convert_rot6d_action_to_axis_angle


class EquiDiffusionModelEval(DiffusionModel):
    """
    Wraps DiffusionModel to convert 10D rot6d outputs to 7D axis-angle for env.
    """
    
    def __init__(self, convert_to_axis_angle: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.convert_to_axis_angle = convert_to_axis_angle
    
    @torch.no_grad()
    def forward(self, cond, deterministic=True):
        """
        Forward pass for sampling actions with rot6d -> axis-angle conversion.
        
        This overrides the parent forward() to avoid compatibility issues.
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Start from noise
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        
        # Choose timesteps based on DDIM or DDPM
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        
        # Denoising loop
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            
            # Get mean and variance (don't pass deterministic to p_mean_var)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = torch.zeros_like(std)
            else:
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
            
            # Add noise (deterministic = no noise)
            if deterministic:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x).clamp_(
                    -self.randn_clip_value, self.randn_clip_value
                )
            x = mean + std * noise

            # Clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        
        # Convert rot6d to axis-angle if needed
        if self.convert_to_axis_angle:
            x = convert_rot6d_action_to_axis_angle(x)
        
        return Sample(x, None)