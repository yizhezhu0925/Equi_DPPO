from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat
from model.diffusion.Equi_Unet_Lab import ConditionalUnet1D
from model.common.Equi_obs_encoder import EquivariantObsEnc

#
class EquiDiffusionUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N=8):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.obs_encoder = EquivariantObsEnc()
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8
            + 2 * [self.group.trivial_repr], # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        z = conv_out[:, 8:9]
        # invert the rot reordering used in getActionGeometricTensor
        rot0 = conv_out[:, 2:3]
        rot1 = conv_out[:, 4:5]
        rot2 = conv_out[:, 6:7]
        rot3 = conv_out[:, 3:4]
        rot4 = conv_out[:, 5:6]
        rot5 = conv_out[:, 7:8]
        rot = torch.cat([rot0, rot1, rot2, rot3, rot4, rot5], dim=1)
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, rot, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        rot = act[:, 3:9]
        g = act[:, 9:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                rot[:, 0].reshape(batch_size, 1),
                rot[:, 3].reshape(batch_size, 1),
                rot[:, 1].reshape(batch_size, 1),
                rot[:, 4].reshape(batch_size, 1),
                rot[:, 2].reshape(batch_size, 1),
                rot[:, 5].reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    def forward(
        self, 
        x, 
        time, 
        cond, 
        **kwargs
    ):
        sample = x
        timestep = time
        B, T = sample.shape[:2]

        if cond is None or "state" not in cond:
            raise ValueError("cond must contain 'state' for EquiDiffusionUNet")
        state = cond["state"]  
        if state.shape[-1] < 9:
            raise ValueError(f"state last dim must be at least 9 (pos+quat+gripper), got {state.shape[-1]}")
        ee_pos = state[..., 0:3]
        ee_quat = state[..., 3:7]#xyzw
        ee_q = state[..., 7:9]

        agentview = cond.get("agentview_image", None)
        if agentview is None and "rgb" in cond:
            agentview = cond["rgb"]
        eye_in_hand = cond.get("robot0_eye_in_hand_image", None)
        if eye_in_hand is None and agentview is not None:
            eye_in_hand = agentview
        if agentview is None or eye_in_hand is None:
            raise ValueError("EquivariantObsEnc requires image inputs: provide 'agentview_image'/'robot0_eye_in_hand_image' or 'rgb'.")

        nobs = {
            "agentview_image": agentview,
            "robot0_eef_pos": ee_pos,
            "robot0_eef_quat": ee_quat,
            "robot0_gripper_qpos": ee_q,
        }

        global_cond = self.obs_encoder(nobs)
        global_cond = rearrange(global_cond, "b t d -> b (t d)")

        local_cond = None
        action_dim = sample.shape[-1]
        if action_dim != 10:
            raise ValueError(f"EquiDiffusionUNet expects 10D actions (xyz+rot6d+g), got {action_dim}")

        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.getActionGeometricTensor(sample)
        
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)

        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)

        if isinstance(timestep, torch.Tensor) and timestep.ndim == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        elif isinstance(timestep, (int, float)) or (isinstance(timestep, torch.Tensor) and timestep.ndim == 0):
             if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=sample.device)
             timestep = repeat(timestep, "1 -> (b f)", b=B, f=self.order)

        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)

        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)

        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        
        out = nn.GeometricTensor(out, self.act_type)
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        out = self.getOutput(out)

        out = rearrange(out, "(b t) n -> b t n", b=B)
        
        return out
