from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat
from Equi_Unet_Lab import ConditionalUnet1D
from model.common.Equi_obs_encoder import EquivariantObsEnc

class EquiDiffusionUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
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
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        cos2 = conv_out[:, 4:5]
        sin2 = conv_out[:, 5:6]
        cos3 = conv_out[:, 6:7]
        sin3 = conv_out[:, 7:8]
        z = conv_out[:, 8:9]
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
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

        nobs = {
            "agentview_image": cond["agentview_image"],
            "robot0_eef_pos": cond["robot0_eef_pos"],
            "robot0_eef_quat": cond["robot0_eef_quat"],
            "robot0_gripper_qpos": cond["robot0_gripper_qpos"],
            "robot0_eye_in_hand_image": cond["robot0_eye_in_hand_image"],
        }

        global_cond = self.obs_encoder(nobs)
        global_cond = rearrange(global_cond, "b t d -> b (t d)")

        local_cond = None

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
 
