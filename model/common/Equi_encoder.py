from escnn import gspaces, nn
import torch
import torch.nn.functional as F




class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize,
            ),
            nn.ReLU(feat_type_hid, inplace=True),
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_hid,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                initialize=initialize,
            ),
        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=1, stride=stride, bias=False, initialize=initialize),
            )

    def forward(self, xx: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class EquivariantResEncoder96Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1, 
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            # 96 -> 48
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),

            # Block 2: 48x48 -> 24x24
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=True),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),

            # Block 3: 24x24 -> 12x12
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=True),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),

            # Block 4: 12x12 -> 6x6
            EquiResBlock(self.group, n_out // 2, n_out, initialize=True),
            EquiResBlock(self.group, n_out, n_out, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 2),

            # Block 5: 6x6 -> 3x3
            EquiResBlock(self.group, n_out, n_out, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 2),

            # Final Block: 3x3 -> 1x1

            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # Output: 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:

            if type(x) is torch.Tensor:
                x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))

            return self.conv(x)
    
class EquivariantResEncoder96CyclicLite(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)
        rep = self.group.regular_repr
        
        self.conv = torch.nn.Sequential(
            # 96
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [rep]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [rep]), inplace=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [rep]), 2),
            # 48
            
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=initialize),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [rep]), 2),
            # 24
            
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=initialize),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [rep]), 2),
            # 12
            
            EquiResBlock(self.group, n_out // 2, n_out, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [rep]), 3),
            # 4
            
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [rep]),
                nn.FieldType(self.group, n_out * [rep]),
                kernel_size=4,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [rep]), inplace=True),
            # 1
        )
    
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
# testq
if __name__ == "__main__":
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Original
    print("=== EquivariantResEncoder96Cyclic (Original) ===")
    encoder_orig = EquivariantResEncoder96Cyclic(obs_channel=3, n_out=128, N=8).to(device)
    x = torch.randn(2, 3, 96, 96, device=device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    out = encoder_orig(x)
    loss = out.tensor.sum()
    loss.backward()
    
    if device.type == "cuda":
        mem_orig = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak memory (with grad): {mem_orig:.1f} MB")
    print(f"Output: {out.tensor.shape}")
    
    del encoder_orig, out, loss
    torch.cuda.empty_cache()
    
    # Lite version
    print("\n=== EquivariantResEncoder96CyclicLite (Lite) ===")
    encoder_lite = EquivariantResEncoder96CyclicLite(obs_channel=3, n_out=128, N=8).to(device)
    x = torch.randn(2, 3, 96, 96, device=device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    out = encoder_lite(x)
    loss = out.tensor.sum()
    loss.backward()
    
    if device.type == "cuda":
        mem_lite = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak memory (with grad): {mem_lite:.1f} MB")
    print(f"Output: {out.tensor.shape}")
    
    # Comparison
    if device.type == "cuda":
        print(f"\n=== Comparison ===")
        print(f"Memory saved: {mem_orig - mem_lite:.1f} MB ({(1 - mem_lite/mem_orig)*100:.1f}%)")