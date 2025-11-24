-# DPPO extension: equivariant perception and Equi-UNet
-
-This note summarizes the new equivariant components added on top of vanilla DPPO and how they plug into the existing pipeline

## New encoder modules
-`model/common/Equi_encoder.py`: Equivariant observation encoder (EquivariantResEncoder96Cyclic). 
-`model/common/Equi_obs_encoder.py`: Takes multi cameras plus end-effector/gripper state; outputs `(B, T, n_hidden * N)` equivariant features (default `n_hidden=128, N=8` → 1024-d).

## New common modules
-`model/common/rotation_transformer.py`
-`tensor_util.py`
-`crop_randomizer.py`

## New Unet modules
-`model/diffusion/Equi_Unet_Lab.py`: Equivariant ConditionalUnet1D (1D conv + FiLM conditioning).
-`model/diffusion/Equi_Unet.py`: `EquiDiffusionUNet` wrapper that:
-  - Packs actions into geometric tensors matching ESCNN FieldType.
-  - Calls `EquivariantObsEnc` to produce `global_cond`, feeds into `ConditionalUnet1D`.


## TODO
The configuration parameters in model/diffusion/Equi_Unet.py are incomplete. Refer to model/diffusion/unet.py to adjust it into a trainable format.
