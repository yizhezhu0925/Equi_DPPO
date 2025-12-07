"""
Utility functions to convert rot6d actions back to axis-angle for environment execution.

CRITICAL: rot6d format is R[:3,:2].reshape(6) in ROW-MAJOR order:
    [R00, R01, R10, R11, R20, R21]
    
This means:
    - First column (indices 0,2,4): [R00, R10, R20]
    - Second column (indices 1,3,5): [R01, R11, R21]
"""

import torch
import numpy as np


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    
    rot6d: (..., 6) - R[:3,:2].reshape(6) in row-major = [R00, R01, R10, R11, R20, R21]
    Returns: (..., 3, 3) rotation matrix
    
    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrix.
    """
    # Extract columns from row-major flattened format
    # First column: [R00, R10, R20] = indices [0, 2, 4]
    # Second column: [R01, R11, R21] = indices [1, 3, 5]
    a1 = rot6d[..., 0::2]  # First column
    a2 = rot6d[..., 1::2]  # Second column
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack as columns to form rotation matrix
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_axis_angle(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.
    
    mat: (..., 3, 3) rotation matrix
    Returns: (..., 3) axis-angle vector (axis * angle)
    
    Uses the formula: R - R^T = 2*sin(θ)*[k]_× where [k]_× is skew-symmetric matrix of axis k
    """
    batch_shape = mat.shape[:-2]
    mat_flat = mat.reshape(-1, 3, 3)
    
    # Compute angle from trace: tr(R) = 1 + 2*cos(θ)
    trace = mat_flat[:, 0, 0] + mat_flat[:, 1, 1] + mat_flat[:, 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1, 1)
    angle = torch.acos(cos_angle)
    
    # Extract axis from skew-symmetric part: (R - R^T) / 2 = sin(θ) * [k]_×
    # The vector form of skew-symmetric [k]_× is [k_x, k_y, k_z]
    # From R - R^T, we get: [R32-R23, R13-R31, R21-R12] = 2*sin(θ)*[k_x, k_y, k_z]
    skew_vec = torch.stack([
        mat_flat[:, 2, 1] - mat_flat[:, 1, 2],  # R32 - R23
        mat_flat[:, 0, 2] - mat_flat[:, 2, 0],  # R13 - R31
        mat_flat[:, 1, 0] - mat_flat[:, 0, 1],  # R21 - R12
    ], dim=-1)
    
    # skew_vec = 2 * sin(angle) * axis
    sin_angle = torch.sin(angle)
    
    # For small angles, use L'Hopital: axis ≈ skew_vec / 2
    # For normal angles: axis = skew_vec / (2 * sin(angle))
    small_angle_mask = sin_angle.abs() < 1e-6
    
    # Avoid division by zero
    safe_sin = torch.where(small_angle_mask, torch.ones_like(sin_angle), sin_angle)
    axis = skew_vec / (2 * safe_sin.unsqueeze(-1) + 1e-8)
    
    # Normalize axis (should already be unit length, but ensure numerical stability)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)
    
    # Handle small angles: result is approximately zero
    axis = torch.where(small_angle_mask.unsqueeze(-1), torch.zeros_like(axis), axis)
    angle = torch.where(small_angle_mask, torch.zeros_like(angle), angle)
    
    # Handle angle = π case (axis from skew part is zero, need diagonal)
    # When angle ≈ π, we need to extract axis from R + I
    near_pi_mask = (angle - np.pi).abs() < 1e-4
    if near_pi_mask.any():
        # R = I + 2*k*k^T - 2*I when θ=π, so R + I = 2*k*k^T
        # Diagonal of R+I gives 2*k_i^2, take sqrt
        diag = torch.stack([
            mat_flat[:, 0, 0] + 1,
            mat_flat[:, 1, 1] + 1,
            mat_flat[:, 2, 2] + 1
        ], dim=-1) / 2  # k_i^2
        diag = torch.clamp(diag, min=0)  # Ensure non-negative before sqrt
        axis_pi = torch.sqrt(diag)
        
        # Determine signs from off-diagonal elements
        # R_ij = 2*k_i*k_j for i≠j when θ=π
        sign_xy = torch.sign(mat_flat[:, 0, 1] + mat_flat[:, 1, 0])
        sign_xz = torch.sign(mat_flat[:, 0, 2] + mat_flat[:, 2, 0])
        
        # Fix signs relative to x component (assume x positive)
        axis_pi[:, 1] = axis_pi[:, 1] * torch.where(sign_xy >= 0, 
                                                      torch.ones_like(sign_xy), 
                                                      -torch.ones_like(sign_xy))
        axis_pi[:, 2] = axis_pi[:, 2] * torch.where(sign_xz >= 0, 
                                                      torch.ones_like(sign_xz), 
                                                      -torch.ones_like(sign_xz))
        
        # Normalize
        axis_pi = axis_pi / (torch.norm(axis_pi, dim=-1, keepdim=True) + 1e-8)
        
        # Use pi-derived axis where needed
        axis = torch.where(near_pi_mask.unsqueeze(-1), axis_pi, axis)
    
    result = axis * angle.unsqueeze(-1)
    return result.reshape(*batch_shape, 3)


def convert_rot6d_action_to_axis_angle(action_rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 10D rot6d action to 7D axis-angle action.
    
    Input:  (..., 10) - [xyz(3), rot6d(6), gripper(1)]
    Output: (..., 7)  - [xyz(3), axis_angle(3), gripper(1)]
    """
    xyz = action_rot6d[..., :3]
    rot6d = action_rot6d[..., 3:9]
    gripper = action_rot6d[..., 9:]
    
    # Convert rot6d -> matrix -> axis-angle
    rot_mat = rot6d_to_matrix(rot6d)
    axis_angle = matrix_to_axis_angle(rot_mat)
    
    return torch.cat([xyz, axis_angle, gripper], dim=-1)


# Numpy versions for direct use
def convert_rot6d_action_to_axis_angle_np(action_rot6d: np.ndarray) -> np.ndarray:
    """Numpy wrapper for action conversion."""
    action_tensor = torch.from_numpy(action_rot6d).float()
    result = convert_rot6d_action_to_axis_angle(action_tensor)
    return result.numpy()


# ============== Verification ==============

def verify_round_trip():
    """Quick self-test for the conversion functions."""
    print("Running rot6d conversion self-test...")
    
    # Test with random axis-angle vectors
    torch.manual_seed(42)
    aa_original = torch.randn(100, 3) * 0.5
    
    # Forward: axis-angle → matrix → rot6d
    def axis_angle_to_matrix(aa):
        angle = torch.linalg.norm(aa, dim=-1, keepdim=True)
        axis = aa / (angle + 1e-8)
        x, y, z = axis.unbind(-1)
        zeros = torch.zeros_like(x)
        K = torch.stack([
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ], dim=-2)
        I = torch.eye(3).expand(*aa.shape[:-1], 3, 3)
        ang = angle.unsqueeze(-1)
        R = I + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)
        return torch.where(ang.abs() < 1e-8, I, R)
    
    def matrix_to_rot6d(mat):
        return mat[..., :3, :2].reshape(*mat.shape[:-2], 6)
    
    mat = axis_angle_to_matrix(aa_original)
    rot6d = matrix_to_rot6d(mat)
    
    # Backward: rot6d → matrix → axis-angle (using our functions)
    mat_rec = rot6d_to_matrix(rot6d)
    aa_rec = matrix_to_axis_angle(mat_rec)
    
    # Check error
    error = torch.abs(aa_original - aa_rec).max().item()
    print(f"  Max round-trip error: {error:.2e}")
    print(f"  Status: {'PASS ✓' if error < 1e-5 else 'FAIL ✗'}")
    return error < 1e-5


if __name__ == "__main__":
    verify_round_trip()