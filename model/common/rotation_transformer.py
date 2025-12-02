import torch
import numpy as np
from typing import Union


def _normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)


def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion (w, x, y, z) to rotation matrix.
    Expected shape (..., 4) -> (..., 3, 3)
    """
    w, x, y, z = quat.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot = torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _matrix_to_rotation6d(mat: torch.Tensor) -> torch.Tensor:
    """Take first two columns of rotation matrix -> 6D representation."""
    return mat[..., :3, :2].reshape(*mat.shape[:-2], 6)


class RotationTransformer:
    """
    Minimal transformer supporting quaternion -> rotation_6d conversion,
    which is the only path used in this codebase.
    """

    def __init__(self, from_rep="quaternion", to_rep="rotation_6d", **kwargs):
        assert from_rep == "quaternion" and to_rep == "rotation_6d"

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # accepts numpy or torch, returns same type
        as_numpy = isinstance(x, np.ndarray)
        if as_numpy:
            x = torch.from_numpy(x)
        x = _normalize_quaternion(x)
        mat = _quat_to_matrix(x)
        rot6d = _matrix_to_rotation6d(mat)
        return rot6d.numpy() if as_numpy else rot6d
