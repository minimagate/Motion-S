"""Rigid, per-trajectory normalization for position/time sequences."""

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryTransform:
    """World-to-local transform produced by :func:`normalize_trajectory`.

    ``rotation`` maps world row-vectors to the local frame. ``origin`` and
    ``scale`` have shape ``[B, 1, D]``; time fields have shape ``[B, 1]``.
    """

    origin: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor
    time_origin: torch.Tensor
    time_scale: torch.Tensor

    def inverse_positions(self, positions: torch.Tensor) -> torch.Tensor:
        scale, origin, rotation = self._expand(positions.ndim)
        return (positions * scale) @ rotation.transpose(-1, -2) + origin

    def inverse_times(self, times: torch.Tensor) -> torch.Tensor:
        return times * self.time_scale + self.time_origin

    def scale_std(self, std: torch.Tensor) -> torch.Tensor:
        """Convert local diagonal standard deviations to world-axis values."""
        # A rotated diagonal covariance is generally not diagonal. Return its
        # world-axis marginal standard deviations.
        scale, _, rotation = self._expand(std.ndim)
        covariance = (std * scale).square() @ rotation.square().transpose(-1, -2)
        return covariance.clamp_min(0).sqrt()

    def _expand(self, ndim: int):
        extra = ndim - 3
        scale, origin = self.scale, self.origin
        rotation = self.rotation
        for _ in range(extra):
            scale, origin = scale.unsqueeze(1), origin.unsqueeze(1)
            rotation = rotation.unsqueeze(1)
        return scale, origin, rotation


def normalize_trajectory(
    positions: torch.Tensor,
    timestamps: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, TrajectoryTransform]:
    """Normalize ``[B,T,D]`` positions and ``[B,T]`` times (D is 2 or 3).

    The latest valid point becomes the origin, the recent velocity defines the
    forward axis, and the recent path extent sets a robust spatial scale.
    Invalid entries are zeroed. The returned transform can invert predictions.
    """
    if positions.ndim != 3 or positions.shape[-1] not in (2, 3):
        raise ValueError("positions must have shape [B,T,2] or [B,T,3]")
    if timestamps.shape != positions.shape[:2] or mask.shape != positions.shape[:2]:
        raise ValueError("timestamps and mask must have shape [B,T]")
    mask = mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("each sequence must contain at least one valid point")
    b, t, d = positions.shape
    device, dtype = positions.device, positions.dtype
    idxs = torch.arange(t, device=device).expand(b, t)
    last_idx = idxs.masked_fill(~mask, -1).max(dim=1).values
    batch = torch.arange(b, device=device)
    origin = positions[batch, last_idx].unsqueeze(1)

    # Find the first valid sample preceding the endpoint for a stable direction.
    previous = mask & (idxs < last_idx[:, None])
    prev_idx = idxs.masked_fill(~previous, -1).max(dim=1).values
    has_previous = prev_idx >= 0
    safe_prev = prev_idx.clamp_min(0)
    velocity = positions[batch, last_idx] - positions[batch, safe_prev]
    velocity = torch.where(has_previous[:, None], velocity, torch.zeros_like(velocity))

    eye = torch.eye(d, device=device, dtype=dtype).expand(b, d, d).clone()
    if d == 2:
        angle = torch.atan2(velocity[:, 1], velocity[:, 0])
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.zeros_like(eye)
        rotation[:, 0, 0], rotation[:, 0, 1] = c, -s
        rotation[:, 1, 0], rotation[:, 1, 1] = s, c
        rotation = torch.where((velocity.norm(dim=-1) > eps)[:, None, None], rotation, eye)
    else:
        direction = velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(eps)
        # Minimal rotation taking the velocity direction to local +x.
        target = torch.zeros_like(direction)
        target[:, 0] = 1
        v = torch.linalg.cross(direction, target, dim=-1)
        dot = (direction * target).sum(-1).clamp(-1, 1)
        vx = torch.zeros_like(eye)
        vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
        vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
        vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
        rot = eye + vx + (vx @ vx) / (1 + dot).clamp_min(eps)[:, None, None]
        # Antiparallel direction: deterministic 180 degree rotation about y.
        opposite = eye.clone()
        opposite[:, 0, 0], opposite[:, 2, 2] = -1, -1
        rot = torch.where((dot < -1 + eps)[:, None, None], opposite, rot)
        usable = velocity.norm(dim=-1) > eps
        rotation = torch.where(usable[:, None, None], rot.transpose(-1, -2), eye)

    local = (positions - origin) @ rotation
    distances = local.norm(dim=-1).masked_fill(~mask, 0)
    scale = (distances.sum(dim=1) / mask.sum(dim=1).clamp_min(1)).clamp_min(eps)
    local = local / scale[:, None, None]
    local = local.masked_fill(~mask[..., None], 0)

    time_origin = timestamps[batch, last_idx].unsqueeze(1)
    first_idx = idxs.masked_fill(~mask, t).min(dim=1).values
    time_span = timestamps[batch, last_idx] - timestamps[batch, first_idx]
    time_scale = time_span.abs().clamp_min(eps).unsqueeze(1)
    local_times = (timestamps - time_origin) / time_scale
    local_times = local_times.masked_fill(~mask, 0)
    transform = TrajectoryTransform(origin, rotation, scale[:, None, None], time_origin, time_scale)
    return local, local_times, transform
