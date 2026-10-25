"""Compact causal TCN for multimodal probabilistic trajectory forecasting."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .normalization import TrajectoryTransform, normalize_trajectory


@dataclass
class TrajectoryDistribution:
    """Forecast in world coordinates: component means/scales and probabilities."""

    means: torch.Tensor  # [B,K,H,D]
    scales: torch.Tensor  # [B,K,H,D], aleatoric diagonal std
    logits: torch.Tensor  # [B,K]
    transform: TrajectoryTransform
    local_means: torch.Tensor
    local_scales: torch.Tensor
    measurement_noise_std: torch.Tensor  # [B], estimated observation noise in world units

    @property
    def weights(self) -> torch.Tensor:
        return self.logits.softmax(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        return (self.weights[:, :, None, None] * self.means).sum(dim=1)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Draw samples shaped ``[B,num_samples,H,D]`` from mixture + noise."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        b, k, h, d = self.means.shape
        component = torch.multinomial(self.weights, num_samples, replacement=True)
        gather = component[:, :, None, None].expand(b, num_samples, h, d)
        means = self.means.transpose(1, 2).unsqueeze(1).expand(b, num_samples, h, k, d)
        scales = self.scales.transpose(1, 2).unsqueeze(1).expand_as(means)
        means = means.gather(3, gather.unsqueeze(3)).squeeze(3)
        scales = scales.gather(3, gather.unsqueeze(3)).squeeze(3)
        return means + torch.randn_like(means) * scales

    def nll(self, targets: torch.Tensor) -> torch.Tensor:
        """Mean Gaussian-mixture negative log likelihood in world coordinates."""
        if targets.shape != self.means.shape[:1] + self.means.shape[2:]:
            raise ValueError("targets must have shape [B,H,D]")
        # Evaluate the density in the model's normalized frame. Rotation has
        # unit determinant; spatial scale contributes the change-of-variables.
        local_targets = (targets - self.transform.origin) @ self.transform.rotation
        local_targets = local_targets / self.transform.scale
        logp = gaussian_mixture_log_prob(
            local_targets, self.local_means, self.local_scales, self.logits
        )
        # The density is over H*D coordinates and this transform uses one
        # isotropic spatial scale per sequence.
        log_scale = self.transform.scale[:, 0, 0].log()
        log_jacobian = targets.shape[1] * targets.shape[2] * log_scale
        return -(logp - log_jacobian).mean()


def gaussian_mixture_log_prob(
    targets: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Stable log density; inputs are ``[B,H,D]``, ``[B,K,H,D]``, ``[B,K,H,D]``."""
    if targets.ndim != 3 or means.ndim != 4 or means.shape != scales.shape:
        raise ValueError("invalid target, mean, or scale shapes")
    if (
        targets.shape != (means.shape[0], means.shape[2], means.shape[3])
        or logits.shape != means.shape[:2]
    ):
        raise ValueError("incompatible mixture shapes")
    safe_scale = scales.clamp_min(1e-6)
    z = (targets[:, None] - means) / safe_scale
    component = -0.5 * (
        z.square()
        + 2 * safe_scale.log()
        + torch.log(torch.tensor(2 * torch.pi, device=means.device, dtype=means.dtype))
    )
    component = component.sum(dim=(-1, -2)) + logits.log_softmax(dim=-1)
    return torch.logsumexp(component, dim=1)


class _ResidualCausalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(width, width, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask[:, None].to(x.dtype)
        y = self.conv1(F.pad(x, (self.left_pad, 0))).transpose(1, 2)
        y = self.dropout(F.gelu(self.norm1(y))).transpose(1, 2) * valid
        y = self.conv2(F.pad(y, (self.left_pad, 0))).transpose(1, 2)
        y = self.norm2(y).transpose(1, 2)
        return F.gelu(x + self.dropout(y)) * valid


class CausalTCNEncoder(nn.Module):
    """Temporal encoder interface shared by potential alternative backbones.

    Inputs are ``features[B,T,C]`` and ``mask[B,T]``; outputs are
    ``states[B,T,output_size]``. A Transformer can replace this module without
    changing trajectory normalization or the probabilistic prediction head.
    """

    def __init__(
        self,
        input_size: int,
        width: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_size = width
        self.input = nn.Conv1d(input_size, width, 1)
        self.blocks = nn.ModuleList(
            [
                _ResidualCausalBlock(width, kernel_size, 2 ** (index % 5), dropout)
                for index in range(depth)
            ]
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input(features.transpose(1, 2)) * mask[:, None].to(features.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return x.transpose(1, 2)


class TrajectoryTCN(nn.Module):
    """Forecast a future horizon from padded 2D or 3D position histories.

    ``forward(positions, timestamps, mask)`` returns a
    :class:`TrajectoryDistribution` with world-coordinate means and aleatoric
    standard deviations. Mixture components represent trajectory ambiguity.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        horizon: int = 12,
        components: int = 5,
        width: int = 192,
        depth: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        min_scale: float = 1e-3,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        if spatial_dim not in (2, 3) or horizon < 1 or components < 1:
            raise ValueError("spatial_dim must be 2/3 and horizon/components positive")
        if width % 8:
            raise ValueError("width must be divisible by 8")
        self.spatial_dim, self.horizon, self.components = spatial_dim, horizon, components
        self.min_scale = min_scale
        self.encoder = encoder or CausalTCNEncoder(
            spatial_dim + 1, width, depth, kernel_size, dropout
        )
        encoder_width = getattr(self.encoder, "output_size", None)
        if not isinstance(encoder_width, int):
            raise ValueError("encoder must expose an integer output_size")
        self.head = nn.Sequential(
            nn.Linear(encoder_width, width),
            nn.GELU(),
            nn.Linear(width, components * (2 * horizon * spatial_dim + 1)),
        )
        self.noise_head = nn.Linear(encoder_width, 1)
        # Zero initialization preserves exact behavior for legacy checkpoints
        # that did not condition on an explicit future sampling schedule.
        self.future_time_projection = nn.Linear(horizon, encoder_width, bias=False)
        nn.init.zeros_(self.future_time_projection.weight)

    def forward(
        self,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        future_timestamps: torch.Tensor | None = None,
    ) -> TrajectoryDistribution:
        if positions.ndim != 3 or positions.shape[-1] != self.spatial_dim:
            raise ValueError(f"positions must have shape [B,T,{self.spatial_dim}]")
        local, local_times, transform = normalize_trajectory(positions, timestamps, mask)
        mask = mask.bool()
        features = torch.cat((local, local_times[..., None]), dim=-1)
        states = self.encoder(features, mask)
        # Last valid index, independent of the amount of right padding.
        idx = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = idx.masked_fill(~mask, -1).max(dim=1).values
        context = states[torch.arange(states.shape[0], device=states.device), last]
        if future_timestamps is None:
            previous = (last - 1).clamp_min(0)
            batch = torch.arange(states.shape[0], device=states.device)
            cadence = (timestamps[batch, last] - timestamps[batch, previous]).clamp_min(1e-6)
            steps = torch.arange(1, self.horizon + 1, device=states.device, dtype=timestamps.dtype)
            future_timestamps = timestamps[batch, last, None] + cadence[:, None] * steps
        if future_timestamps.shape != (positions.shape[0], self.horizon):
            raise ValueError(f"future_timestamps must have shape [B,{self.horizon}]")
        local_future_times = (future_timestamps - transform.time_origin) / transform.time_scale
        context = context + self.future_time_projection(local_future_times)
        raw = self.head(context).view(-1, self.components, 2 * self.horizon * self.spatial_dim + 1)
        n = self.horizon * self.spatial_dim
        means_local = raw[..., :n].view(-1, self.components, self.horizon, self.spatial_dim)
        scales_local = F.softplus(raw[..., n : 2 * n]).view_as(means_local) + self.min_scale
        logits = raw[..., -1]
        measurement_noise_local = F.softplus(self.noise_head(context).squeeze(-1)) + self.min_scale
        measurement_noise_std = measurement_noise_local * transform.scale[:, 0, 0]
        means = transform.inverse_positions(means_local)
        # Marginal world-axis scales after applying the inverse rotation.
        scales = transform.scale_std(scales_local)
        return TrajectoryDistribution(
            means,
            scales,
            logits,
            transform,
            means_local,
            scales_local,
            measurement_noise_std,
        )


__all__ = [
    "CausalTCNEncoder",
    "TrajectoryDistribution",
    "TrajectoryTCN",
    "gaussian_mixture_log_prob",
]
