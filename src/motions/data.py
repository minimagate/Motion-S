"""Procedural trajectory datasets for motion forecasting experiments.

All randomness is derived from a stable trajectory seed. Dataset indexing is
therefore deterministic and does not depend on worker order or global RNG state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

Family = Literal[
    "linear",
    "accelerated",
    "circular",
    "oscillatory",
    "ballistic",
    "stochastic",
    "goal-directed",
    "hybrid",
]
FAMILIES: tuple[Family, ...] = (
    "linear",
    "accelerated",
    "circular",
    "oscillatory",
    "ballistic",
    "stochastic",
    "goal-directed",
    "hybrid",
)
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitProfile:
    """Rules for a reproducible train/validation/test partition.

    ``held_out_families`` is applied to the test split. In ``unseen_parameters``
    test trajectories use a disjoint speed/acceleration/noise regime; in
    ``unseen_combinations`` withholds joint speed/noise regimes while retaining
    each marginal regime in training. These profiles are useful evaluation
    protocols, not claims of universal statistical independence.
    """

    name: str = "standard"
    held_out_families: tuple[Family, ...] = ()
    unseen_parameters: bool = False
    unseen_combinations: bool = False


SPLIT_PROFILES: dict[str, SplitProfile] = {
    "standard": SplitProfile(),
    "unseen_parameters": SplitProfile(name="unseen_parameters", unseen_parameters=True),
    "unseen_combinations": SplitProfile(name="unseen_combinations", unseen_combinations=True),
    "held_out_family": SplitProfile(name="held_out_family", held_out_families=("hybrid",)),
}

# Default research protocol: hybrid dynamics and two joint speed/noise regimes
# are absent from fitting data. Evaluation reconstructs each challenge in
# isolation so its meaning stays explicit.
RESEARCH_PROFILE = SplitProfile(
    name="research_train", held_out_families=("hybrid",), unseen_combinations=True
)


def get_split_profile(profile: str | SplitProfile) -> SplitProfile:
    if isinstance(profile, SplitProfile):
        return profile
    try:
        return SPLIT_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown split profile {profile!r}; choose from {tuple(SPLIT_PROFILES)}"
        ) from exc


def _seed(base: int, split: str, index: int) -> int:
    # Avoid Python's randomized hash(); stable across processes and versions.
    split_id = {"train": 1, "val": 2, "test": 3}[split]
    return (int(base) * 1_000_003 + split_id * 97_409 + int(index) * 65_537) % (2**63 - 1)


def _unit(generator: torch.Generator, dim: int) -> torch.Tensor:
    v = torch.randn(dim, generator=generator)
    return v / v.norm().clamp_min(1e-8)


def _trajectory(
    family: Family,
    *,
    dim: int,
    steps: int,
    dt: float,
    speed: float,
    acceleration: float,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a clean (steps, dim) trajectory, including its initial point."""
    origin = torch.randn(dim, generator=generator) * (0.5 * scale)
    direction = _unit(generator, dim)
    side = _unit(generator, dim)
    t = torch.arange(steps, dtype=torch.float32) * dt
    phase = float(torch.rand((), generator=generator) * (2 * torch.pi))

    if family == "linear":
        pos = origin + (t * speed)[:, None] * direction
    elif family == "accelerated":
        pos = origin + (t * speed + 0.5 * acceleration * t.square())[:, None] * direction
    elif family == "circular":
        # Embed a randomly oriented 2D plane in either 2D or 3D.
        u = direction
        v = side - torch.dot(side, u) * u
        v = v / v.norm().clamp_min(1e-8)
        radius = scale * (0.35 + float(torch.rand((), generator=generator)))
        angle = phase + (speed / max(radius, 1e-5)) * t
        pos = origin + radius * (torch.cos(angle)[:, None] * u + torch.sin(angle)[:, None] * v)
    elif family == "oscillatory":
        amp = scale * (0.2 + float(torch.rand((), generator=generator)))
        pos = (
            origin
            + (speed * t)[:, None] * direction
            + (amp * torch.sin(2.0 * t + phase))[:, None] * side
        )
    elif family == "ballistic":
        gravity = torch.zeros(dim)
        gravity[-1] = -9.81 * scale * 0.08
        launch = direction * speed
        pos = origin + t[:, None] * launch + 0.5 * t.square()[:, None] * gravity
    elif family == "goal-directed":
        goal = origin + _unit(generator, dim) * scale * (
            1.5 + float(torch.rand((), generator=generator))
        )
        p = origin
        points = [p]
        for _ in range(steps - 1):
            delta = goal - p
            p = p + (delta / delta.norm().clamp_min(1e-8)) * speed * dt
            points.append(p)
        pos = torch.stack(points)
    elif family == "stochastic":
        increments = torch.randn((steps - 1, dim), generator=generator) * (speed * dt)
        increments += direction * (speed * dt * 0.35)
        pos = torch.cat((origin[None], origin[None] + increments.cumsum(0)), dim=0)
    elif family == "hybrid":
        bend = max(1, steps // 2)
        pos = origin + (t * speed)[:, None] * direction
        pos = pos + (acceleration * 0.5 * t.square())[:, None] * side
        pos = pos + (scale * 0.15 * torch.sin(1.7 * t + phase))[:, None] * direction
        # A small velocity change gives the family a piecewise component.
        pos[bend:] += ((t[bend:] - t[bend : bend + 1]) * speed * 0.35)[:, None] * side
    else:  # pragma: no cover - guarded by public validation
        raise ValueError(f"Unsupported family {family!r}")
    return pos.to(torch.float32)


class TrajectoryDataset(Dataset[dict[str, object]]):
    """Lazy deterministic generator of 2D or 3D motion forecasting examples.

    Each item contains ``timestamps`` (observed timestamps), noisy ``observed``
    positions, clean ``future`` positions, ``family`` (string), and scalar
    ``noise_std``. Observation length is sampled per item. ``horizon`` is the
    number of future points; ``dt`` may be a positive scalar or inclusive range.
    """

    def __init__(
        self,
        size: int,
        *,
        split: SplitName = "train",
        seed: int = 0,
        dim: int = 2,
        obs_length: int | tuple[int, int] = (8, 16),
        horizon: int = 12,
        dt: float | tuple[float, float] = (0.05, 0.2),
        noise_std: float | tuple[float, float] = (0.0, 0.08),
        families: Iterable[Family] = FAMILIES,
        profile: str | SplitProfile = "standard",
    ) -> None:
        if size < 0 or dim not in (2, 3) or horizon < 1:
            raise ValueError("size must be nonnegative, dim must be 2 or 3, and horizon positive")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.size, self.split, self.seed, self.dim, self.horizon = (
            size,
            split,
            int(seed),
            dim,
            horizon,
        )
        self.obs_length = self._range(obs_length, "obs_length", integer=True)
        self.dt = self._range(dt, "dt")
        self.noise_range = self._range(noise_std, "noise_std")
        if self.obs_length[0] < 1 or self.dt[0] <= 0 or self.noise_range[0] < 0:
            raise ValueError(
                "observation length and dt must be positive; noise_std must be nonnegative"
            )
        self.profile = get_split_profile(profile)
        supplied = tuple(families)
        if not supplied or any(f not in FAMILIES for f in supplied):
            raise ValueError("families must contain known family names")
        if self.profile.held_out_families:
            held = set(self.profile.held_out_families)
            supplied = tuple(f for f in supplied if (f in held) == (split == "test"))
        self.families = supplied
        if not self.families:
            raise ValueError("profile removes every requested family from this split")

    @staticmethod
    def _range(value: float | int | tuple, name: str, integer: bool = False) -> tuple:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"{name} range must have two endpoints")
            lo, hi = value
        else:
            lo = hi = value
        if lo > hi:
            raise ValueError(f"{name} lower bound must not exceed upper bound")
        return (int(lo), int(hi)) if integer else (float(lo), float(hi))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        g = torch.Generator(device="cpu").manual_seed(_seed(self.seed, self.split, index))
        family = self.families[int(torch.randint(len(self.families), (), generator=g))]
        obs_n = int(torch.randint(self.obs_length[0], self.obs_length[1] + 1, (), generator=g))
        dt = self.dt[0] + float(torch.rand((), generator=g)) * (self.dt[1] - self.dt[0])
        noise_u = float(torch.rand((), generator=g))
        noise = self.noise_range[0] + noise_u * (self.noise_range[1] - self.noise_range[0])
        speed_range, accel_range = (0.25, 2.5), (0.05, 1.0)
        if self.split == "test" and self.profile.unseen_parameters:
            speed_range, accel_range = (2.7, 4.0), (1.2, 2.0)
            noise = min(self.noise_range[1] * 1.5, noise + self.noise_range[1] * 0.5)
        if self.profile.unseen_combinations:
            # Both speed and noise marginal regimes appear in training. Their
            # correlation is reversed at test, withholding only the pairing.
            high_speed = bool(torch.randint(2, (), generator=g))
            high_noise = high_speed if self.split == "test" else not high_speed
            speed_range = (1.5, 2.5) if high_speed else (0.25, 1.25)
            noise_mid = (self.noise_range[0] + self.noise_range[1]) / 2
            selected_noise_range = (
                (noise_mid, self.noise_range[1]) if high_noise else (self.noise_range[0], noise_mid)
            )
            noise = selected_noise_range[0] + noise_u * (
                selected_noise_range[1] - selected_noise_range[0]
            )
        speed = speed_range[0] + float(torch.rand((), generator=g)) * (
            speed_range[1] - speed_range[0]
        )
        accel = accel_range[0] + float(torch.rand((), generator=g)) * (
            accel_range[1] - accel_range[0]
        )
        scale = 0.5 + float(torch.rand((), generator=g)) * 2.0
        clean = _trajectory(
            family,
            dim=self.dim,
            steps=obs_n + self.horizon,
            dt=dt,
            speed=speed,
            acceleration=accel,
            scale=scale,
            generator=g,
        )
        observed = clean[:obs_n] + torch.randn((obs_n, self.dim), generator=g) * noise
        timestamps = torch.arange(obs_n, dtype=torch.float32) * dt
        future_timestamps = torch.arange(obs_n, obs_n + self.horizon, dtype=torch.float32) * dt
        return {
            "timestamps": timestamps,
            "future_timestamps": future_timestamps,
            "observed": observed,
            "future": clean[obs_n:],
            "family": family,
            "noise_std": torch.tensor(noise, dtype=torch.float32),
        }


def make_splits(
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 0,
    profile: str | SplitProfile = "standard",
    **dataset_kwargs: object,
) -> dict[SplitName, TrajectoryDataset]:
    """Construct explicit train/val/test datasets sharing configuration."""
    return {
        name: TrajectoryDataset(size, split=name, seed=seed, profile=profile, **dataset_kwargs)
        for name, size in (("train", train_size), ("val", val_size), ("test", test_size))
    }


def collate_trajectories(items: list[dict[str, object]]) -> dict[str, object]:
    """Right-pad variable histories and create an explicit validity mask."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    observed = [item["observed"] for item in items]
    timestamps = [item["timestamps"] for item in items]
    if not all(isinstance(value, torch.Tensor) for value in observed + timestamps):
        raise TypeError("observed positions and timestamps must be tensors")
    lengths = torch.tensor([value.shape[0] for value in observed], dtype=torch.long)
    batch_size, max_length = len(items), int(lengths.max())
    dim = observed[0].shape[-1]
    padded_positions = torch.zeros(batch_size, max_length, dim, dtype=torch.float32)
    padded_times = torch.zeros(batch_size, max_length, dtype=torch.float32)
    mask = torch.arange(max_length)[None, :] < lengths[:, None]
    for index, (positions, times) in enumerate(zip(observed, timestamps)):
        padded_positions[index, : lengths[index]] = positions
        padded_times[index, : lengths[index]] = times
    return {
        "observed": padded_positions,
        "timestamps": padded_times,
        "mask": mask,
        "future": torch.stack([item["future"] for item in items]),
        "future_timestamps": torch.stack([item["future_timestamps"] for item in items]),
        "noise_std": torch.stack([item["noise_std"] for item in items]),
        "family": [item["family"] for item in items],
    }


__all__ = [
    "FAMILIES",
    "SPLIT_PROFILES",
    "RESEARCH_PROFILE",
    "SplitProfile",
    "TrajectoryDataset",
    "collate_trajectories",
    "get_split_profile",
    "make_splits",
]
