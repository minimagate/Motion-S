"""Leakage-safe loading and windowing for real trajectory corpora."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

RealSplit = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Track:
    """One actor trajectory in world coordinates."""

    scene: str
    actor_id: str
    timestamps: np.ndarray
    positions: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.scene}:{self.actor_id}"


def _stable_order(value: str, seed: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def load_eth(root: str | Path) -> list[Track]:
    """Load official ETH Walking Pedestrians ``obsmat.txt`` annotations.

    The archive contains ETH and Hotel scenes. Positions use the ground-plane
    ``pos_x`` and ``pos_y`` columns in metres. The scene metadata states that
    annotations are sampled at 2.5 Hz; frame-number differences are converted
    using each scene's native annotation stride.
    """
    root = Path(root)
    scene_steps = {"seq_eth": 6.0, "seq_hotel": 10.0}
    tracks: list[Track] = []
    for scene, frame_step in scene_steps.items():
        path = root / scene / "obsmat.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing ETH annotations: {path}")
        rows = np.loadtxt(path, dtype=np.float64)
        for actor in np.unique(rows[:, 1]):
            actor_rows = rows[rows[:, 1] == actor]
            actor_rows = actor_rows[np.argsort(actor_rows[:, 0])]
            timestamps = (actor_rows[:, 0] - actor_rows[0, 0]) * (0.4 / frame_step)
            positions = actor_rows[:, [2, 4]]
            finite = np.isfinite(timestamps) & np.isfinite(positions).all(axis=1)
            timestamps, positions = timestamps[finite], positions[finite]
            if len(timestamps) >= 2 and np.all(np.diff(timestamps) > 0):
                tracks.append(
                    Track(
                        scene=scene.removeprefix("seq_"),
                        actor_id=str(int(actor)),
                        timestamps=timestamps.astype(np.float32),
                        positions=positions.astype(np.float32),
                    )
                )
    return tracks


def split_tracks(
    tracks: list[Track],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[RealSplit, list[Track]]:
    """Assign complete actor tracks to splits, stratified by scene."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    total = sum(ratios)
    ratios = tuple(value / total for value in ratios)
    result: dict[RealSplit, list[Track]] = {"train": [], "val": [], "test": []}
    for scene in sorted({track.scene for track in tracks}):
        scene_tracks = sorted(
            (track for track in tracks if track.scene == scene),
            key=lambda track: _stable_order(track.key, seed),
        )
        train_end = round(len(scene_tracks) * ratios[0])
        val_end = train_end + round(len(scene_tracks) * ratios[1])
        result["train"].extend(scene_tracks[:train_end])
        result["val"].extend(scene_tracks[train_end:val_end])
        result["test"].extend(scene_tracks[val_end:])
    return result


def _contiguous_segments(track: Track, max_gap: float) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(track.timestamps) > max_gap) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(track.timestamps)]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


class RealTrajectoryDataset(Dataset[dict[str, object]]):
    """Deterministic windows whose actors never cross data splits."""

    def __init__(
        self,
        tracks: list[Track],
        *,
        observations: int = 8,
        horizon: int = 12,
        stride: int = 4,
        max_gap: float = 0.6,
        max_tracks: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        if observations < 2 or horizon < 1 or stride < 1:
            raise ValueError("observations >= 2 and positive horizon/stride are required")
        eligible = [
            track
            for track in tracks
            if any(
                end - start >= observations + horizon
                for start, end in _contiguous_segments(track, max_gap)
            )
        ]
        eligible.sort(key=lambda track: _stable_order(track.key, subset_seed))
        if max_tracks is not None:
            if max_tracks < 1:
                raise ValueError("max_tracks must be positive")
            eligible = eligible[:max_tracks]
        self.tracks = eligible
        self.observations = observations
        self.horizon = horizon
        self.dim = 2
        self.windows: list[tuple[int, int]] = []
        length = observations + horizon
        for track_index, track in enumerate(self.tracks):
            for start, end in _contiguous_segments(track, max_gap):
                for offset in range(start, end - length + 1, stride):
                    self.windows.append((track_index, offset))
        if not self.windows:
            raise ValueError("no complete trajectory windows are available")

    @property
    def track_keys(self) -> set[str]:
        return {track.key for track in self.tracks}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        track_index, start = self.windows[index]
        track = self.tracks[track_index]
        split = start + self.observations
        end = split + self.horizon
        timestamps = torch.from_numpy(track.timestamps[start:split].copy())
        future_timestamps = torch.from_numpy(track.timestamps[split:end].copy())
        observed = torch.from_numpy(track.positions[start:split].copy())
        future = torch.from_numpy(track.positions[split:end].copy())
        return {
            "timestamps": timestamps,
            "future_timestamps": future_timestamps,
            "observed": observed,
            "future": future,
            "family": "real",
            "noise_std": torch.tensor(float("nan"), dtype=torch.float32),
            "track_id": track.key,
            "scene": track.scene,
        }


def make_eth_splits(
    root: str | Path,
    *,
    seed: int,
    observations: int = 8,
    horizon: int = 12,
    stride: int = 4,
) -> dict[RealSplit, RealTrajectoryDataset]:
    split = split_tracks(load_eth(root), seed=seed)
    return {
        name: RealTrajectoryDataset(
            tracks,
            observations=observations,
            horizon=horizon,
            stride=stride,
            subset_seed=seed,
        )
        for name, tracks in split.items()
    }


__all__ = [
    "RealTrajectoryDataset",
    "Track",
    "load_eth",
    "make_eth_splits",
    "split_tracks",
]
