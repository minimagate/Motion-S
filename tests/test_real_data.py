import numpy as np

from motions.real_data import RealTrajectoryDataset, Track, split_tracks


def _tracks(count=30):
    return [
        Track(
            scene="a" if index % 2 else "b",
            actor_id=str(index),
            timestamps=np.arange(24, dtype=np.float32) * 0.4,
            positions=np.stack(
                (np.arange(24, dtype=np.float32), np.full(24, index, dtype=np.float32)), axis=1
            ),
        )
        for index in range(count)
    ]


def test_track_splits_are_disjoint_deterministic_and_scene_stratified():
    first = split_tracks(_tracks(), seed=4)
    second = split_tracks(_tracks(), seed=4)
    keys = {name: {track.key for track in tracks} for name, tracks in first.items()}
    assert keys == {name: {track.key for track in tracks} for name, tracks in second.items()}
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"])
    assert {track.scene for track in first["test"]} == {"a", "b"}


def test_windows_never_cross_gaps_or_actor_boundaries():
    track = _tracks(1)[0]
    timestamps = track.timestamps.copy()
    timestamps[12:] += 2.0
    dataset = RealTrajectoryDataset(
        [Track(track.scene, track.actor_id, timestamps, track.positions)],
        observations=4,
        horizon=4,
        stride=1,
    )
    for item in dataset:
        all_times = np.concatenate((item["timestamps"].numpy(), item["future_timestamps"].numpy()))
        assert np.diff(all_times).max() <= 0.6
