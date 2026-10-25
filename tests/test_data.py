import pytest
import torch

from motions.data import FAMILIES, SPLIT_PROFILES, TrajectoryDataset, make_splits


def test_item_shapes_and_clean_future():
    ds = TrajectoryDataset(4, dim=3, obs_length=(5, 7), horizon=9, seed=4)
    item = ds[0]
    assert item["observed"].shape[1] == 3
    assert 5 <= item["observed"].shape[0] <= 7
    assert item["future"].shape == (9, 3)
    assert item["timestamps"].shape == (item["observed"].shape[0],)
    assert item["future_timestamps"].shape == (9,)
    assert item["family"] in FAMILIES
    assert item["noise_std"].ndim == 0


def test_index_is_deterministic_and_splits_have_disjoint_seed_streams():
    splits = make_splits(train_size=8, val_size=8, test_size=8, seed=11)
    a, b = splits["train"][3], splits["train"][3]
    assert torch.equal(a["observed"], b["observed"])
    assert not torch.equal(a["observed"], splits["val"][3]["observed"])


def test_profiles_are_explicit_and_hold_out_hybrid_from_test():
    assert {"standard", "unseen_parameters", "unseen_combinations", "held_out_family"} <= set(
        SPLIT_PROFILES
    )
    ds = TrajectoryDataset(100, split="test", seed=2, profile="held_out_family")
    assert all(ds[i]["family"] == "hybrid" for i in range(len(ds)))


def test_all_families_generate_finite_trajectories():
    # Pick one deterministic sample per family explicitly.
    for i, family in enumerate(FAMILIES):
        item = TrajectoryDataset(1, families=(family,), obs_length=6, horizon=4, seed=i)[0]
        assert item["family"] == family
        assert torch.isfinite(item["observed"]).all()
        assert torch.isfinite(item["future"]).all()


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        TrajectoryDataset(1, dim=4)
    with pytest.raises(ValueError):
        TrajectoryDataset(1, noise_std=-1)


def test_sampling_intervals_are_variable_and_increasing():
    item = TrajectoryDataset(1, obs_length=12, horizon=8, dt_jitter=0.5, seed=23)[0]
    timestamps = torch.cat((item["timestamps"], item["future_timestamps"]))
    intervals = torch.diff(timestamps)
    assert torch.all(intervals > 0)
    assert intervals.std() > 0
