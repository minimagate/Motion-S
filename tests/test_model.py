import torch

from motions.model import TrajectoryTCN
from motions.normalization import normalize_trajectory


def _batch(dim=2):
    torch.manual_seed(7)
    positions = torch.randn(3, 9, dim).cumsum(dim=1)
    timestamps = torch.arange(9, dtype=torch.float32).expand(3, 9) * 0.2
    mask = torch.tensor([[1] * 9, [1] * 6 + [0] * 3, [1] * 4 + [0] * 5], dtype=torch.bool)
    return positions, timestamps, mask


def test_normalization_round_trip_2d_and_3d():
    for dim in (2, 3):
        positions, times, mask = _batch(dim)
        norm, norm_times, transform = normalize_trajectory(positions, times, mask)
        restored = transform.inverse_positions(norm)
        restored_times = transform.inverse_times(norm_times)
        assert torch.allclose(restored[mask], positions[mask], atol=1e-5)
        assert torch.allclose(restored_times[mask], times[mask], atol=1e-5)
        assert torch.count_nonzero(norm[~mask]) == 0


def test_model_output_shapes_masks_and_finite_loss():
    positions, times, mask = _batch(2)
    model = TrajectoryTCN(horizon=5, components=3, width=32, depth=2, dropout=0)
    result = model(positions, times, mask)
    assert result.means.shape == (3, 3, 5, 2)
    assert result.scales.shape == result.means.shape
    assert result.logits.shape == (3, 3)
    assert result.measurement_noise_std.shape == (3,)
    assert result.mean.shape == (3, 5, 2)
    assert result.sample(4).shape == (3, 4, 5, 2)
    target = torch.randn(3, 5, 2)
    loss = result.nll(target)
    assert torch.isfinite(loss)
    loss.backward()


def test_default_model_has_compact_parameter_count():
    model = TrajectoryTCN()
    count = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= count <= 5_000_000


def test_padding_does_not_change_forecast():
    positions, times, mask = _batch(3)
    model = TrajectoryTCN(
        spatial_dim=3, horizon=2, components=2, width=32, depth=2, dropout=0
    ).eval()
    first = model(positions[1:2, :6], times[1:2, :6], mask[1:2, :6])
    second = model(positions[1:2], times[1:2], mask[1:2])
    assert torch.allclose(first.means, second.means, atol=1e-5)
