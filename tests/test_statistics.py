from motions.statistics import cluster_bootstrap_ci95, mean_ci95


def test_seed_interval_and_cluster_bootstrap_are_deterministic():
    interval = mean_ci95([1.0, 2.0, 3.0])
    assert interval["mean"] == 2.0
    assert interval["low"] < 1.0 < interval["high"]
    first = cluster_bootstrap_ci95([1, 2, 10, 12], ["a", "a", "b", "b"], seed=7, resamples=100)
    second = cluster_bootstrap_ci95([1, 2, 10, 12], ["a", "a", "b", "b"], seed=7, resamples=100)
    assert first == second
    assert first["clusters"] == 2
