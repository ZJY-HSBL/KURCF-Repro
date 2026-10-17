"""Tests for controlled full-rating study and deterministic aggregation."""
from argparse import Namespace
import copy
import math
import numpy as np
import pytest

from kurcf.core import KURCF, weighted_kmeans
from kurcf.study import run_study
from scripts.summarize_study import aggregate, stats, overview_svg, sensitivity_svg

ATTRS = {
    1: [1, 0, 0], 2: [1, 0, 0], 3: [0, 1, 0],
    4: [0, 1, 0], 5: [0, 0, 1], 6: [0, 0, 1],
}
RATINGS = [
    (user, item, float((user * 3 + item) % 5 + 1))
    for user in range(1, 6) for item in range(1, 7)
    if (user + item) % 6 != 0
]


def args(seed=0):
    return Namespace(
        dataset="ml-100k", data_dir="unused", seed=seed, clusters=2,
        neighbors=3, alpha=.3, test_ratio=.2, top_k=2,
        rating_cases=0, ranking_users=0,
        alpha_grid=[0, .3, 1], cluster_grid=[2, 3],
    )


def test_uniform_weight_ablation():
    x = np.array([[0., 1.], [2., 1.], [4., 1.], [8., 1.]])
    var = weighted_kmeans(x, 2, seed=3)
    uniform = weighted_kmeans(x, 2, seed=3, feature_weighting="uniform")
    np.testing.assert_allclose(var.weights, [1, 0])
    np.testing.assert_allclose(uniform.weights, [.5, .5])
    with pytest.raises(ValueError):
        weighted_kmeans(x, 2, feature_weighting="unknown")
    with pytest.raises(ValueError):
        KURCF(feature_weighting="unknown")


def test_full_rating_study_and_shared_population(monkeypatch):
    monkeypatch.setattr("kurcf.study.load_movielens", lambda *a: (RATINGS, ATTRS))
    run = run_study(args(seed=1))
    assert run["schema_version"] == 1
    assert len(run["variants"]) == 7
    assert run["protocol"]["rating_cases_requested"] == 0
    assert run["variants"]["kurcf"]["n"] == run["protocol"]["test_interactions"]
    assert run["variants"]["kurcf"]["users"] == run["protocol"]["ranking_users_evaluated"]
    assert len(run["alpha_sweep"]) == 3
    assert len(run["cluster_sweep"]) == 2
    assert run["variants"]["kurcf"]["mae"] == pytest.approx(
        next(row["mae"] for row in run["alpha_sweep"] if row["alpha"] == .3)
    )
    assert run["variants"]["kurcf"]["mae"] == pytest.approx(
        next(row["mae"] for row in run["cluster_sweep"] if row["clusters"] == 2)
    )
    for variant in run["variants"].values():
        assert variant["n"] == run["protocol"]["test_interactions"]
        assert math.isfinite(variant["mae"])
        assert variant["users"] == run["protocol"]["ranking_users_evaluated"]


def test_invalid_study_arguments(monkeypatch):
    monkeypatch.setattr("kurcf.study.load_movielens", lambda *a: (RATINGS, ATTRS))
    bad = args()
    bad.rating_cases = -1
    with pytest.raises(ValueError):
        run_study(bad)
    bad = args()
    bad.alpha_grid = [1.1]
    with pytest.raises(ValueError):
        run_study(bad)


def test_aggregate_detects_duplicate_seed_and_mismatched_protocol(monkeypatch):
    monkeypatch.setattr("kurcf.study.load_movielens", lambda *a: (RATINGS, ATTRS))
    first, second = run_study(args(seed=0)), run_study(args(seed=1))
    summary = aggregate([copy.deepcopy(first), copy.deepcopy(second)], min_seeds=2)
    assert summary["protocol"]["seeds"] == [0, 1]
    assert len(summary["metrics"]) == 7
    assert len(summary["alpha_sweep"]) == 3
    assert "<svg" in overview_svg(summary)
    assert "<svg" in sensitivity_svg(summary, "alpha_sweep", "alpha", "Sensitivity")
    assert summary["metrics"]["kurcf"]["mae"]["n_seeds"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([first, first], min_seeds=2)
    modified = copy.deepcopy(second)
    modified["protocol"]["neighbors"] = 99
    with pytest.raises(ValueError, match="non-comparable"):
        aggregate([first, modified], min_seeds=2)


def test_bootstrap_is_deterministic():
    assert stats([1., 2., 3.]) == stats([1., 2., 3.])
    assert stats([2., 2., 2.])["ci95"] == [2., 2.]
    with pytest.raises(ValueError):
        stats([float("nan")])
