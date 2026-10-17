"""Multi-objective validation selection must be deterministic and Pareto-correct."""
import pytest
from kurcf.objectives import pareto_frontier, select_objectives


def candidates():
    return [
        {"clusters": 2, "alpha": 0., "mae": .70, "rmse": .91,
         "recall_at_k": .12, "precision_at_k": .04},
        {"clusters": 3, "alpha": .2, "mae": .73, "rmse": .93,
         "recall_at_k": .19, "precision_at_k": .06},
        {"clusters": 4, "alpha": .5, "mae": .78, "rmse": .97,
         "recall_at_k": .09, "precision_at_k": .03},
    ]


def test_objectives_choose_different_validation_optima():
    result = select_objectives(candidates())
    assert result["rating_mae"]["clusters"] == 2
    assert result["ranking_recall"]["clusters"] == 3
    assert [row["clusters"] for row in result["pareto"]] == [2, 3]


def test_deterministic_order_and_no_mutation():
    original = candidates()
    result = select_objectives(list(reversed(original)))
    assert result["rating_mae"]["clusters"] == 2
    assert result["ranking_recall"]["clusters"] == 3
    assert len(original) == 3


def test_invalid_rows_rejected():
    with pytest.raises(ValueError):
        select_objectives([])
    for change in ({"mae": float("nan")}, {"recall_at_k": 1.1},
                   {"clusters": 1}, {"alpha": -1}):
        rows = candidates()
        rows[0].update(change)
        with pytest.raises(ValueError):
            pareto_frontier(rows)
