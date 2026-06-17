from leader_dt.evaluation.penalized_objective import (
    PenalizedObjectiveWeights,
    compute_penalized_score,
    add_penalized_score_to_metric_dictionary,
)


def test_compute_penalized_score() -> None:
    penalty_weights = PenalizedObjectiveWeights(
        freshness_violation_weight=10.0,
        terminal_cpu_violation_weight=1000.0,
        accuracy_violation_weight=100.0,
    )

    score = compute_penalized_score(
        average_weighted_aoi_float=9.0,
        freshness_violation_count_integer=2,
        terminal_cpu_violation_count_integer=1,
        accuracy_violation_count_integer=3,
        penalty_weights=penalty_weights,
    )

    assert score == 1329.0


def test_add_penalized_score_to_metric_dictionary() -> None:
    metric_dictionary = {
        "average_weighted_aoi_float": 9.0,
        "freshness_violation_count_integer": 2,
        "terminal_cpu_violation_count_integer": 1,
        "accuracy_violation_count_integer": 3,
    }

    updated_metric_dictionary = add_penalized_score_to_metric_dictionary(
        metric_dictionary
    )

    assert updated_metric_dictionary["penalized_score_float"] == 1329.0
    assert updated_metric_dictionary["penalized_objective_freshness_weight"] == 10.0
    assert updated_metric_dictionary["penalized_objective_terminal_cpu_weight"] == 1000.0
    assert updated_metric_dictionary["penalized_objective_accuracy_weight"] == 100.0