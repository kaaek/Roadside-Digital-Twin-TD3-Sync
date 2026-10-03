from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy, ProximityGreedyPolicy
from leader_dt.config import SimulationConfig
from leader_dt.evaluation.monte_carlo import TrainedModelGroup
from leader_dt.evaluation.policy_factory import split_model_paths, training_seeds_from_model_paths
from leader_dt.evaluation.sensitivity import SensitivityEvaluator


def test_sensitivity_pools_a_trained_model_group():
    trial_count = 2
    group = TrainedModelGroup(
        policies=[GreedyWeightedAoiPolicy(), ProximityGreedyPolicy()],
        training_seeds=[1, 2],
    )
    results = SensitivityEvaluator(SimulationConfig()).run_sweep_with_trials(
        parameter_name="vehicle_count",
        parameter_values=[10],
        policy_dictionary={"Group": group, "Greedy": GreedyWeightedAoiPolicy()},
        trial_count=trial_count,
        seed_start=50_000,
    )
    group_result = results[0].policy_results["Group"]
    assert len(group_result.per_trial_metric_list) == 2 * trial_count
    assert [row["training_seed_integer"] for row in group_result.per_trial_metric_list] == [1, 1, 2, 2]
    assert "average_weighted_aoi_float" in group_result.metric_between_seed_std_dictionary
    assert results[0].policy_results["Greedy"].metric_between_seed_std_dictionary == {}


def test_model_path_helpers():
    paths = split_model_paths("results/td3/seed_1/models/best.zip, results/td3/seed_3/models/best.zip")
    assert paths == ["results/td3/seed_1/models/best.zip", "results/td3/seed_3/models/best.zip"]
    assert training_seeds_from_model_paths(paths) == [1, 3]
    assert training_seeds_from_model_paths(["results/td3/models/best.zip"]) is None
    assert split_model_paths(None) == []
