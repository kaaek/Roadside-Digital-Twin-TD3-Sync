import importlib.util

import pytest

from leader_dt.config import SimulationConfig, SystemConfig


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("z3") is None,
    reason="z3-solver is not installed",
)


def test_z3_checker_builds_small_report() -> None:
    from leader_dt.verification.z3_feasibility import Z3FeasibilityChecker

    config = SimulationConfig(
        system=SystemConfig(
            time_horizon_slots=4,
            vehicle_count=2,
            sensors_per_vehicle=1,
            freshness_threshold_slots=4,
            max_vehicle_count_for_action_space=2,
            max_sensors_per_vehicle_for_action_space=1,
        ),
        random_seed=1,
    )
    report = Z3FeasibilityChecker(config, seed=1).check(timeout_milliseconds=5_000)
    assert report.z3_status_string in {"sat", "unsat", "unknown"}
    assert "pair_count" in report.strictness_diagnostics_dictionary
