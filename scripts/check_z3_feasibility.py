"""Run Z3 satisfiability checks for the paper constraint system."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from leader_dt.config import (
    CommunicationConfig,
    DataGenerationConfig,
    RoadConfig,
    SimulationConfig,
    SystemConfig,
)
from leader_dt.verification.z3_feasibility import Z3FeasibilityChecker


def build_config_from_arguments(args: argparse.Namespace) -> SimulationConfig:
    base_config = SimulationConfig(random_seed=args.seed)
    system_config = replace(
        base_config.system,
        time_horizon_slots=args.time_horizon_slots,
        vehicle_count=args.vehicle_count,
        sensors_per_vehicle=args.sensors_per_vehicle,
        freshness_threshold_slots=args.freshness_threshold_slots,
        accuracy_threshold=args.accuracy_threshold,
        leader_cpu_frequency_cycles_per_second=args.cpu_frequency,
        include_leader_as_provider=not args.exclude_leader,
        max_vehicle_count_for_action_space=max(
            base_config.system.max_vehicle_count_for_action_space,
            args.vehicle_count,
        ),
        max_sensors_per_vehicle_for_action_space=max(
            base_config.system.max_sensors_per_vehicle_for_action_space,
            args.sensors_per_vehicle,
        ),
    )
    road_config = replace(
        base_config.road,
        vehicle_speed_meter_per_second=args.vehicle_speed,
        defective_zone_end_meter=base_config.road.defective_zone_start_meter + args.zone_size,
    )
    communication_config = replace(
        base_config.communication,
        uplink_bandwidth_hz=args.uplink_bandwidth,
    )
    data_generation_config = replace(
        base_config.data_generation,
        low_multiplier=args.data_size_low_multiplier,
        high_multiplier=args.data_size_high_multiplier,
    )
    return SimulationConfig(
        system=system_config,
        communication=communication_config,
        road=road_config,
        data_generation=data_generation_config,
        random_seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vehicle-count", type=int, default=40)
    parser.add_argument("--sensors-per-vehicle", type=int, default=4)
    parser.add_argument("--time-horizon-slots", type=int, default=40)
    parser.add_argument("--freshness-threshold-slots", type=int, default=10)
    parser.add_argument("--accuracy-threshold", type=float, default=0.80)
    parser.add_argument("--cpu-frequency", type=float, default=1.5e6)
    parser.add_argument("--uplink-bandwidth", type=float, default=6.0e5)
    parser.add_argument("--vehicle-speed", type=float, default=15.0)
    parser.add_argument("--zone-size", type=float, default=400.0)
    parser.add_argument("--data-size-low-multiplier", type=float, default=0.80)
    parser.add_argument("--data-size-high-multiplier", type=float, default=1.20)
    parser.add_argument("--exclude-leader", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--optimize-objective", action="store_true")
    parser.add_argument("--relax-freshness", action="store_true")
    parser.add_argument("--relax-terminal-cpu", action="store_true")
    parser.add_argument(
        "--output",
        default="results/metrics/z3_feasibility_report.json",
    )
    args = parser.parse_args()

    simulation_config = build_config_from_arguments(args)
    checker = Z3FeasibilityChecker(simulation_config=simulation_config, seed=args.seed)
    report = checker.check(
        timeout_milliseconds=args.timeout_ms,
        optimize_objective_boolean=args.optimize_objective,
        enforce_freshness_boolean=not args.relax_freshness,
        enforce_terminal_cpu_boolean=not args.relax_terminal_cpu,
    )
    report.save_json(args.output)

    print("Z3 status:", report.z3_status_string)
    print("Elapsed seconds:", round(report.elapsed_seconds_float, 3))
    print("Objective value:", report.objective_value_float)
    print("Note:", report.note_string)
    print("Relaxed statuses:", report.relaxed_status_dictionary)
    print("Strictness diagnostics:", report.strictness_diagnostics_dictionary)
    print("Saved", Path(args.output))


if __name__ == "__main__":
    main()
