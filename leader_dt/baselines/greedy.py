"""Greedy exact-pair baselines for leader-assisted DT synchronization.

The primary Greedy policy in this module is CPU-aware.  It keeps the same
weighted-AoI urgency term used by the earlier Greedy baseline, but subtracts a
normalized CPU-backlog penalty so that expensive uploads are discouraged when
processing capacity is tight.
"""
from __future__ import annotations

import numpy as np
from leader_dt import constants
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class GreedyWeightedAoiPolicy:
    """Select a feasible pair using CPU-aware weighted-AoI urgency.

    For each feasible sensor-vehicle pair ``i`` at slot ``t``, the policy
    computes

    ``score(i,t) = w_s A_s(t) - lambda_cpu * max(0, B(t)+C_i(t)-F*Delta) / (F*Delta)``

    where ``w_s`` is the priority weight of the sensor type, ``A_s(t)`` is the
    sensor-type AoI in slots, ``B(t)`` is the current CPU backlog in cycles,
    ``C_i(t)`` is the estimated added CPU cycles for the candidate upload, and
    ``F*Delta`` is the per-slot CPU capacity in cycles.

    Unit interpretation: ``w_s A_s(t)`` is a weighted-AoI score measured in
    weighted-slot score units. The CPU backlog ratio is dimensionless because
    it divides cycles by cycles. Therefore, ``lambda_cpu`` is a score-space
    coefficient measured in weighted-slot score units per normalized CPU
    backlog slot, which makes the two terms compatible in the objective score.
    """

    def __init__(
        self,
        lambda_cpu: float = constants.DEFAULT_GREEDY_CPU_LAMBDA,
        requested_accuracy_fraction: float = constants.DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION,
    ) -> None:
        """Create a CPU-aware Greedy policy.

        Args:
            lambda_cpu: Penalty weight applied to the normalized predicted CPU
                backlog. A value of zero recovers the old CPU-blind weighted-AoI
                Greedy behavior, except for the configurable requested accuracy
                fraction.
            requested_accuracy_fraction: Fraction of each candidate payload to
                request. ``1.0`` requests full data; using the system accuracy
                threshold approximates a minimum-accuracy Greedy variant.
        """
        self.lambda_cpu = float(lambda_cpu)
        self.requested_accuracy_fraction = float(np.clip(requested_accuracy_fraction, 0.0, 1.0))

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        """Return the CPU-aware Greedy action for the current environment state."""
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")

        feasible_pair_indices = environment.dynamics.get_feasible_pair_indices(environment.state)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        action[-1] = np.float32(self.requested_accuracy_fraction)

        if not feasible_pair_indices or self.requested_accuracy_fraction <= 0.0:
            return action

        slot_index = min(
            environment.state.time_slot_index,
            environment.simulation_config.system.time_horizon_slots - 1,
        )
        priority_weight_array_by_sensor_type = environment.scenario.priority_weight_array_by_sensor_type()
        sensor_type_aoi_slots_array = environment.state.sensor_type_aoi_slots_array
        available_data_size_bits_array = environment.scenario.available_data_size_bits_matrix[slot_index]
        uplink_capacity_bits_array = environment.dynamics.compute_uplink_capacity_bits_array(environment.state)
        cpu_cycles_per_bit_array = environment.scenario.cpu_cycles_per_bit_array_by_pair()
        slot_cpu_capacity_cycles_float = self._compute_slot_cpu_capacity_cycles(environment)
        current_cpu_backlog_cycles_float = float(environment.state.cpu_backlog_cycles_float)

        scores = np.full(environment.scenario.pair_count, -np.inf, dtype=np.float64)
        for pair in environment.scenario.sensor_pair_index.pairs:
            pair_id = int(pair.pair_id)
            sensor_type_id = int(pair.sensor_type_id)
            urgency_score_float = (
                priority_weight_array_by_sensor_type[sensor_type_id]
                * sensor_type_aoi_slots_array[sensor_type_id]
            )
            estimated_added_cycles_float = self._estimate_added_cpu_cycles_for_pair(
                pair_index=pair_id,
                requested_accuracy_fraction=self.requested_accuracy_fraction,
                available_data_size_bits_array=available_data_size_bits_array,
                uplink_capacity_bits_array=uplink_capacity_bits_array,
                cpu_cycles_per_bit_array=cpu_cycles_per_bit_array,
            )
            normalized_predicted_backlog_float = max(
                0.0,
                current_cpu_backlog_cycles_float
                + estimated_added_cycles_float
                - slot_cpu_capacity_cycles_float,
            ) / max(slot_cpu_capacity_cycles_float, constants.EPSILON_FLOAT)
            scores[pair_id] = urgency_score_float - self.lambda_cpu * normalized_predicted_backlog_float

        feasible_pair_indices_array = np.asarray(feasible_pair_indices, dtype=int)
        selected_pair = int(feasible_pair_indices_array[np.argmax(scores[feasible_pair_indices_array])])
        action[selected_pair] = 1.0
        return action

    @staticmethod
    def _compute_slot_cpu_capacity_cycles(environment: LeaderSynchronizationEnv) -> float:
        """Return the number of CPU cycles available during one scheduling slot."""
        system_config = environment.simulation_config.system
        return float(
            system_config.leader_cpu_frequency_cycles_per_second
            * system_config.slot_duration_seconds
        )

    @staticmethod
    def _estimate_added_cpu_cycles_for_pair(
        pair_index: int,
        requested_accuracy_fraction: float,
        available_data_size_bits_array: np.ndarray,
        uplink_capacity_bits_array: np.ndarray,
        cpu_cycles_per_bit_array: np.ndarray,
    ) -> float:
        """Estimate the CPU cycles added if ``pair_index`` is scheduled now.

        The estimate mirrors ``ActionDecoder``: requested bits are the requested
        accuracy fraction times available data; collected bits are limited by
        requested bits, available data, and uplink capacity. Added cycles equal
        collected bits multiplied by the pair's CPU cycles per bit.
        """
        requested_bits_float = float(
            requested_accuracy_fraction * available_data_size_bits_array[int(pair_index)]
        )
        estimated_collected_bits_float = min(
            requested_bits_float,
            float(available_data_size_bits_array[int(pair_index)]),
            float(uplink_capacity_bits_array[int(pair_index)]),
        )
        return float(estimated_collected_bits_float * cpu_cycles_per_bit_array[int(pair_index)])


class GreedyMaxAoiPolicy:
    """Select feasible pair with largest sensor-type AoI and request full data."""

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        """Return the max-AoI Greedy action for the current environment state."""
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")
        feasible = environment.dynamics.get_feasible_pair_indices(environment.state)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if not feasible:
            return action
        sensor_type_aoi = environment.state.sensor_type_aoi_slots_array
        scores = np.zeros(environment.scenario.pair_count, dtype=np.float64)
        for pair in environment.scenario.sensor_pair_index.pairs:
            scores[int(pair.pair_id)] = sensor_type_aoi[int(pair.sensor_type_id)]
        selected_pair = int(feasible[int(np.argmax(scores[feasible]))])
        action[selected_pair] = 1.0
        action[-1] = 1.0
        return action
