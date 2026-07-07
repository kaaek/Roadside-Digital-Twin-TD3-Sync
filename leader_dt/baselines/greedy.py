"""Greedy exact-pair baselines for leader-assisted DT synchronization.

The primary Greedy policy is CPU-aware: it uses weighted AoI as the urgency
term and subtracts a normalized CPU-backlog penalty.  The Greedy variants also
maintain an episode-level supplier blacklist.  If a selected vehicle-sensor
pair requires at least as much service time as the carrier vehicle has left in
Zone B, that vehicle is no longer accepted as a supplier for that specific
sensor type for the rest of the episode.
"""
from __future__ import annotations

import numpy as np

from leader_dt import constants
from leader_dt.domain.sensor_pairs import SensorVehiclePair
from leader_dt.simulator.environment import LeaderSynchronizationEnv


class _EpisodeSupplierBlacklistMixin:
    """Track vehicle-sensor supplier exclusions for one rollout episode."""

    def _initialize_supplier_blacklist(self) -> None:
        self._blacklisted_vehicle_sensor_pairs: set[tuple[int, int]] = set()
        self._active_environment_id: int | None = None
        self._last_seen_time_slot_index: int | None = None

    def _prepare_episode_state(self, environment: LeaderSynchronizationEnv) -> None:
        """Clear persistent exclusions when a new episode/environment starts."""
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")

        environment_id = id(environment)
        time_slot_index = int(environment.state.time_slot_index)
        is_new_environment = self._active_environment_id != environment_id
        is_episode_restart = (
            time_slot_index == 0
            and self._last_seen_time_slot_index is not None
            and self._last_seen_time_slot_index != 0
        )

        if is_new_environment or is_episode_restart:
            self._blacklisted_vehicle_sensor_pairs.clear()

        self._active_environment_id = environment_id
        self._last_seen_time_slot_index = time_slot_index

    def _is_pair_blacklisted(self, pair: SensorVehiclePair) -> bool:
        return (int(pair.vehicle_id), int(pair.sensor_type_id)) in self._blacklisted_vehicle_sensor_pairs

    def _blacklist_pair_for_rest_of_episode(self, pair: SensorVehiclePair) -> None:
        self._blacklisted_vehicle_sensor_pairs.add((int(pair.vehicle_id), int(pair.sensor_type_id)))

    def _filter_blacklisted_pairs(
        self,
        environment: LeaderSynchronizationEnv,
        feasible_pair_indices: list[int],
    ) -> list[int]:
        """Remove vehicle-sensor pairs blacklisted by previous failed service checks."""
        eligible_pair_indices: list[int] = []
        for pair_index in feasible_pair_indices:
            pair = environment.scenario.sensor_pair_index.get_pair(int(pair_index))
            if not self._is_pair_blacklisted(pair):
                eligible_pair_indices.append(int(pair_index))
        return eligible_pair_indices

    def _blacklist_if_selected_pair_cannot_finish_before_exit(
        self,
        environment: LeaderSynchronizationEnv,
        selected_pair_index: int,
        requested_accuracy_fraction: float,
        available_data_size_bits_array: np.ndarray,
        uplink_capacity_bits_array: np.ndarray,
        uplink_rate_bits_per_second_array: np.ndarray,
        cpu_cycles_per_bit_array: np.ndarray,
    ) -> None:
        """Persistently exclude the selected supplier if service time is too long.

        The current selected refresh attempt is still returned to the simulator.
        The exclusion affects only future time slots in the same episode.
        """
        selected_pair_index = int(selected_pair_index)
        collected_bits_float = self._estimate_collected_bits_for_pair(
            pair_index=selected_pair_index,
            requested_accuracy_fraction=requested_accuracy_fraction,
            available_data_size_bits_array=available_data_size_bits_array,
            uplink_capacity_bits_array=uplink_capacity_bits_array,
        )
        if collected_bits_float <= 0.0:
            return

        pair = environment.scenario.sensor_pair_index.get_pair(selected_pair_index)
        if self._service_time_exceeds_or_matches_time_until_zone_exit(
            environment=environment,
            pair=pair,
            pair_index=selected_pair_index,
            collected_bits_float=collected_bits_float,
            uplink_rate_bits_per_second_array=uplink_rate_bits_per_second_array,
            cpu_cycles_per_bit_array=cpu_cycles_per_bit_array,
        ):
            self._blacklist_pair_for_rest_of_episode(pair)

    @staticmethod
    def _estimate_collected_bits_for_pair(
        pair_index: int,
        requested_accuracy_fraction: float,
        available_data_size_bits_array: np.ndarray,
        uplink_capacity_bits_array: np.ndarray,
    ) -> float:
        requested_bits_float = float(
            requested_accuracy_fraction * available_data_size_bits_array[int(pair_index)]
        )
        return float(
            min(
                requested_bits_float,
                float(available_data_size_bits_array[int(pair_index)]),
                float(uplink_capacity_bits_array[int(pair_index)]),
            )
        )

    @staticmethod
    def _service_time_exceeds_or_matches_time_until_zone_exit(
        environment: LeaderSynchronizationEnv,
        pair: SensorVehiclePair,
        pair_index: int,
        collected_bits_float: float,
        uplink_rate_bits_per_second_array: np.ndarray,
        cpu_cycles_per_bit_array: np.ndarray,
    ) -> bool:
        vehicle_id = int(pair.vehicle_id)
        vehicle_position_meter = float(environment.state.vehicle_positions_meter_array[vehicle_id])
        vehicle_speed_meter_per_second = float(environment.scenario.vehicle_speed_meter_per_second_array[vehicle_id])
        distance_to_zone_exit_meter = max(
            0.0,
            float(environment.simulation_config.road.defective_zone_end_meter) - vehicle_position_meter,
        )
        if vehicle_speed_meter_per_second <= constants.EPSILON_FLOAT:
            time_until_vehicle_exits_zone_float = float("inf")
        else:
            time_until_vehicle_exits_zone_float = distance_to_zone_exit_meter / vehicle_speed_meter_per_second

        uplink_rate_bits_per_second_float = max(
            float(uplink_rate_bits_per_second_array[int(pair_index)]),
            constants.EPSILON_FLOAT,
        )
        upload_time_seconds_float = collected_bits_float / uplink_rate_bits_per_second_float

        cpu_cycles_per_bit_float = max(float(cpu_cycles_per_bit_array[int(pair_index)]), constants.EPSILON_FLOAT)
        processing_capacity_bits_per_second_float = (
            float(environment.simulation_config.system.leader_cpu_frequency_cycles_per_second)
            / cpu_cycles_per_bit_float
        )
        processing_time_seconds_float = collected_bits_float / max(
            processing_capacity_bits_per_second_float,
            constants.EPSILON_FLOAT,
        )
        service_time_seconds_float = upload_time_seconds_float + processing_time_seconds_float
        return service_time_seconds_float >= time_until_vehicle_exits_zone_float


class GreedyWeightedAoiPolicy(_EpisodeSupplierBlacklistMixin):
    """Select a feasible pair using CPU-aware weighted-AoI urgency.

    For each feasible sensor-vehicle pair ``i`` at slot ``t``, the policy
    computes

    ``score(i,t) = w_s A_s(t) - lambda_cpu * max(0, B(t)+C_i(t)-F*Delta) / (F*Delta)``

    where ``w_s`` is the priority weight of the sensor type, ``A_s(t)`` is the
    sensor-type AoI in slots, ``B(t)`` is the current CPU backlog in cycles,
    ``C_i(t)`` is the estimated added CPU cycles for the candidate upload, and
    ``F*Delta`` is the per-slot CPU capacity in cycles.  If the selected pair's
    service time is at least the time left before its carrier exits Zone B, that
    vehicle-sensor pair is blacklisted for the rest of the episode.
    """

    def __init__(
        self,
        lambda_cpu: float = constants.DEFAULT_GREEDY_CPU_LAMBDA,
        requested_accuracy_fraction: float = constants.DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION,
    ) -> None:
        """Create a CPU-aware Greedy policy."""
        self.lambda_cpu = float(lambda_cpu)
        self.requested_accuracy_fraction = float(np.clip(requested_accuracy_fraction, 0.0, 1.0))
        self._initialize_supplier_blacklist()

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        """Return the CPU-aware Greedy action for the current environment state."""
        self._prepare_episode_state(environment)

        feasible_pair_indices = environment.dynamics.get_feasible_pair_indices(environment.state)
        eligible_pair_indices = self._filter_blacklisted_pairs(environment, feasible_pair_indices)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        action[-1] = np.float32(self.requested_accuracy_fraction)

        if not eligible_pair_indices or self.requested_accuracy_fraction <= 0.0:
            return action

        shared_arrays = self._build_shared_scoring_arrays(environment)
        scores = np.full(environment.scenario.pair_count, -np.inf, dtype=np.float64)
        for pair_index in eligible_pair_indices:
            scores[int(pair_index)] = self._score_pair(
                environment=environment,
                pair_index=int(pair_index),
                **shared_arrays,
            )

        eligible_pair_indices_array = np.asarray(eligible_pair_indices, dtype=int)
        selected_pair_index = int(eligible_pair_indices_array[np.argmax(scores[eligible_pair_indices_array])])
        self._blacklist_if_selected_pair_cannot_finish_before_exit(
            environment=environment,
            selected_pair_index=selected_pair_index,
            requested_accuracy_fraction=self.requested_accuracy_fraction,
            available_data_size_bits_array=shared_arrays["available_data_size_bits_array"],
            uplink_capacity_bits_array=shared_arrays["uplink_capacity_bits_array"],
            uplink_rate_bits_per_second_array=shared_arrays["uplink_rate_bits_per_second_array"],
            cpu_cycles_per_bit_array=shared_arrays["cpu_cycles_per_bit_array"],
        )
        action[selected_pair_index] = 1.0
        return action

    def _build_shared_scoring_arrays(self, environment: LeaderSynchronizationEnv) -> dict[str, np.ndarray | float]:
        slot_index = min(
            int(environment.state.time_slot_index),
            environment.simulation_config.system.time_horizon_slots - 1,
        )
        uplink_rate_bits_per_second_array = environment.dynamics.compute_uplink_rate_array_by_pair(environment.state)
        return {
            "priority_weight_array_by_sensor_type": environment.scenario.priority_weight_array_by_sensor_type(),
            "sensor_type_aoi_slots_array": environment.state.sensor_type_aoi_slots_array,
            "available_data_size_bits_array": environment.scenario.available_data_size_bits_matrix[slot_index],
            "uplink_rate_bits_per_second_array": uplink_rate_bits_per_second_array,
            "uplink_capacity_bits_array": uplink_rate_bits_per_second_array
            * environment.simulation_config.system.slot_duration_seconds,
            "cpu_cycles_per_bit_array": environment.scenario.cpu_cycles_per_bit_array_by_pair(),
            "slot_cpu_capacity_cycles_float": self._compute_slot_cpu_capacity_cycles(environment),
            "current_cpu_backlog_cycles_float": float(environment.state.cpu_backlog_cycles_float),
        }

    def _score_pair(
        self,
        environment: LeaderSynchronizationEnv,
        pair_index: int,
        priority_weight_array_by_sensor_type: np.ndarray,
        sensor_type_aoi_slots_array: np.ndarray,
        available_data_size_bits_array: np.ndarray,
        uplink_rate_bits_per_second_array: np.ndarray,
        uplink_capacity_bits_array: np.ndarray,
        cpu_cycles_per_bit_array: np.ndarray,
        slot_cpu_capacity_cycles_float: float,
        current_cpu_backlog_cycles_float: float,
    ) -> float:
        pair = environment.scenario.sensor_pair_index.get_pair(pair_index)
        sensor_type_id = int(pair.sensor_type_id)
        urgency_score_float = (
            priority_weight_array_by_sensor_type[sensor_type_id]
            * sensor_type_aoi_slots_array[sensor_type_id]
        )
        estimated_added_cycles_float = self._estimate_added_cpu_cycles_for_pair(
            pair_index=pair_index,
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
        return float(urgency_score_float - self.lambda_cpu * normalized_predicted_backlog_float)

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
        """Estimate CPU cycles added if ``pair_index`` is scheduled now."""
        estimated_collected_bits_float = _EpisodeSupplierBlacklistMixin._estimate_collected_bits_for_pair(
            pair_index=pair_index,
            requested_accuracy_fraction=requested_accuracy_fraction,
            available_data_size_bits_array=available_data_size_bits_array,
            uplink_capacity_bits_array=uplink_capacity_bits_array,
        )
        return float(estimated_collected_bits_float * cpu_cycles_per_bit_array[int(pair_index)])


class ProximityGreedyPolicy(GreedyWeightedAoiPolicy):
    """Pick the closest carrier vehicle, then its most urgent sensor.

    This policy uses the same requested accuracy, CPU-aware sensor scoring, and
    service-time blacklist as ``GreedyWeightedAoiPolicy``.  The difference is
    the selection order: it first chooses the closest eligible carrier vehicle
    to the leader and then schedules the highest-scoring sensor owned by that
    vehicle.
    """

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        """Return the proximity-based Greedy action for the current state."""
        self._prepare_episode_state(environment)

        feasible_pair_indices = environment.dynamics.get_feasible_pair_indices(environment.state)
        eligible_pair_indices = self._filter_blacklisted_pairs(environment, feasible_pair_indices)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        action[-1] = np.float32(self.requested_accuracy_fraction)

        if not eligible_pair_indices or self.requested_accuracy_fraction <= 0.0:
            return action

        distance_array_by_pair = environment.dynamics.compute_distance_array_by_pair(environment.state)
        eligible_pairs = [environment.scenario.sensor_pair_index.get_pair(index) for index in eligible_pair_indices]
        eligible_vehicle_ids = sorted({int(pair.vehicle_id) for pair in eligible_pairs})
        closest_vehicle_id = min(
            eligible_vehicle_ids,
            key=lambda vehicle_id: min(
                distance_array_by_pair[int(pair.pair_id)]
                for pair in eligible_pairs
                if int(pair.vehicle_id) == vehicle_id
            ),
        )
        candidate_pair_indices = [
            int(pair.pair_id)
            for pair in eligible_pairs
            if int(pair.vehicle_id) == closest_vehicle_id
        ]

        shared_arrays = self._build_shared_scoring_arrays(environment)
        scores = np.full(environment.scenario.pair_count, -np.inf, dtype=np.float64)
        for pair_index in candidate_pair_indices:
            scores[int(pair_index)] = self._score_pair(
                environment=environment,
                pair_index=int(pair_index),
                **shared_arrays,
            )

        candidate_pair_indices_array = np.asarray(candidate_pair_indices, dtype=int)
        selected_pair_index = int(candidate_pair_indices_array[np.argmax(scores[candidate_pair_indices_array])])
        self._blacklist_if_selected_pair_cannot_finish_before_exit(
            environment=environment,
            selected_pair_index=selected_pair_index,
            requested_accuracy_fraction=self.requested_accuracy_fraction,
            available_data_size_bits_array=shared_arrays["available_data_size_bits_array"],
            uplink_capacity_bits_array=shared_arrays["uplink_capacity_bits_array"],
            uplink_rate_bits_per_second_array=shared_arrays["uplink_rate_bits_per_second_array"],
            cpu_cycles_per_bit_array=shared_arrays["cpu_cycles_per_bit_array"],
        )
        action[selected_pair_index] = 1.0
        return action


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
