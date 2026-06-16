"""Configuration dataclasses for the simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from leader_dt import constants

@dataclass(frozen=True)
class SystemConfig:
    time_horizon_slots: int = constants.DEFAULT_TIME_HORIZON_SLOTS
    slot_duration_seconds: float = constants.DEFAULT_SLOT_DURATION_SECONDS
    vehicle_count: int = constants.DEFAULT_VEHICLE_COUNT
    sensor_type_count: int = constants.DEFAULT_SENSOR_TYPE_COUNT
    sensors_per_vehicle: int = constants.DEFAULT_SENSORS_PER_VEHICLE
    freshness_threshold_slots: int = constants.DEFAULT_FRESHNESS_THRESHOLD_SLOTS
    accuracy_threshold: float = constants.DEFAULT_ACCURACY_THRESHOLD
    leader_cpu_frequency_cycles_per_second: float = constants.DEFAULT_LEADER_CPU_FREQUENCY_CYCLES_PER_SECOND
    include_leader_as_provider: bool = True
    max_vehicle_count_for_action_space: int = constants.DEFAULT_MAX_VEHICLE_COUNT_FOR_ACTION_SPACE
    max_sensors_per_vehicle_for_action_space: int = constants.DEFAULT_MAX_SENSORS_PER_VEHICLE_FOR_ACTION_SPACE

    @property
    def max_pair_count_for_action_space(self) -> int:
        return self.max_vehicle_count_for_action_space * self.max_sensors_per_vehicle_for_action_space

@dataclass(frozen=True)
class CommunicationConfig:
    uplink_bandwidth_hz: float = constants.DEFAULT_UPLINK_BANDWIDTH_HZ
    max_transmit_power_watt: float = constants.DEFAULT_V2L_MAX_TRANSMIT_POWER_WATT
    noise_power_spectral_density_watt_per_hz: float = constants.DEFAULT_NOISE_POWER_SPECTRAL_DENSITY_WATT_PER_HZ
    pathloss_exponent: float = constants.DEFAULT_UPLINK_PATHLOSS_EXPONENT
    reference_distance_meter: float = constants.REFERENCE_DISTANCE_METER

@dataclass(frozen=True)
class RoadConfig:
    lane_length_meter: float = constants.DEFAULT_LANE_LENGTH_METER
    defective_zone_start_meter: float = constants.DEFAULT_DEFECTIVE_ZONE_START_METER
    defective_zone_end_meter: float = constants.DEFAULT_DEFECTIVE_ZONE_END_METER
    vehicle_speed_meter_per_second: float = constants.DEFAULT_VEHICLE_SPEED_METER_PER_SECOND
    vehicle_speed_jitter_std_meter_per_second: float = constants.DEFAULT_VEHICLE_SPEED_JITTER_STD_METER_PER_SECOND

@dataclass(frozen=True)
class DataGenerationConfig:
    low_multiplier: float = constants.DEFAULT_DATA_SIZE_LOW_MULTIPLIER
    high_multiplier: float = constants.DEFAULT_DATA_SIZE_HIGH_MULTIPLIER

@dataclass(frozen=True)
class SimulationConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    random_seed: int | None = None

@dataclass(frozen=True)
class Td3TrainingConfig:
    total_timesteps: int = constants.DEFAULT_TOTAL_TIMESTEPS
    learning_rate: float = constants.DEFAULT_LEARNING_RATE
    learning_starts: int = constants.DEFAULT_LEARNING_STARTS
    buffer_size: int = constants.DEFAULT_BUFFER_SIZE
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    gamma: float = constants.DEFAULT_GAMMA
    tau: float = constants.DEFAULT_TAU
    policy_delay: int = constants.DEFAULT_POLICY_DELAY
    train_frequency_steps: int = constants.DEFAULT_TRAIN_FREQUENCY_STEPS
    gradient_steps: int = constants.DEFAULT_GRADIENT_STEPS
    actor_hidden_layers: tuple[int, int] = constants.DEFAULT_ACTOR_HIDDEN_LAYERS
    critic_hidden_layers: tuple[int, int] = constants.DEFAULT_CRITIC_HIDDEN_LAYERS
    device: str = constants.DEFAULT_DEVICE

@dataclass(frozen=True)
class MonteCarloConfig:
    trial_count: int = constants.DEFAULT_MONTE_CARLO_TRIAL_COUNT
    seed_start: int = constants.DEFAULT_SEED_START

@dataclass(frozen=True)
class SensitivityConfig:
    trial_count_per_point: int = constants.DEFAULT_TRIAL_COUNT_PER_SLOT
