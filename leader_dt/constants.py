"""
Constants for the leader-assisted DT synchronization model.
Config dataclasses import these values as defaults.
"""
from __future__ import annotations

REFERENCE_DISTANCE_METER: float = 1.0
EPSILON_FLOAT: float = 1.0e-12

DEFAULT_TIME_HORIZON_SLOTS: int = 40
DEFAULT_SLOT_DURATION_SECONDS: float = 1.0
DEFAULT_VEHICLE_COUNT: int = 40
DEFAULT_SENSOR_TYPE_COUNT: int = 8
DEFAULT_SENSORS_PER_VEHICLE: int = 4

# Fixed-capacity dimensions for RL action/observation padding.
# These let one TD3 policy keep a stable shape across vehicle-count sweeps.
DEFAULT_MAX_VEHICLE_COUNT_FOR_ACTION_SPACE: int = 80
DEFAULT_MAX_SENSORS_PER_VEHICLE_FOR_ACTION_SPACE: int = 4
DEFAULT_FRESHNESS_THRESHOLD_SLOTS: int = 10
DEFAULT_ACCURACY_THRESHOLD: float = 0.80

DEFAULT_UPLINK_BANDWIDTH_HZ: float = 6.0e5
DEFAULT_V2L_MAX_TRANSMIT_POWER_WATT: float = 1.0
DEFAULT_NOISE_POWER_SPECTRAL_DENSITY_WATT_PER_HZ: float = 1.0e-12
DEFAULT_UPLINK_PATHLOSS_EXPONENT: float = 2.5

DEFAULT_LEADER_CPU_FREQUENCY_CYCLES_PER_SECOND: float = 1.5e6

DEFAULT_LANE_LENGTH_METER: float = 1000.0
DEFAULT_DEFECTIVE_ZONE_START_METER: float = 300.0
DEFAULT_DEFECTIVE_ZONE_END_METER: float = 700.0
DEFAULT_VEHICLE_SPEED_METER_PER_SECOND: float = 15.0
DEFAULT_VEHICLE_SPEED_JITTER_STD_METER_PER_SECOND: float = 1.0

# Data-size sampling: each delta_i(t) is sampled uniformly around the nominal
# sensor-type payload size. This keeps the paper's time-dependent delta_i(t)
# while remaining simple and reproducible.
DEFAULT_DATA_SIZE_LOW_MULTIPLIER: float = 0.80
DEFAULT_DATA_SIZE_HIGH_MULTIPLIER: float = 1.20

# Sensor definitions are ordered by sensor_type_id.
DEFAULT_SENSOR_DEFINITIONS: tuple[dict, ...] = (
    {
        "name": "Front Camera",
        "priority_weight": 3.0,
        "cpu_cycles_per_bit": 15.0,
        "sensing_delay_slots": 1.0,
        "nominal_data_size_bits": 300_000.0,
    },
    {
        "name": "Radar",
        "priority_weight": 2.2,
        "cpu_cycles_per_bit": 2.0,
        "sensing_delay_slots": 0.2,
        "nominal_data_size_bits": 80_000.0,
    },
    {
        "name": "Engine Temperature",
        "priority_weight": 2.0,
        "cpu_cycles_per_bit": 3.0,
        "sensing_delay_slots": 0.3,
        "nominal_data_size_bits": 60_000.0,
    },
    {
        "name": "Battery BMS",
        "priority_weight": 1.8,
        "cpu_cycles_per_bit": 2.0,
        "sensing_delay_slots": 0.2,
        "nominal_data_size_bits": 50_000.0,
    },
    {
        "name": "Tyre Pressure TPMS",
        "priority_weight": 1.5,
        "cpu_cycles_per_bit": 8.0,
        "sensing_delay_slots": 0.8,
        "nominal_data_size_bits": 200_000.0,
    },
    {
        "name": "IMU Accelerometer",
        "priority_weight": 1.3,
        "cpu_cycles_per_bit": 11.0,
        "sensing_delay_slots": 0.7,
        "nominal_data_size_bits": 220_000.0,
    },
    {
        "name": "Ambient Weather",
        "priority_weight": 1.2,
        "cpu_cycles_per_bit": 10.0,
        "sensing_delay_slots": 0.6,
        "nominal_data_size_bits": 250_000.0,
    },
    {
        "name": "Fuel Level",
        "priority_weight": 1.0,
        "cpu_cycles_per_bit": 1.0,
        "sensing_delay_slots": 0.1,
        "nominal_data_size_bits": 40_000.0,
    },
)
DEFAULT_TOTAL_TIMESTEPS: int = 500_000
DEFAULT_LEARNING_RATE: float = 5.0e-4
DEFAULT_LEARNING_STARTS: int = 10_000
DEFAULT_BUFFER_SIZE: int = 300_000
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_GAMMA: float = 0.99
DEFAULT_TAU: float = 0.005
DEFAULT_POLICY_DELAY: int = 2
DEFAULT_TRAIN_FREQUENCY_STEPS: int = 10
DEFAULT_GRADIENT_STEPS: int = 1
DEFAULT_ACTOR_HIDDEN_LAYERS: tuple[int, int] = (64, 64)
DEFAULT_CRITIC_HIDDEN_LAYERS: tuple[int, int] = (64, 64)
DEFAULT_DEVICE: str = "cpu"

DEFAULT_MONTE_CARLO_TRIAL_COUNT: int = 30
DEFAULT_SEED_START: int = 1

DEFAULT_TRIAL_COUNT_PER_SLOT: int = 10