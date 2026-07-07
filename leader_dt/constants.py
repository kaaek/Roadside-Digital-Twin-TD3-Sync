"""
Constants for the leader-assisted DT synchronization model.
Config dataclasses import these values as defaults.
"""
from __future__ import annotations

REFERENCE_DISTANCE_METER: float = 1.0
EPSILON_FLOAT: float = 1.0e-12

DEFAULT_TIME_HORIZON_SLOTS: int = 40
DEFAULT_SLOT_DURATION_SECONDS: float = 1.0
DEFAULT_VEHICLE_COUNT: int = 20
DEFAULT_SENSOR_TYPE_COUNT: int = 8
DEFAULT_SENSORS_PER_VEHICLE: int = 4

# Fixed-capacity dimensions for RL action/observation padding.
# These let one TD3 policy keep a stable shape across vehicle-count sweeps.
DEFAULT_MAX_VEHICLE_COUNT_FOR_ACTION_SPACE: int = 80
DEFAULT_MAX_SENSORS_PER_VEHICLE_FOR_ACTION_SPACE: int = 4
DEFAULT_FRESHNESS_THRESHOLD_SLOTS: int = 10
DEFAULT_ACCURACY_THRESHOLD: float = 0.80

DEFAULT_UPLINK_BANDWIDTH_HZ: float = 1_200_000.0
DEFAULT_V2L_MAX_TRANSMIT_POWER_WATT: float = 1.0
DEFAULT_NOISE_POWER_SPECTRAL_DENSITY_WATT_PER_HZ: float = 1.0e-12
DEFAULT_UPLINK_PATHLOSS_EXPONENT: float = 2.5

DEFAULT_LEADER_CPU_FREQUENCY_CYCLES_PER_SECOND: float = 2_500_000.0

DEFAULT_LANE_LENGTH_METER: float = 2000.0
DEFAULT_DEFECTIVE_ZONE_START_METER: float = 0.0
DEFAULT_DEFECTIVE_ZONE_END_METER: float = 2000.0
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
    {
        "name": "LiDAR Point Cloud",
        "priority_weight": 2.8,
        "cpu_cycles_per_bit": 18.0,
        "sensing_delay_slots": 1.2,
        "nominal_data_size_bits": 360_000.0,
    },
    {
        "name": "Brake System Status",
        "priority_weight": 2.6,
        "cpu_cycles_per_bit": 3.0,
        "sensing_delay_slots": 0.2,
        "nominal_data_size_bits": 70_000.0,
    },
    {
        "name": "Road Friction Estimate",
        "priority_weight": 2.5,
        "cpu_cycles_per_bit": 9.0,
        "sensing_delay_slots": 0.6,
        "nominal_data_size_bits": 180_000.0,
    },
    {
        "name": "Lane Marking Detector",
        "priority_weight": 2.4,
        "cpu_cycles_per_bit": 14.0,
        "sensing_delay_slots": 0.9,
        "nominal_data_size_bits": 260_000.0,
    },
    {
        "name": "V2X Beacon Monitor",
        "priority_weight": 2.1,
        "cpu_cycles_per_bit": 4.0,
        "sensing_delay_slots": 0.3,
        "nominal_data_size_bits": 90_000.0,
    },
    {
        "name": "Steering Angle Sensor",
        "priority_weight": 1.9,
        "cpu_cycles_per_bit": 2.5,
        "sensing_delay_slots": 0.2,
        "nominal_data_size_bits": 55_000.0,
    },
    {
        "name": "GNSS Position Fix",
        "priority_weight": 1.7,
        "cpu_cycles_per_bit": 2.0,
        "sensing_delay_slots": 0.1,
        "nominal_data_size_bits": 45_000.0,
    },
    {
        "name": "Acoustic Hazard Sensor",
        "priority_weight": 1.6,
        "cpu_cycles_per_bit": 7.0,
        "sensing_delay_slots": 0.5,
        "nominal_data_size_bits": 160_000.0,
    },
)
DEFAULT_TOTAL_TIMESTEPS: int = 500_000

# CPU-aware Greedy baseline defaults. ``lambda`` is interpreted as a
# score-space penalty coefficient in weighted-AoI units per normalized CPU
# backlog slot. The requested accuracy fraction controls how much of a
# pair's available payload the Greedy policy asks to upload.
DEFAULT_GREEDY_CPU_LAMBDA: float = 3.0
DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION: float = 1.0

# TD3 convergence-training defaults.  These are intentionally centralized so
# convergence experiments, CLI defaults, reports, and plots all stay aligned.
DEFAULT_TD3_CONVERGENCE_EVAL_FREQUENCY_STEPS: int = 25_000
DEFAULT_TD3_CONVERGENCE_EVALUATION_EPISODES: int = 20
DEFAULT_TD3_CONVERGENCE_PATIENCE_EVALUATIONS: int = 10
DEFAULT_TD3_CONVERGENCE_MINIMUM_TIMESTEPS: int = 500_000
DEFAULT_TD3_CONVERGENCE_MAXIMUM_TIMESTEPS: int = 3_000_000
DEFAULT_TD3_CONVERGENCE_MINIMUM_REWARD_IMPROVEMENT: float = 1.0e-6
DEFAULT_TD3_CONVERGENCE_EVALUATION_SEED_START: int = 10_000
DEFAULT_TD3_CONVERGENCE_SB3_LOG_INTERVAL: int = 10
DEFAULT_TD3_CONVERGENCE_OUTPUT_DIRECTORY: str = "results/convergence_td3"
DEFAULT_TD3_CONVERGENCE_BEST_MODEL_NAME: str = "best_td3"
DEFAULT_TD3_CONVERGENCE_LATEST_MODEL_NAME: str = "latest_td3"
DEFAULT_LEARNING_RATE: float = 5.0e-4
DEFAULT_LEARNING_STARTS: int = 10_000
DEFAULT_BUFFER_SIZE: int = 1_000_000
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_GAMMA: float = 0.99
DEFAULT_TAU: float = 0.005
DEFAULT_POLICY_DELAY: int = 2
DEFAULT_TRAIN_FREQUENCY_STEPS: int = 10
DEFAULT_GRADIENT_STEPS: int = 1

# TD3 exploration, target-policy smoothing, and experiment-management defaults.
# The action-noise default is lowered from the earlier 0.20 to 0.10 so the
# CLI can directly compare 0.05 and 0.10 smoke/tuning runs.
DEFAULT_ACTION_NOISE_SIGMA: float = 0.10
DEFAULT_TARGET_POLICY_NOISE: float = 0.20
DEFAULT_TARGET_NOISE_CLIP: float = 0.30
DEFAULT_TENSORBOARD_LOG_DIRECTORY: str = "results/tensorboard"
DEFAULT_MONITOR_LOG_DIRECTORY: str = "results/monitor"
DEFAULT_TD3_CHECKPOINT_FREQUENCY_STEPS: int = 100_000
DEFAULT_TD3_CHECKPOINT_OUTPUT_DIRECTORY: str = "results/checkpoints"
DEFAULT_ACTOR_HIDDEN_LAYERS: tuple[int, int] = (64, 64)
DEFAULT_CRITIC_HIDDEN_LAYERS: tuple[int, int] = (64, 64)
DEFAULT_DEVICE: str = "cpu"

DEFAULT_MONTE_CARLO_TRIAL_COUNT: int = 30
DEFAULT_SEED_START: int = 1

DEFAULT_TRIAL_COUNT_PER_SLOT: int = 10
# PPO defaults. PPO is kept separate from TD3 because it is an on-policy
# algorithm with rollout-buffer and clipped-policy-update hyperparameters.
DEFAULT_PPO_TOTAL_TIMESTEPS: int = 3_000_000
DEFAULT_PPO_LEARNING_RATE: float = 3.0e-4
DEFAULT_PPO_N_STEPS: int = 2_048
DEFAULT_PPO_BATCH_SIZE: int = 256
DEFAULT_PPO_N_EPOCHS: int = 10
DEFAULT_PPO_GAMMA: float = 0.99
DEFAULT_PPO_GAE_LAMBDA: float = 0.95
DEFAULT_PPO_CLIP_RANGE: float = 0.20
DEFAULT_PPO_ENT_COEF: float = 0.0
DEFAULT_PPO_VF_COEF: float = 0.5
DEFAULT_PPO_MAX_GRAD_NORM: float = 0.5
DEFAULT_PPO_TENSORBOARD_LOG_DIRECTORY: str = "results/tensorboard_ppo"
DEFAULT_PPO_MONITOR_LOG_DIRECTORY: str = "results/monitor_ppo"
DEFAULT_PPO_CHECKPOINT_FREQUENCY_STEPS: int = 100_000
DEFAULT_PPO_CHECKPOINT_OUTPUT_DIRECTORY: str = "results/checkpoints_ppo"

# PPO convergence-training defaults. These mirror the TD3 convergence workflow
# while preserving PPO-specific model names and output directories.
DEFAULT_PPO_CONVERGENCE_EVAL_FREQUENCY_STEPS: int = 100_000
DEFAULT_PPO_CONVERGENCE_EVALUATION_EPISODES: int = 50
DEFAULT_PPO_CONVERGENCE_PATIENCE_EVALUATIONS: int = 999_999
DEFAULT_PPO_CONVERGENCE_MINIMUM_TIMESTEPS: int = 3_000_000
DEFAULT_PPO_CONVERGENCE_MAXIMUM_TIMESTEPS: int = 3_000_000
DEFAULT_PPO_CONVERGENCE_MINIMUM_REWARD_IMPROVEMENT: float = 50.0
DEFAULT_PPO_CONVERGENCE_EVALUATION_SEED_START: int = 10_000
DEFAULT_PPO_CONVERGENCE_SB3_LOG_INTERVAL: int = 10
DEFAULT_PPO_CONVERGENCE_OUTPUT_DIRECTORY: str = "results/convergence_ppo"
DEFAULT_PPO_CONVERGENCE_BEST_MODEL_NAME: str = "best_ppo"
DEFAULT_PPO_CONVERGENCE_LATEST_MODEL_NAME: str = "latest_ppo"

