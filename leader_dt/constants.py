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


DEFAULT_FRESHNESS_THRESHOLD_SLOTS: int = 10
DEFAULT_ACCURACY_THRESHOLD: float = 0.80

# Vehicle-to-leader sidelink, following 3GPP TR 37.885 (highway LOS, 5.9 GHz ITS band):
# PL_dB = 32.4 + 20 log10(fc_GHz) + 10 n log10(d_m), log-normal shadowing sigma = 3 dB,
# UE transmit power 23 dBm, noise figure 9 dB.  The leader's uploads get one
# LTE-V2X sub-channel (10 resource blocks = 1.8 MHz) of the 10 MHz ITS channel,
# which is shared with the surrounding V2X traffic.
DEFAULT_UPLINK_BANDWIDTH_HZ: float = 1_800_000.0
DEFAULT_V2L_MAX_TRANSMIT_POWER_WATT: float = 0.2  # 23 dBm
# Thermal noise -174 dBm/Hz plus a 9 dB receiver noise figure = -165 dBm/Hz.
DEFAULT_NOISE_POWER_SPECTRAL_DENSITY_WATT_PER_HZ: float = 10.0 ** ((-174.0 + 9.0) / 10.0) / 1000.0
DEFAULT_UPLINK_PATHLOSS_EXPONENT: float = 2.0
DEFAULT_CARRIER_FREQUENCY_GHZ: float = 5.9
DEFAULT_SHADOWING_STD_DB: float = 3.0

# One application core of an automotive SoC / on-board unit (1.5-2.5 GHz class).
DEFAULT_LEADER_CPU_FREQUENCY_CYCLES_PER_SECOND: float = 2.0e9

DEFAULT_LANE_LENGTH_METER: float = 2000.0
DEFAULT_DEFECTIVE_ZONE_START_METER: float = 0.0
DEFAULT_DEFECTIVE_ZONE_END_METER: float = 2000.0
DEFAULT_VEHICLE_SPEED_METER_PER_SECOND: float = 23
DEFAULT_VEHICLE_SPEED_JITTER_STD_METER_PER_SECOND: float = 4

# Data-size sampling: each delta_i(t) is sampled uniformly around the nominal
# sensor-type payload size. This keeps the paper's time-dependent delta_i(t)
# while remaining simple and reproducible.
DEFAULT_DATA_SIZE_LOW_MULTIPLIER: float = 0.50
DEFAULT_DATA_SIZE_HIGH_MULTIPLIER: float = 2.00

# Sample arrival process: each vehicle-sensor pair receives Poisson(lambda) new
# samples per slot and keeps only its latest unsent sample, so a pair holds a
# fresh sample in a slot with probability 1 - exp(-lambda).  ``None`` keeps the
# legacy model where every pair has a fresh sample in every slot.  The default
# 0.10 gives each pair a new sample in ~10% of slots (about one every 10 s).
DEFAULT_SAMPLE_ARRIVAL_RATE_PER_SLOT: float | None = 0.10

# Sensor definitions are ordered by sensor_type_id.  Payload sizes are one
# update for the digital twin: a compressed frame / point cloud for perception
# sensors, a short status message or sample window for telemetry.  Processing
# cost follows the MEC literature: hundreds of cycles/bit for perception data
# (detection, fusion), tens for parsing telemetry.  Sensing delay is the
# capture/acquisition time in slots (1 slot = 1 s): ~33 ms camera frame, ~100 ms
# LiDAR sweep or GNSS fix, ~10 ms for bus-read telemetry.  The default 8 types
# (the first 8 entries) include LiDAR so that heavy perception data is present.
DEFAULT_SENSOR_DEFINITIONS: tuple[dict, ...] = (
    {
        "name": "Front Camera",
        "priority_weight": 3.0,
        "cpu_cycles_per_bit": 500.0,
        "sensing_delay_slots": 0.033,
        "nominal_data_size_bits": 3_000_000.0,
    },
    {
        "name": "Radar",
        "priority_weight": 2.2,
        "cpu_cycles_per_bit": 100.0,
        "sensing_delay_slots": 0.05,
        "nominal_data_size_bits": 400_000.0,
    },
    {
        "name": "Engine Temperature",
        "priority_weight": 2.0,
        "cpu_cycles_per_bit": 20.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 8_000.0,
    },
    {
        "name": "Battery BMS",
        "priority_weight": 1.8,
        "cpu_cycles_per_bit": 30.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 16_000.0,
    },
    {
        "name": "Tyre Pressure TPMS",
        "priority_weight": 1.5,
        "cpu_cycles_per_bit": 10.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 1_000.0,
    },
    {
        "name": "IMU Accelerometer",
        "priority_weight": 1.3,
        "cpu_cycles_per_bit": 50.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 10_000.0,
    },
    {
        "name": "LiDAR Point Cloud",
        "priority_weight": 2.8,
        "cpu_cycles_per_bit": 800.0,
        "sensing_delay_slots": 0.1,
        "nominal_data_size_bits": 10_000_000.0,
    },
    {
        "name": "Fuel Level",
        "priority_weight": 1.0,
        "cpu_cycles_per_bit": 10.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 250.0,
    },
    {
        "name": "Ambient Weather",
        "priority_weight": 1.2,
        "cpu_cycles_per_bit": 10.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 1_000.0,
    },
    {
        "name": "Brake System Status",
        "priority_weight": 2.6,
        "cpu_cycles_per_bit": 20.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 1_000.0,
    },
    {
        "name": "Road Friction Estimate",
        "priority_weight": 2.5,
        "cpu_cycles_per_bit": 100.0,
        "sensing_delay_slots": 0.05,
        "nominal_data_size_bits": 4_000.0,
    },
    {
        "name": "Lane Marking Detector",
        "priority_weight": 2.4,
        "cpu_cycles_per_bit": 200.0,
        "sensing_delay_slots": 0.033,
        "nominal_data_size_bits": 16_000.0,
    },
    {
        "name": "V2X Beacon Monitor",
        "priority_weight": 2.1,
        "cpu_cycles_per_bit": 50.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 20_000.0,
    },
    {
        "name": "Steering Angle Sensor",
        "priority_weight": 1.9,
        "cpu_cycles_per_bit": 10.0,
        "sensing_delay_slots": 0.01,
        "nominal_data_size_bits": 1_000.0,
    },
    {
        "name": "GNSS Position Fix",
        "priority_weight": 1.7,
        "cpu_cycles_per_bit": 20.0,
        "sensing_delay_slots": 0.1,
        "nominal_data_size_bits": 1_000.0,
    },
    {
        "name": "Acoustic Hazard Sensor",
        "priority_weight": 1.6,
        "cpu_cycles_per_bit": 300.0,
        "sensing_delay_slots": 0.05,
        "nominal_data_size_bits": 256_000.0,
    },
)
DEFAULT_TOTAL_TIMESTEPS: int = 500_000

# CPU-aware Greedy baseline defaults. ``lambda`` is interpreted as a
# score-space penalty coefficient in weighted-AoI units per normalized CPU
# backlog slot. The requested accuracy fraction controls how much of a
# pair's available payload the Greedy policy asks to upload.  10.0 was picked on
# the realistic defaults (50 seeds): same weighted AoI as 3.0, with terminal CPU
# violations down from 62% to 22% of episodes.
DEFAULT_GREEDY_CPU_LAMBDA: float = 10.0
DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION: float = 1.0

# TD3 convergence-training defaults.  These are intentionally centralized so
# convergence experiments, CLI defaults, reports, and plots all stay aligned.
DEFAULT_TD3_CONVERGENCE_EVAL_FREQUENCY_STEPS: int = 100_000
DEFAULT_TD3_CONVERGENCE_EVALUATION_EPISODES: int = 50
DEFAULT_TD3_CONVERGENCE_PATIENCE_EVALUATIONS: int = 999_999
# DEFAULT_TD3_CONVERGENCE_MINIMUM_TIMESTEPS: int = 10_000_000
# DEFAULT_TD3_CONVERGENCE_MAXIMUM_TIMESTEPS: int = 10_000_000
DEFAULT_TD3_CONVERGENCE_MINIMUM_TIMESTEPS: int = 1_000_000
DEFAULT_TD3_CONVERGENCE_MAXIMUM_TIMESTEPS: int = 1_000_000
DEFAULT_TD3_CONVERGENCE_MINIMUM_REWARD_IMPROVEMENT: float = 0.0
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
DEFAULT_ACTION_NOISE_SIGMA: float = 0.05
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
# DEFAULT_PPO_CONVERGENCE_MINIMUM_TIMESTEPS: int = 10_000_000
# DEFAULT_PPO_CONVERGENCE_MAXIMUM_TIMESTEPS: int = 10_000_000
DEFAULT_PPO_CONVERGENCE_MINIMUM_TIMESTEPS: int = 1_000_000
DEFAULT_PPO_CONVERGENCE_MAXIMUM_TIMESTEPS: int = 1_000_000
DEFAULT_PPO_CONVERGENCE_MINIMUM_REWARD_IMPROVEMENT: float = 0.0
DEFAULT_PPO_CONVERGENCE_EVALUATION_SEED_START: int = 10_000
DEFAULT_PPO_CONVERGENCE_SB3_LOG_INTERVAL: int = 10
DEFAULT_PPO_CONVERGENCE_OUTPUT_DIRECTORY: str = "results/convergence_ppo"
DEFAULT_PPO_CONVERGENCE_BEST_MODEL_NAME: str = "best_ppo"
DEFAULT_PPO_CONVERGENCE_LATEST_MODEL_NAME: str = "latest_ppo"

