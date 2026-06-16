"""Vehicle-to-leader uplink communication model."""
from __future__ import annotations

import numpy as np
from leader_dt.config import CommunicationConfig

class UplinkRateModel:
    """Computes R_i^up(t) using the Okumara-Hata pathloss equation."""

    def __init__(self, communication_config: CommunicationConfig) -> None:
        self.communication_config = communication_config

    def compute_rate_bits_per_second(self, distance_meter: float) -> float:
        config = self.communication_config
        bounded_distance = max(float(distance_meter), config.reference_distance_meter)
        pathloss_gain = (bounded_distance / config.reference_distance_meter) ** (-config.pathloss_exponent)
        signal_to_noise_ratio = (
            config.max_transmit_power_watt * pathloss_gain
        ) / max(config.noise_power_spectral_density_watt_per_hz * config.uplink_bandwidth_hz, 1e-12)
        return float(config.uplink_bandwidth_hz * np.log2(1.0 + signal_to_noise_ratio))

    def compute_rate_vector_bits_per_second(self, distance_meter_array: np.ndarray) -> np.ndarray:
        return np.asarray([self.compute_rate_bits_per_second(distance) for distance in distance_meter_array], dtype=np.float64)
