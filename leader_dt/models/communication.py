"""Vehicle-to-leader uplink communication model."""
from __future__ import annotations

import numpy as np
from leader_dt.config import CommunicationConfig

class UplinkRateModel:
    """Computes R_i^up(t) with the 3GPP TR 37.885 highway LOS pathloss and Shannon capacity.

    PL_dB = 32.4 + 20 log10(fc_GHz) + 10 n log10(d), minus a per-link shadowing
    term in dB (positive shadowing = stronger signal).
    """

    def __init__(self, communication_config: CommunicationConfig) -> None:
        self.communication_config = communication_config

    def compute_pathloss_db(self, distance_meter: float | np.ndarray) -> np.ndarray:
        config = self.communication_config
        bounded_distance = np.maximum(np.asarray(distance_meter, dtype=np.float64), config.reference_distance_meter)
        return (
            32.4
            + 20.0 * np.log10(config.carrier_frequency_ghz)
            + 10.0 * config.pathloss_exponent * np.log10(bounded_distance / config.reference_distance_meter)
        )

    def compute_rate_vector_bits_per_second(
        self,
        distance_meter_array: np.ndarray,
        shadowing_db_array: np.ndarray | None = None,
    ) -> np.ndarray:
        config = self.communication_config
        channel_gain_db = -self.compute_pathloss_db(distance_meter_array)
        if shadowing_db_array is not None:
            channel_gain_db = channel_gain_db + np.asarray(shadowing_db_array, dtype=np.float64)
        signal_to_noise_ratio = (
            config.max_transmit_power_watt * 10.0 ** (channel_gain_db / 10.0)
        ) / max(config.noise_power_spectral_density_watt_per_hz * config.uplink_bandwidth_hz, 1e-30)
        return np.asarray(config.uplink_bandwidth_hz * np.log2(1.0 + signal_to_noise_ratio), dtype=np.float64)

    def compute_rate_bits_per_second(self, distance_meter: float, shadowing_db: float = 0.0) -> float:
        return float(self.compute_rate_vector_bits_per_second(np.array([distance_meter]), np.array([shadowing_db]))[0])
