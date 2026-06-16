"""Accuracy model for equations C6a/C6b"""
from __future__ import annotations

from leader_dt import constants

class AccuracyModel:
    def __init__(self, minimum_accuracy_threshold: float) -> None:
        self.minimum_accuracy_threshold = float(minimum_accuracy_threshold)

    def compute_accuracy(self, collected_bits_float: float, available_data_size_bits_float: float) -> float:
        return float(collected_bits_float / max(available_data_size_bits_float, constants.EPSILON_FLOAT))

    def satisfies_accuracy(self, collected_bits_float: float, available_data_size_bits_float: float) -> bool:
        return self.compute_accuracy(collected_bits_float, available_data_size_bits_float) >= self.minimum_accuracy_threshold

    def minimum_required_bits(self, available_data_size_bits_float: float) -> float:
        return self.minimum_accuracy_threshold * available_data_size_bits_float
