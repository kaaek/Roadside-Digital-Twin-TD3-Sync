"""Shared type aliases and enums."""
from __future__ import annotations

from enum import Enum
from typing import NewType

SlotIndex = NewType("SlotIndex", int)
VehicleId = NewType("VehicleId", int)
SensorTypeId = NewType("SensorTypeId", int)
SensorPairId = NewType("SensorPairId", int)

class ZoneName(str, Enum):
    DEFECTIVE = "Zone B"

class SchedulingMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
