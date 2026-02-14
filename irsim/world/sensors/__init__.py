"""
Sensor classes for IR-SIM simulation.

This package contains:
- lidar2d: 2D LiDAR sensor implementation
- range_only: Range-only sensor for tracking goals or pedestrians
- sensor_factory: Sensor factory for creating sensors
"""

from .lidar2d import Lidar2D
from .range_only import RangeOnly
from .sensor_factory import SensorFactory

__all__ = ["Lidar2D", "RangeOnly", "SensorFactory"]
