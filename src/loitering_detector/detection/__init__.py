"""
Computer vision detection and object tracking.

Exposes the detection manager, configuration schemas, and
pluggable detection strategies.

Classes
-------
DetectionManager
    Backend coordinator for inference and tracking.
DetectionConfig
    Configuration schema for the detection backend.
DetectionStrategy
    Abstract base class for detection implementations.
YOLODetection
    YOLO-based detection strategy.
"""

import logging

from .manager import DetectionManager
from .config import DetectionConfig
from .strategy import DetectionStrategy, YOLODetection

__all__ = [
    "DetectionManager",
    "DetectionConfig",
    "DetectionStrategy",
    "YOLODetection",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
