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

from .config import DetectionConfig, SupervisionByteTrackConfig
from .manager import DetectionManager
from .providers.ultralytics import YOLODetection
from .results import BoundingBox, DetectionResult
from .strategy import DetectionStrategy
from .trackers import SupervisionByteTrack, TrackerInterface

__all__ = [
    "DetectionManager",
    "DetectionConfig",
    "SupervisionByteTrackConfig",
    "DetectionStrategy",
    "YOLODetection",
    "BoundingBox",
    "DetectionResult",
    "TrackerInterface",
    "SupervisionByteTrack",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
