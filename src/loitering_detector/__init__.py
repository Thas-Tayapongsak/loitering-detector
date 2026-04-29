"""
Loitering Detector - A modular system for real-time loitering detection.

This module exposes the primary interface for building and running
the loitering detection system.

Classes
-------
LoiteringDetectionSystem
    The central coordinator for detection and monitoring.
LoiteringEngine
    The core logic for tracking object loitering state.
AlertManager
    Manager for loitering notifications and persistence.
StreamManager
    Orchestrator for video stream capture.
DetectionManager
    Backend manager for vision inference and tracking.
"""

from loitering_detector import config
from loitering_detector.core.alerts import AlertManager
from loitering_detector.core.loitering import LoiteringEngine
from loitering_detector.core.system import LoiteringDetectionSystem
from loitering_detector.detection.manager import DetectionManager
from loitering_detector.stream.manager import StreamManager

__all__ = [
    "LoiteringDetectionSystem",
    "LoiteringEngine",
    "AlertManager",
    "StreamManager",
    "DetectionManager",
    "config",
]
