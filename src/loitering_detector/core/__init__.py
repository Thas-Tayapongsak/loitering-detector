"""
Core logic and orchestration for loitering detection.

Exposes the central system coordinator, the loitering state engine,
and the alert notification manager.

Classes
-------
LoiteringDetectionSystem
    The high-level orchestrator for the entire application.
LoiteringEngine
    The state engine for calculating loitering durations.
AlertManager
    The component responsible for managing loitering alerts.
"""

import logging

from .alerts import AlertManager
from .interfaces import DetectedObject, GeometryEngine, LoiteringStateRepository
from .loitering import LoiteringEngine
from .system import LoiteringDetectionSystem

__all__ = [
    "LoiteringDetectionSystem",
    "LoiteringEngine",
    "AlertManager",
    "DetectedObject",
    "LoiteringStateRepository",
    "GeometryEngine",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
