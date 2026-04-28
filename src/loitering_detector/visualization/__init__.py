"""
Visual feedback and results overlay.

Provides tools for visualizing detection and loitering results on
video frames.

Classes
-------
DebugVisualizer
    Component for drawing detection overlays and alerts.
"""

import logging

from .visualizer import DebugVisualizer

__all__ = [
    "DebugVisualizer",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
