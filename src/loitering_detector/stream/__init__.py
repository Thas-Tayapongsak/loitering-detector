"""
Video stream capture and management.

Provides the high-level manager and configuration schemas for
handling multiple video sources in parallel.

Classes
-------
StreamManager
    Orchestrator for background frame capture.
StreamConfig
    Configuration schema for a video stream.
"""

import logging

from .manager import StreamManager
from .config import StreamConfig

__all__ = [
    "StreamManager",
    "StreamConfig",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
