"""Production headless entry point for the loitering detector."""

import logging

from loitering_detector.config import SystemConfig
from loitering_detector.core.system import LoiteringDetectionSystem


def run(config: SystemConfig) -> None:
    """Core headless execution logic."""
    with LoiteringDetectionSystem(config=config) as system:
        logging.info("Loitering Detection System started in production mode.")
        for _ in system.detect():
            pass
