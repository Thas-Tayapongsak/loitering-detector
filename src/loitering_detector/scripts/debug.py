"""Debug entry point: launches the loitering detector with the DebugVisualizer UI.

For local development only. Requires ``opencv-python`` (not headless).
"""

import logging
import sys

from loitering_detector.core.system import LoiteringDetectionSystem
from loitering_detector.visualization import DebugVisualizer


def run(config) -> None:
    """Core debug UI execution logic."""
    import cv2  # Lazy import: keeps this module importable in headless environments.

    with LoiteringDetectionSystem(config=config) as system:
        logging.info("Starting Debug Visualizer. Opening windows...")
        try:
            visualizer = DebugVisualizer(system)
            visualizer.run()
        except cv2.error as e:
            if "gui" in str(e).lower() or "not supported" in str(e).lower():
                logging.critical(
                    "OpenCV GUI support is missing. "
                    "Ensure you have 'opencv-python' installed, NOT 'opencv-python-headless'."
                )
                sys.exit(1)
            else:
                raise
