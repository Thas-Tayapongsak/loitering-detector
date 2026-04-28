"""
Debugging and calibration utility for visual results.

This module provides the DebugVisualizer which is intended for local
development and troubleshooting. It is not designed for production use
as it requires a GUI environment.
"""

__all__ = ["DebugVisualizer"]

import cv2
import numpy as np

from loitering_detector.core.system import LoiteringDetectionSystem


class DebugVisualizer:
    """
    Component for visualizing detection results and ROI overlays.

    [DEBUG ONLY] This component is intended for calibration and troubleshooting.
    It requires a local GUI environment (OpenCV high-gui) and is NOT suitable
    for headless production server deployments.

    Attributes
    ----------
    system : LoiteringDetectionSystem
        The system instance providing the streams and results.
    closed_windows : set[str]
        A tracking set for windows that have been closed by the user.
    roi_px_polygons : dict[int, np.ndarray]
        Cached ROI polygons in pixel coordinates.
    """

    def __init__(self, system: LoiteringDetectionSystem):
        """
        Initialize the visualizer.

        Parameters
        ----------
        system : LoiteringDetectionSystem
            The detection system to visualize.
        """
        self.system = system
        self.closed_windows: set[str] = set()
        self.roi_px_polygons: dict[int, np.ndarray] = {}

    def run(self) -> None:
        """
        Run the visualization loop, processing detection results and displaying them in real-time.
        """
        streams = list(self.system.streams.values())

        for stream in streams:
            cv2.namedWindow(stream.config.name, cv2.WINDOW_NORMAL)

        for results in self.system.detect():
            if results:
                for stream_id, result in results.items():
                    stream_name = self.system.streams[stream_id].config.name

                    # Plot the frame just-in-time for display
                    annotated_frame = result.plot()
                    height, width = annotated_frame.shape[:2]

                    # Cache pixel-space polygon on first frame (dimensions don't change mid-stream)
                    if stream_id not in self.roi_px_polygons:
                        roi_norm = self.system.roi_polygons[stream_id]
                        self.roi_px_polygons[stream_id] = np.array(
                            [(x * width, y * height) for x, y in roi_norm], np.int32
                        )

                    annotated_frame = cv2.polylines(
                        annotated_frame,
                        [self.roi_px_polygons[stream_id]],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=max(1, int(0.005 * min(height, width))),
                    )
                    cv2.imshow(stream_name, annotated_frame)

            # 'q' or 'ESC'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            # Check for window closes and disconnections
            all_closed = True
            for stream in streams:
                if not stream.is_running:
                    # Stream stopped (disconnection), ensure window is closed
                    if stream.config.name not in self.closed_windows:
                        try:
                            cv2.destroyWindow(stream.config.name)
                        except cv2.error:
                            pass
                        self.closed_windows.add(stream.config.name)
                    continue

                is_visible = False
                if stream.config.name not in self.closed_windows:
                    if (
                        cv2.getWindowProperty(stream.config.name, cv2.WND_PROP_VISIBLE)
                        < 1
                    ):
                        self.closed_windows.add(stream.config.name)
                        stream.stop()
                    else:
                        is_visible = True

                if is_visible:
                    all_closed = False

            if all_closed:
                break

        # Cleanup all windows after the loop finishes
        cv2.destroyAllWindows()
