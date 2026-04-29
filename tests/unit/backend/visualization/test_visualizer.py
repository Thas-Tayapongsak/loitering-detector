"""Tests for the DebugVisualizer component, responsible for real-time video overlays and OpenCV window management."""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from loitering_detector.visualization.visualizer import DebugVisualizer as Visualizer

# Constants

ESC_KEY = 27
QUIT_KEY = ord("q")

STREAM_1_ID = 1
WINDOW_NAME_PREFIX = "Stream-"

WINDOW_CLOSED = 0
WINDOW_OPEN = 1


# Fixtures


def _make_mock_system(stream_ids=None, running=None):
    """Build a mock system with configurable streams."""
    if stream_ids is None:
        stream_ids = [STREAM_1_ID]
    if running is None:
        running = dict.fromkeys(stream_ids, True)

    system = MagicMock()
    streams = {}
    for sid in stream_ids:
        s = MagicMock()
        s.config.id = sid
        s.config.name = f"Stream-{sid}"
        type(s).is_running = PropertyMock(return_value=running.get(sid, True))
        streams[sid] = s

    system.streams = streams
    system.roi_polygons = {
        sid: [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)] for sid in stream_ids
    }
    return system


@pytest.fixture
def mock_system():
    return _make_mock_system([STREAM_1_ID])


# Tests


class TestVisualizerLifecycle:
    """Tests for initialization and main loop control."""

    def test_initialization(self, mock_system):
        """
        Verify that the visualizer correctly initializes its internal state with a provided detection system.

        Given: a mock loitering detection system
        When: the Visualizer is instantiated
        Then: the system reference is stored and the closed windows tracking set is empty
        """
        # When: the Visualizer is initialized
        vis = Visualizer(mock_system)

        # Then: internal state is correct
        assert vis.system is mock_system
        assert vis.closed_windows == set()

    @patch("loitering_detector.visualization.visualizer.cv2")
    def test_esc_key_breaks_loop(self, mock_cv2, mock_system):
        """
        Verify that the visualization loop terminates when the ESC key is pressed.

        Given: a running visualizer
        When: the user presses the ESC key (ASCII 27)
        Then: the main execution loop terminates
        """
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result.orig_shape = (100, 100)

        mock_system.detect.return_value = iter([{STREAM_1_ID: mock_result}])
        # Given: user presses ESC
        mock_cv2.waitKey.return_value = ESC_KEY

        vis = Visualizer(mock_system)

        # When: running the visualizer
        vis.run()

        # Then: the loop terminates after waitKey returns ESC
        mock_cv2.waitKey.assert_called()


class TestVisualizerWindowManagement:
    """Tests for OpenCV window lifecycle."""

    @patch("loitering_detector.visualization.visualizer.cv2")
    def test_closed_window_stops_stream(self, mock_cv2, mock_system):
        """
        Verify that closing a stream window in the UI causes the corresponding stream to stop.

        Given: an active stream window
        When: the window is closed by the user (simulated by visibility check)
        Then: the corresponding StreamManager is automatically stopped
        """
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result.orig_shape = (100, 100)

        mock_system.detect.return_value = iter([{STREAM_1_ID: mock_result}, None])
        stream = mock_system.streams[STREAM_1_ID]

        # Given: the window visibility check returns closed
        mock_cv2.waitKey.return_value = 0
        mock_cv2.getWindowProperty.return_value = WINDOW_CLOSED

        vis = Visualizer(mock_system)

        # When: the visualizer runs its loop
        vis.run()

        # Then: the stream's stop method is called
        stream.stop.assert_called()

    @patch("loitering_detector.visualization.visualizer.cv2")
    def test_disconnected_stream_cleans_up_window(self, mock_cv2, mock_system):
        """
        Verify that the visualizer automatically destroys windows for streams that have disconnected.

        Given: a stream that is no longer running
        When: the visualizer processes its status
        Then: the OpenCV window for that stream is destroyed
        """
        mock_system.detect.return_value = iter([None])
        stream = mock_system.streams[STREAM_1_ID]
        # Given: the stream has disconnected
        type(stream).is_running = PropertyMock(return_value=False)

        vis = Visualizer(mock_system)

        # When: the visualizer runs
        vis.run()

        # Then: the window is destroyed
        mock_cv2.destroyWindow.assert_called_with(f"{WINDOW_NAME_PREFIX}{STREAM_1_ID}")


class TestVisualizerRendering:
    """Tests for frame rendering and ROI overlays."""

    @patch("loitering_detector.visualization.visualizer.cv2")
    def test_roi_drawing_and_caching(self, mock_cv2, mock_system):
        """
        Verify that ROI polygons are correctly scaled to pixel space, drawn on the frame, and cached.

        Given: a system with normalized ROI coordinates
        When: a frame is rendered for display
        Then: the ROI is drawn in pixel space using polylines and cached for future frames
        """
        mock_result = MagicMock()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        mock_result.plot.return_value = frame
        mock_result.orig_shape = (200, 400)

        mock_system.detect.return_value = iter([{STREAM_1_ID: mock_result}])
        mock_cv2.waitKey.return_value = QUIT_KEY
        mock_cv2.getWindowProperty.return_value = WINDOW_OPEN

        vis = Visualizer(mock_system)

        # When: rendering the frame
        vis.run()

        # Then: polylines is called to draw the ROI
        assert mock_cv2.polylines.called
        # Then: ROI coordinates are cached in pixel space
        assert STREAM_1_ID in vis.roi_px_polygons
