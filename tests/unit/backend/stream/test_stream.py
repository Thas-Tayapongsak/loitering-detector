"""Tests for stream management, ingestion strategies, and background capture threads."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loitering_detector.stream.config import StreamConfig
from loitering_detector.stream.manager import StreamManager
from loitering_detector.stream.strategy import LiveStream

# Constants

STREAM_ID = 1
STREAM_SOURCE = "rtsp://localhost:554/stream"
STREAM_NAME = "test_stream"
STREAM_TIMEOUT = 0.1

MJPEG_SOURCE = "http://localhost:8080/stream"
RTSP_PORT = 554

FRAME_ID_LIMIT = 1000000


# Fixtures


@pytest.fixture
def stream_config() -> StreamConfig:
    """Default stream configuration for testing."""
    return StreamConfig(
        id=STREAM_ID, source=STREAM_SOURCE, name=STREAM_NAME, timeout=STREAM_TIMEOUT
    )


@pytest.fixture
def livestream(stream_config: StreamConfig) -> LiveStream:
    """A LiveStream instance with a fast timeout."""
    return LiveStream(source=stream_config.source, timeout=stream_config.timeout)


# Tests


class TestStreamManager:
    """Tests for the StreamManager orchestrator."""

    @patch("loitering_detector.stream.manager.LiveStream")
    def test_lifecycle(
        self,
        mock_strategy_cls: MagicMock,
        stream_config: StreamConfig,
        mock_frame: np.ndarray,
    ) -> None:
        """
        Test the full lifecycle of the StreamManager from startup through frame reading to shutdown.

        Given: a valid stream configuration
        When: the StreamManager is started and a frame is read
        Then: the frame matches the source and resources are released on exit
        """
        mock_strategy = mock_strategy_cls.return_value
        mock_strategy.get_frame.return_value = (mock_frame, 1)
        mock_strategy.disconnected = False

        # When: the StreamManager is started
        with StreamManager(stream_config) as manager:
            assert manager.is_running is True
            # When: a frame is read
            frame = manager.read()
            # Then: the correct frame is returned
            assert frame is not None
            assert np.array_equal(frame, mock_frame)

        # Then: resources are released and the manager is stopped
        assert manager.is_running is False
        assert mock_strategy.release.called

    @patch("loitering_detector.stream.manager.LiveStream")
    def test_auto_cleanup_on_failure(
        self, mock_strategy_cls: MagicMock, stream_config: StreamConfig
    ) -> None:
        """
        Test that the StreamManager automatically stops and cleans up if the underlying strategy disconnects.

        Given: a running StreamManager
        When: the underlying stream strategy disconnects
        Then: the manager stops running and returns None for further reads
        """
        mock_strategy = mock_strategy_cls.return_value
        mock_strategy.disconnected = True

        manager = StreamManager(stream_config)
        manager.is_running = True
        manager._strategy = mock_strategy

        # When: reading from a disconnected strategy
        frame = manager.read()

        # Then: the manager stops and returns None
        assert frame is None
        assert manager.is_running is False


class TestLiveStreamConnection:
    """Tests for LiveStream connection and reachability logic."""

    @patch("cv2.VideoCapture")
    @patch("loitering_detector.stream.strategy.socket.create_connection")
    def test_connect_mjpeg(
        self,
        mock_socket: MagicMock,
        mock_video_capture: MagicMock,
        stream_config: StreamConfig,
        mock_frame: np.ndarray,
    ) -> None:
        """
        Test connecting to a network-based MJPEG or RTSP stream via OpenCV's VideoCapture.

        Given: a valid MJPEG/RTSP source URL
        When: connect() is called
        Then: the underlying VideoCapture is initialized correctly
        """
        stream_config.source = MJPEG_SOURCE
        mock_cap = mock_video_capture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)

        strategy = LiveStream(stream_config.source, timeout=stream_config.timeout)

        # When: the strategy attempts to connect
        strategy.connect()

        # Then: VideoCapture is opened for the source
        assert strategy._cap is not None
        assert mock_video_capture.called

    @patch("cv2.VideoCapture")
    def test_connect_local_device(
        self,
        mock_video_capture: MagicMock,
        livestream: LiveStream,
        mock_frame: np.ndarray,
    ) -> None:
        """
        Test connecting to a local device camera via integer index.

        Given: an integer camera source (e.g., 0)
        When: connect() is called
        Then: the source is resolved as an int and VideoCapture is initialized
        """
        livestream.source = 0
        mock_cap = mock_video_capture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)

        # When: attempting to connect
        livestream.connect()

        # Then: the resolved source is an int and connection succeeds
        assert livestream._cap is not None
        assert livestream.resolved_source == 0

    @patch("loitering_detector.stream.strategy.socket.create_connection")
    def test_source_reachability(
        self, mock_conn: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify socket-based reachability check for RTSP/HTTP sources.

        Given: a network stream source
        When: checking source reachability
        Then: a socket connection is attempted on the correct port
        """
        # When: checking reachability for an RTSP source
        reachable = livestream._is_source_reachable("rtsp://example.com/live")

        # Then: it returns True (mocked) and uses the default RTSP port
        assert reachable is True
        mock_conn.assert_called_with(
            ("example.com", RTSP_PORT), timeout=livestream.timeout
        )

    @patch("loitering_detector.stream.strategy.socket.create_connection")
    def test_source_reachability_http_https(
        self, mock_conn: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify reachability check infers correct default ports for HTTP and HTTPS.

        Given: HTTP and HTTPS stream sources
        When: checking source reachability
        Then: the correct port (80 or 443) is used for the connection attempt
        """
        # When: checking HTTP
        livestream._is_source_reachable("http://example.com/stream.mjpg")
        # Then: it uses port 80
        mock_conn.assert_called_with(("example.com", 80), timeout=livestream.timeout)

        # When: checking HTTPS
        livestream._is_source_reachable("https://example.com/stream.mjpg")
        # Then: it uses port 443
        mock_conn.assert_called_with(("example.com", 443), timeout=livestream.timeout)

    def test_source_reachability_local_path(self, livestream: LiveStream) -> None:
        """
        Verify reachability check bypasses socket connections for local file paths.

        Given: a local file path source
        When: checking source reachability
        Then: it immediately returns True without attempting network connections
        """
        # When: checking a local path
        reachable = livestream._is_source_reachable("/local/path/video.mp4")

        # Then: it is considered reachable
        assert reachable is True

    @patch("cv2.VideoCapture")
    def test_connect_aborted_when_stopped(
        self, mock_video_capture: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that connect() aborts early if the stream is already marked as stopped.

        Given: a LiveStream marked as stopped
        When: connect() is called
        Then: the underlying capture device is not initialized
        """
        livestream.source = 0
        livestream._is_stopped = True

        # When: attempting to connect
        livestream.connect()

        # Then: the connection is aborted
        assert livestream._cap is None

    @patch("cv2.VideoCapture")
    def test_connect_fails_not_opened(
        self, mock_video_capture: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that a ValueError is raised if the video capture fails to open.

        Given: a source that VideoCapture cannot open
        When: connect() is called
        Then: a ValueError is raised and resources are released
        """
        livestream.source = 0
        mock_cap = mock_video_capture.return_value
        mock_cap.isOpened.return_value = False

        # When/Then: connecting raises ValueError
        with pytest.raises(ValueError, match="Unable to open video source"):
            livestream.connect()

    @patch("cv2.VideoCapture")
    def test_connect_fails_no_frames(
        self, mock_video_capture: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that a ConnectionError is raised if the video capture opens but cannot read frames.

        Given: a source that opens but returns no frames on the first read
        When: connect() is called
        Then: a ConnectionError is raised
        """
        livestream.source = 0
        mock_cap = mock_video_capture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        # When/Then: connecting raises ConnectionError
        with pytest.raises(ConnectionError, match="Unable to read from video source"):
            livestream.connect()

    @patch("cv2.VideoCapture")
    def test_connect_unexpected_state_none_cap(
        self, mock_video_capture: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify RuntimeError is raised if internal state is invalid after initialization.

        Given: a mocked environment where _cap remains None unexpectedly
        When: connect() is called
        Then: a RuntimeError is raised
        """
        livestream.source = 0
        mock_video_capture.return_value = None

        # When/Then: connecting raises RuntimeError
        with pytest.raises(RuntimeError, match="Unexpected state"):
            livestream.connect()

    @patch("cv2.VideoCapture")
    @patch("loitering_detector.stream.strategy.yt_dlp.YoutubeDL")
    @patch.object(LiveStream, "_is_source_reachable")
    def test_connect_youtube_resolution(
        self,
        mock_reachable: MagicMock,
        mock_yt: MagicMock,
        mock_cv2: MagicMock,
        livestream: LiveStream,
        mock_frame: np.ndarray,
    ) -> None:
        """
        Verify that YouTube URLs are resolved to direct video streams via yt-dlp.

        Given: a YouTube video URL
        When: connect() is called
        Then: yt-dlp extracts the direct URL before connecting
        """
        livestream.source = "https://youtube.com/watch?v=123"
        mock_reachable.return_value = True

        mock_ydl = mock_yt.return_value.__enter__.return_value
        mock_ydl.extract_info.return_value = {"url": "http://resolved.url/video.mp4"}

        mock_cap = mock_cv2.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)

        # When: connecting to a youtube source
        livestream.connect()

        # Then: the source is resolved correctly
        assert livestream.resolved_source == "http://resolved.url/video.mp4"
        mock_yt.assert_called_once()

    @patch("loitering_detector.stream.strategy.yt_dlp.YoutubeDL")
    def test_connect_youtube_resolution_error(
        self, mock_yt: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that a ConnectionError is raised if yt-dlp fails to extract the video URL.

        Given: a YouTube video URL that cannot be resolved
        When: connect() is called
        Then: a ConnectionError is raised
        """
        livestream.source = "https://youtube.com/watch?v=123"

        from yt_dlp.utils import DownloadError

        mock_ydl = mock_yt.return_value.__enter__.return_value
        mock_ydl.extract_info.side_effect = DownloadError("Extraction failed")

        # When/Then: connecting raises ConnectionError
        with pytest.raises(ConnectionError, match="Unable to extract video URL"):
            livestream._resolve_youtube_url()


class TestLiveStreamBuffer:
    """Tests for frame buffering and ID management."""

    def test_frame_id_management(
        self, livestream: LiveStream, mock_frame: np.ndarray
    ) -> None:
        """
        Test that frame IDs increment and wrap around at the configured limit.

        Given: a LiveStream strategy
        When: frames are added to the buffer
        Then: the frame ID increments or wraps correctly
        """
        # When: the first frame is buffered
        livestream._update_buffer(mock_frame)
        # Then: ID increments to 1
        assert livestream._frame_id == 1

        # When: the ID is near the limit and a new frame is buffered
        livestream._frame_id = FRAME_ID_LIMIT - 1
        livestream._update_buffer(mock_frame)
        # Then: ID wraps back to 0
        assert livestream._frame_id == 0

    def test_read_only_view(
        self, livestream: LiveStream, mock_frame: np.ndarray
    ) -> None:
        """
        Verify that returned frames are read-only to prevent downstream corruption.

        Given: a buffered frame
        When: get_frame() is called
        Then: the returned frame is marked as non-writeable
        """
        livestream._update_buffer(mock_frame)

        # When: retrieving the frame
        frame, _ = livestream.get_frame()

        # Then: the frame buffer is read-only
        assert frame is not None
        assert not frame.flags.writeable

    def test_get_frame_empty_buffer(self, livestream: LiveStream) -> None:
        """
        Verify that get_frame returns None and -1 if the buffer is empty.

        Given: a newly initialized LiveStream with an empty buffer
        When: get_frame() is called
        Then: it safely returns None and an invalid frame ID
        """
        # When: retrieving from an empty buffer
        frame, frame_id = livestream.get_frame()

        # Then: no frame is returned
        assert frame is None
        assert frame_id == -1


class TestLiveStreamResilience:
    """Tests for timeout, reconnection, and error handling."""

    def test_timeout_detection(self, livestream: LiveStream) -> None:
        """
        Test that the strategy correctly detects and handles network timeouts.

        Given: an active stream strategy where reads return no frames
        When: the elapsed time exceeds the timeout threshold
        Then: the strategy is marked as disconnected
        """
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_cap.isOpened.return_value = True
        livestream._cap = mock_cap

        # Given: time has passed beyond the timeout
        livestream._last_read_time = time.perf_counter() - 1.0

        with patch("time.sleep"):
            # When: the update loop processes the lack of frames
            livestream._update_loop()

        # Then: it detects a timeout and disconnects
        assert livestream.disconnected is True

    def test_update_loop_sleeps_on_temporary_failure(
        self, livestream: LiveStream
    ) -> None:
        """
        Verify that the update loop sleeps and retries if a frame read temporarily fails.

        Given: a stream that fails to read once, but succeeds on the next attempt
        When: the update loop runs
        Then: it sleeps briefly before retrying, without disconnecting
        """
        mock_cap = MagicMock()
        # Mock reading to fail once, then succeed
        mock_cap.read.side_effect = [(False, None), (True, np.zeros((10, 10, 3)))]
        mock_cap.isOpened.return_value = True
        livestream._cap = mock_cap
        livestream._last_read_time = time.perf_counter()

        def stop_loop(*args: object) -> None:
            livestream._is_stopped = True

        with patch("time.sleep", side_effect=stop_loop) as mock_sleep:
            # When: the update loop runs
            livestream._update_loop()

            # Then: it slept to wait for the next frame
            mock_sleep.assert_called_once()
            assert livestream.disconnected is False

    @patch("cv2.VideoCapture")
    def test_update_loop_cv2_error_handling(
        self, mock_cv2: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that the update loop catches cv2.error exceptions and handles them gracefully.

        Given: a stream where cv2.VideoCapture.read() raises cv2.error
        When: the update loop processes the exception
        Then: it treats it as a temporary failure and does not crash
        """
        mock_cap = MagicMock()
        import cv2

        mock_cap.read.side_effect = cv2.error("Simulated hardware error")
        mock_cap.isOpened.return_value = True
        livestream._cap = mock_cap
        livestream._last_read_time = time.perf_counter()

        def stop_loop(*args: object) -> None:
            livestream._is_stopped = True

        with patch("time.sleep", side_effect=stop_loop) as mock_sleep:
            # When: the update loop encounters the error
            livestream._update_loop()

            # Then: it handled it cleanly
            mock_sleep.assert_called_once()
            assert livestream.disconnected is False

    @patch.object(LiveStream, "connect")
    def test_reconnection_attempt(
        self, mock_connect: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that the stream attempts to reconnect if it detects the capture device is closed.

        Given: a disconnected stream
        When: its connection status is checked
        Then: it calls connect() to attempt recovery
        """
        livestream._cap = None

        def set_cap() -> None:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            livestream._cap = mock_cap

        mock_connect.side_effect = set_cap

        # When: checking if connected
        is_connected = livestream._is_cap_connected()

        # Then: it attempted reconnection and succeeded
        assert is_connected is True
        mock_connect.assert_called_once()

    @patch.object(LiveStream, "connect")
    def test_reconnection_failure_handling(
        self, mock_connect: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that if a reconnection attempt fails, the stream is marked as disconnected.

        Given: a disconnected stream that cannot reach its source
        When: its connection status is checked and reconnection fails
        Then: the stream is cleanly marked as disconnected
        """
        livestream._cap = None
        mock_connect.side_effect = ValueError("Network Error")

        # When: checking if connected and the attempt fails
        is_connected = livestream._is_cap_connected()

        # Then: it failed gracefully
        assert is_connected is False
        assert livestream.disconnected is True

    @patch.object(LiveStream, "_release_cap")
    def test_release_terminates_cleanly(
        self, mock_release: MagicMock, livestream: LiveStream
    ) -> None:
        """
        Verify that calling release() stops the loop and releases the capture device.

        Given: a running stream strategy
        When: release() is called
        Then: the stop flag is set and the capture is released
        """
        livestream._is_updating = False

        # When: releasing the stream
        livestream.release()

        # Then: internal state is updated
        assert livestream._is_stopped is True
        mock_release.assert_called_once()

    @patch.object(LiveStream, "_is_cap_connected")
    @patch.object(LiveStream, "_update_loop")
    @patch.object(LiveStream, "_release_cap")
    def test_update_aborts_if_not_connected(
        self,
        mock_release: MagicMock,
        mock_loop: MagicMock,
        mock_connected: MagicMock,
        livestream: LiveStream,
    ) -> None:
        """
        Verify that the update method aborts immediately if the capture device is not connected.

        Given: a stream that cannot connect
        When: update() is called by the background thread
        Then: the update loop is bypassed and resources are released
        """
        mock_connected.return_value = False

        # When: calling update on an unconnected stream
        livestream.update()

        # Then: the loop is not entered
        mock_loop.assert_not_called()
        mock_release.assert_called_once()
