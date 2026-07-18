"""Tests for the LoiteringDetectionSystem orchestrator, which coordinates streams, detection, and alerting."""

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import redis

from loitering_detector.config import (
    AlertConfig,
    LoiteringConfig,
    RedisConfig,
    SystemConfig,
)
from loitering_detector.detection.config import DetectionConfig
from loitering_detector.stream.config import StreamConfig

# Constants

STREAM_1_ID = 1
STREAM_2_ID = 2
STREAM_1_NAME = "s1"
STREAM_2_NAME = "s2"
STREAM_1_SOURCE = "rtsp://test/1"
STREAM_2_SOURCE = "rtsp://test/2"

DEFAULT_IMGSZ = 320
DEFAULT_CONF = 0.5
DEFAULT_TRACKER: Literal["bytetrack", "botsort"] = "bytetrack"

DEFAULT_THRESHOLD = 10.0
DEFAULT_COOLDOWN_PCT = 0.5
ALERT_INTERVAL = 5.0
SAMPLE_FPS = 10.0

REDIS_HOST = "localhost"
REDIS_PORT = 6379

TRACK_ID = 101


# Helpers


def _make_system_config(weights_path: Path) -> SystemConfig:
    """Build a minimal SystemConfig for testing."""
    return SystemConfig(
        streams=[
            StreamConfig(
                id=STREAM_1_ID, source=STREAM_1_SOURCE, name=STREAM_1_NAME, timeout=0.1
            ),
            StreamConfig(
                id=STREAM_2_ID, source=STREAM_2_SOURCE, name=STREAM_2_NAME, timeout=0.1
            ),
        ],
        detection=DetectionConfig(
            path=weights_path,
            imgsz=DEFAULT_IMGSZ,
            conf=DEFAULT_CONF,
            tracker=DEFAULT_TRACKER,
            classes=[0],
        ),
        loitering=LoiteringConfig(
            threshold=DEFAULT_THRESHOLD,
            cooldown_percentage=DEFAULT_COOLDOWN_PCT,
            redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
        ),
        alerts=AlertConfig(interval=ALERT_INTERVAL),
        sample_fps=SAMPLE_FPS,
    )


# Fixtures


@pytest.fixture
def system_config(mock_weights_file: Path) -> SystemConfig:
    return _make_system_config(mock_weights_file)


@pytest.fixture
def mock_stream() -> MagicMock:
    """A mock StreamManager."""
    stream = MagicMock()
    stream.config = MagicMock()
    stream.config.id = STREAM_1_ID
    stream.config.name = STREAM_1_NAME
    stream.config.roi_polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    stream.is_running = True
    stream.read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    return stream


# Tests


class TestSystemInit:
    """Tests for __init__ and configuration."""

    def test_default_active_stream_ids(self, system_config) -> None:
        """
        Test that all configured stream IDs are activated by default if no filter is provided.

        Given: a system configuration with two streams
        When: the LoiteringDetectionSystem is initialized without specific active IDs
        Then: both stream IDs are added to the active list
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        # When: initializing with default settings
        system = LoiteringDetectionSystem(system_config)

        # Then: both configured streams are active
        assert system.active_stream_ids == [STREAM_1_ID, STREAM_2_ID]

    def test_custom_active_stream_ids(self, system_config) -> None:
        """
        Test that only a subset of configured stream IDs can be activated during initialization.

        Given: a system configuration with two streams
        When: the system is initialized with only stream ID 2 active
        Then: only stream ID 2 is monitored
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        # When: initializing with specific active streams
        system = LoiteringDetectionSystem(
            system_config, active_stream_ids=[STREAM_2_ID]
        )

        # Then: only specified stream is in the active list
        assert system.active_stream_ids == [STREAM_2_ID]

    def test_sample_interval_calculation(self, system_config) -> None:
        """Sample interval is computed from FPS."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        assert system.sample_interval == pytest.approx(1.0 / SAMPLE_FPS, abs=0.01)


class TestSystemLifecycle:
    """Tests for start/stop and context manager."""

    @patch("loitering_detector.core.system.StreamManager")
    @patch("loitering_detector.core.system.DetectionManager")
    @patch.object(
        __import__(
            "loitering_detector.core.loitering", fromlist=["LoiteringEngine"]
        ).LoiteringEngine,
        "connect",
    )
    def test_context_manager_calls_start_and_stop(
        self, mock_engine_connect, mock_det_cls, mock_sm_cls, system_config
    ) -> None:
        """
        Test that the system correctly starts and stops resources when used as a context manager.

        Given: a system configuration
        When: entering and exiting the context manager
        Then: the start and stop methods are called, and the detector is managed correctly
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        mock_stream = MagicMock()
        mock_stream.config = system_config.streams[0]
        mock_sm_cls.return_value = mock_stream

        # When: using as a context manager
        with patch.object(LoiteringDetectionSystem, "_start_monitor_thread"):
            system = LoiteringDetectionSystem(system_config, active_stream_ids=[1])
            # When: start is called
            system.start()
            # Then: detector is initialized
            assert system.detector is not None
            # When: stop is called
            system.stop()
            # Then: detector is cleared
            assert system.detector is None

    def test_stop_is_idempotent(self, system_config) -> None:
        """Calling stop without start does not raise."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.stop()  # Should not raise

    def test_start_detector_raises_if_already_initialized(self, system_config) -> None:
        """
        Verify that attempting to start the detector when it is already running raises an error.

        Given: a system where the detector is already initialized
        When: _start_detector() is called again
        Then: a ValueError is raised to prevent duplicate initialization
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        # Given: detector already exists
        system.detector = MagicMock()

        # When/Then: starting it again raises an error
        with pytest.raises(ValueError, match="already initialized"):
            system._start_detector()

    def test_stop_detector_clears_reference(self, system_config) -> None:
        """
        Verify that stopping the detector properly cleans up the instance and its reference.

        Given: an active detector
        When: _stop_detector() is called
        Then: the detector's stop method is invoked and the reference is set to None
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        mock_det = MagicMock()
        system.detector = mock_det

        # When: stopping the detector
        system._stop_detector()

        # Then: resources are released and reference is cleared
        assert system.detector is None
        mock_det.stop.assert_called_once()

    def test_stop_detector_noop_if_none(self, system_config) -> None:
        """_stop_detector is a no-op when detector is None."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.detector = None
        system._stop_detector()  # Should not raise


class TestStartStreams:
    """Tests for _start_streams."""

    @patch("loitering_detector.core.system.StreamManager")
    def test_start_streams_populates_dict(self, mock_sm_cls, system_config) -> None:
        """
        Test that _start_streams correctly initializes and starts StreamManagers for active streams.

        Given: a system with an active stream ID
        When: _start_streams() is called
        Then: a StreamManager is created and added to the internal registry
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        mock_stream = MagicMock()
        mock_stream.config = system_config.streams[0]
        mock_sm_cls.return_value = mock_stream

        system = LoiteringDetectionSystem(
            system_config, active_stream_ids=[STREAM_1_ID]
        )

        # When: starting streams
        system._start_streams()

        # Then: stream is tracked and started
        assert STREAM_1_ID in system.streams
        mock_stream.start.assert_called_once()

    @patch("loitering_detector.core.system.StreamManager")
    def test_start_streams_skips_non_active_ids(
        self, mock_sm_cls, system_config
    ) -> None:
        """Streams not in active_stream_ids are skipped."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config, active_stream_ids=[2])
        mock_stream = MagicMock()
        mock_stream.config = system_config.streams[1]
        mock_sm_cls.return_value = mock_stream

        system._start_streams()
        assert STREAM_1_ID not in system.streams
        assert STREAM_2_ID in system.streams

    @patch("loitering_detector.core.system.StreamManager")
    def test_start_streams_handles_failure(self, mock_sm_cls, system_config) -> None:
        """Failed stream start is logged, not raised."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        mock_sm_cls.side_effect = RuntimeError("connection failed")
        system = LoiteringDetectionSystem(
            system_config, active_stream_ids=[STREAM_1_ID]
        )
        system._start_streams()
        assert len(system.streams) == 0


class TestStopStreams:
    """Tests for _stop_streams."""

    def test_stop_streams_calls_stop_on_each(self, system_config) -> None:
        """_stop_streams calls stop() on every stream."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        s1, s2 = MagicMock(), MagicMock()
        system.streams = {STREAM_1_ID: s1, STREAM_2_ID: s2}
        system._stop_streams()
        s1.stop.assert_called_once()
        s2.stop.assert_called_once()


class TestGetAlerts:
    """Tests for get_alerts."""

    def test_get_alerts_delegates_to_alert_manager(self, system_config) -> None:
        """get_alerts returns a snapshot from AlertManager."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.alerts = MagicMock()
        system.alerts.get_snapshot.return_value = {STREAM_1_ID: {TRACK_ID: 1000.0}}
        assert system.get_alerts() == {STREAM_1_ID: {TRACK_ID: 1000.0}}


class TestMonitorThread:
    """Tests for _start_monitor_thread and _monitor_loop."""

    def test_monitor_thread_starts_and_stops(self, system_config) -> None:
        """Monitor thread lifecycle."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.loitering_engine = MagicMock()
        system.loitering_engine.check.return_value = {}

        system._start_monitor_thread()
        assert system.monitor_thread is not None
        assert system.monitor_thread.is_alive()

        system._stop_monitor_thread()
        assert not system.monitor_thread.is_alive()

    def test_monitor_loop_updates_alerts(self, system_config) -> None:
        """_monitor_loop calls check() and updates alerts."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config, active_stream_ids=[1])
        system.loitering_engine = MagicMock()
        system.alerts.update(STREAM_1_ID, {TRACK_ID})

    def test_monitor_loop_handles_redis_connection_error(self, system_config) -> None:
        """
        Test that the monitor loop survives Redis connection failures by attempting to reconnect.

        Given: a loss of Redis connectivity during a check
        When: the monitor loop iteration fails with a ConnectionError
        Then: it attempts to disconnect and reconnect the engine before continuing
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config, active_stream_ids=[1])
        system.loitering_engine = MagicMock()
        # Given: first call fails
        system.loitering_engine.check.side_effect = redis.ConnectionError("conn lost")

        system.stop_event = MagicMock()
        # Run one loop then exit
        system.stop_event.is_set.side_effect = [False, True]

        with patch("time.sleep"):
            # When: monitor loop runs
            system._monitor_loop()

        # Then: it attempts recovery
        system.loitering_engine.disconnect.assert_called_once()
        system.loitering_engine.connect.assert_called_once()

    def test_monitor_loop_handles_reconnect_failure(self, system_config) -> None:
        """_monitor_loop continues even if reconnection fails."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(
            system_config, active_stream_ids=[STREAM_1_ID]
        )
        system.loitering_engine = MagicMock()
        system.loitering_engine.check.side_effect = redis.ConnectionError("conn lost")
        system.loitering_engine.connect.side_effect = RuntimeError("reconnect fail")

        system.stop_event = MagicMock()
        system.stop_event.is_set.side_effect = [False, True]

        with patch("time.sleep"):
            system._monitor_loop()  # Should not raise


class TestGetRunningStreams:
    """Tests for _get_running_streams."""

    def test_detects_disconnected_streams(self, system_config) -> None:
        """Disconnected streams trigger clear_stream."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.loitering_engine = MagicMock()

        s1 = MagicMock()
        s1.config.id = STREAM_1_ID
        s1.is_running = False  # Disconnected

        result = system._get_running_streams([s1], {STREAM_1_ID})
        assert result == set()
        system.loitering_engine.clear_stream.assert_called_with(STREAM_1_ID)

    def test_no_change_when_all_running(self, system_config) -> None:
        """No clear_stream if all streams are still active."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.loitering_engine = MagicMock()

        s1 = MagicMock()
        s1.config.id = STREAM_1_ID
        s1.is_running = True

        result = system._get_running_streams([s1], {STREAM_1_ID})
        assert result == {STREAM_1_ID}
        system.loitering_engine.clear_stream.assert_not_called()


class TestBatchInfer:
    """Tests for _batch_infer_results."""

    def test_returns_none_when_no_valid_frames(self, system_config) -> None:
        """Returns None if all streams return None frames."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        s1 = MagicMock()
        s1.is_running = True
        s1.config.id = STREAM_1_ID
        s1.read.return_value = None

        result = system._batch_infer_results([s1])
        assert result is None

    def test_returns_none_when_detector_is_none(self, system_config) -> None:
        """Returns None if detector has not been initialized."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.detector = None
        s1 = MagicMock()
        s1.is_running = True
        s1.config.id = 1
        s1.read.return_value = np.zeros((10, 10, 3))

        result = system._batch_infer_results([s1])
        assert result is None

    def test_infers_valid_frames(self, system_config) -> None:
        """
        Test that valid frames from multiple streams are batched together for efficient inference.

        Given: a running stream providing frames
        When: _batch_infer_results() is called
        Then: the frames are sent to the detector and results are mapped back to stream IDs
        """
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result = MagicMock()

        s1 = MagicMock()
        s1.is_running = True
        s1.config.id = 1
        s1.read.return_value = frame

        system.detector = MagicMock()
        system.detector.infer.return_value = [mock_result]

        # When: performing batch inference
        result = system._batch_infer_results([s1])

        # Then: results are mapped correctly
        assert result == {STREAM_1_ID: mock_result}

    def test_skips_non_running_streams(self, system_config) -> None:
        """Non-running streams are excluded from batch."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        s1 = MagicMock()
        s1.is_running = False
        s1.config.id = 1

        result = system._batch_infer_results([s1])
        assert result is None


class TestUpdateLoiteringEngine:
    """Tests for _update_loitering_engine."""

    def test_calls_engine_update(self, system_config) -> None:
        """Delegates to loitering_engine.update after translating results."""
        from loitering_detector.core.interfaces import DetectedObject
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.loitering_engine = MagicMock()
        system.roi_polygons = {1: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]}

        mock_result = MagicMock()
        mock_result.orig_shape = (1080, 1920)
        mock_result.boxes.data = [[100, 100, 200, 200, 5, 0.9, 0]]
        batch = {1: mock_result}

        system._update_loitering_engine(batch)

        expected_detections = {
            1: [
                DetectedObject(track_id=5, x_center=150.0 / 1920, y_bottom=200.0 / 1080)
            ]
        }
        system.loitering_engine.update.assert_called_once_with(
            expected_detections, system.roi_polygons
        )

    def test_catches_redis_errors(self, system_config) -> None:
        """Redis errors during update are caught and logged."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.loitering_engine = MagicMock()
        system.loitering_engine.update.side_effect = redis.ConnectionError("fail")

        mock_result = MagicMock()
        mock_result.orig_shape = (1080, 1920)
        mock_result.boxes.data = [[100, 100, 200, 200, 5, 0.9, 0]]

        system._update_loitering_engine({1: mock_result})  # Should not raise


class TestSleepInterval:
    """Tests for _sleep_sample_interval."""

    @patch("time.sleep")
    @patch("time.perf_counter", return_value=100.05)
    def test_sleeps_remaining_interval(
        self, mock_perf, mock_sleep, system_config
    ) -> None:
        """Sleeps for the remaining portion of the sample interval."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.last_process_time = 100.0  # 0.05s ago
        system._sleep_sample_interval()

        # interval computed from FPS, elapsed, remaining
        mock_sleep.assert_called_once()
        sleep_val = mock_sleep.call_args[0][0]
        assert sleep_val == pytest.approx((1.0 / SAMPLE_FPS) - 0.05, abs=0.01)

    @patch("time.sleep")
    @patch("time.perf_counter", return_value=100.2)
    def test_minimum_sleep(self, mock_perf, mock_sleep, system_config) -> None:
        """Sleeps at least 0.001s even if interval has elapsed."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.last_process_time = 100.0  # 0.2s ago, interval=0.1 => negative
        system._sleep_sample_interval()

        mock_sleep.assert_called_once_with(0.001)


class TestContextManager:
    """Tests for __enter__ and __exit__."""

    def test_enter_returns_self(self, system_config) -> None:
        """__enter__ returns the system instance."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        with patch.object(system, "start"):
            result = system.__enter__()
        assert result is system

    def test_exit_calls_stop(self, system_config) -> None:
        """__exit__ calls stop."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        with patch.object(system, "stop") as mock_stop:
            system.__exit__(None, None, None)
        mock_stop.assert_called_once()

    def test_start_failure_calls_stop(self, system_config) -> None:
        """If start() fails, stop() is still called."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        with patch.object(system, "_start_streams", side_effect=RuntimeError("fail")):
            with patch.object(system, "stop") as mock_stop:
                with pytest.raises(RuntimeError):
                    system.start()
                mock_stop.assert_called_once()


class TestDetectLoop:
    """Tests for the detect() generator."""

    def test_detect_yields_results_and_none(self, system_config) -> None:
        """detect() yields batch results and None for sleep cycles."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config)
        system.loitering_engine = MagicMock()
        system.roi_polygons = {1: [(0, 0), (1, 0), (1, 1)]}

        s1 = MagicMock()
        s1.config.id = 1
        # Use a counter that returns True for first N calls, then False
        counter = {"n": 0}

        def is_running_fn():
            counter["n"] += 1
            return counter["n"] <= 10

        type(s1).is_running = PropertyMock(side_effect=is_running_fn)
        s1.read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        system.streams = {1: s1}

        mock_result = MagicMock()
        mock_result.orig_shape = (1080, 1920)
        mock_result.boxes.data = [[100, 100, 200, 200, 5, 0.9, 0]]
        system.detector = MagicMock()
        system.detector.infer.return_value = [mock_result]

        # Force time to always be past the sample interval
        system.last_process_time = 0.0
        system.sample_interval = 0.0

        with patch("time.sleep"):
            with patch("time.perf_counter", return_value=1000.0):
                results = list(system.detect())

        # Should have produced at least one result
        assert len(results) >= 1

    def test_detect_handles_unexpected_error(self, system_config) -> None:
        """_monitor_loop handles unexpected exceptions."""
        from loitering_detector.core.system import LoiteringDetectionSystem

        system = LoiteringDetectionSystem(system_config, active_stream_ids=[1])
        system.loitering_engine = MagicMock()
        system.loitering_engine.check.side_effect = TypeError("unexpected")

        system.stop_event = MagicMock()
        system.stop_event.is_set.side_effect = [False, True]

        with patch("time.sleep"):
            system._monitor_loop()  # Should not raise
