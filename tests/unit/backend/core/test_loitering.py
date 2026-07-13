"""Tests for loitering detection logic, state repository interfaces, and geometry adapters."""

from unittest.mock import MagicMock, patch

import pytest

from loitering_detector.config import LoiteringConfig, RedisConfig
from loitering_detector.core.interfaces import DetectedObject, GeometryEngine
from loitering_detector.core.loitering import LoiteringEngine
from loitering_detector.infrastructure.geometry import OpenCVGeometryEngine
from loitering_detector.infrastructure.persistence.in_memory import (
    InMemoryStateRepository,
)
from loitering_detector.infrastructure.persistence.redis import RedisStateRepository

# Constants
REDIS_HOST = "localhost"
REDIS_PORT = 6379

DEFAULT_THRESHOLD = 10.0
DEFAULT_COOLDOWN_PCT = 0.5
DEFAULT_COOLDOWN_SEC = DEFAULT_THRESHOLD * DEFAULT_COOLDOWN_PCT

STREAM_ID = 1
TRACK_ID = 10
TRACK_KEY = f"state:loitering:{STREAM_ID}:{TRACK_ID}"


# Fixtures
@pytest.fixture
def loitering_config():
    """Default loitering configuration for testing."""
    return LoiteringConfig(
        threshold=DEFAULT_THRESHOLD,
        cooldown_percentage=DEFAULT_COOLDOWN_PCT,
        redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
    )


@pytest.fixture
def in_memory_repo():
    return InMemoryStateRepository()


class StubGeometryEngine(GeometryEngine):
    """Stub geometry engine that considers points with x > 0 inside the ROI."""

    def is_inside(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        return point[0] > 0.0


@pytest.fixture
def stub_geometry():
    return StubGeometryEngine()


@pytest.fixture
def engine(loitering_config, in_memory_repo, stub_geometry):
    """A LoiteringEngine instance with in-memory persistence and stub geometry."""
    return LoiteringEngine(
        loitering_config, repository=in_memory_repo, geometry=stub_geometry
    )


# Tests


class TestLoiteringEngineInit:
    """Tests for class instantiation and property delegation."""

    def test_cooldown_calculation(
        self, loitering_config, in_memory_repo, stub_geometry
    ):
        """Test that the cooldown duration is correctly calculated on init."""
        engine = LoiteringEngine(
            loitering_config, repository=in_memory_repo, geometry=stub_geometry
        )
        assert engine.cooldown == DEFAULT_COOLDOWN_SEC

    def test_injected_dependencies(
        self, loitering_config, in_memory_repo, stub_geometry
    ):
        """Verify that dependencies are correctly stored in the engine."""
        engine = LoiteringEngine(
            loitering_config, repository=in_memory_repo, geometry=stub_geometry
        )
        assert engine.repository is in_memory_repo
        assert engine.geometry is stub_geometry


class TestLoiteringEngineDomainLogic:
    """Tests for core domain loitering rules using InMemoryStateRepository (US0006, US0007)."""

    def test_record_presence_inside_roi(self, engine, in_memory_repo):
        """Test that an object inside the ROI has its presence recorded."""
        detections = {
            STREAM_ID: [DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)]
        }
        polygon = [(0.0, 0.0), (1.0, 1.0)]

        with patch("time.time", return_value=1000.0):
            engine.update(detections, {STREAM_ID: polygon})

        # Verify entry timestamp is set in repo
        key = f"state:loitering:{STREAM_ID}:{TRACK_ID}"
        assert in_memory_repo._presence[key] == 1000.0
        assert in_memory_repo._active_expiry[key] == 1000.0 + DEFAULT_COOLDOWN_SEC
        assert (
            in_memory_repo._base_expiry[key]
            == 1000.0 + DEFAULT_THRESHOLD + DEFAULT_COOLDOWN_SEC
        )
        assert key in in_memory_repo._stream_indices[STREAM_ID]

    def test_remove_presence_outside_roi(self, engine, in_memory_repo):
        """Test that an object outside the ROI is removed from active sentinel status."""
        detections_inside = {
            STREAM_ID: [DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)]
        }
        detections_outside = {
            STREAM_ID: [DetectedObject(track_id=TRACK_ID, x_center=-0.5, y_bottom=0.8)]
        }
        polygon = [(0.0, 0.0), (1.0, 1.0)]

        with patch("time.time", return_value=1000.0):
            engine.update(detections_inside, {STREAM_ID: polygon})

        with patch("time.time", return_value=1002.0):
            engine.update(detections_outside, {STREAM_ID: polygon})

        key = f"state:loitering:{STREAM_ID}:{TRACK_ID}"
        # Active sentinel is deleted (not in active_expiry)
        assert key not in in_memory_repo._active_expiry
        # Base record is still stored with cooldown TTL (1002 + cooldown)
        assert in_memory_repo._presence[key] == 1000.0
        assert in_memory_repo._base_expiry[key] == 1002.0 + DEFAULT_COOLDOWN_SEC

    def test_identify_loiterers(self, engine):
        """Test loiterer identification based on thresholds."""
        detections = {
            STREAM_ID: [DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)]
        }
        polygon = [(0.0, 0.0), (1.0, 1.0)]

        # T = 1000.0: Enter ROI
        with patch("time.time", return_value=1000.0):
            engine.update(detections, {STREAM_ID: polygon})

        # T = 1005.0: Check (duration = 5s, threshold = 10s) -> Should not loiter
        with patch("time.time", return_value=1005.0):
            engine.update(detections, {STREAM_ID: polygon})
            res_not_yet = engine.check([STREAM_ID])
        assert res_not_yet == {}

        # T = 1011.0: Check (duration = 11s, threshold = 10s) -> Should loiter
        with patch("time.time", return_value=1011.0):
            engine.update(detections, {STREAM_ID: polygon})
            res_loitering = engine.check([STREAM_ID])
        assert res_loitering == {STREAM_ID: [TRACK_ID]}

    def test_state_cooldown_reentry(self, engine, in_memory_repo):
        """Verify that returning to ROI during cooldown maintains the same start time."""
        polygon = [(0.0, 0.0), (1.0, 1.0)]

        # T = 1000.0: Enter ROI
        with patch("time.time", return_value=1000.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # T = 1002.0: Leave ROI (goes to cooldown)
        with patch("time.time", return_value=1002.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=-0.5, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # T = 1004.0: Re-enter ROI (cooldown is 5s, so entry is still preserved)
        with patch("time.time", return_value=1004.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        key = f"state:loitering:{STREAM_ID}:{TRACK_ID}"
        # Start timestamp should still be T=1000.0, not T=1004.0
        assert in_memory_repo._presence[key] == 1000.0


class TestOpenCVGeometryEngine:
    """Tests for OpenCV point containment checks using OpenCVGeometryEngine (US0005)."""

    def test_is_inside_polygon(self):
        engine = OpenCVGeometryEngine()
        polygon = [(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)]

        assert engine.is_inside((0.3, 0.2), polygon) is True
        assert engine.is_inside((0.0, 0.0), polygon) is False


class TestRedisStateRepository:
    """Tests for Redis state repository client interaction and Lua registration."""

    @patch("loitering_detector.infrastructure.persistence.redis.redis.Redis")
    def test_redis_connect_registers_scripts(self, mock_redis_cls, loitering_config):
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.scan_iter.return_value = []

        repo = RedisStateRepository(loitering_config.redis)
        repo.connect()

        assert repo.redis is mock_client
        assert mock_client.register_script.call_count == 2

    def test_redis_record_presence_script_call(self, loitering_config, mock_redis):
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)
        mock_script = MagicMock()
        repo._record_script = mock_script

        repo.record_presence(
            STREAM_ID, TRACK_ID, 1000.0, DEFAULT_THRESHOLD, DEFAULT_COOLDOWN_SEC
        )

        mock_script.assert_called_once_with(
            keys=[
                TRACK_KEY,
                f"state:loitering:{STREAM_ID}:index",
                f"{TRACK_KEY}:active",
            ],
            args=[1000.0, 15, DEFAULT_COOLDOWN_SEC],
        )

    def test_redis_remove_presence_script_call(self, loitering_config, mock_redis):
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)
        mock_script = MagicMock()
        repo._remove_script = mock_script

        repo.remove_presence(STREAM_ID, TRACK_ID, 1000.0, DEFAULT_COOLDOWN_SEC)

        mock_script.assert_called_once_with(
            keys=[
                f"{TRACK_KEY}:active",
                f"state:loitering:{STREAM_ID}:index",
                TRACK_KEY,
            ],
            args=[DEFAULT_COOLDOWN_SEC],
        )

    def test_redis_clear_stream_state(self, loitering_config, mock_redis):
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)
        mock_redis.smembers.return_value = {TRACK_KEY}

        repo.clear_stream_state(STREAM_ID)

        mock_redis.delete.assert_called_once_with(TRACK_KEY, f"{TRACK_KEY}:active")
        mock_redis.srem.assert_called_once_with(
            f"state:loitering:{STREAM_ID}:index", TRACK_KEY
        )
