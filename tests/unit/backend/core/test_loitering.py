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
def loitering_config() -> LoiteringConfig:
    """Default loitering configuration for testing."""
    return LoiteringConfig(
        threshold=DEFAULT_THRESHOLD,
        cooldown_percentage=DEFAULT_COOLDOWN_PCT,
        redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
    )


@pytest.fixture
def in_memory_repo() -> InMemoryStateRepository:
    return InMemoryStateRepository()


class StubGeometryEngine(GeometryEngine):
    """Stub geometry engine that considers points with x > 0 inside the ROI."""

    def is_inside(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        return point[0] > 0.0


@pytest.fixture
def stub_geometry() -> StubGeometryEngine:
    return StubGeometryEngine()


@pytest.fixture
def engine(
    loitering_config: LoiteringConfig,
    in_memory_repo: InMemoryStateRepository,
    stub_geometry: StubGeometryEngine,
) -> LoiteringEngine:
    """A LoiteringEngine instance with in-memory persistence and stub geometry."""
    return LoiteringEngine(
        loitering_config, repository=in_memory_repo, geometry=stub_geometry
    )


# Tests


class TestLoiteringEngineInit:
    """Tests for class instantiation and property delegation."""

    def test_cooldown_calculation(
        self,
        loitering_config: LoiteringConfig,
        in_memory_repo: InMemoryStateRepository,
        stub_geometry: StubGeometryEngine,
    ) -> None:
        """Test that the cooldown duration is correctly calculated on init."""
        engine = LoiteringEngine(
            loitering_config, repository=in_memory_repo, geometry=stub_geometry
        )
        assert engine.cooldown == DEFAULT_COOLDOWN_SEC

    def test_injected_dependencies(
        self,
        loitering_config: LoiteringConfig,
        in_memory_repo: InMemoryStateRepository,
        stub_geometry: StubGeometryEngine,
    ) -> None:
        """Verify that dependencies are correctly stored in the engine."""
        engine = LoiteringEngine(
            loitering_config, repository=in_memory_repo, geometry=stub_geometry
        )
        assert engine.repository is in_memory_repo
        assert engine.geometry is stub_geometry


class TestLoiteringEngineDomainLogic:
    """Tests for core domain loitering rules using InMemoryStateRepository (US0006, US0007)."""

    def test_record_presence_inside_roi(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
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

    def test_remove_presence_outside_roi(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
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

    def test_identify_loiterers(self, engine: LoiteringEngine) -> None:
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

    def test_state_cooldown_reentry(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
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

    def test_update_empty_detections_and_polygons(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
        """Verify that updating with empty detections or polygons is handled gracefully."""
        # Both empty
        engine.update({}, {})

        # Empty detections, non-empty polygons
        polygon = [(0.0, 0.0), (1.0, 1.0)]
        engine.update({STREAM_ID: []}, {STREAM_ID: polygon})

        # Non-empty detections, empty polygons
        detections = {
            STREAM_ID: [DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)]
        }
        engine.update(detections, {})

        # Verify nothing was added to the repository
        assert not in_memory_repo._presence
        assert not in_memory_repo._active_expiry
        assert not in_memory_repo._base_expiry
        assert not in_memory_repo._stream_indices

    def test_multi_stream_index_management(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
        """Verify index management in InMemoryStateRepository with multiple concurrent streams."""
        polygon = [(0.0, 0.0), (1.0, 1.0)]
        stream_1 = 1
        stream_2 = 2

        # Detections for both streams
        detections = {
            stream_1: [DetectedObject(track_id=10, x_center=0.8, y_bottom=0.8)],
            stream_2: [DetectedObject(track_id=20, x_center=0.8, y_bottom=0.8)],
        }

        with patch("time.time", return_value=1000.0):
            engine.update(detections, {stream_1: polygon, stream_2: polygon})

        # Verify indices exist for both streams
        key_1 = f"state:loitering:{stream_1}:10"
        key_2 = f"state:loitering:{stream_2}:20"
        assert key_1 in in_memory_repo._stream_indices[stream_1]
        assert key_2 in in_memory_repo._stream_indices[stream_2]

        # Clear stream 1 state
        engine.clear_stream(stream_1)

        # Verify stream 1 index is cleared, but stream 2 remains intact
        assert stream_1 not in in_memory_repo._stream_indices
        assert key_1 not in in_memory_repo._presence
        assert key_2 in in_memory_repo._stream_indices[stream_2]
        assert in_memory_repo._presence[key_2] == 1000.0


class TestOpenCVGeometryEngine:
    """Tests for OpenCV point containment checks using OpenCVGeometryEngine (US0005)."""

    def test_is_inside_polygon(self) -> None:
        engine = OpenCVGeometryEngine()
        polygon = [(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)]

        assert engine.is_inside((0.3, 0.2), polygon) is True
        assert engine.is_inside((0.0, 0.0), polygon) is False

    def test_is_inside_polygon_boundary(self) -> None:
        engine = OpenCVGeometryEngine()
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        # Point exactly on vertex
        assert engine.is_inside((0.0, 0.0), polygon) is True
        # Point exactly on edge
        assert engine.is_inside((0.5, 0.0), polygon) is True
        # Point slightly outside
        assert engine.is_inside((0.5, -0.0001), polygon) is False


class TestRedisStateRepository:
    """Tests for Redis state repository client interaction and Lua registration."""

    @patch("loitering_detector.infrastructure.persistence.redis.redis.Redis")
    def test_redis_connect_registers_scripts(
        self, mock_redis_cls: MagicMock, loitering_config: LoiteringConfig
    ) -> None:
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.scan_iter.return_value = []

        repo = RedisStateRepository(loitering_config.redis)
        repo.connect()

        assert repo.redis is mock_client
        assert mock_client.register_script.call_count == 2

    def test_redis_record_presence_script_call(
        self, loitering_config: LoiteringConfig, mock_redis: MagicMock
    ) -> None:
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

    def test_redis_remove_presence_script_call(
        self, loitering_config: LoiteringConfig, mock_redis: MagicMock
    ) -> None:
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

    def test_redis_clear_stream_state(
        self, loitering_config: LoiteringConfig, mock_redis: MagicMock
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)
        mock_redis.smembers.return_value = {TRACK_KEY}

        repo.clear_stream_state(STREAM_ID)

        mock_redis.delete.assert_called_once_with(TRACK_KEY, f"{TRACK_KEY}:active")
        mock_redis.srem.assert_called_once_with(
            f"state:loitering:{STREAM_ID}:index", TRACK_KEY
        )

    def test_redis_connect_exception(self, loitering_config: LoiteringConfig) -> None:
        with patch(
            "loitering_detector.infrastructure.persistence.redis.redis.Redis"
        ) as mock_redis_cls:
            mock_client = MagicMock()
            mock_redis_cls.return_value = mock_client
            mock_client.ping.side_effect = Exception("Connection timed out")

            repo = RedisStateRepository(loitering_config.redis)
            with pytest.raises(ConnectionError) as exc_info:
                repo.connect()

            assert "Failed to connect to Redis" in str(exc_info.value)
            assert repo.redis is None

    @patch("loitering_detector.infrastructure.persistence.redis.redis.Redis")
    def test_redis_connect_with_stale_keys(
        self, mock_redis_cls: MagicMock, loitering_config: LoiteringConfig
    ) -> None:
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.scan_iter.return_value = ["state:loitering:1:99"]

        repo = RedisStateRepository(loitering_config.redis)
        repo.connect()

        mock_client.delete.assert_called_once_with("state:loitering:1:99")

    def test_redis_disconnect(
        self, loitering_config: LoiteringConfig, mock_redis: MagicMock
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)
        repo._record_script = MagicMock()
        repo._remove_script = MagicMock()

        repo.disconnect()

        mock_redis.close.assert_called_once()
        assert repo.redis is None
        assert repo._record_script is None
        assert repo._remove_script is None

    def test_redis_unconnected_record_presence_raises_value_error(
        self, loitering_config: LoiteringConfig
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=None)
        with pytest.raises(ValueError) as exc_info:
            repo.record_presence(
                STREAM_ID, TRACK_ID, 1000.0, DEFAULT_THRESHOLD, DEFAULT_COOLDOWN_SEC
            )
        assert "Redis not connected" in str(exc_info.value)

    def test_redis_unconnected_remove_presence_raises_value_error(
        self, loitering_config: LoiteringConfig
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=None)
        with pytest.raises(ValueError) as exc_info:
            repo.remove_presence(STREAM_ID, TRACK_ID, 1000.0, DEFAULT_COOLDOWN_SEC)
        assert "Redis not connected" in str(exc_info.value)

    def test_redis_clear_stream_state_unconnected(
        self, loitering_config: LoiteringConfig
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=None)
        # Should return early without raising
        repo.clear_stream_state(STREAM_ID)

    def test_redis_register_lua_scripts_unconnected(
        self, loitering_config: LoiteringConfig
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=None)
        # Should return early without raising
        repo._register_lua_scripts()

    def test_redis_get_loitering_objects_unconnected(
        self, loitering_config: LoiteringConfig
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=None)
        res = repo.get_loitering_objects([STREAM_ID], DEFAULT_THRESHOLD, 1000.0)
        assert res == {}

    def test_redis_get_loitering_objects_empty_index(
        self, loitering_config: LoiteringConfig, mock_redis: MagicMock
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)
        mock_redis.smembers.return_value = set()

        res = repo.get_loitering_objects([STREAM_ID], DEFAULT_THRESHOLD, 1000.0)
        assert res == {}
        mock_redis.smembers.assert_called_once_with(
            f"state:loitering:{STREAM_ID}:index"
        )

    def test_redis_get_loitering_objects_success_and_stale_cleanup(
        self, loitering_config: LoiteringConfig, mock_redis: MagicMock
    ) -> None:
        repo = RedisStateRepository(loitering_config.redis, redis_client=mock_redis)

        # Mock stream keys in the set index (using a list to enforce deterministic order in test)
        stream_keys = [
            "state:loitering:1:10",
            "state:loitering:1:20",
            "state:loitering:1:30",
            "state:loitering:1:40",
        ]
        mock_redis.smembers.return_value = stream_keys

        # Active flags returned via mget
        # 10 is active, 20 is stale (None), 30 is active, 40 is active
        mock_redis.mget.side_effect = [
            ["1", None, "1", "1"],  # First mget call for active status
            [
                "980.0",
                "995.0",
                "invalid_timestamp",
            ],  # Second mget call for active start times
        ]

        res = repo.get_loitering_objects([1], threshold=10.0, now=1000.0)

        # Verify stale key 20 was removed from the index
        mock_redis.srem.assert_called_once_with(
            "state:loitering:1:index", "state:loitering:1:20"
        )

        # Verify result:
        # Track 10: 1000.0 - 980.0 = 20.0 >= 10.0 -> Loitering
        # Track 30: 1000.0 - 995.0 = 5.0 < 10.0 -> Not loitering yet
        # Track 40: raises ValueError inside and skipped
        assert res == {1: [10]}


class TestLoiteringEngineRobustness:
    """Additional robustness and error path tests for LoiteringEngine context/logging."""

    def test_context_manager_lifecycle(
        self, loitering_config: LoiteringConfig, stub_geometry: StubGeometryEngine
    ) -> None:
        mock_repo = MagicMock(spec=RedisStateRepository)
        with LoiteringEngine(
            loitering_config, repository=mock_repo, geometry=stub_geometry
        ) as engine:
            assert engine.repository is mock_repo

        mock_repo.connect.assert_called_once()
        mock_repo.disconnect.assert_called_once()

    def test_clear_stream_handles_exception_gracefully(
        self, loitering_config: LoiteringConfig, stub_geometry: StubGeometryEngine
    ) -> None:
        mock_repo = MagicMock(spec=RedisStateRepository)
        mock_repo.clear_stream_state.side_effect = Exception("Redis error")
        engine = LoiteringEngine(
            loitering_config, repository=mock_repo, geometry=stub_geometry
        )

        # Should log but not raise
        engine.clear_stream(STREAM_ID)
        mock_repo.clear_stream_state.assert_called_once_with(STREAM_ID)

    def test_check_handles_exception_gracefully(
        self, loitering_config: LoiteringConfig, stub_geometry: StubGeometryEngine
    ) -> None:
        mock_repo = MagicMock(spec=RedisStateRepository)
        mock_repo.get_loitering_objects.side_effect = Exception("Connection closed")
        engine = LoiteringEngine(
            loitering_config, repository=mock_repo, geometry=stub_geometry
        )

        # Should log and return empty dict
        res = engine.check([STREAM_ID])
        assert res == {}
        mock_repo.get_loitering_objects.assert_called_once()
