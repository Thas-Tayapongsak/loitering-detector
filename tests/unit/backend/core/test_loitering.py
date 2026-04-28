"""Tests for loitering detection logic, ROI geometry, and Redis-based state persistence."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import redis

from loitering_detector.config import LoiteringConfig, RedisConfig
from loitering_detector.core.loitering import LoiteringEngine


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
def mock_redis():
    """A mocked Redis client."""
    mock = MagicMock(spec=redis.Redis)
    mock.decode_responses = True
    return mock


@pytest.fixture
def engine(loitering_config, mock_redis):
    """A LoiteringEngine instance with mocked Redis."""
    return LoiteringEngine(loitering_config, redis_client=mock_redis)


# Tests


class TestLoiteringEngineInit:
    """Tests for class instantiation and basic property calculations."""

    def test_cooldown_calculation(self, loitering_config):
        """
        Test that the cooldown duration is correctly calculated based on the threshold and percentage.

        Given: a loitering threshold and cooldown percentage
        When: the LoiteringEngine is initialized
        Then: the cooldown duration is correctly calculated (threshold * percentage)
        """
        # When: the engine is initialized with threshold and percentage
        engine = LoiteringEngine(loitering_config)

        # Then: the cooldown is calculated correctly
        assert engine.cooldown == DEFAULT_COOLDOWN_SEC

    def test_injected_redis_client(self, loitering_config, mock_redis):
        """
        Verify that an external Redis client can be correctly injected into the engine.

        Given: an existing Redis client mock
        When: the LoiteringEngine is initialized with the client
        Then: the engine uses the provided client for operations
        """
        # When: injecting a mock redis client
        engine = LoiteringEngine(loitering_config, redis_client=mock_redis)

        # Then: the engine stores the reference
        assert engine.redis is mock_redis


class TestLoiteringEngineLifecycle:
    """Tests for Redis connection management."""

    @patch("loitering_detector.core.loitering.redis.Redis")
    def test_connect_lifecycle(self, mock_redis_cls, loitering_config):
        """
        Test the connection lifecycle and registration of Lua scripts in Redis.

        Given: a Redis configuration
        When: connect() is called
        Then: the engine connects to Redis, pings it, and registers the required Lua scripts
        """
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.scan_iter.return_value = []

        engine = LoiteringEngine(loitering_config)

        # When: connecting to Redis
        engine.connect()

        # Then: scripts are registered and ping is performed
        assert engine.redis is mock_client
        assert mock_client.register_script.call_count == 2

    def test_disconnect_clears_references(self, engine, mock_redis):
        """
        Verify that disconnect() properly cleans up connections and script handles.

        Given: a connected engine
        When: disconnect() is called
        Then: the connection is closed and script handles are cleared
        """
        # When: disconnecting
        engine.disconnect()

        # Then: client is closed and references are cleared
        mock_redis.close.assert_called_once()
        assert engine.redis is None
        assert engine._record_script is None


class TestLoiteringEngineGeometry:
    """Tests for ROI and coordinate handling (US0005)."""

    def test_is_inside_roi(self, engine):
        """
        Test the point-in-polygon logic for identifying if a detection is within an ROI.

        Given: a polygonal ROI definition
        When: checking points against the polygon
        Then: it correctly identifies points as inside or outside the bounds
        """
        # Given: a triangle in normalized space
        roi = np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]], dtype=np.float32)

        # Then: (0.3, 0.2) is inside, (0.0, 0.0) is outside
        assert engine._is_inside((0.3, 0.2), roi) is True
        assert engine._is_inside((0.0, 0.0), roi) is False

    def test_coordinate_scaling(self, engine, mock_redis):
        """
        Test that normalized ROI coordinates are correctly scaled to the dimensions of the input frames.

        Given: a normalized ROI and a 1080p frame
        When: a detection at (0.3, 0.2) normalized is processed
        Then: it is correctly scaled to pixel coordinates and recorded in state
        """
        roi_polygons = {1: [(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)]}
        mock_result = MagicMock()
        mock_result.orig_shape = (1080, 1920)
        # Person at (0.3, 0.2) normalized => (576, 216) pixel
        mock_result.boxes.data = [[570, 100, 582, 216, 5, 0.9, 0]]

        engine._record_script = MagicMock()

        # When: processing results with scaling
        engine.update({1: mock_result}, roi_polygons)

        # Then: scaling triggers a record for track 5 on stream 1
        engine._record_script.assert_called()
        assert (
            f"state:loitering:{STREAM_ID}:5"
            in engine._record_script.call_args[1]["keys"]
        )


class TestLoiteringEnginePersistence:
    """Tests for Redis state management (US0006, US0007)."""

    def test_atomic_record_logic(self, engine):
        """
        Test that the parameters for atomic state recording in Redis are correctly generated.

        Given: a new object track
        When: recording object state in Redis
        Then: the correct Lua script keys and arguments (timestamps and TTLs) are passed
        """
        mock_script = MagicMock()
        engine._record_script = mock_script

        # When: recording a track at T=1000.0
        with patch("time.time", return_value=1000.0):
            engine._record(TRACK_KEY, STREAM_ID)

        # Then: script is called with correct keys and calculated TTLs
        mock_script.assert_called_once_with(
            keys=[
                TRACK_KEY,
                f"state:loitering:{STREAM_ID}:index",
                f"{TRACK_KEY}:active",
            ],
            args=[1000.0, 15, DEFAULT_COOLDOWN_SEC],
        )

    def test_cooldown_expiry_logic(self, engine):
        """
        Test that the cooldown duration is correctly applied when an object leaves the monitored ROI.

        Given: an object leaving the ROI
        When: _remove() is called
        Then: the cooldown TTL is applied to the object's persistence record in Redis
        """
        mock_script = MagicMock()
        engine._remove_script = mock_script
        # When: removing a track
        engine._remove(TRACK_KEY, STREAM_ID)

        # Then: the cooldown TTL is correctly passed to the removal script
        assert mock_script.call_args[1]["args"][0] == DEFAULT_COOLDOWN_SEC

    def test_state_cleanup(self, engine, mock_redis):
        """
        Test that the system automatically cleans up loitering records that have exceeded their cooldown.

        Given: a track index containing a stale record (no active sentinel)
        When: check() is called
        Then: the stale record is removed from the tracking index
        """
        mock_redis.smembers.return_value = {TRACK_KEY}
        # Given: the record is stale (mget returns None for active sentinel)
        mock_redis.mget.return_value = [None]

        # When: checking for loitering status
        engine.check([STREAM_ID])

        # Then: the stale track is removed from the index
        mock_redis.srem.assert_called_with(
            f"state:loitering:{STREAM_ID}:index", TRACK_KEY
        )


class TestLoiteringEngineIdentification:
    """Tests for loitering detection logic."""

    def test_identify_loiterers(self, engine, mock_redis):
        """
        Test that objects residing in the ROI longer than the threshold are correctly identified as loitering.

        Given: an active object that has been in the ROI for 20s (threshold 10s)
        When: check() is called
        Then: the object ID is returned as loitering
        """
        mock_redis.smembers.return_value = {TRACK_KEY}
        # Given: the object is active and started 20s ago
        mock_redis.mget.side_effect = [["1"], [str(time.time() - 20)]]

        # When: checking status
        result = engine.check([STREAM_ID])

        # Then: track is identified as loitering
        assert result == {STREAM_ID: [TRACK_ID]}
