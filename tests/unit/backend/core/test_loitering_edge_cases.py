"""Unit tests for complex loitering detection edge cases."""

from unittest.mock import patch

import pytest

from loitering_detector.config import LoiteringConfig, RedisConfig
from loitering_detector.core.interfaces import DetectedObject, GeometryEngine
from loitering_detector.core.loitering import LoiteringEngine
from loitering_detector.infrastructure.persistence.in_memory import (
    InMemoryStateRepository,
)

# Constants
DEFAULT_THRESHOLD = 10.0
DEFAULT_COOLDOWN_PCT = 0.5
DEFAULT_COOLDOWN_SEC = DEFAULT_THRESHOLD * DEFAULT_COOLDOWN_PCT

STREAM_ID = 1
TRACK_ID = 10


class StubGeometryEngine(GeometryEngine):
    """Stub geometry engine that considers points with x > 0 inside the ROI."""

    def is_inside(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        return point[0] > 0.0


@pytest.fixture
def loitering_config() -> LoiteringConfig:
    """Default loitering configuration for testing."""
    return LoiteringConfig(
        threshold=DEFAULT_THRESHOLD,
        cooldown_percentage=DEFAULT_COOLDOWN_PCT,
        redis=RedisConfig(host="localhost", port=6379),
    )


@pytest.fixture
def in_memory_repo() -> InMemoryStateRepository:
    return InMemoryStateRepository()


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


class TestLoiteringEdgeCases:
    """Edge cases for object detection, cooldown transitions, and isolation."""

    def test_reentry_within_cooldown_retains_entry_time(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
        """
        Scenario 1: Object inside ROI -> leaves ROI -> returns within cooldown window.
        Assert that loitering duration is cumulative (start time preserved).
        """
        polygon = [(0.0, 0.0), (1.0, 1.0)]

        # T = 1000.0: Object enters ROI (Inside)
        with patch("time.time", return_value=1000.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # T = 1002.0: Object leaves ROI (Outside) -> Starts cooldown (5s)
        # Expires at 1002.0 + 5.0 = 1007.0
        with patch("time.time", return_value=1002.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=-0.5, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # T = 1004.0: Object returns to ROI (Inside) - within cooldown window
        with patch("time.time", return_value=1004.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # Start time should still be 1000.0
        key = f"state:loitering:{STREAM_ID}:{TRACK_ID}"
        assert in_memory_repo._presence[key] == 1000.0

        # T = 1011.0: Check loitering (duration since 1000.0 is 11s >= threshold 10s)
        # Call update first to simulate the object still being detected in the ROI at T=1011.0.
        with patch("time.time", return_value=1011.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )
            res = engine.check([STREAM_ID])
        assert res == {STREAM_ID: [TRACK_ID]}

    def test_reentry_after_cooldown_expiry_resets_entry_time(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
        """
        Scenario 2: Object inside ROI -> leaves ROI -> returns after cooldown window has expired.
        Assert that loitering timer resets (new start time).
        """
        polygon = [(0.0, 0.0), (1.0, 1.0)]

        # T = 1000.0: Object enters ROI
        with patch("time.time", return_value=1000.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # T = 1002.0: Object leaves ROI (cooldown expires at 1007.0)
        with patch("time.time", return_value=1002.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=-0.5, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        # T = 1008.0: Object returns to ROI (Outside cooldown window)
        # Cooldown expired, should reset presence start time to 1008.0
        with patch("time.time", return_value=1008.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )

        key = f"state:loitering:{STREAM_ID}:{TRACK_ID}"
        assert in_memory_repo._presence[key] == 1008.0

        # T = 1011.0: Check loitering (duration 3s < threshold 10s)
        # Call update first to simulate the object still being detected in the ROI at T=1011.0.
        with patch("time.time", return_value=1011.0):
            engine.update(
                {
                    STREAM_ID: [
                        DetectedObject(track_id=TRACK_ID, x_center=0.8, y_bottom=0.8)
                    ]
                },
                {STREAM_ID: polygon},
            )
            res = engine.check([STREAM_ID])
        assert res == {}

    def test_track_states_isolated_across_different_streams(
        self, engine: LoiteringEngine, in_memory_repo: InMemoryStateRepository
    ) -> None:
        """
        Scenario 3: Same track ID on different stream IDs.
        Assert that track states are completely isolated.
        """
        polygon = [(0.0, 0.0), (1.0, 1.0)]
        stream_a = 1
        stream_b = 2
        track_id = 10

        # T = 1000.0: Track 10 appears on Stream A and Stream B
        with patch("time.time", return_value=1000.0):
            engine.update(
                {
                    stream_a: [
                        DetectedObject(track_id=track_id, x_center=0.8, y_bottom=0.8)
                    ],
                    stream_b: [
                        DetectedObject(track_id=track_id, x_center=0.8, y_bottom=0.8)
                    ],
                },
                {stream_a: polygon, stream_b: polygon},
            )

        key_a = f"state:loitering:{stream_a}:{track_id}"
        key_b = f"state:loitering:{stream_b}:{track_id}"

        # Both exist in repo
        assert key_a in in_memory_repo._presence
        assert key_b in in_memory_repo._presence

        # Clear stream A
        engine.clear_stream(stream_a)

        # Stream A should be cleared, Stream B should be intact
        assert key_a not in in_memory_repo._presence
        assert key_b in in_memory_repo._presence
