"""Loitering detection engine and persistence delegation logic."""

import logging
import time
from types import TracebackType

from loitering_detector.config import LoiteringConfig
from loitering_detector.core.interfaces import (
    DetectedObject,
    GeometryEngine,
    LoiteringStateRepository,
)

logger = logging.getLogger(__name__)


class LoiteringEngine:
    """
    Engine for detecting loitering behavior using injected state persistence and geometry helpers.

    Coordinates between computer vision detection results (generic DTOs) and a persistent state
    store (via the LoiteringStateRepository interface) to track how long objects remain within defined
    Regions of Interest (ROIs) (via the GeometryEngine interface).
    """

    def __init__(
        self,
        config: LoiteringConfig,
        repository: LoiteringStateRepository,
        geometry: GeometryEngine,
    ):
        """
        Initialize the loitering engine.

        Parameters
        ----------
        config : LoiteringConfig
            Configuration settings for loitering detection.
        repository : LoiteringStateRepository
            Persistent state repository adapter.
        geometry : GeometryEngine
            Spatial geometry calculations adapter.
        """
        self.config: LoiteringConfig = config
        self.repository: LoiteringStateRepository = repository
        self.geometry: GeometryEngine = geometry
        self.threshold: float = config.threshold
        self.cooldown: float = round(self.threshold * config.cooldown_percentage)

    def connect(self) -> None:
        """Establish connection to the repository."""
        self.repository.connect()

    def disconnect(self) -> None:
        """Close connection to the repository."""
        self.repository.disconnect()

    def __enter__(self) -> "LoiteringEngine":
        """Establish connection on context entry."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Close connection on context exit."""
        self.disconnect()
        return None

    def clear_stream(self, stream_id: int) -> None:
        """
        Clear all tracking state for a specific stream.

        Parameters
        ----------
        stream_id : int
            The ID of the stream to clear.
        """
        try:
            self.repository.clear_stream_state(stream_id)
        except Exception as e:
            logger.error("Error in clear_stream for stream %s: %s", stream_id, e)

    def update(
        self,
        detections: dict[int, list[DetectedObject]],
        roi_polygons: dict[int, list[tuple[float, float]]],
    ) -> None:
        """
        Process detection results and update loitering state in the repository.

        Parameters
        ----------
        detections : dict[int, list[DetectedObject]]
            Mapping of stream IDs to list of detected object DTOs.
        roi_polygons : dict[int, list[tuple[float, float]]]
            Mapping of stream IDs to ROI polygon coordinates (normalized).
        """
        for stream_id, objects in detections.items():
            roi = roi_polygons.get(stream_id)
            if not roi:
                continue

            for obj in objects:
                point = (obj.x_center, obj.y_bottom)
                if self.geometry.is_inside(point, roi):
                    self.repository.record_presence(
                        stream_id=stream_id,
                        track_id=obj.track_id,
                        timestamp=time.time(),
                        threshold=self.threshold,
                        cooldown=self.cooldown,
                    )
                else:
                    self.repository.remove_presence(
                        stream_id=stream_id,
                        track_id=obj.track_id,
                        timestamp=time.time(),
                        cooldown=self.cooldown,
                    )

    def check(self, stream_ids: list[int]) -> dict[int, list[int]]:
        """
        Identify objects that have exceeded the loitering threshold.

        Parameters
        ----------
        stream_ids : list[int]
            List of stream IDs to check for loitering objects.

        Returns
        -------
        dict[int, list[int]]
            Mapping of stream IDs to lists of track IDs that are loitering.
        """
        try:
            return self.repository.get_loitering_objects(
                stream_ids, self.threshold, time.time()
            )
        except Exception as e:
            logger.error("Error in tracker.check(): %s", e)
            return {}
