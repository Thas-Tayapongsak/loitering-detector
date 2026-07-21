"""
Object detection and tracking logic manager.

This module provides the orchestrator that combines detection strategies
(like YOLO) with multi-object tracking algorithms. It handles batch
inference and maintains independent tracker states for each stream.

Classes
-------
DetectionManager
    High-level coordinator for computer vision inference and tracking.
"""

__all__ = ["DetectionManager"]

import logging
from collections.abc import Callable

import numpy as np

from loitering_detector.detection.config import (
    DetectionConfig,
    SupervisionByteTrackConfig,
)
from loitering_detector.detection.providers.ultralytics import YOLODetection
from loitering_detector.detection.results import DetectionResult
from loitering_detector.detection.strategy import DetectionStrategy
from loitering_detector.detection.trackers import (
    SupervisionByteTrack,
    TrackerInterface,
)

logger = logging.getLogger(__name__)

# Registry mapping tracker names to their classes and default argument classes.
TRACKER_REGISTRY = {
    "bytetrack": (SupervisionByteTrack, SupervisionByteTrackConfig),
}


class DetectionManager:
    """
    Manager for detection strategies and multi-stream tracking.

    This class coordinates batch inference across multiple video streams.
    It uses a `DetectionStrategy` to get raw predictions and then updates
    independent tracker instances (one per stream) to maintain object
    identities over time.

    Features:
    - Optimizes GPU utilization by processing multiple frames in a single forward pass.
    - Maintains unique tracker states for each active stream to prevent cross-stream ID pollution.
    - Supports pluggable detection strategies and Supervision multi-object tracking.

    Attributes
    ----------
    config : DetectionConfig
        Configuration settings for detection and tracking.
    active_stream_ids : list of int
        List of stream IDs that this manager is currently processing.
    trackers : dict of int to TrackerInterface
        Active tracker instances mapped by stream ID.

    Examples
    --------
    Perform batch inference across two streams:

    >>> config = DetectionConfig(path="yolov8n.pt", tracker="bytetrack")
    >>> detector = DetectionManager(config, active_stream_ids=[1, 2])
    >>> results = detector.infer(frames=[frame1, frame2], stream_ids=[1, 2])
    >>> detector.stop()
    """

    def __init__(
        self,
        config: DetectionConfig,
        active_stream_ids: list[int],
        strategy: DetectionStrategy | None = None,
        tracker_factory: Callable[[], TrackerInterface] | None = None,
    ):
        """
        Initialize the detection manager.

        Parameters
        ----------
        config : DetectionConfig
            Configuration for the detection and tracking.
        active_stream_ids : list of int
            List of stream IDs that this manager will process.
        strategy : DetectionStrategy, optional, default None
            Specific detection strategy to use.
        tracker_factory : Callable, optional, default None
            Factory function that returns a new tracker instance.
        """
        self.config: DetectionConfig = config
        self.active_stream_ids: list[int] = active_stream_ids

        # Core Components
        self._strategy: DetectionStrategy = (
            strategy if strategy is not None else self._init_strategy()
        )  # Pluggable detection backend
        self._tracker_factory: Callable[[], TrackerInterface] = (
            tracker_factory
            if tracker_factory is not None
            else self._get_default_tracker_factory()
        )  # Factory for per-stream trackers

        # Active Tracker State
        self.trackers: dict[int, TrackerInterface] = {
            stream_id: self._tracker_factory() for stream_id in self.active_stream_ids
        }

    def infer(
        self, frames: list[np.ndarray], stream_ids: list[int]
    ) -> list[DetectionResult]:
        """
        Perform batch inference and update trackers for each stream.

        Parameters
        ----------
        frames : list of numpy.ndarray
            Batch of image frames to process.
        stream_ids : list of int
            Stream IDs corresponding to each frame in the batch.

        Returns
        -------
        list of DetectionResult
            Detection and tracking results for each frame.

        Raises
        ------
        ValueError
            If the batch size of frames does not match the stream IDs.
        """
        if len(frames) != len(stream_ids):
            raise ValueError(
                f"Batch size mismatch: {len(frames)} frames vs "
                f"{len(stream_ids)} stream IDs"
            )

        results = self._strategy.predict(frames)

        updated_results = []
        for result, stream_id in zip(results, stream_ids, strict=True):
            updated_results.append(self._update_result_with_tracks(result, stream_id))

        return updated_results

    def stop(self) -> None:
        """
        Stop the detection manager and release all resources.

        Clears tracker states and shuts down the underlying detection
        strategy.
        """
        self.trackers.clear()
        self._strategy.stop()

    def _update_result_with_tracks(
        self, result: DetectionResult, stream_id: int
    ) -> DetectionResult:
        """Update a prediction result with identities from the tracker."""
        tracker = self.trackers.get(stream_id)
        if tracker is None:
            raise ValueError(f"Tracker not found for stream_id: {stream_id}")

        return tracker.update(result)

    def _get_default_tracker_factory(self) -> Callable[[], TrackerInterface]:
        """Retrieve the default tracker factory from the registry."""
        tracker_entry = TRACKER_REGISTRY.get(self.config.tracker)
        if tracker_entry is None:
            raise ValueError(f"Unknown tracker: {self.config.tracker}")
        cls, _ = tracker_entry
        return lambda: cls()

    def _init_strategy(self) -> DetectionStrategy:
        """Initialize the default detection strategy."""
        # Currently only YOLO is supported by default.
        return YOLODetection(self.config)
