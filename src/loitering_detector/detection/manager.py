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
import torch
from ultralytics.engine.results import Results
from ultralytics.trackers import BOTSORT, BYTETracker

from loitering_detector.detection.config import (
    BoTSORTArgs,
    BYTETrackArgs,
    DetectionConfig,
)
from loitering_detector.detection.strategy import DetectionStrategy, YOLODetection

logger = logging.getLogger(__name__)

# Registry mapping tracker names to their classes and default argument classes.
TRACKER_REGISTRY = {
    "bytetrack": (BYTETracker, BYTETrackArgs),
    "botsort": (BOTSORT, BoTSORTArgs),
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
    - Supports different detection strategies and tracking algorithms (BYTETrack, BoT-SORT).

    Attributes
    ----------
    config : DetectionConfig
        Configuration settings for detection and tracking.
    active_stream_ids : list of int
        List of stream IDs that this manager is currently processing.
    trackers : dict of int to (BYTETracker or BOTSORT)
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
        tracker_factory: Callable[[], BYTETracker | BOTSORT] | None = None,
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
        self._tracker_factory: Callable[[], BYTETracker | BOTSORT] = (
            tracker_factory
            if tracker_factory is not None
            else self._get_default_tracker_factory()
        )  # Factory for per-stream trackers

        # Active Tracker State
        self.trackers: dict[int, BYTETracker | BOTSORT] = {
            stream_id: self._tracker_factory() for stream_id in self.active_stream_ids
        }

    def infer(self, frames: list[np.ndarray], stream_ids: list[int]) -> list[Results]:
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
        list of Results
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

        for result, stream_id in zip(results, stream_ids, strict=True):
            self._update_result_with_tracks(result, stream_id)

        return results

    def stop(self) -> None:
        """
        Stop the detection manager and release all resources.

        Clears tracker states and shuts down the underlying detection
        strategy.
        """
        self.trackers.clear()
        self._strategy.stop()

    def _update_result_with_tracks(self, result: Results, stream_id: int) -> None:
        """Update a prediction result with identities from the tracker."""
        if result.boxes is None or result.boxes.data is None:
            return

        tracker = self.trackers.get(stream_id)
        if tracker is None:
            raise ValueError(f"Tracker not found for stream_id: {stream_id}")

        tracks = tracker.update(result.boxes, result.orig_img)

        # Update the result boxes with tracking data if available
        if len(tracks) > 0 and len(tracks[0]) >= 7:
            result.update(
                boxes=torch.as_tensor(tracks[:, :7], device=result.boxes.data.device)
            )
        else:
            result.update(boxes=torch.empty(0, 7, device=result.boxes.data.device))

    def _get_default_tracker_factory(self) -> Callable[[], BYTETracker | BOTSORT]:
        """Retrieve the default tracker factory from the registry."""
        tracker_entry = TRACKER_REGISTRY.get(self.config.tracker)
        if tracker_entry is None:
            raise ValueError(f"Unknown tracker: {self.config.tracker}")
        cls, args_cls = tracker_entry
        return lambda: cls(args_cls())

    def _init_strategy(self) -> DetectionStrategy:
        """Initialize the default detection strategy."""
        # Currently only YOLO is supported by default.
        return YOLODetection(self.config)
