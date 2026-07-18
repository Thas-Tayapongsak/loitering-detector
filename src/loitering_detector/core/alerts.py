"""
Alert management and notification logic.

This module coordinates the logging and tracking of loitering events
across multiple video streams. It ensures that alerts are only logged
at specified intervals and handles the lifecycle of an alert from
detection to clearance.

Classes
-------
AlertConfig
    Configuration schema for alert settings.
AlertManager
    Manager for loitering alerts across multiple streams.
"""

__all__ = ["AlertManager", "AlertConfig"]

import logging
import threading
import time

from loitering_detector.config import AlertConfig
from loitering_detector.core.interfaces import AlertSeverity

logger = logging.getLogger(__name__)

# Type alias for the internal alert state
# {stream_id: {track_id: last_alert_time}}
type AlertState = dict[int, dict[int, float]]


class AlertManager:
    """
    Manages loitering alerts across multiple streams.

    Handles the state of active loiterers and coordinates logging of
    new, continuing, and cleared loitering events.

    Features:
    - Thread-safe state management for multi-stream environments.
    - Automated logging of alert lifecycles.
    - Interval-based repeat alerts to prevent log flooding.

    Attributes
    ----------
    config : AlertConfig
        The configuration settings for the alert manager.
    alerts : AlertState
        Nested dictionary mapping stream IDs to track IDs and their last alert time.

    Examples
    --------
    >>> config = AlertConfig(interval=1.0)
    >>> manager = AlertManager(config)
    >>> manager.update(stream_id=1, current_loiterers={101})
    >>> # [ALERT] New loitering detected: Stream 1, Object 101
    """

    def __init__(self, config: AlertConfig):
        """
        Initialize the alert manager.

        Parameters
        ----------
        config : AlertConfig
            The configuration for the alert manager.
        """
        self.config: AlertConfig = config
        self.alerts: AlertState = {}

        self._lock: threading.Lock = threading.Lock()

    def update(self, stream_id: int, current_loiterers: set[int]) -> None:
        """
        Update the alert state for a given stream based on fresh loitering data.

        Coordinates the logging of:
        1. New loitering detections.
        2. Continuing loitering events (if interval elapsed).
        3. Cleared loitering events (when objects leave).

        Parameters
        ----------
        stream_id : int
            The ID of the stream being updated.
        current_loiterers : set[int]
            The set of track IDs currently identified as loitering.
        """
        current_time = time.time()
        with self._lock:
            active_for_stream = self.alerts.setdefault(stream_id, {})

            # 1 & 2: New and continuing loiterers
            for track_id in current_loiterers:
                if track_id not in active_for_stream:
                    logger.info(
                        f"[{AlertSeverity.ALERT.value}] New loitering detected: Stream %s, Object %s",
                        stream_id,
                        track_id,
                    )
                    active_for_stream[track_id] = current_time
                else:
                    last_alert = active_for_stream[track_id]
                    if current_time - last_alert >= self.config.interval:
                        logger.info(
                            f"[{AlertSeverity.INFO.value}] Object %s is still loitering in Stream %s",
                            track_id,
                            stream_id,
                        )
                        active_for_stream[track_id] = current_time

            # 3: Objects that have left
            past_loiterers = list(active_for_stream.keys())
            for track_id in past_loiterers:
                if track_id not in current_loiterers:
                    logger.info(
                        f"[{AlertSeverity.CLEARED.value}] Object %s has left Stream %s",
                        track_id,
                        stream_id,
                    )
                    del active_for_stream[track_id]

    def get_snapshot(self) -> AlertState:
        """
        Return a copy of the current active alerts.

        Returns
        -------
        AlertState
            A copy of the active alerts dictionary.
        """
        with self._lock:
            return {sid: dict(objs) for sid, objs in self.alerts.items()}
