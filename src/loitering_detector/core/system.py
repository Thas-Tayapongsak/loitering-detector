import logging
import threading
import time
from typing import Generator

import redis
from ultralytics.engine.results import Results

from loitering_detector.config import SystemConfig
from loitering_detector.core.alerts import AlertManager
from loitering_detector.core.loitering import LoiteringEngine
from loitering_detector.detection import DetectionManager
from loitering_detector.stream import StreamManager

logger = logging.getLogger(__name__)


class LoiteringDetectionSystem:
    """
    Central coordinator for the loitering detection system.

    Manages the lifecycle and interaction of video streams, detection backends,
    and loitering engines. Orchestrates background monitoring threads and
    provides a high-level API for running detection loops.

    Attributes
    ----------
    config : SystemConfig
        The global system configuration.
    active_stream_ids : list[int]
        The IDs of streams currently active in the system.
    streams : dict[int, StreamManager]
        Mapping of active stream IDs to their respective managers.
    detector : DetectionManager or None
        The backend manager for computer vision inference.
    loitering_engine : LoiteringEngine
        The engine responsible for tracking object state and loitering duration.
    roi_polygons : dict[int, list[tuple[float, float]]]
        Cached ROI polygons for each stream, normalized to [0, 1].
    sample_fps : float
        The targeted sampling rate for the detection loop.
    sample_interval : float
        The calculated time interval between samples.
    last_process_time : float
        Timestamp of the last processed frame batch.
    alerts : AlertManager
        Component for managing and logging loitering notifications.
    stop_event : threading.Event
        Event used to signal background threads to terminate.
    monitor_thread : threading.Thread or None
        Background thread for loitering status monitoring.
    """

    def __init__(
        self,
        config: SystemConfig,
        active_stream_ids: list[int] | None = None,
    ):
        """
        Initialize the loitering detection system.

        Parameters
        ----------
        config : SystemConfig
            The configuration settings for the system.
        active_stream_ids : list[int], optional
            Specific stream IDs to activate. If None, all configured streams are used.
        """
        self.config = config

        if active_stream_ids is None:
            self.active_stream_ids = [stream.id for stream in self.config.streams]
        else:
            self.active_stream_ids = active_stream_ids

        self.streams: dict[int, StreamManager] = {}
        self.detector: DetectionManager | None = None
        self.loitering_engine = LoiteringEngine(self.config.loitering)
        self.roi_polygons: dict[int, list[tuple[float, float]]] = {}
        self.sample_fps: float = self.config.sample_fps
        self.sample_interval: float = 1.0 / self.sample_fps
        self.last_process_time: float = 0.0

        # Monitoring state
        self.alerts = AlertManager(self.config.alerts)
        self.stop_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None

    def __enter__(self):
        """Establish system resources on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up system resources on context exit."""
        self.stop()

    def get_alerts(self) -> dict[int, dict[int, float]]:
        """
        Return a snapshot of currently active loitering alerts.

        Safe to call from any thread.

        Returns
        -------
        dict[int, dict[int, float]]
            A mapping of stream IDs to track IDs and their last alert time.
        """
        return self.alerts.get_snapshot()

    def start(self) -> None:
        """
        Start the loitering detection system and background monitoring.

        Raises
        ------
        Exception
            If any component (streams, detector, or engine) fails to initialize.
        """
        try:
            self._start_streams()
            self._start_detector()
            self.loitering_engine.connect()
            self._start_monitor_thread()
        except Exception as e:
            logger.error("Failed to start loitering detection system: %s", e)
            self.stop()
            raise

    def stop(self) -> None:
        """Clean up all system resources, including streams and background threads."""
        self._stop_monitor_thread()
        self._stop_streams()
        self._stop_detector()
        self.loitering_engine.disconnect()

    def detect(self) -> Generator[dict[int, Results] | None, None, None]:
        """
        Main detection loop that processes frames from all active streams.

        Yields
        ------
        dict[int, Results] or None
            A mapping of stream IDs to detection results for each processed batch,
            or None if the loop is sleeping between samples.
        """
        streams = list(self.streams.values())
        tracked_active_streams = {s.config.id for s in streams}

        while any(stream.is_running for stream in streams):
            tracked_active_streams = self._get_running_streams(
                streams, tracked_active_streams
            )

            now = time.perf_counter()
            if (now - self.last_process_time) >= self.sample_interval:
                self.last_process_time = now

                batch_results = self._batch_infer_results(streams)
                if batch_results:
                    self._update_loitering_engine(batch_results)
                    yield batch_results
                    continue

            self._sleep_sample_interval()
            yield None

    def _start_streams(self) -> None:
        """Initialize and start all configured video streams."""
        for stream_config in self.config.streams:
            if stream_config.id not in self.active_stream_ids:
                continue
            try:
                stream = StreamManager(stream_config)
                stream.start()
                self.streams[stream_config.id] = stream
                self.roi_polygons[stream_config.id] = stream.config.roi_polygon
            except Exception as e:
                logger.error("Failed to start stream %s: %s", stream_config.id, e)

    def _stop_streams(self) -> None:
        """Stop all video streams."""
        for stream in self.streams.values():
            stream.stop()

    def _start_detector(self) -> None:
        """Initialize the detection manager."""
        if self.detector is not None:
            raise ValueError("Detector is already initialized.")

        self.detector = DetectionManager(
            self.config.detection, active_stream_ids=self.active_stream_ids
        )

    def _stop_detector(self) -> None:
        """Stop the detection manager and release its resources."""
        if self.detector is not None:
            self.detector.stop()
            self.detector = None

    def _start_monitor_thread(self) -> None:
        """Start the background monitor thread."""
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitor_thread(self) -> None:
        """Stop the background monitor thread."""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        """Background loop to check for loitering objects and log alerts."""
        while not self.stop_event.is_set():
            try:
                loitering_data = self.loitering_engine.check(self.active_stream_ids)

                # Update alerts for each active stream
                for stream_id in self.active_stream_ids:
                    current_loiterers = set(loitering_data.get(stream_id, []))
                    self.alerts.update(stream_id, current_loiterers)

            except redis.ConnectionError as e:
                logger.error(
                    "[ERROR] Redis connection error in monitor loop: %s. "
                    "Attempting reconnection...",
                    e,
                )
                try:
                    self.loitering_engine.disconnect()
                    self.loitering_engine.connect()
                    logger.info("[INFO] Successfully reconnected to Redis.")
                except Exception as reconnect_err:
                    logger.error("[ERROR] Redis reconnection failed: %s", reconnect_err)
            except Exception as e:
                logger.error(
                    "[ERROR] Unexpected exception in monitoring loop: %s",
                    e,
                    exc_info=True,
                )

            # Sleep for a bit to avoid hammering Redis
            time.sleep(1.0)

    def _get_running_streams(
        self, streams: list[StreamManager], tracked_ids: set[int]
    ) -> set[int]:
        """Check for disconnected streams and clean up tracking state."""
        current_active = {s.config.id for s in streams if s.is_running}
        disconnected = tracked_ids - current_active
        for stream_id in disconnected:
            logger.info("Stream %s disconnected. Clearing tracking state.", stream_id)
            self.loitering_engine.clear_stream(stream_id)
        return current_active

    def _batch_infer_results(
        self, streams: list[StreamManager]
    ) -> dict[int, Results] | None:
        """Acquire frames from running streams and perform inference."""
        running_streams = [s for s in streams if s.is_running]
        frames = [s.read() for s in running_streams]

        valid_streams = []
        valid_frames = []
        for stream, frame in zip(running_streams, frames):
            if frame is not None:
                valid_streams.append(stream)
                valid_frames.append(frame)

        if not valid_frames:
            return None

        if self.detector is None:
            return None

        valid_stream_ids = [stream.config.id for stream in valid_streams]
        results = self.detector.infer(valid_frames, valid_stream_ids)
        return {sid: res for sid, res in zip(valid_stream_ids, results)}

    def _update_loitering_engine(self, batch_results: dict[int, Results]) -> None:
        """Update the loitering engine with fresh detection results."""
        try:
            self.loitering_engine.update(batch_results, self.roi_polygons)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error("Redis error in detect loop: %s", e)

    def _sleep_sample_interval(self) -> None:
        """Sleep for the remaining time in the current sample interval."""
        now = time.perf_counter()
        remaining = self.sample_interval - (now - self.last_process_time)
        time.sleep(max(0.001, remaining))
