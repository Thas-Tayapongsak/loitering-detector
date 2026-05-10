"""Loitering detection engine and persistence logic."""

import logging
import time
from typing import cast

import cv2
import numpy as np
import redis
from ultralytics.engine.results import Results

from loitering_detector.config import LoiteringConfig

logger = logging.getLogger(__name__)


class LoiteringEngine:
    """
    Engine for detecting loitering behavior using Redis-backed state persistence.

    Coordinates between computer vision detection results and a persistent state
    store (Redis) to track how long objects remain within defined Regions of
    Interest (ROIs). Uses Lua scripts to ensure atomic updates to object
    timestamps and stream-specific indices.

    Attributes
    ----------
    redis : redis.Redis or None
        The Redis client used for state persistence.
    config : LoiteringConfig
        Configuration settings for loitering detection.
    threshold : float
        The time threshold (in seconds) after which an object is considered loitering.
    cooldown : float
        The time period (in seconds) to keep an object's state after it leaves the ROI.
    _roi_cache : dict[int, np.ndarray]
        Local cache of ROI polygons to avoid redundant conversions.
    _record_script : redis.commands.core.Script or None
        Registered Lua script for recording object presence.
    _remove_script : redis.commands.core.Script or None
        Registered Lua script for removing object presence.
    """

    def __init__(
        self, config: LoiteringConfig, redis_client: redis.Redis | None = None
    ):
        """
        Initialize the loitering engine.

        Parameters
        ----------
        config : LoiteringConfig
            Configuration settings for loitering detection.
        redis_client : redis.Redis, optional
            An existing Redis client to use. If None, one will be created during `connect()`.
        """
        self.redis: redis.Redis | None = redis_client
        self.config: LoiteringConfig = config
        self.threshold: float = config.threshold
        self.cooldown: float = round(self.threshold * config.cooldown_percentage)
        self._roi_cache: dict[int, np.ndarray] = {}
        self._record_script: redis.commands.core.Script | None = None
        self._remove_script: redis.commands.core.Script | None = None

    def connect(self) -> None:
        """
        Establish connection to Redis and initialize state.

        Raises
        ------
        ConnectionError
            If the connection to Redis fails.
        """
        if self.redis is not None:
            return

        logger.info(
            "Connecting to Redis at %s:%s...",
            self.config.redis.host,
            self.config.redis.port,
        )
        client = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            decode_responses=True,
            socket_connect_timeout=2.0,
        )
        self.redis = client

        try:
            client.ping()
            logger.info("Redis connection established.")
            self._register_lua_scripts()

            stale_keys = list(client.scan_iter("state:loitering:*"))
            if stale_keys:
                client.delete(*stale_keys)
        except Exception as e:
            self.redis = None  # Reset on failure
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    def disconnect(self) -> None:
        """Close the Redis connection and clean up Lua scripts."""
        if self.redis:
            try:
                self.redis.close()
            finally:
                self.redis = None
                self._record_script = None
                self._remove_script = None

    def __enter__(self):
        """Establish Redis connection on context entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close Redis connection on context exit."""
        self.disconnect()

    def clear_stream(self, stream_id: int) -> None:
        """
        Clear all tracking state for a specific stream.

        Parameters
        ----------
        stream_id : int
            The ID of the stream to clear.
        """
        client = self.redis
        if client is None:
            return
        try:
            stream_index = self._index_key(stream_id)
            stream_keys = list(cast(set[str], client.smembers(stream_index)))
            if stream_keys:
                active_keys = [self._active_key(k) for k in stream_keys]
                client.delete(*stream_keys, *active_keys)
                client.srem(stream_index, *stream_keys)

            # Clear local ROI cache
            if stream_id in self._roi_cache:
                del self._roi_cache[stream_id]
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error("Redis error in clear_stream for stream %s: %s", stream_id, e)

    def update(
        self,
        batch_results: dict[int, Results],
        roi_polygons: dict[int, list[tuple[float, float]]],
    ) -> None:
        """
        Process detection results and update loitering state in Redis.

        Parameters
        ----------
        batch_results : dict[int, Results]
            Mapping of stream IDs to Ultralytics detection results.
        roi_polygons : dict[int, list[tuple[float, float]]]
            Mapping of stream IDs to ROI polygon coordinates (normalized).
        """
        client = self.redis
        if client is None:
            return

        for stream_id, result in batch_results.items():
            if stream_id not in roi_polygons:
                continue

            # Get ROI from cache or convert if not present
            roi = self._roi_cache.get(stream_id)
            if roi is None:
                roi = np.array(roi_polygons[stream_id], dtype=np.float32)
                self._roi_cache[stream_id] = roi

            if result.boxes is None or result.boxes.data is None:
                continue

            boxes = np.array(result.boxes.data)
            orig_h, orig_w = result.orig_shape

            for box in boxes:
                # Ensure track_id is present
                if len(box) < 7:
                    continue

                track_id = int(box[4])
                x_center = ((box[0] + box[2]) / 2.0) / orig_w
                y_bottom = box[3] / orig_h
                point = (x_center, y_bottom)

                key = self._track_key(stream_id, track_id)
                if self._is_inside(point, roi):
                    self._record(key, stream_id)
                else:
                    self._remove(key, stream_id)

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
        loitering_objects: dict[int, list[int]] = {}
        client = self.redis
        if client is None:
            return loitering_objects

        try:
            for stream_id in stream_ids:
                stream_loiterers = self._check_stream(stream_id)
                if stream_loiterers:
                    loitering_objects[stream_id] = stream_loiterers
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error("Redis error in tracker.check(): %s", e)

        return loitering_objects

    def _check_stream(self, stream_id: int) -> list[int]:
        """Identify loitering objects for a single stream."""
        loiterers: list[int] = []
        client = self.redis
        if client is None:
            return loiterers

        stream_index = self._index_key(stream_id)
        stream_keys = list(cast(set[str], client.smembers(stream_index)))
        if not stream_keys:
            return loiterers

        active_keys = [self._active_key(k) for k in stream_keys]
        active_values = cast(list[str | None], client.mget(active_keys))

        stale_keys: list[str] = []
        loitering_keys: list[str] = []
        for k, v in zip(stream_keys, active_values, strict=True):
            if v is not None:
                loitering_keys.append(k)
            else:
                stale_keys.append(k)

        if stale_keys:
            client.srem(stream_index, *stale_keys)

        if not loitering_keys:
            return loiterers

        now = time.time()
        start_timestamps = cast(list[str | None], client.mget(loitering_keys))
        for key, start_timestamp in zip(loitering_keys, start_timestamps, strict=True):
            if start_timestamp is None:
                continue
            try:
                start_time = float(start_timestamp)
                if (now - start_time) >= self.threshold:
                    track_id = int(key.split(":")[-1])
                    loiterers.append(track_id)
            except (ValueError, TypeError, IndexError):
                continue

        return loiterers

    def _is_inside(self, point: tuple[float, float], polygon: np.ndarray) -> bool:
        """Check if a point is within a polygon ROI."""
        res = cv2.pointPolygonTest(polygon, point, False)
        return res >= 0

    def _record(self, key: str, stream_id: int) -> None:
        """Record an object's presence in the ROI using an atomic Lua script."""
        try:
            if self._record_script is None:
                raise ValueError("Redis not connected or Lua script not registered.")
            # Atomic update via Lua script: SET nx, EXPIRE base, SADD index, SET active
            # This ensures no partial state if the app crashes mid-update.
            self._record_script(
                keys=[key, self._index_key(stream_id), self._active_key(key)],
                args=[
                    time.time(),
                    int(self.threshold) + self.cooldown,
                    self.cooldown,
                ],
            )
        except (redis.ConnectionError, redis.TimeoutError, TypeError) as e:
            logger.warning("Redis error during _record: %s", e)

    def _remove(self, key: str, stream_id: int) -> None:
        """Remove an object's active status using an atomic Lua script."""
        try:
            if self._remove_script is None:
                raise ValueError("Redis not connected or Lua script not registered.")
            # Atomic update via Lua script: DEL active, SREM index, EXPIRE base to cooldown
            self._remove_script(
                keys=[self._active_key(key), self._index_key(stream_id), key],
                args=[self.cooldown],
            )
        except (redis.ConnectionError, redis.TimeoutError, TypeError) as e:
            logger.warning("Redis error during _remove: %s", e)

    def _register_lua_scripts(self) -> None:
        """Register atomic Lua scripts with the Redis client."""
        client = self.redis
        if client is None:
            return

        # KEYS[1]: base_key, KEYS[2]: stream_index, KEYS[3]: active_key
        # ARGV[1]: timestamp, ARGV[2]: heartbeat_ttl, ARGV[3]: active_ttl
        lua_record = """
        local is_new = redis.call('SET', KEYS[1], ARGV[1], 'NX')
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        redis.call('SADD', KEYS[2], KEYS[1])
        redis.call('SET', KEYS[3], '1', 'EX', ARGV[3])
        return is_new
        """

        # KEYS[1]: active_key, KEYS[2]: stream_index, KEYS[3]: base_key
        # ARGV[1]: cooldown
        lua_remove = """
        redis.call('DEL', KEYS[1])
        redis.call('SREM', KEYS[2], KEYS[3])
        local ttl = redis.call('TTL', KEYS[3])
        if ttl >= -1 then
            if ttl == -1 or ttl > tonumber(ARGV[1]) then
                redis.call('EXPIRE', KEYS[3], ARGV[1])
            end
        end
        return ttl
        """
        self._record_script = client.register_script(lua_record)
        self._remove_script = client.register_script(lua_remove)

    def _track_key(self, stream_id: int, track_id: int) -> str:
        """Generate the Redis key for a specific track."""
        return f"state:loitering:{stream_id}:{track_id}"

    def _active_key(self, track_key: str) -> str:
        """Generate the active sentinel key for a track."""
        return f"{track_key}:active"

    def _index_key(self, stream_id: int) -> str:
        """Generate the stream-specific index key."""
        return f"state:loitering:{stream_id}:index"
