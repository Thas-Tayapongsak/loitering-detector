"""Redis-backed persistence implementation for loitering tracking state."""

import logging
from typing import cast

import redis

from loitering_detector.config import RedisConfig
from loitering_detector.core.interfaces import LoiteringStateRepository

logger = logging.getLogger(__name__)


class RedisStateRepository(LoiteringStateRepository):
    """
    State repository implementing atomic Redis operations with Lua scripts.
    """

    def __init__(
        self, config: RedisConfig, redis_client: redis.Redis | None = None
    ) -> None:
        """
        Initialize the Redis state repository.

        Parameters
        ----------
        config : RedisConfig
            Redis connection configuration.
        redis_client : redis.Redis, optional
            An existing Redis client instance.
        """
        self.config: RedisConfig = config
        self.redis: redis.Redis | None = redis_client
        self._record_script: redis.commands.core.Script | None = None
        self._remove_script: redis.commands.core.Script | None = None

    def connect(self) -> None:
        """Establish Redis connection and register Lua scripts."""
        if self.redis is not None:
            return

        logger.info(
            "Connecting to Redis at %s:%s...",
            self.config.host,
            self.config.port,
        )
        client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
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
            self.redis = None
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    def disconnect(self) -> None:
        """Disconnect and clear scripts."""
        if self.redis:
            try:
                self.redis.close()
            finally:
                self.redis = None
                self._record_script = None
                self._remove_script = None

    def record_presence(
        self,
        stream_id: int,
        track_id: int,
        timestamp: float,
        threshold: float,
        cooldown: float,
    ) -> None:
        """Record object presence using atomic Lua script."""
        if self._record_script is None:
            raise ValueError("Redis not connected or Lua script not registered.")
        key = self._track_key(stream_id, track_id)
        self._record_script(
            keys=[key, self._index_key(stream_id), self._active_key(key)],
            args=[
                timestamp,
                int(threshold) + cooldown,
                cooldown,
            ],
        )

    def remove_presence(
        self, stream_id: int, track_id: int, timestamp: float, cooldown: float
    ) -> None:
        """Remove object presence using atomic Lua script."""
        if self._remove_script is None:
            raise ValueError("Redis not connected or Lua script not registered.")
        key = self._track_key(stream_id, track_id)
        self._remove_script(
            keys=[self._active_key(key), self._index_key(stream_id), key],
            args=[cooldown],
        )

    def get_loitering_objects(
        self, stream_ids: list[int], threshold: float, now: float
    ) -> dict[int, list[int]]:
        """Identify loitering objects across streams."""
        loitering_objects: dict[int, list[int]] = {}
        client = self.redis
        if client is None:
            return loitering_objects

        for stream_id in stream_ids:
            stream_index = self._index_key(stream_id)
            stream_keys = list(cast(set[str], client.smembers(stream_index)))
            if not stream_keys:
                continue

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
                continue

            start_timestamps = cast(list[str | None], client.mget(loitering_keys))
            loiterers: list[int] = []
            for key, start_timestamp in zip(
                loitering_keys, start_timestamps, strict=True
            ):
                if start_timestamp is None:
                    continue
                try:
                    start_time = float(start_timestamp)
                    if (now - start_time) >= threshold:
                        track_id = int(key.split(":")[-1])
                        loiterers.append(track_id)
                except (ValueError, TypeError, IndexError):
                    continue

            if loiterers:
                loitering_objects[stream_id] = loiterers

        return loitering_objects

    def clear_stream_state(self, stream_id: int) -> None:
        """Clear state for a single stream."""
        client = self.redis
        if client is None:
            return
        stream_index = self._index_key(stream_id)
        stream_keys = list(cast(set[str], client.smembers(stream_index)))
        if stream_keys:
            active_keys = [self._active_key(k) for k in stream_keys]
            client.delete(*stream_keys, *active_keys)
            client.srem(stream_index, *stream_keys)

    def _register_lua_scripts(self) -> None:
        """Register Lua scripts with Redis."""
        client = self.redis
        if client is None:
            return

        lua_record = """
        local is_new = redis.call('SET', KEYS[1], ARGV[1], 'NX')
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        redis.call('SADD', KEYS[2], KEYS[1])
        redis.call('SET', KEYS[3], '1', 'EX', ARGV[3])
        return is_new
        """

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
        return f"state:loitering:{stream_id}:{track_id}"

    def _active_key(self, track_key: str) -> str:
        return f"{track_key}:active"

    def _index_key(self, stream_id: int) -> str:
        return f"state:loitering:{stream_id}:index"
