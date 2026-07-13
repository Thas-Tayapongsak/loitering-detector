"""In-memory persistence implementation for loitering tracking state."""

from loitering_detector.core.interfaces import LoiteringStateRepository


class InMemoryStateRepository(LoiteringStateRepository):
    """
    In-memory state repository using standard Python dictionaries.
    Simulates Redis state transition, active/base TTL, and index set operations.
    """

    def __init__(self) -> None:
        self._presence: dict[str, float] = {}
        self._active_expiry: dict[str, float] = {}
        self._base_expiry: dict[str, float] = {}
        self._stream_indices: dict[int, set[str]] = {}

    def connect(self) -> None:
        """No-op connection for in-memory store."""
        pass

    def disconnect(self) -> None:
        """No-op disconnection for in-memory store."""
        pass

    def record_presence(
        self,
        stream_id: int,
        track_id: int,
        timestamp: float,
        threshold: float,
        cooldown: float,
    ) -> None:
        key = self._track_key(stream_id, track_id)

        # Simulates Redis Lua script NX SET for KEYS[1]
        # Only set self._presence[key] if the base key doesn't exist or is expired
        if key not in self._presence or timestamp >= self._base_expiry.get(key, 0.0):
            self._presence[key] = timestamp

        self._active_expiry[key] = timestamp + cooldown
        self._base_expiry[key] = timestamp + threshold + cooldown
        self._stream_indices.setdefault(stream_id, set()).add(key)

    def remove_presence(
        self, stream_id: int, track_id: int, timestamp: float, cooldown: float
    ) -> None:
        key = self._track_key(stream_id, track_id)

        # DEL KEYS[1] active_key
        self._active_expiry.pop(key, None)

        # SREM index_key KEYS[3]
        if stream_id in self._stream_indices:
            self._stream_indices[stream_id].discard(key)

        # Update key TTL to cooldown
        if key in self._presence:
            self._base_expiry[key] = timestamp + cooldown

    def get_loitering_objects(
        self, stream_ids: list[int], threshold: float, now: float
    ) -> dict[int, list[int]]:
        loitering_objects: dict[int, list[int]] = {}

        for stream_id in stream_ids:
            index = self._stream_indices.get(stream_id)
            if not index:
                continue

            stale_keys = []
            active_keys = []

            for key in list(index):
                # Check base key TTL expiry (like Redis TTL)
                base_exp = self._base_expiry.get(key, 0.0)
                active_exp = self._active_expiry.get(key, 0.0)

                # If base key is expired, clean up
                if now >= base_exp:
                    stale_keys.append(key)
                elif now < active_exp:
                    # Key is active
                    active_keys.append(key)
                else:
                    # Key exists but active sentinel has expired
                    stale_keys.append(key)

            for key in stale_keys:
                index.discard(key)
                self._presence.pop(key, None)
                self._active_expiry.pop(key, None)
                self._base_expiry.pop(key, None)

            loiterers = []
            for key in active_keys:
                start_time = self._presence.get(key)
                if start_time is not None and (now - start_time) >= threshold:
                    track_id = int(key.split(":")[-1])
                    loiterers.append(track_id)

            if loiterers:
                loitering_objects[stream_id] = loiterers

        return loitering_objects

    def clear_stream_state(self, stream_id: int) -> None:
        index = self._stream_indices.pop(stream_id, None)
        if index:
            for key in index:
                self._presence.pop(key, None)
                self._active_expiry.pop(key, None)
                self._base_expiry.pop(key, None)

    def _track_key(self, stream_id: int, track_id: int) -> str:
        return f"state:loitering:{stream_id}:{track_id}"
