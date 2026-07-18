"""Domain interfaces and DTOs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class AlertSeverity(StrEnum):
    ALERT = "ALERT"
    INFO = "INFO"
    CLEARED = "CLEARED"


@dataclass(frozen=True)
class DetectedObject:
    """Domain representation of a tracked object detected in a frame."""

    track_id: int
    x_center: float  # Normalized X coordinate (0.0 to 1.0)
    y_bottom: float  # Normalized Y coordinate (0.0 to 1.0)


class LoiteringStateRepository(ABC):
    """Interface for persisting loitering tracking state."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the persistence store."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the persistence store."""
        pass

    @abstractmethod
    def record_presence(
        self,
        stream_id: int,
        track_id: int,
        timestamp: float,
        threshold: float,
        cooldown: float,
    ) -> None:
        """
        Record the presence of an object within the ROI.

        Must implement atomic logic to capture the initial entry timestamp,
        and update the heartbeat sentinel.
        """
        pass

    @abstractmethod
    def remove_presence(
        self, stream_id: int, track_id: int, timestamp: float, cooldown: float
    ) -> None:
        """
        Remove the active presence marker, triggering a cooldown period.

        The object's initial entry timestamp is retained for the duration of the
        cooldown to handle intermittent/fragmented tracking.
        """
        pass

    @abstractmethod
    def get_loitering_objects(
        self, stream_ids: list[int], threshold: float, now: float
    ) -> dict[int, list[int]]:
        """Identify and return all objects exceeding the loitering threshold."""
        pass

    @abstractmethod
    def clear_stream_state(self, stream_id: int) -> None:
        """Clear all tracking state and indices associated with the stream."""
        pass


class GeometryEngine(ABC):
    """Interface for spatial containment operations."""

    @abstractmethod
    def is_inside(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        """Determine if a point (x, y) is located inside a polygon boundary."""
        pass
