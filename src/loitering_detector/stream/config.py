"""
Configuration schemas for the `stream` module.

Defines the Pydantic models used to configure individual video streams.

Classes
-------
StreamConfig
    Configuration for a video stream.
"""

__all__ = ["StreamConfig"]

from pydantic import BaseModel, Field, field_validator


class StreamConfig(BaseModel):
    """
    Configuration for a video stream.

    Used to initialize a `StreamManager`.

    Examples
    --------
    >>> config = StreamConfig(
    ...     id=0,
    ...     source="rtsp://[IP_ADDRESS]/live/1/01",
    ...     name="stream1",
    ...     timeout=5.0,
    ...     roi_polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)]
    ... )
    >>> with StreamManager(config) as stream:
    ...     frame = stream.read()

    Attributes
    ----------
    id : int
        ID of the stream. Non-negative and unique.
    source : str or int
        Source of the stream (e.g., RTSP URL, camera index).
    name : str
        Name of the stream.
    roi_polygon : list, optional, default [(0, 0), (0, 1), (1, 1), (1, 0)]
        Region of interest polygon defined by normalized (x, y)
        coordinates. At least 3 vertices.
    timeout : float, optional, default 5.0
        Timeout in seconds.
    """

    id: int = Field(ge=0)
    source: str | int
    name: str
    roi_polygon: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        min_length=3,
    )
    timeout: float = Field(default=5.0, gt=0)

    @field_validator("roi_polygon")
    @classmethod
    def validate_roi_polygon(
        cls, v: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Validate the ROI polygon coordinates.

        Raises
        ------
        ValueError
            Any coordinate is outside the [0, 1] range.
        """
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"ROI coordinates must be in [0, 1], got ({x}, {y})")
        return v
