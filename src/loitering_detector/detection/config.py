"""
Configuration schemas for detection and tracking.

Provides Pydantic and dataclass models for configuring detection models
(e.g., YOLO) and multi-object tracking algorithms (e.g., BYTETrack,
BoT-SORT).

Classes
-------
DetectionConfig
    Primary configuration for the detection backend.
BYTETrackArgs
    Arguments for the BYTETrack tracker algorithm.
BoTSORTArgs
    Arguments for the BoT-SORT tracker algorithm.
TrackerType
    Supported multi-object tracking algorithms.
"""

__all__ = [
    "DetectionConfig",
    "BYTETrackArgs",
    "BoTSORTArgs",
    "TrackerType",
]

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field, FilePath


class TrackerType(StrEnum):
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"


class DetectionConfig(BaseModel):
    """
    Configuration for the detection manager and strategies.

    Defines the parameters for model inference and tracking selection.
    Compatible with Ultralytics YOLO models and various tracking
    backends.

    Attributes
    ----------
    path : FilePath
        Path to the model weights file (e.g., .pt, .onnx, or .engine).
    imgsz : int, optional, default 640
        Input image size for the detection model (must be > 0).
    conf : float, optional, default 0.5
        Confidence threshold for detections [0.0, 1.0].
    tracker : {"bytetrack", "botsort"}, optional, default "bytetrack"
        Object tracking algorithm to use.
    classes : list of int, optional, default [0]
        List of class IDs to detect (e.g., [0] for persons).

    Examples
    --------
    Initialize with a standard YOLO model:

    >>> config = DetectionConfig(path="yolov8n.pt", conf=0.25)

    Configure for specific class detection (e.g., persons and cars):

    >>> config = DetectionConfig(
    ...     path="models/yolov8s.engine",
    ...     classes=[0, 2],
    ...     tracker="botsort"
    ... )
    """

    path: FilePath
    imgsz: int = Field(640, gt=0)
    conf: float = Field(0.5, ge=0.0, le=1.0)
    tracker: TrackerType = TrackerType.BYTETRACK
    classes: list[int] = Field(default_factory=lambda: [0])


@dataclass(frozen=True)
class BYTETrackArgs:
    """
    Arguments for the BYTETrack tracker algorithm.

    Taken from ultralytics/cfg/trackers/bytetrack.yaml
    """

    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True


@dataclass(frozen=True)
class BoTSORTArgs:
    """
    Arguments for the BoT-SORT tracker algorithm.

    Taken from ultralytics/cfg/trackers/botsort.yaml
    """

    tracker_type: str = "botsort"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    # BoT-SORT specifics
    gmc_method: str = "sparseOptFlow"
    # ReID model related thresh
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False
    model: str = "auto"
