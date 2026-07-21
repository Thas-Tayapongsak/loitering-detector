"""
Configuration schemas for detection and tracking.

Provides Pydantic and dataclass models for configuring detection models
(e.g., YOLO) and multi-object tracking algorithms (e.g., Supervision ByteTrack).

Classes
-------
DetectionConfig
    Primary configuration for the detection backend.
SupervisionByteTrackConfig
    Arguments for the Supervision ByteTrack tracker algorithm.
TrackerType
    Supported multi-object tracking algorithms.
"""

__all__ = [
    "DetectionConfig",
    "TrackersByteTrackConfig",
    "TrackerType",
]

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field, FilePath


class TrackerType(StrEnum):
    BYTETRACK = "bytetrack"


class DetectionConfig(BaseModel):
    """
    Configuration for the detection manager and strategies.

    Defines the parameters for model inference and tracking selection.
    Compatible with detection strategies and Supervision multi-object tracking.

    Attributes
    ----------
    path : FilePath
        Path to the model weights file (e.g., .pt, .onnx, or .engine).
    imgsz : int, optional, default 640
        Input image size for the detection model (must be > 0).
    conf : float, optional, default 0.5
        Confidence threshold for detections [0.0, 1.0].
    tracker : {"bytetrack"}, optional, default "bytetrack"
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
    ...     tracker="bytetrack"
    ... )
    """

    path: FilePath
    imgsz: int = Field(640, gt=0)
    conf: float = Field(0.5, ge=0.0, le=1.0)
    tracker: TrackerType = TrackerType.BYTETRACK
    classes: list[int] = Field(default_factory=lambda: [0])


@dataclass(frozen=True)
class TrackersByteTrackConfig:
    """
    Configuration arguments for Roboflow ByteTrackTracker algorithm.

    Attributes
    ----------
    lost_track_buffer : int, default 30
        Number of frames to keep a lost track in memory before deletion.
    frame_rate : int, default 30
        Video frame rate for internal speed and motion estimations.
    track_activation_threshold : float, default 0.7
        Detection confidence threshold required to activate a new track.
    minimum_consecutive_frames : int, default 2
        Minimum consecutive frame detections required to confirm a new track ID.
    minimum_iou_threshold : float, default 0.1
        IoU threshold for matching detections to existing tracks.
    high_conf_det_threshold : float, default 0.6
        High-confidence detection threshold for first-stage tracking association.
    """

    lost_track_buffer: int = 30
    frame_rate: int = 30
    track_activation_threshold: float = 0.7
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold: float = 0.1
    high_conf_det_threshold: float = 0.6
