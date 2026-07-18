"""
Detection strategy implementations.

Defines the abstract interface and concrete implementations for object
detection models. These strategies are used by the `DetectionManager` to
perform inference on video frames.

Classes
-------
DetectionStrategy
    Abstract base class for all detection implementations.
YOLODetection
    Concrete implementation using Ultralytics YOLO models.
"""

__all__ = ["DetectionStrategy", "YOLODetection"]

from abc import ABC, abstractmethod

import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.engine.results import Results

from loitering_detector.detection.config import DetectionConfig


class DetectionStrategy(ABC):
    """
    Abstract base class defining the interface for detection models.

    Subclasses must implement methods for model inference and resource
    cleanup.
    """

    @abstractmethod
    def predict(self, frames: list[np.ndarray]) -> list[Results]:
        """
        Perform inference on a batch of frames.

        Parameters
        ----------
        frames : list of numpy.ndarray
            Batch of image frames to process.

        Returns
        -------
        list of Results
            Detection results for each frame.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release any resources held by the strategy."""
        pass


class YOLODetection(DetectionStrategy):
    """
    YOLO-based implementation of a detection strategy.

    Uses Ultralytics YOLO models to perform object detection and
    classification on image frames.

    Examples
    --------
    Initialize and run a standalone detection:

    >>> config = DetectionConfig(path="yolov8n.pt")
    >>> detector = YOLODetection(config)
    >>> results = detector.predict([frame1, frame2])
    >>> detector.stop()
    """

    def __init__(self, config: DetectionConfig):
        """
        Initialize the YOLO model.

        Parameters
        ----------
        config : DetectionConfig
            Configuration for the YOLO model.
        """
        self.config = config
        self.model = YOLO(config.path)

    def predict(self, frames: list[np.ndarray]) -> list[Results]:
        """
        Perform inference using the YOLO model.

        Parameters
        ----------
        frames : list of numpy.ndarray
            Batch of image frames to process.

        Returns
        -------
        list of Results
            Detection results for each frame from the YOLO model.
        """
        return self.model.predict(
            source=frames,
            batch=len(frames),
            classes=self.config.classes,
            verbose=False,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
        )

    def stop(self) -> None:
        """
        Release the YOLO model reference.

        Deleting the model reference allows for earlier garbage collection
        and helps free up GPU memory in multi-process or high-load
        environments.
        """
        del self.model
