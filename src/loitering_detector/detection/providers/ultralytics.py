__all__ = ["YOLODetection"]

import numpy as np
import supervision as sv

from loitering_detector.detection.config import DetectionConfig
from loitering_detector.detection.results import DetectionResult
from loitering_detector.detection.strategy import DetectionStrategy


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
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Please install 'loitering-detector[ultralytics]' "
                "or 'loitering-detector[ultralytics-headless]' to use YOLODetection."
            ) from e

        self.config = config
        self.model = YOLO(config.path)

    def predict(self, frames: list[np.ndarray]) -> list[DetectionResult]:
        """
        Perform inference using the YOLO model.

        Parameters
        ----------
        frames : list of numpy.ndarray
            Batch of image frames to process.

        Returns
        -------
        list of DetectionResult
            Detection results for each frame from the YOLO model.
        """
        raw_results = self.model.predict(
            source=frames,
            batch=len(frames),
            classes=self.config.classes,
            verbose=False,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
        )

        return [
            DetectionResult.from_supervision(
                detections=sv.Detections.from_ultralytics(r),
                orig_shape=r.orig_shape,
                orig_img=r.orig_img,
            )
            for r in raw_results
        ]

    def stop(self) -> None:
        """
        Release the YOLO model reference.

        Deleting the model reference allows for earlier garbage collection
        and helps free up GPU memory in multi-process or high-load
        environments.
        """
        if hasattr(self, "model"):
            del self.model
