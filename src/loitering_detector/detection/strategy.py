"""
Detection strategy implementations.

Defines the abstract interface and concrete implementations for object
detection models. These strategies are used by the `DetectionManager` to
perform inference on video frames.

Classes
-------
DetectionStrategy
    Abstract base class for all detection implementations.
"""

__all__ = ["DetectionStrategy"]

from abc import ABC, abstractmethod

import numpy as np

from loitering_detector.detection.results import DetectionResult


class DetectionStrategy(ABC):
    """
    Abstract base class defining the interface for detection models.

    Subclasses must implement methods for model inference and resource
    cleanup.
    """

    @abstractmethod
    def predict(self, frames: list[np.ndarray]) -> list[DetectionResult]:
        """
        Perform inference on a batch of frames.

        Parameters
        ----------
        frames : list of numpy.ndarray
            Batch of image frames to process.

        Returns
        -------
        list of DetectionResult
            Detection results for each frame.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release any resources held by the strategy."""
        pass
