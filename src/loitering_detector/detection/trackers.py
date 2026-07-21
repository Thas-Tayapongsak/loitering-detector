__all__ = [
    "TrackerInterface",
    "SupervisionByteTrack",
]

from abc import ABC, abstractmethod

import supervision as sv

from loitering_detector.detection.results import DetectionResult


class TrackerInterface(ABC):
    @abstractmethod
    def update(self, detection_result: DetectionResult) -> DetectionResult:
        pass


class SupervisionByteTrack(TrackerInterface):
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detection_result: DetectionResult) -> DetectionResult:
        sv_dets = detection_result.to_supervision()
        updated_dets = self.tracker.update_with_detections(detections=sv_dets)
        return DetectionResult.from_supervision(
            detections=updated_dets,
            orig_shape=detection_result.orig_shape,
            orig_img=detection_result.orig_img,
        )
