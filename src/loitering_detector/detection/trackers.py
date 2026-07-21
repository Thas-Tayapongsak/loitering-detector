__all__ = [
    "TrackerInterface",
    "TrackersByteTrack",
]

from abc import ABC, abstractmethod

from trackers import ByteTrackTracker

from loitering_detector.detection.results import DetectionResult


class TrackerInterface(ABC):
    @abstractmethod
    def update(self, detection_result: DetectionResult) -> DetectionResult:
        pass


class TrackersByteTrack(TrackerInterface):
    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
    ):
        self.tracker = ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=float(frame_rate),
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )

    def update(self, detection_result: DetectionResult) -> DetectionResult:
        sv_dets = detection_result.to_supervision()
        updated_dets = self.tracker.update(detections=sv_dets)
        return DetectionResult.from_supervision(
            detections=updated_dets,
            orig_shape=detection_result.orig_shape,
            orig_img=detection_result.orig_img,
        )
