"""
Unit tests for tracker interface and SupervisionByteTrack wrapper.
"""

import numpy as np

from loitering_detector.detection.results import BoundingBox, DetectionResult
from loitering_detector.detection.trackers import (
    SupervisionByteTrack,
    TrackerInterface,
)


def test_supervision_bytetrack_initialization() -> None:
    tracker = SupervisionByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    assert isinstance(tracker, TrackerInterface)


def test_supervision_bytetrack_update_empty() -> None:
    tracker = SupervisionByteTrack()
    result = DetectionResult(boxes=[], orig_shape=(100, 100))
    updated_result = tracker.update(result)

    assert len(updated_result.boxes) == 0
    assert updated_result.orig_shape == (100, 100)


def test_supervision_bytetrack_tracking_sequence() -> None:
    tracker = SupervisionByteTrack(frame_rate=30)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Frame 1: Initial detection of an object
    box_f1 = BoundingBox(
        x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0
    )
    res_f1 = DetectionResult(boxes=[box_f1], orig_shape=(640, 640), orig_img=dummy_img)
    updated_f1 = tracker.update(res_f1)

    assert len(updated_f1.boxes) == 1
    box1 = updated_f1.boxes[0]
    assert box1.is_tracked
    assert box1.track_id is not None
    initial_track_id = box1.track_id

    # Frame 2: Slightly shifted detection of the same object
    box_f2 = BoundingBox(
        x1=105.0, y1=105.0, x2=205.0, y2=205.0, confidence=0.88, class_id=0
    )
    res_f2 = DetectionResult(boxes=[box_f2], orig_shape=(640, 640), orig_img=dummy_img)
    updated_f2 = tracker.update(res_f2)

    assert len(updated_f2.boxes) == 1
    box2 = updated_f2.boxes[0]
    assert box2.is_tracked
    assert box2.track_id == initial_track_id
