"""
Unit tests for tracker interface and TrackersByteTrack wrapper.
"""

import numpy as np

from loitering_detector.detection.results import BoundingBox, DetectionResult
from loitering_detector.detection.trackers import (
    TrackerInterface,
    TrackersByteTrack,
)


def test_trackers_bytetrack_initialization() -> None:
    tracker = TrackersByteTrack(
        lost_track_buffer=30,
        frame_rate=30,
        track_activation_threshold=0.7,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
    )
    assert isinstance(tracker, TrackerInterface)


def test_trackers_bytetrack_update_empty() -> None:
    tracker = TrackersByteTrack()
    result = DetectionResult(boxes=[], orig_shape=(100, 100))
    updated_result = tracker.update(result)

    assert len(updated_result.boxes) == 0
    assert updated_result.orig_shape == (100, 100)


def test_trackers_bytetrack_tracking_sequence() -> None:
    tracker = TrackersByteTrack(frame_rate=30)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Frame 1: Initial detection spawns tentative track
    box_f1 = BoundingBox(
        x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0
    )
    res_f1 = DetectionResult(boxes=[box_f1], orig_shape=(640, 640), orig_img=dummy_img)
    updated_f1 = tracker.update(res_f1)
    assert len(updated_f1.boxes) == 1

    # Frame 2: Second detection confirms track and assigns tracker_id
    box_f2 = BoundingBox(
        x1=105.0, y1=105.0, x2=205.0, y2=205.0, confidence=0.88, class_id=0
    )
    res_f2 = DetectionResult(boxes=[box_f2], orig_shape=(640, 640), orig_img=dummy_img)
    updated_f2 = tracker.update(res_f2)

    assert len(updated_f2.boxes) == 1
    box2 = updated_f2.boxes[0]
    assert box2.is_tracked
    assert box2.track_id is not None
    confirmed_track_id = box2.track_id

    # Frame 3: Subsequent detection maintains confirmed track_id
    box_f3 = BoundingBox(
        x1=110.0, y1=110.0, x2=210.0, y2=210.0, confidence=0.85, class_id=0
    )
    res_f3 = DetectionResult(boxes=[box_f3], orig_shape=(640, 640), orig_img=dummy_img)
    updated_f3 = tracker.update(res_f3)

    assert len(updated_f3.boxes) == 1
    box3 = updated_f3.boxes[0]
    assert box3.is_tracked
    assert box3.track_id == confirmed_track_id
