"""
Unit tests for detection results DTOs and Supervision conversion adapter.
"""

import numpy as np
import pytest
import supervision as sv

from loitering_detector.detection.results import BoundingBox, DetectionResult


def test_bounding_box_initialization():
    box = BoundingBox(
        x1=10.0,
        y1=20.0,
        x2=50.0,
        y2=60.0,
        confidence=0.95,
        class_id=0,
    )
    assert box.track_id is None
    assert not box.is_tracked

    tracked_box = BoundingBox(
        x1=10.0,
        y1=20.0,
        x2=50.0,
        y2=60.0,
        confidence=0.95,
        class_id=0,
        track_id=42,
    )
    assert tracked_box.track_id == 42
    assert tracked_box.is_tracked


def test_to_supervision_empty():
    result = DetectionResult(boxes=[], orig_shape=(100, 100))
    sv_dets = result.to_supervision()
    assert len(sv_dets) == 0


def test_to_supervision_untracked():
    box = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0, confidence=0.9, class_id=0)
    result = DetectionResult(boxes=[box], orig_shape=(100, 100))
    sv_dets = result.to_supervision()

    assert len(sv_dets) == 1
    assert sv_dets.confidence is not None
    assert sv_dets.class_id is not None
    assert np.allclose(sv_dets.xyxy[0], [10.0, 20.0, 50.0, 60.0])
    assert sv_dets.confidence[0] == pytest.approx(0.9)
    assert sv_dets.class_id[0] == 0
    assert sv_dets.tracker_id is None


def test_to_supervision_partially_tracked():
    tracked_box = BoundingBox(
        x1=10.0, y1=20.0, x2=50.0, y2=60.0, confidence=0.9, class_id=0, track_id=7
    )
    untracked_box = BoundingBox(
        x1=15.0, y1=25.0, x2=55.0, y2=65.0, confidence=0.8, class_id=0, track_id=None
    )
    result = DetectionResult(boxes=[tracked_box, untracked_box], orig_shape=(100, 100))
    sv_dets = result.to_supervision()

    assert len(sv_dets) == 2
    assert sv_dets.tracker_id is not None
    assert sv_dets.tracker_id[0] == 7
    assert sv_dets.tracker_id[1] == -1


def test_from_supervision_empty():
    sv_dets = sv.Detections.empty()
    result = DetectionResult.from_supervision(sv_dets, orig_shape=(100, 100))
    assert len(result.boxes) == 0
    assert result.orig_shape == (100, 100)


def test_from_supervision_conversion():
    xyxy = np.array(
        [[10.0, 20.0, 50.0, 60.0], [15.0, 25.0, 55.0, 65.0]], dtype=np.float32
    )
    confidence = np.array([0.9, 0.8], dtype=np.float32)
    class_id = np.array([0, 1], dtype=int)
    tracker_id = np.array([7, -1], dtype=int)

    sv_dets = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )

    result = DetectionResult.from_supervision(sv_dets, orig_shape=(100, 100))
    assert len(result.boxes) == 2

    box1 = result.boxes[0]
    assert box1.x1 == 10.0
    assert box1.y1 == 20.0
    assert box1.confidence == pytest.approx(0.9)
    assert box1.class_id == 0
    assert box1.is_tracked
    assert box1.track_id == 7

    box2 = result.boxes[1]
    assert not box2.is_tracked
    assert box2.track_id is None


def test_plot_raises_without_image():
    result = DetectionResult(boxes=[], orig_shape=(100, 100), orig_img=None)
    with pytest.raises(ValueError, match="No image to plot"):
        result.plot()


def test_plot_renders_scene():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = BoundingBox(
        x1=10.0, y1=20.0, x2=50.0, y2=60.0, confidence=0.9, class_id=0, track_id=1
    )
    result = DetectionResult(boxes=[box], orig_shape=(100, 100), orig_img=dummy_img)

    plotted = result.plot()
    assert isinstance(plotted, np.ndarray)
    assert plotted.shape == (100, 100, 3)
