"""Tests for the detection and inference pipeline, including model loading, batch processing, and multi-stream tracking."""

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from loitering_detector.detection.config import DetectionConfig
from loitering_detector.detection.manager import DetectionManager
from loitering_detector.detection.strategy import YOLODetection

# Constants

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.5
DEFAULT_TRACKER: Literal["bytetrack", "botsort"] = "bytetrack"
DEFAULT_CLASSES = [0]

STREAM_1_ID = 1
STREAM_2_ID = 2

TRACK_ID = 5


# Fixtures


@pytest.fixture
def detection_config(mock_weights_file: Path) -> DetectionConfig:
    """Default detection configuration for testing."""
    return DetectionConfig(
        path=mock_weights_file,
        imgsz=DEFAULT_IMGSZ,
        conf=DEFAULT_CONF,
        tracker=DEFAULT_TRACKER,
        classes=DEFAULT_CLASSES,
    )


@pytest.fixture
def mock_results() -> MagicMock:
    """Create a mock Results object from Ultralytics."""
    mock = MagicMock(spec=Results)
    mock.boxes = MagicMock()
    mock.boxes.data = torch.tensor([[0, 0, 10, 10, 0.9, 0]])
    mock.orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
    return mock


# Tests


class TestDetectionInference:
    """Tests for the DetectionManager and YOLODetection strategy including batch inference and tracking isolation."""

    @patch("loitering_detector.detection.strategy.YOLO")
    def test_yolo_model_loading(
        self, mock_yolo_cls: MagicMock, detection_config: DetectionConfig
    ) -> None:
        """
        Test that the YOLOv8 model is correctly loaded using the configured weights path.

        Given: a path to a YOLOv8 weight file
        When: the DetectionStrategy is initialized
        Then: the model is successfully loaded into the underlying runtime
        """
        # When: the DetectionStrategy is initialized
        strategy = YOLODetection(detection_config)

        # Then: the model is successfully loaded
        assert strategy.model is not None
        mock_yolo_cls.assert_called_with(detection_config.path)

    @patch("loitering_detector.detection.strategy.YOLO")
    def test_detection_manager_init(
        self, mock_yolo_cls: MagicMock, detection_config: DetectionConfig
    ) -> None:
        """
        Test that the DetectionManager correctly isolates trackers for multiple independent streams.

        Given: Stream A (ID: 1) and Stream B (ID: 2)
        When: the DetectionManager is initialized
        Then: each stream is assigned its own distinct tracker instance
        """
        # Given: Stream IDs
        stream_ids = [STREAM_1_ID, STREAM_2_ID]

        # When: DetectionManager is initialized
        with patch(
            "loitering_detector.detection.manager.TRACKER_REGISTRY"
        ) as mock_registry:
            # Mock the registry to return simple mocks for trackers
            # side_effect ensures that each call returns a NEW MagicMock
            mock_tracker_cls = MagicMock(
                side_effect=lambda *args, **kwargs: MagicMock()
            )
            mock_args_cls = MagicMock()
            mock_registry.get.return_value = (mock_tracker_cls, mock_args_cls)

            manager = DetectionManager(detection_config, active_stream_ids=stream_ids)

            # Then: the manager contains two isolated trackers
            assert len(manager.trackers) == 2
            assert STREAM_1_ID in manager.trackers
            assert STREAM_2_ID in manager.trackers
            assert manager.trackers[STREAM_1_ID] is not manager.trackers[STREAM_2_ID]

    def test_detection_manager_infer_mismatch(
        self, detection_config: DetectionConfig
    ) -> None:
        """
        Ensure that an error is raised when the number of frames does not match the number of stream IDs.

        Given: a list of one frame
        When: calling infer() with two stream IDs
        Then: a ValueError is raised due to the batch size mismatch
        """
        manager = DetectionManager(
            detection_config, active_stream_ids=[STREAM_1_ID], strategy=MagicMock()
        )

        # When: calling infer with mismatched batch sizes
        # Then: it raises a ValueError
        with pytest.raises(ValueError, match="Batch size mismatch"):
            manager.infer(
                frames=[np.zeros((10, 10, 3))], stream_ids=[STREAM_1_ID, STREAM_2_ID]
            )

    @patch("loitering_detector.detection.strategy.YOLO")
    def test_yolo_strategy_predict(
        self, mock_yolo_cls: MagicMock, detection_config: DetectionConfig
    ) -> None:
        """
        Verify that inference parameters like confidence and image size are correctly passed to the YOLO model.

        Given: a detection configuration with specific confidence and image size
        When: predict() is called on a frame
        Then: the underlying model receives the correct hyperparameters
        """
        mock_yolo = mock_yolo_cls.return_value
        strategy = YOLODetection(detection_config)
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

        # When: performing a prediction
        strategy.predict(frames)

        # Then: hyperparameters match the configuration
        mock_yolo.predict.assert_called_with(
            source=frames,
            batch=1,
            classes=DEFAULT_CLASSES,
            verbose=False,
            conf=DEFAULT_CONF,
            imgsz=DEFAULT_IMGSZ,
        )

    @patch("loitering_detector.detection.manager.TRACKER_REGISTRY")
    def test_tracker_selection(
        self, mock_registry: MagicMock, detection_config: DetectionConfig
    ) -> None:
        """
        Verify that the specified tracking algorithm is correctly instantiated from the registry.

        Given: a configuration specifying the "botsort" tracker
        When: the DetectionManager is initialized
        Then: the tracker class associated with "botsort" is instantiated
        """
        # Given: "botsort" configuration
        detection_config.tracker = "botsort"
        mock_tracker_cls = MagicMock()
        mock_args_cls = MagicMock()
        mock_registry.get.return_value = (mock_tracker_cls, mock_args_cls)

        # When: initializing the manager
        manager = DetectionManager(
            detection_config, active_stream_ids=[STREAM_1_ID], strategy=MagicMock()
        )

        # Then: the botsort tracker is instantiated
        mock_tracker_cls.assert_called()
        assert manager.config.tracker == "botsort"

    @patch("loitering_detector.detection.manager.TRACKER_REGISTRY")
    def test_id_persistence_tracking(
        self,
        mock_registry: MagicMock,
        detection_config: DetectionConfig,
        mock_results: MagicMock,
    ) -> None:
        """
        Verify that object tracking IDs are correctly persisted and mapped onto detection results across frames.

        Given: a tracker that consistently assigns ID 5 to a detected object
        When: a frame is processed through the manager
        Then: the resulting detection is updated with Track ID 5
        """
        # Given: a tracker that identifies an object as ID 5
        mock_tracker = MagicMock()
        # Mock tracker update to return [x1, y1, x2, y2, id, conf, cls]
        # TC0009 requires tracking ID 5
        mock_tracker.update.return_value = np.array([[0, 0, 10, 10, 5, 0.9, 0]])

        mock_tracker_cls = MagicMock(return_value=mock_tracker)
        mock_args_cls = MagicMock()
        mock_registry.get.return_value = (mock_tracker_cls, mock_args_cls)

        strategy = MagicMock()
        strategy.predict.return_value = [mock_results]

        manager = DetectionManager(
            detection_config, active_stream_ids=[STREAM_1_ID], strategy=strategy
        )

        # When: processing a frame
        manager.infer(frames=[np.zeros((100, 100, 3))], stream_ids=[STREAM_1_ID])

        # Then: the detection results are updated with the tracker ID 5
        assert mock_results.update.called
        updated_boxes = mock_results.update.call_args[1]["boxes"]
        assert updated_boxes[0, 4] == TRACK_ID
