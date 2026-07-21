__all__ = ["BoundingBox", "DetectionResult"]

from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    track_id: int | None = None

    @property
    def is_tracked(self) -> bool:
        return self.track_id is not None


class DetectionResult:
    def __init__(
        self,
        boxes: list[BoundingBox],
        orig_shape: tuple[int, int],
        orig_img: np.ndarray | None = None,
    ):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.orig_img = orig_img

    def to_supervision(self) -> sv.Detections:
        if not self.boxes:
            return sv.Detections.empty()

        # Unpack BoundingBox into numpy arrays
        xyxy = np.array(
            [[b.x1, b.y1, b.x2, b.y2] for b in self.boxes], dtype=np.float32
        )
        confidence = np.array([b.confidence for b in self.boxes], dtype=np.float32)
        class_id = np.array([b.class_id for b in self.boxes], dtype=int)

        # If all box has None, then there is no tracked boxes
        if all(not b.is_tracked for b in self.boxes):
            tracker_id = None
        else:
            tracker_id = np.array(
                [b.track_id if b.is_tracked else -1 for b in self.boxes],
                dtype=int,
            )

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )

    @classmethod
    def from_supervision(
        cls,
        detections: sv.Detections,
        orig_shape: tuple[int, int],
        orig_img: np.ndarray | None = None,
    ) -> "DetectionResult":
        if len(detections) == 0:
            return cls(boxes=[], orig_shape=orig_shape, orig_img=orig_img)

        confidences = (
            detections.confidence
            if detections.confidence is not None
            else [1.0] * len(detections)
        )
        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else [None] * len(detections)
        )
        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else [-1] * len(detections)
        )

        boxes = [
            BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=conf,
                class_id=cls_id,
                track_id=int(trk_id) if trk_id is not None and trk_id != -1 else None,
            )
            for box, conf, cls_id, trk_id in zip(
                detections.xyxy, confidences, class_ids, tracker_ids, strict=True
            )
        ]

        return cls(boxes=boxes, orig_shape=orig_shape, orig_img=orig_img)

    def plot(self) -> np.ndarray:
        if self.orig_img is None:
            raise ValueError("No image to plot")

        scene = self.orig_img.copy()
        detections = self.to_supervision()

        if len(detections) > 0:
            labels = [
                f"#{b.track_id} {b.class_id}" if b.is_tracked else f"{b.class_id}"
                for b in self.boxes
            ]
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            scene = np.asarray(
                box_annotator.annotate(scene=scene, detections=detections)
            )
            scene = np.asarray(
                label_annotator.annotate(
                    scene=scene, detections=detections, labels=labels  # type: ignore[arg-type]
                )
            )

        return scene
