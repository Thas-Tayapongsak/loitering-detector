"""OpenCV-based implementation of GeometryEngine interface."""

import cv2
import numpy as np

from loitering_detector.core.interfaces import GeometryEngine


class OpenCVGeometryEngine(GeometryEngine):
    """
    Geometry engine that encapsulates OpenCV's pointPolygonTest.
    Caches the np.ndarray representation of polygons to avoid redundant casting.
    """

    def __init__(self) -> None:
        self._roi_cache: dict[int, np.ndarray] = {}

    def is_inside(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        """
        Check if a point is inside the given polygon using cv2.pointPolygonTest.
        """
        poly_id = id(polygon)
        np_poly = self._roi_cache.get(poly_id)
        if np_poly is None:
            np_poly = np.array(polygon, dtype=np.float32)
            self._roi_cache[poly_id] = np_poly

        res = cv2.pointPolygonTest(np_poly, point, False)
        return res >= 0
