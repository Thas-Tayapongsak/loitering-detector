"""Global fixtures for backend unit tests."""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import redis

# Keep reference to the real redis.Redis class for spec definition,
# since redis.Redis itself will be monkeypatched during tests.
_ORIGINAL_REDIS_CLASS = redis.Redis


@pytest.fixture
def mock_frame() -> np.ndarray:
    """A dummy 100x100 RGB frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_weights_file(tmp_path: Path) -> Path:
    """Create a dummy weights file for configuration and model loading tests."""
    path = tmp_path / "mock.pt"
    path.write_text("dummy weights")
    return path


@pytest.fixture
def create_mock_weights() -> Generator[Callable[[str], str], None, None]:
    """Fixture that returns a helper to create a mock weights file at any path."""
    import os

    created_files = []

    def _create(path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("dummy weights for offline testing")
        created_files.append(path)
        return path

    yield _create

    # Cleanup created files after test
    for file_path in created_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


def generate_synthetic_frame(
    width: int = 640,
    height: int = 480,
    channels: int = 3,
    color: tuple[int, int, int] = (0, 0, 0),
    draw_shape: str | None = None,
    shape_coords: tuple[Any, ...] | None = None,
) -> np.ndarray:
    """
    Generate a synthetic mock video stream frame as a NumPy array.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        channels: Number of image channels (default 3 for RGB).
        color: Base background color.
        draw_shape: Optional shape to draw ('rect' or 'circle').
        shape_coords: Coordinates for the shape (x1, y1, x2, y2) or (cx, cy, r).

    Returns:
        np.ndarray: Synthetic frame of shape (height, width, channels) and dtype uint8.
    """
    frame = np.zeros((height, width, channels), dtype=np.uint8)
    if color != (0, 0, 0):
        frame[:] = color

    if draw_shape and shape_coords:
        import cv2

        if draw_shape == "rect":
            x1, y1, x2, y2 = shape_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        elif draw_shape == "circle":
            cx, cy, r = shape_coords
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1)

    return frame


@pytest.fixture
def synthetic_frame_generator() -> Callable[..., np.ndarray]:
    """Fixture providing a generator for synthetic video frames."""
    return generate_synthetic_frame


@pytest.fixture
def mock_redis() -> MagicMock:
    """A standard mock Redis client for unit testing."""
    mock = MagicMock(spec=_ORIGINAL_REDIS_CLASS)
    mock.decode_responses = True

    # Mock script registration to return a callable MagicMock
    mock_script = MagicMock()
    mock.register_script.return_value = mock_script

    return mock


@pytest.fixture(autouse=True)
def stub_redis_network(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Autouse fixture that patches redis.Redis globally during backend unit tests
    to prevent any accidental external network calls to a Redis server.
    """
    mock_client = MagicMock(spec=_ORIGINAL_REDIS_CLASS)
    mock_client.decode_responses = True
    mock_client.ping.return_value = True

    mock_script = MagicMock()
    mock_client.register_script.return_value = mock_script

    # Monkeypatch the redis.Redis class to return the mock client
    monkeypatch.setattr(redis, "Redis", MagicMock(return_value=mock_client))
    return mock_client
