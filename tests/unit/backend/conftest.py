"""Global fixtures for backend unit tests."""

import numpy as np
import pytest


@pytest.fixture
def mock_frame():
    """A dummy 100x100 RGB frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_weights_file(tmp_path):
    """Create a dummy weights file for configuration and model loading tests."""
    path = tmp_path / "mock.pt"
    path.write_text("dummy weights")
    return path
