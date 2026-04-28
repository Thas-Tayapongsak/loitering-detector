"""
Utility functions for the `stream` module.

Provides helper functions and context managers to handle lower-level
aspects of video stream management.

Functions
---------
opencv_ffmpeg_capture_options_context
    Safely set `OPENCV_FFMPEG_CAPTURE_OPTIONS` for `cv2.VideoCapture`.
"""

__all__ = ["opencv_ffmpeg_capture_options_context"]

import os
import threading
from contextlib import contextmanager

_lock = threading.Lock()


@contextmanager
def opencv_ffmpeg_capture_options_context(
    reconnect: int | None = 1,
    reconnect_streamed: int | None = 1,
    reconnect_delay_max: int | None = 5,
    **extra_options,
):
    """
    Safely set `OPENCV_FFMPEG_CAPTURE_OPTIONS` for `cv2.VideoCapture`.

    Used while instantiating a `cv2.VideoCapture` one at a time.

    Uses a global lock to prevent threads from mutating environment
    variables at the same time. Specifically, it is used for setting
    `OPENCV_FFMPEG_CAPTURE_OPTIONS` which has no supported
    `cv2.CAP_PROP_` counterpart.

    Parameters
    ----------
    reconnect : int, optional, default 1
        Enable/disable automatic reconnection.
    reconnect_streamed : int, optional, default 1
        Enable/disable reconnection for streamed content.
    reconnect_delay_max : int, optional, default 5
        Maximum delay (seconds) between reconnection attempts.
    **extra_options : dict
        Additional FFmpeg options as key-value pairs.

    Examples
    --------
    >>> with opencv_ffmpeg_capture_options_context(
    ...     reconnect=1,
    ...     reconnect_streamed=1,
    ...     reconnect_delay_max=5
    ... ):
    ...     cap = cv2.VideoCapture("rtsp://example.com/stream")
    """
    options = {
        "reconnect": reconnect,
        "reconnect_streamed": reconnect_streamed,
        "reconnect_delay_max": reconnect_delay_max,
    }
    options.update(extra_options)

    # Filter out None values.
    # Join into "key;val|key;val" format
    new_options = "|".join([f"{k};{v}" for k, v in options.items() if v is not None])

    with _lock:
        old_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = new_options
        try:
            yield
        finally:
            # Reset back to original value.
            if old_options is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old_options
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
