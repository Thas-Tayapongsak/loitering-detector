"""
Video stream strategy implementations.

Defines the abstract interface and implementations for video streams.

Classes
-------
StreamStrategy
    Abstract base class for all stream strategies.
LiveStream
    Implementation for live video sources (local device, network stream,
    YouTube URL).
"""

__all__ = ["StreamStrategy", "LiveStream"]

import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import cv2
import numpy as np
import yt_dlp
from yt_dlp.utils import DownloadError

from loitering_detector.stream.utils import opencv_ffmpeg_capture_options_context

logger = logging.getLogger(__name__)


class StreamStrategy(ABC):
    """
    Abstract base class defining the interface for video stream sources.

    Subclasses must implement methods for:
    - `connect()`: Connect to a video source
    - `update()`: Update the frame buffer (continuous capturing)
    - `get_frame()`: Retrieve the latest frame from the buffer
    - `release()`: Release all associated resources
    """

    @property
    @abstractmethod
    def disconnected(self) -> bool:
        """Indicate if the stream has lost connection."""
        pass

    @abstractmethod
    def connect(self) -> None:
        """
        Initialize the connection to the video source.

        Raises
        ------
        ConnectionError
            If the source is unreachable or refuses connection.
        ValueError
            If the source configuration is invalid.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Buffer frames from the source continuously.

        This method is intended to be run in a separate background thread.
        """
        pass

    @abstractmethod
    def get_frame(self) -> tuple[np.ndarray | None, int]:
        """
        Retrieve the most recent frame from the stream buffer.

        Returns
        -------
        frame : numpy.ndarray or None
            Image frame as a NumPy array, or None if no frame is
            available.
        frame_id : int
            A unique ID of the current frame, used to check for new
            frames. Returns -1 if no frame is available.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Stop the stream capture and release all associated resources.
        """
        pass


class LiveStream(StreamStrategy):
    """
    Implementation of a video stream for live sources.

    Manages a live video feed using a producer-consumer model.
    A background thread continuously polls the source and maintains a
    single-frame buffer. `get_frame()` retrieves the latest available
    frame without waiting for network I/O.

    Features:
    - Handles local cameras, RTSP/HTTP streams, and direct YouTube links
    - Performs URI resolution and TCP connection tests before opening
      the stream
    - Handles automatic reconnection if the stream drops or times out
    - Thread-safe frame access

    Attributes
    ----------
    source : str or int
        The raw connection string or camera index.
    timeout : float
        Maximum time in seconds to wait for a connection or a new frame.
    resolved_source : str or int, optional
        The final URI after resolution (e.g., direct YouTube video link).

    Examples
    --------
    Connecting to a local camera:

    >>> stream = LiveStream(source=0)
    >>> stream.connect()
    >>> frame, frame_id = stream.get_frame()
    >>> stream.release()

    Using a network stream with a custom timeout:

    >>> stream = LiveStream(source="rtsp://example.com/live", timeout=10.0)
    >>> stream.connect()
    >>> # Typically run update() in a background thread...
    """

    def __init__(self, source: str | int, timeout: float = 5.0):
        """
        Initialize the live stream.

        Parameters
        ----------
        source : str or int
            The source of the stream (e.g., RTSP URL, YouTube link, or
            camera index).
        timeout : float, optional, default 5.0
            The connection and read timeout in seconds.
        """
        # Connection Configuration
        self.source: str | int = source
        self.resolved_source: str | int | None = None
        self.timeout: float = timeout

        # Internal Capture State
        self._cap: cv2.VideoCapture | None = None
        self._lock: threading.Lock = threading.Lock()

        # Buffer
        self._current_frame: np.ndarray | None = None
        self._frame_id: int = 0
        self._last_read_time: float = 0.0

        # Lifecycle Flags
        self._disconnected: bool = False
        self._is_stopped: bool = False
        self._is_updating: bool = False

    @property
    def disconnected(self) -> bool:
        """Indicate if the stream has lost connection."""
        return self._disconnected

    def connect(self) -> None:
        """
        Initialize the connection to the video source.

        Determine connection strategy based on source type.
        Hardware cameras (int) are opened directly. Network sources (str) undergo
        URI resolution (e.g., YouTube extraction) and a TCP pre-flight check
        before being initialized with custom OpenCV/FFmpeg options.

        Raises
        ------
        RuntimeError
            If the source is not connected after initialization.
        ConnectionError
            If the source is unreachable or refuses connection.
        ValueError
            If the source configuration is invalid.
        """
        # Connect based on source type
        _is_local = isinstance(self.source, int) or (
            isinstance(self.source, str) and self.source.isdigit()
        )
        if _is_local:
            self._connect_local_device()
        else:
            self._connect_network_source()

        # Guard against stopping during connection
        if self._is_stopped:
            self._release_cap()
            return

        if self._cap is None:
            raise RuntimeError("Unexpected state: self._cap is None after connection.")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            self._release_cap()
            raise ValueError(f"Unable to open video source: {self.source}")

        # Confirm connection
        grabbed, frame = self._cap.read()
        if not grabbed:
            self._release_cap()
            raise ConnectionError(
                f"Unable to read from video source: {self.source} "
                f"(resolved: {self.resolved_source})"
            )

        # Get first frame
        self._update_buffer(frame)

    def update(self) -> None:
        """
        Buffer frames from the source continuously.

        This method is intended to be run in a separate background thread.
        """
        self._is_updating = True
        try:
            if not self._is_cap_connected():
                return
            self._update_loop()
        finally:
            self._is_updating = False
            self._release_cap()

    def get_frame(self) -> tuple[np.ndarray | None, int]:
        """
        Retrieve the most recent frame from the stream buffer.

        Returns
        -------
        frame : numpy.ndarray or None
            Image frame as a NumPy array, or None if no frame is
            available.
        frame_id : int
            A unique ID of the current frame, used to check for new
            frames. Returns -1 if no frame is available.
        """
        with self._lock:
            if self._current_frame is not None:
                view = self._current_frame.view()
                view.flags.writeable = False
                return view, self._frame_id
            else:
                return None, -1

    def release(self) -> None:
        """
        Stop the stream capture and release all associated resources.

        Signals the background thread to exit gracefully before closing
        the underlying video capture.
        """
        self._is_stopped = True
        if not self._is_updating:
            self._release_cap()

    def _connect_local_device(self) -> None:
        """Connect cv2.VideoCapture to a local device."""
        self.source = int(self.source)
        self.resolved_source = self.source

        if self._is_stopped:
            return

        with opencv_ffmpeg_capture_options_context(
            reconnect=1, reconnect_streamed=1, reconnect_delay_max=5
        ):
            self._cap = cv2.VideoCapture(self.resolved_source)

    def _connect_network_source(self) -> None:
        """Connect cv2.VideoCapture to a network source."""
        self.resolved_source = self.source

        # Resolve youtube url
        if "youtube.com" in str(self.source) or "youtu.be" in str(self.source):
            self.resolved_source = self._resolve_youtube_url()

        if not self._is_source_reachable(str(self.resolved_source)):
            raise ConnectionError(f"Host unreachable: {self.resolved_source}")

        if self._is_stopped:
            return

        with opencv_ffmpeg_capture_options_context(
            reconnect=1,
            reconnect_streamed=1,
            reconnect_delay_max=5,
            allowed_extensions="ALL",
        ):
            timeout_ms = int(self.timeout * 1000)
            ffmpeg_options = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                timeout_ms,
                cv2.CAP_PROP_HW_ACCELERATION,
                cv2.VIDEO_ACCELERATION_ANY,
            ]
            self._cap = cv2.VideoCapture(
                self.resolved_source, cv2.CAP_FFMPEG, ffmpeg_options
            )

    def _update_loop(self) -> None:
        """Continuously grab frames and update the internal buffer."""
        while not self._is_stopped:
            if self._cap is None:
                break

            try:
                grabbed, frame = self._cap.read()
            except cv2.error:
                grabbed, frame = False, None

            now = time.perf_counter()
            if grabbed and frame is not None:
                self._update_buffer(frame)
            else:
                if self._cap is None or not self._cap.isOpened():
                    self._disconnected = True
                    break

                _is_timeout = (now - self._last_read_time) > self.timeout
                if _is_timeout:
                    self._disconnected = True
                    break

                time.sleep(0.01)

    def _update_buffer(self, frame: np.ndarray) -> None:
        """Update the stream buffer."""
        self._last_read_time = time.perf_counter()
        with self._lock:
            self._current_frame = frame
            self._frame_id = (self._frame_id + 1) % 1000000

    def _release_cap(self) -> None:
        """Release the cv2.VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _resolve_youtube_url(self) -> str:
        """Extract direct video URL from YouTube link using yt-dlp."""
        if not isinstance(self.source, str):
            raise TypeError("YouTube source must be a string URL.")
        ydl_opts = {"format": "best", "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(self.source, download=False)
            except DownloadError as e:
                raise ConnectionError(
                    f"Unable to extract video URL from YouTube link: {self.source}"
                ) from e
            video_url = info_dict.get("url", None)
            if video_url is None:
                raise ValueError(
                    f"Unable to extract video URL from YouTube link: {self.source}"
                )
            assert isinstance(video_url, str)
            return video_url

    def _is_source_reachable(self, url: str) -> bool:
        """Check if a network host is reachable via TCP."""
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            if not host:
                return True  # Likely local or relative path

            port = parsed.port
            if not port:
                if parsed.scheme == "rtsp":
                    port = 554
                elif parsed.scheme == "http":
                    port = 80
                elif parsed.scheme == "https":
                    port = 443
                else:
                    return True  # Unknown scheme, skip check

            with socket.create_connection((host, port), timeout=self.timeout):
                return True
        except Exception:
            return False

    def _is_cap_connected(self) -> bool:
        """Check if VideoCapture is connected and try to reconnect if not."""
        if self._cap is not None and self._cap.isOpened():
            return True

        try:
            self.connect()
            return self._cap is not None and self._cap.isOpened()
        except Exception as e:
            logger.error(
                "Connection failed for %s (resolved: %s): %s",
                self.source,
                self.resolved_source,
                e,
            )
            self._disconnected = True
            return False
