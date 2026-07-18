"""
High-level video stream management.

This module coordinates the specialized stream strategies to provide
a simplified, thread-safe API for frame capture. It handles the
background thread lifecycle and frame deduplication.

Classes
-------
StreamManager
    Manager for handling the lifecycle of a video stream.
"""

__all__ = ["StreamManager"]

import threading
from types import TracebackType

import numpy as np

from loitering_detector.stream.config import StreamConfig
from loitering_detector.stream.strategy import LiveStream, StreamStrategy


class StreamManager:
    """
    Manager for handling the lifecycle of a video stream.

    Orchestrates a `StreamStrategy` to provide a simplified, thread-safe
    API for frame capture. This manager handles the background thread
    lifecycle, automatic disconnection cleanup, and frame deduplication
    via unique frame IDs.

    Features:
    - Automates the start, stop, and join logic for background capture threads.
    - Implements the Context Manager pattern for guaranteed resource cleanup.
    - Ensures the same frame is never processed twice by tracking internal frame IDs.

    Attributes
    ----------
    config : StreamConfig
        The configuration settings for this stream.
    is_running : bool
        True if the background capture thread is active.
    last_frame_id : int
        ID of the most recently processed frame, used for deduplication.

    Examples
    --------
    Using the manager as a context manager (recommended):

    >>> with StreamManager(config) as stream:
    ...     while True:
    ...         frame = stream.read()
    ...         if frame is not None:
    ...             process(frame)

    Manual lifecycle management:

    >>> stream = StreamManager(config)
    >>> stream.start()
    >>> frame = stream.read()
    >>> stream.stop()
    """

    def __init__(self, config: StreamConfig):
        """
        Initialize the video stream manager.

        Parameters
        ----------
        config : StreamConfig
            The configuration for the stream.
        """
        self.config: StreamConfig = config

        # State
        self.is_running: bool = False
        self.last_frame_id: int = -1

        # Threading State
        self._thread: threading.Thread | None = None
        self._strategy: StreamStrategy | None = None

    def __enter__(self) -> "StreamManager":
        """Start the stream on context entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Stop the stream on context exit."""
        self.stop()
        return None

    def start(self) -> None:
        """Initialize the strategy and start the capture thread."""
        if self.is_running:
            return

        self._init_strategy()
        self._start_strategy_thread()
        self.is_running = True

    def read(self) -> np.ndarray | None:
        """
        Read a frame from the stream.

        Returns
        -------
        frame : numpy.ndarray or None
            The image frame as a NumPy array, or
            None if no frame is available or the frame is stale.
        """
        if not self.is_running or self._strategy is None:
            return None

        if self._strategy.disconnected:
            self.stop()
            return None

        frame, frame_id = self._strategy.get_frame()

        # Fail to read or stale frame, return None
        if frame is None or frame_id == self.last_frame_id:
            return None

        self.last_frame_id = frame_id
        return frame

    def stop(self) -> None:
        """Signal the thread to stop and release resources."""
        self.is_running = False
        self._stop_strategy_thread()

    def _init_strategy(self) -> None:
        """Initialize the specific capture strategy based on source type."""
        _is_live_stream = isinstance(self.config.source, int) or (
            isinstance(self.config.source, str)
            and (
                self.config.source.startswith(("rtsp://", "http://", "https://"))
                or self.config.source.isdigit()
            )
        )
        if _is_live_stream:
            self._strategy = LiveStream(self.config.source, timeout=self.config.timeout)
        else:
            raise NotImplementedError(
                f"Unsupported stream source type: {self.config.source}"
            )

    def _start_strategy_thread(self) -> None:
        """Start the stream strategy thread."""
        assert self._strategy is not None
        self._thread = threading.Thread(target=self._strategy.update, daemon=True)
        self._thread.start()

    def _stop_strategy_thread(self) -> None:
        """Signal the strategy to stop and join the background thread."""
        if self._strategy is not None:
            self._strategy.release()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
