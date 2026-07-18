"""Tests for the loitering alert lifecycle and management logic."""

from unittest.mock import MagicMock, patch

import pytest

from loitering_detector.config import AlertConfig
from loitering_detector.core.alerts import AlertManager

# Constants

ALERT_INTERVAL = 5.0
MSG_NEW = "[ALERT] New loitering detected"
MSG_CONTINUING = "is still loitering"
MSG_CLEARED = "[CLEARED] Object"

STREAM_ID = 1
TRACK_ID = 101


# Fixtures


@pytest.fixture
def alert_config() -> AlertConfig:
    """Default alert configuration for testing."""
    return AlertConfig(interval=ALERT_INTERVAL)


@pytest.fixture
def alert_manager(alert_config: AlertConfig) -> AlertManager:
    """An AlertManager instance."""
    return AlertManager(alert_config)


# Tests


class TestAlertLifecycle:
    """Tests for the lifecycle of loitering alerts including new, continuing, and cleared events."""

    @patch("loitering_detector.core.alerts.logger")
    def test_initial_loitering_alert(
        self, mock_logger: MagicMock, alert_manager: AlertManager
    ) -> None:
        """
        Test that a new alert log is generated immediately when an object is detected loitering.

        Given: an object is currently loitering
        When: the AlertManager processes the state
        Then: a "NEW" alert log is generated immediately
        """
        # When: the AlertManager processes the state
        alert_manager.update(STREAM_ID, {TRACK_ID})

        # Then: a "NEW" alert log is generated immediately
        mock_logger.info.assert_called()
        args = mock_logger.info.call_args[0]
        assert MSG_NEW in args[0]
        assert args[1] == STREAM_ID
        assert args[2] == TRACK_ID

    @patch("loitering_detector.core.alerts.logger")
    def test_no_alert_when_interval_not_reached(
        self, mock_logger: MagicMock, alert_manager: AlertManager
    ) -> None:
        """
        Test that no duplicate alert is generated if the configured interval has not yet elapsed.

        Given: an object is already loitering (alerted at T=1000)
        When: the interval has NOT elapsed (T=1000 + ALERT_INTERVAL - 1)
        Then: no continuing alert is generated
        """
        start_time = 1000.0

        with patch("time.time", return_value=start_time):
            alert_manager.update(STREAM_ID, {TRACK_ID})

        mock_logger.reset_mock()

        # When: the interval has NOT elapsed
        with patch("time.time", return_value=start_time + ALERT_INTERVAL - 1.0):
            alert_manager.update(STREAM_ID, {TRACK_ID})

        # Then: NO continuing alert is logged
        assert not any(
            MSG_CONTINUING in str(call) for call in mock_logger.info.call_args_list
        )

    @patch("loitering_detector.core.alerts.logger")
    def test_continuing_alert_when_interval_reached(
        self, mock_logger: MagicMock, alert_manager: AlertManager
    ) -> None:
        """
        Test that a continuing alert is generated once the configured interval has elapsed.

        Given: an object is already loitering (alerted at T=1000)
        When: the interval has elapsed (T=1000 + ALERT_INTERVAL)
        Then: a "CONTINUING" alert is generated
        """
        start_time = 1000.0

        with patch("time.time", return_value=start_time):
            alert_manager.update(STREAM_ID, {TRACK_ID})

        mock_logger.reset_mock()

        # When: the interval is reached
        with patch("time.time", return_value=start_time + ALERT_INTERVAL):
            alert_manager.update(STREAM_ID, {TRACK_ID})

        # Then: a "CONTINUING" alert is generated
        mock_logger.info.assert_called()
        args = mock_logger.info.call_args[0]
        assert MSG_CONTINUING in args[0]
        assert args[1] == TRACK_ID

    @patch("loitering_detector.core.alerts.logger")
    def test_alert_cleared_log_generated(
        self, mock_logger: MagicMock, alert_manager: AlertManager
    ) -> None:
        """
        Test that a cleared alert log is generated when a loitering object leaves the ROI.

        Given: an object that was previously loitering
        When: the object leaves the ROI
        Then: a "CLEARED" alert log is generated
        """
        # Given: an object that was previously loitering
        alert_manager.update(STREAM_ID, {TRACK_ID})
        mock_logger.reset_mock()

        # When: the object leaves the ROI
        alert_manager.update(STREAM_ID, set())

        # Then: a "CLEARED" alert log is generated
        mock_logger.info.assert_called()
        args = mock_logger.info.call_args[0]
        assert MSG_CLEARED in args[0]
        assert args[1] == TRACK_ID
        assert args[2] == STREAM_ID

    def test_object_removed_from_state_on_exit(
        self, alert_manager: AlertManager
    ) -> None:
        """
        Test that a loitering object is removed from the internal tracking state upon leaving the ROI.

        Given: an object that was previously loitering
        When: the object leaves the ROI
        Then: the object is removed from internal state
        """
        # Given: an object that was previously loitering
        alert_manager.update(STREAM_ID, {TRACK_ID})
        assert TRACK_ID in alert_manager.alerts[STREAM_ID]

        # When: the object leaves the ROI
        alert_manager.update(STREAM_ID, set())

        # Then: the object is removed from internal state
        assert TRACK_ID not in alert_manager.alerts[STREAM_ID]
