"""Tests for configuration models and settings loader."""

import os
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from loitering_detector.config import (
    AlertConfig,
    LoiteringConfig,
    RedisConfig,
    Settings,
    SystemConfig,
    get_config,
)

# Constants

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PORT_ALT = 6380

DEFAULT_THRESHOLD = 30.0
DEFAULT_COOLDOWN = 0.1
DEFAULT_FPS = 10.0

STREAM_ID = 0
STREAM_SOURCE = "rtsp://localhost/test"
STREAM_NAME = "test_stream"

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.5
DEFAULT_TRACKER = "botsort"


# Fixtures


@pytest.fixture
def sample_config_yaml(tmp_path, mock_weights_file):
    """Write a valid system config YAML and return its path."""
    config_data = {
        "streams": [
            {
                "id": STREAM_ID,
                "source": STREAM_SOURCE,
                "name": STREAM_NAME,
                "timeout": 5.0,
                "roi_polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            }
        ],
        "detection": {
            "path": str(mock_weights_file),
            "imgsz": DEFAULT_IMGSZ,
            "conf": DEFAULT_CONF,
            "tracker": DEFAULT_TRACKER,
            "classes": [0],
        },
        "loitering": {
            "threshold": DEFAULT_THRESHOLD,
            "cooldown_percentage": DEFAULT_COOLDOWN,
            "redis": {"host": REDIS_HOST, "port": REDIS_PORT},
        },
        "alerts": {"interval": 60.0},
        "sample_fps": DEFAULT_FPS,
    }
    config_path = tmp_path / "system_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


# Tests


class TestRedisConfig:
    """Tests for the RedisConfig data model validation."""

    def test_valid_config(self):
        """
        Test that a valid Redis configuration can be instantiated.

        Given: host and port strings
        When: RedisConfig is initialized
        Then: the fields are correctly assigned
        """
        # When: initializing with valid data
        config = RedisConfig(host="redis.local", port=REDIS_PORT_ALT)

        # Then: fields match input
        assert config.host == "redis.local"
        assert config.port == REDIS_PORT_ALT

    def test_default_missing_field_raises(self):
        """
        Test that missing required fields raise a validation error.

        Given: a missing port field
        When: RedisConfig is initialized
        Then: a validation error is raised
        """
        # Then: initializing without a port raises an exception
        with pytest.raises(ValidationError):
            RedisConfig(host=REDIS_HOST)  # port is required


class TestAlertConfig:
    """Tests for AlertConfig model."""

    def test_defaults(self):
        config = AlertConfig()
        assert config.interval == 60.0

    def test_custom_interval(self):
        config = AlertConfig(interval=30.0)
        assert config.interval == 30.0

    def test_zero_interval(self):
        config = AlertConfig(interval=0.0)
        assert config.interval == 0.0

    def test_negative_interval_raises(self):
        with pytest.raises(ValidationError):
            AlertConfig(interval=-1.0)


class TestLoiteringConfig:
    """Tests for the LoiteringConfig data model and its boundary validations."""

    def test_valid_config(self):
        """
        Test that a complete loitering configuration is valid.

        Given: threshold, cooldown percentage, and Redis config
        When: LoiteringConfig is initialized
        Then: the fields are correctly assigned
        """
        # When: initializing with valid nested data
        config = LoiteringConfig(
            threshold=60.0,
            cooldown_percentage=DEFAULT_COOLDOWN,
            redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
        )
        # Then: data is stored correctly
        assert config.threshold == 60.0
        assert config.cooldown_percentage == DEFAULT_COOLDOWN

    def test_cooldown_boundary_values(self):
        """
        Test that the cooldown percentage accepts values at the boundaries of [0, 1].

        Given: boundary values (0.0 and 1.0)
        When: LoiteringConfig is initialized
        Then: the values are accepted as valid
        """
        # When: testing lower boundary 0.0
        config_zero = LoiteringConfig(
            threshold=60.0,
            cooldown_percentage=0.0,
            redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
        )
        assert config_zero.cooldown_percentage == 0.0

        # When: testing upper boundary 1.0
        config_one = LoiteringConfig(
            threshold=60.0,
            cooldown_percentage=1.0,
            redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
        )
        assert config_one.cooldown_percentage == 1.0

    def test_cooldown_over_one_raises(self):
        """
        Test that a cooldown percentage greater than 1.0 raises a validation error.

        Given: an invalid percentage (1.5)
        When: LoiteringConfig is initialized
        Then: a validation error is raised
        """
        # Then: values > 1.0 raise an exception
        with pytest.raises(ValidationError):
            LoiteringConfig(
                threshold=60.0,
                cooldown_percentage=1.5,
                redis=RedisConfig(host=REDIS_HOST, port=REDIS_PORT),
            )


class TestSettings:
    """Tests for the YAML-based settings loader and environment variable overrides."""

    def test_loads_from_yaml(self, sample_config_yaml):
        """
        Test that the Settings class correctly parses a YAML configuration file.

        Given: a valid YAML config file
        When: Settings is initialized with the file path
        Then: the nested configuration objects are correctly populated
        """
        # When: loading from YAML
        settings = Settings(sample_config_yaml)

        # Then: top-level and nested fields are correct
        assert settings.system.sample_fps == DEFAULT_FPS
        assert len(settings.system.streams) == 1
        assert settings.system.loitering.threshold == DEFAULT_THRESHOLD

    def test_env_override_redis_host(self, sample_config_yaml):
        """
        Test that the REDIS_HOST environment variable overrides values in the YAML file.

        Given: a REDIS_HOST environment variable
        When: Settings is initialized
        Then: the loitering config uses the environment variable's value
        """
        # Given: REDIS_HOST is set in environment
        with patch.dict(os.environ, {"REDIS_HOST": "redis-prod.local"}):
            # When: loading settings
            settings = Settings(sample_config_yaml)
            # Then: environment value wins
            assert settings.system.loitering.redis.host == "redis-prod.local"

    def test_env_override_redis_port(self, sample_config_yaml):
        """
        Test that the REDIS_PORT environment variable overrides values in the YAML file.

        Given: a REDIS_PORT environment variable
        When: Settings is initialized
        Then: the loitering config uses the environment variable's integer value
        """
        # Given: REDIS_PORT is set in environment
        with patch.dict(os.environ, {"REDIS_PORT": str(REDIS_PORT_ALT)}):
            # When: loading settings
            settings = Settings(sample_config_yaml)
            # Then: environment value wins and is cast to int
            assert settings.system.loitering.redis.port == REDIS_PORT_ALT

    def test_both_env_overrides(self, sample_config_yaml):
        """
        Test that both host and port can be simultaneously overridden by environment variables.
        """
        with patch.dict(os.environ, {"REDIS_HOST": "custom", "REDIS_PORT": "9999"}):
            settings = Settings(sample_config_yaml)
            assert settings.system.loitering.redis.host == "custom"
            assert settings.system.loitering.redis.port == 9999


class TestGetConfig:
    """Tests for get_config helper."""

    def test_returns_system_config(self, sample_config_yaml):
        config = get_config(sample_config_yaml)
        assert isinstance(config, SystemConfig)
        assert config.sample_fps == DEFAULT_FPS

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            get_config("/nonexistent/config.yml")
