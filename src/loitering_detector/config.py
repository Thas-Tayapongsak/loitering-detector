from pathlib import Path
import yaml  # type: ignore
import os
from pydantic import BaseModel, Field

from loitering_detector.stream.config import StreamConfig
from loitering_detector.detection.config import DetectionConfig


class RedisConfig(BaseModel):
    host: str
    port: int


class AlertConfig(BaseModel):
    """
    Configuration for loitering alerts.

    Attributes
    ----------
    interval : float
        The interval in seconds between repeating "still loitering" logs.
    """

    interval: float = Field(60.0, ge=0.0)


class LoiteringConfig(BaseModel):
    """
    Configuration for the loitering engine.

    Attributes
    ----------
    threshold : float
        The time threshold (in seconds) after which an object is considered loitering.
    cooldown_percentage : float
        The percentage of the threshold to use as a cooldown period [0.0, 1.0].
    redis : RedisConfig
        The configuration for the Redis persistence layer.
    """

    threshold: float = Field(60.0, ge=0.0)
    cooldown_percentage: float = Field(0.1, ge=0.0, le=1.0)
    redis: RedisConfig


class SystemConfig(BaseModel):
    streams: list[StreamConfig]
    detection: DetectionConfig
    loitering: LoiteringConfig
    alerts: AlertConfig
    sample_fps: float


class Settings:
    """Centralized config management."""

    def __init__(self, path: str | Path):
        self._raw = self._read_yaml(path)

        # Override with Environment Variables for Docker
        if "REDIS_HOST" in os.environ:
            self._raw["loitering"]["redis"]["host"] = os.environ["REDIS_HOST"]
        if "REDIS_PORT" in os.environ:
            self._raw["loitering"]["redis"]["port"] = int(os.environ["REDIS_PORT"])

        self.system = SystemConfig(**self._raw)

    def _read_yaml(self, path: str | Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)


def get_config(path: str | Path = "system_config.yml") -> SystemConfig:
    return Settings(path).system
