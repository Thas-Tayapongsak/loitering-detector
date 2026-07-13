"""Infrastructure adapters and implementations."""

from .geometry import OpenCVGeometryEngine
from .persistence.in_memory import InMemoryStateRepository
from .persistence.redis import RedisStateRepository

__all__ = [
    "OpenCVGeometryEngine",
    "InMemoryStateRepository",
    "RedisStateRepository",
]
