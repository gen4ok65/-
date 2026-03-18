"""Music lyrics generation package."""

from .config import LyricsModelConfig, TrainingConfig, GenerationConfig
from .model import StyleConditionedTransformer

__all__ = [
    "LyricsModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    "StyleConditionedTransformer",
]
