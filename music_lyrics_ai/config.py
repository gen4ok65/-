from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelPreset:
    key: str
    repo_id: str
    label: str
    description: str
    supports_audio_prompt: bool = True


MODEL_PRESETS: dict[str, ModelPreset] = {
    "small": ModelPreset(
        key="small",
        repo_id="facebook/musicgen-small",
        label="MusicGen Small",
        description="Самый лёгкий пресет для быстрого локального запуска на ноутбуке.",
        supports_audio_prompt=True,
    ),
    "medium": ModelPreset(
        key="medium",
        repo_id="facebook/musicgen-medium",
        label="MusicGen Medium",
        description="Качественнее small, но требует заметно больше видеопамяти и времени.",
        supports_audio_prompt=True,
    ),
    "melody": ModelPreset(
        key="melody",
        repo_id="facebook/musicgen-melody",
        label="MusicGen Melody",
        description="Лучший базовый выбор для генерации по описанию и по аудио-примеру/мелодии.",
        supports_audio_prompt=True,
    ),
    "large": ModelPreset(
        key="large",
        repo_id="facebook/musicgen-large",
        label="MusicGen Large",
        description="Самый тяжёлый пресет из набора. Нужен мощный GPU или много терпения на CPU.",
        supports_audio_prompt=True,
    ),
}


@dataclass(slots=True)
class GenerationRequest:
    description: str
    style: str = "cinematic"
    lyrics: str = ""
    reference_notes: str = ""
    reference_audio_path: str | None = None
    duration_seconds: int = 12
    guidance_scale: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    seed: int = 1234


@dataclass(slots=True)
class GenerationResult:
    audio_path: str
    prompt_used: str
    seed: int
    duration_seconds: int
    model_id: str
    sample_rate: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(slots=True)
class AppSettings:
    model_preset: str = "melody"
    device: str = "auto"
    host: str = "127.0.0.1"
    port: int = 7860
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    output_dir: Path = Path("artifacts/renders")
    api_key_store: Path = Path("artifacts/api_keys.json")

    @property
    def model(self) -> ModelPreset:
        if self.model_preset not in MODEL_PRESETS:
            supported = ", ".join(sorted(MODEL_PRESETS))
            raise ValueError(f"Unknown model preset '{self.model_preset}'. Supported presets: {supported}")
        return MODEL_PRESETS[self.model_preset]
