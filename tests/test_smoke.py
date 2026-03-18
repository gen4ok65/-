from __future__ import annotations

from pathlib import Path

from music_lyrics_ai.api_keys import APIKeyStore
from music_lyrics_ai.config import AppSettings, GenerationRequest
from music_lyrics_ai.service import MusicStudioService, build_conditioned_prompt


class FakeBackend:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.last_request: GenerationRequest | None = None

    def generate(self, request: GenerationRequest):
        self.last_request = request
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.output_dir / "fake.wav"
        audio_path.write_bytes(b"RIFFfake")
        return type(
            "FakeResult",
            (),
            {
                "audio_path": str(audio_path),
                "prompt_used": build_conditioned_prompt(request),
                "seed": request.seed,
                "duration_seconds": request.duration_seconds,
                "model_id": "fake/musicgen",
                "sample_rate": 32000,
                "metadata": {"preset": "fake"},
            },
        )()


def test_api_key_store_create_and_validate(tmp_path: Path) -> None:
    store = APIKeyStore(tmp_path / "api_keys.json")
    key = store.create_key("desktop-client")

    assert key.startswith("mla_")
    assert store.validate(key) is True
    assert store.validate("invalid") is False
    assert store.list_keys()[0]["label"] == "desktop-client"


def test_service_normalizes_request_and_uses_backend(tmp_path: Path) -> None:
    settings = AppSettings(output_dir=tmp_path / "renders", api_key_store=tmp_path / "keys.json")
    backend = FakeBackend(settings.output_dir)
    service = MusicStudioService(settings, backend=backend)

    result = service.generate(
        GenerationRequest(
            description="  huge cyberpunk battle cue  ",
            style=" synthwave ",
            lyrics="  We run through neon fire  ",
            reference_notes="  big drop at 10 sec  ",
            duration_seconds=99,
            guidance_scale=0.5,
            temperature=0.01,
            seed=77,
        )
    )

    assert backend.last_request is not None
    assert backend.last_request.description == "huge cyberpunk battle cue"
    assert backend.last_request.style == "synthwave"
    assert backend.last_request.duration_seconds == 25
    assert backend.last_request.guidance_scale == 1.0
    assert backend.last_request.temperature == 0.1
    assert Path(result.audio_path).exists()
    assert "Lyrics or vocal idea" in result.prompt_used
    assert "Reference notes" in result.prompt_used
