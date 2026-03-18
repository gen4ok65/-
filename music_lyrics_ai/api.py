from __future__ import annotations

from pathlib import Path

from .api_keys import APIKeyStore
from .config import AppSettings, GenerationRequest
from .service import MusicStudioService


def create_api_app(service: MusicStudioService, key_store: APIKeyStore, settings: AppSettings):
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="Music Lyrics AI API", version="0.2.0")

    class GeneratePayload(BaseModel):
        description: str = Field(..., min_length=3)
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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model_repo": settings.model.repo_id,
            "output_dir": str(settings.output_dir),
        }

    @app.get("/v1/files/{filename}")
    def download_file(filename: str, x_api_key: str | None = Header(default=None)):
        if not key_store.validate(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        file_path = Path(settings.output_dir) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    @app.post("/v1/generate")
    def generate(payload: GeneratePayload, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
        if not key_store.validate(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        result = service.generate(
            GenerationRequest(
                description=payload.description,
                style=payload.style,
                lyrics=payload.lyrics,
                reference_notes=payload.reference_notes,
                reference_audio_path=payload.reference_audio_path,
                duration_seconds=payload.duration_seconds,
                guidance_scale=payload.guidance_scale,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                seed=payload.seed,
            )
        )
        filename = Path(result.audio_path).name
        return {
            "audio_path": result.audio_path,
            "download_url": f"http://{settings.api_host}:{settings.api_port}/v1/files/{filename}",
            "prompt_used": result.prompt_used,
            "seed": result.seed,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            "model_id": result.model_id,
            "metadata": result.metadata,
        }

    return app
