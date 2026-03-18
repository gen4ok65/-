from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import AppSettings, GenerationRequest, GenerationResult


def _log(message: str) -> None:
    print(f"[music-lyrics-ai] {message}", file=sys.stderr, flush=True)


class MusicBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    def warmup(self) -> str | None:
        ...


def build_conditioned_prompt(request: GenerationRequest) -> str:
    chunks = [
        f"Style: {request.style.strip() or 'cinematic'}.",
        f"Music brief: {(request.description or '').strip() or 'instrumental modern soundtrack'}.",
    ]
    if request.lyrics.strip():
        chunks.append(f"Lyrics or vocal idea: {request.lyrics.strip()}.")
    if request.reference_notes.strip():
        chunks.append(f"Reference notes: {request.reference_notes.strip()}.")
    chunks.append(
        "Make it sound polished, dynamic, emotionally clear, and ready for a modern production demo."
    )
    return " ".join(chunks)


class HuggingFaceMusicBackend:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._processor = None
        self._model = None
        self._torch = None

    def _lazy_load(self) -> tuple[object, object, object]:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return self._processor, self._model, self._torch

        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        repo_id = self.settings.model.repo_id
        device = self._resolve_device(torch)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        _log(
            f"Loading pretrained model {repo_id} on device={device}. If this is the first run, files may download for several minutes. Do not close the terminal or press Ctrl+C."
        )
        _log("Downloading/loading processor...")
        processor = AutoProcessor.from_pretrained(repo_id)
        _log("Downloading/loading model weights...")
        model = MusicgenForConditionalGeneration.from_pretrained(repo_id, torch_dtype=dtype)
        model.to(device)
        model.generation_config.do_sample = True
        _log(f"Model ready: {repo_id} on {device}.")

        self._processor = processor
        self._model = model
        self._torch = torch
        return processor, model, torch

    def _resolve_device(self, torch_module: object) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        if torch_module.cuda.is_available():
            return "cuda"
        if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _duration_to_tokens(seconds: int) -> int:
        bounded_seconds = max(1, min(seconds, 25))
        return max(32, int(bounded_seconds * 50))

    @staticmethod
    def _safe_slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
        return slug.strip("-") or "track"

    @staticmethod
    def _prepare_audio_prompt(path: str) -> tuple[object, int]:
        import soundfile as sf

        audio_array, sample_rate = sf.read(path, dtype="float32")
        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array, int(sample_rate)

    @staticmethod
    def _move_inputs_to_device(inputs: object, device: str) -> object:
        moved: dict[str, object] = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved

    def warmup(self) -> str:
        self._lazy_load()
        return self.settings.model.repo_id

    def generate(self, request: GenerationRequest) -> GenerationResult:
        processor, model, torch = self._lazy_load()
        prompt = build_conditioned_prompt(request)
        device = self._resolve_device(torch)
        _log(
            f"Starting generation: style={request.style}, duration={request.duration_seconds}s, seed={request.seed}, device={device}"
        )

        if request.reference_audio_path:
            audio_array, sample_rate = self._prepare_audio_prompt(request.reference_audio_path)
            inputs = processor(
                audio=audio_array,
                sampling_rate=sample_rate,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")

        prepared_inputs = self._move_inputs_to_device(inputs, device)
        generator_device = "cpu" if device == "mps" else device
        generator = torch.Generator(device=generator_device).manual_seed(int(request.seed))

        _log("Running model.generate(...). This can be slow on CPU.")
        with torch.inference_mode():
            audio_values = model.generate(
                **prepared_inputs,
                do_sample=True,
                guidance_scale=float(request.guidance_scale),
                temperature=float(request.temperature),
                top_k=int(request.top_k),
                top_p=float(request.top_p),
                max_new_tokens=self._duration_to_tokens(request.duration_seconds),
                generator=generator,
            )

        waveform = audio_values[0].detach().float().cpu().numpy()
        if getattr(waveform, "ndim", 1) == 2 and waveform.shape[0] == 1:
            waveform = waveform[0]
        elif getattr(waveform, "ndim", 1) == 2:
            waveform = waveform.T

        import numpy as np
        from scipy.io import wavfile

        waveform = np.clip(waveform, -1.0, 1.0)
        int_waveform = (waveform * 32767).astype(np.int16)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = Path(self.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp}-{self._safe_slug(request.style)}-{request.seed}.wav"
        output_path = output_dir / filename
        sample_rate = int(model.config.audio_encoder.sampling_rate)
        wavfile.write(output_path, sample_rate, int_waveform)
        _log(f"Saved audio to {output_path}")

        metadata = asdict(request)
        metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
        metadata["resolved_model_repo"] = self.settings.model.repo_id
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return GenerationResult(
            audio_path=str(output_path),
            prompt_used=prompt,
            seed=int(request.seed),
            duration_seconds=int(request.duration_seconds),
            model_id=self.settings.model.repo_id,
            sample_rate=sample_rate,
            metadata={"metadata_path": str(metadata_path), "preset": self.settings.model.key},
        )


class MusicStudioService:
    def __init__(self, settings: AppSettings, backend: MusicBackend | None = None) -> None:
        self.settings = settings
        self.backend = backend or HuggingFaceMusicBackend(settings)

    def warmup(self) -> str | None:
        if hasattr(self.backend, "warmup"):
            return self.backend.warmup()
        return None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        cleaned_request = GenerationRequest(
            description=request.description.strip(),
            style=request.style.strip() or "cinematic",
            lyrics=request.lyrics.strip(),
            reference_notes=request.reference_notes.strip(),
            reference_audio_path=request.reference_audio_path,
            duration_seconds=max(4, min(int(request.duration_seconds), 25)),
            guidance_scale=max(1.0, float(request.guidance_scale)),
            temperature=max(0.1, float(request.temperature)),
            top_k=max(0, int(request.top_k)),
            top_p=max(0.0, min(float(request.top_p), 1.0)),
            seed=int(request.seed),
        )
        return self.backend.generate(cleaned_request)
