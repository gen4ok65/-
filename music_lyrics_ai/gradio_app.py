from __future__ import annotations

import json
from pathlib import Path

from .api_keys import APIKeyStore
from .config import AppSettings, GenerationRequest
from .service import MusicStudioService, build_conditioned_prompt


STYLE_PRESETS = [
    "cinematic",
    "electronic",
    "ambient",
    "rock",
    "pop",
    "trap",
    "lofi",
    "orchestral",
    "synthwave",
    "drill",
]


def build_demo(service: MusicStudioService, key_store: APIKeyStore, settings: AppSettings):
    import gradio as gr

    def generate_music(
        description: str,
        style: str,
        lyrics: str,
        reference_notes: str,
        reference_audio: str | None,
        duration_seconds: int,
        guidance_scale: float,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ):
        request = GenerationRequest(
            description=description,
            style=style,
            lyrics=lyrics,
            reference_notes=reference_notes,
            reference_audio_path=reference_audio,
            duration_seconds=duration_seconds,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        result = service.generate(request)
        payload = {
            "audio_path": result.audio_path,
            "model_id": result.model_id,
            "seed": result.seed,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            **result.metadata,
        }
        return result.audio_path, result.prompt_used, json.dumps(payload, ensure_ascii=False, indent=2)

    def preview_prompt(description: str, style: str, lyrics: str, reference_notes: str) -> str:
        request = GenerationRequest(
            description=description,
            style=style,
            lyrics=lyrics,
            reference_notes=reference_notes,
        )
        return build_conditioned_prompt(request)

    def create_api_key(label: str):
        key = key_store.create_key(label)
        list_view = json.dumps(key_store.list_keys(), ensure_ascii=False, indent=2)
        curl = (
            f"curl -X POST http://{settings.api_host}:{settings.api_port}/v1/generate \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -H 'X-API-Key: {key}' \\\n"
            f"  -d '{{\"description\": \"dark cyberpunk trailer with distorted bass\", \"style\": \"synthwave\"}}'"
        )
        return key, list_view, curl

    existing_keys = json.dumps(key_store.list_keys(), ensure_ascii=False, indent=2)

    with gr.Blocks(title="Music Lyrics AI Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
# Music Lyrics AI Studio

Полноценная локальная студия поверх **{settings.model.label}** (`{settings.model.repo_id}`):
- генерация по текстовому описанию,
- усиление промпта текстом песни,
- генерация по аудио-примеру / мелодии,
- developer-меню с API-ключами для отдельного REST API.
"""
        )

        with gr.Tab("Studio"):
            with gr.Row():
                with gr.Column(scale=3):
                    description = gr.Textbox(
                        label="Описание трека",
                        lines=4,
                        placeholder="Например: epic hybrid trailer with choirs, huge drums and futuristic synths",
                    )
                    style = gr.Dropdown(STYLE_PRESETS, value=settings.model.key if settings.model.key in STYLE_PRESETS else "cinematic", label="Стиль")
                    lyrics = gr.Textbox(
                        label="Текст / вокальная идея",
                        lines=5,
                        placeholder="Необязательно. Например: We rise above the fire tonight...",
                    )
                    reference_notes = gr.Textbox(
                        label="Референсы и комментарии",
                        lines=3,
                        placeholder="Например: intro should be sparse, then big drop at 10 sec, female vocal texture",
                    )
                    reference_audio = gr.Audio(
                        label="Аудио-пример / мелодия (необязательно)",
                        type="filepath",
                    )
                with gr.Column(scale=2):
                    duration_seconds = gr.Slider(4, 25, value=12, step=1, label="Длительность (сек.)")
                    guidance_scale = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Guidance scale")
                    temperature = gr.Slider(0.3, 2.0, value=1.0, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 512, value=250, step=1, label="Top-k")
                    top_p = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Top-p")
                    seed = gr.Number(value=1234, precision=0, label="Seed")
                    preview_button = gr.Button("Показать итоговый промпт", variant="secondary")
                    generate_button = gr.Button("Сгенерировать музыку", variant="primary")

            audio_output = gr.Audio(label="Результат", type="filepath")
            prompt_output = gr.Textbox(label="Промпт, ушедший в модель", lines=7)
            metadata_output = gr.Code(label="Метаданные генерации", language="json")

            preview_button.click(
                preview_prompt,
                inputs=[description, style, lyrics, reference_notes],
                outputs=[prompt_output],
            )
            generate_button.click(
                generate_music,
                inputs=[
                    description,
                    style,
                    lyrics,
                    reference_notes,
                    reference_audio,
                    duration_seconds,
                    guidance_scale,
                    temperature,
                    top_k,
                    top_p,
                    seed,
                ],
                outputs=[audio_output, prompt_output, metadata_output],
            )

        with gr.Tab("Developer"):
            gr.Markdown(
                f"""
### REST API

1. Создайте ключ ниже.
2. Запустите API-команду: `lyrics-ai serve-api --preset {settings.model.key}`.
3. Подключайте сервис откуда угодно по адресу `http://{settings.api_host}:{settings.api_port}/v1/generate`.

Сгенерированные wav/json файлы сохраняются в `{Path(settings.output_dir)}`.
"""
            )
            dev_label = gr.Textbox(label="Название ключа", value="desktop-client")
            create_key_button = gr.Button("Создать API-ключ", variant="primary")
            api_key_output = gr.Textbox(label="Новый API-ключ", lines=2)
            key_list_output = gr.Code(label="Список ключей", value=existing_keys, language="json")
            curl_output = gr.Code(label="Готовый curl пример", language="bash")
            create_key_button.click(create_api_key, inputs=[dev_label], outputs=[api_key_output, key_list_output, curl_output])

    return demo
