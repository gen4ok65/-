from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path

from .api import create_api_app
from .api_keys import APIKeyStore
from .config import AppSettings, MODEL_PRESETS
from .gradio_app import build_demo
from .service import MusicStudioService


def _cli_log(message: str) -> None:
    print(f"[music-lyrics-ai] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a local text-to-music studio with Gradio UI, pretrained MusicGen models, and developer API keys."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="melody", help="Pretrained MusicGen checkpoint preset")
    shared.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    shared.add_argument("--output-dir", default="artifacts/renders", help="Directory for generated wav/json files")
    shared.add_argument("--api-key-store", default="artifacts/api_keys.json", help="JSON file used to store API key hashes")

    launch = subparsers.add_parser("launch", parents=[shared], help="Run the Gradio application")
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=7860)
    launch.add_argument("--share", action="store_true", help="Enable Gradio share link")

    api = subparsers.add_parser("serve-api", parents=[shared], help="Run the REST API")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    preload = subparsers.add_parser("preload-model", parents=[shared], help="Download/load the selected model before opening UI/API")
    preload.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    preload.add_argument("--port", type=int, default=7860, help=argparse.SUPPRESS)

    doctor = subparsers.add_parser("doctor", parents=[shared], help="Print a startup checklist for the local environment")
    doctor.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    doctor.add_argument("--port", type=int, default=7860, help=argparse.SUPPRESS)

    create_key = subparsers.add_parser("create-key", parents=[shared], help="Generate an API key and print it")
    create_key.add_argument("--label", default="desktop-client")

    return parser


def _settings_from_args(args: argparse.Namespace, api_mode: bool = False) -> AppSettings:
    settings = AppSettings(
        model_preset=args.preset,
        device=args.device,
        output_dir=args.output_dir,
        api_key_store=args.api_key_store,
    )
    if api_mode:
        settings.api_host = args.host
        settings.api_port = args.port
    else:
        settings.host = args.host
        settings.port = args.port
    return settings


def _module_status(module_name: str) -> str:
    return "ok" if importlib.util.find_spec(module_name) is not None else "missing"


def _run_doctor(settings: AppSettings) -> None:
    _cli_log("Environment diagnostics:")
    _cli_log(f"  Python: {platform.python_version()} ({sys.executable})")
    _cli_log(f"  Platform: {platform.platform()}")
    _cli_log(f"  Preset: {settings.model.key} -> {settings.model.repo_id}")
    _cli_log(f"  Output dir: {Path(settings.output_dir).resolve()}")
    _cli_log(f"  API key store: {Path(settings.api_key_store).resolve()}")
    _cli_log("Dependency status:")
    for module_name in ["gradio", "fastapi", "uvicorn", "torch", "transformers", "soundfile"]:
        _cli_log(f"  {module_name}: {_module_status(module_name)}")
    _cli_log("Recommended next steps on Windows:")
    _cli_log("  1) python -m pip install -e .[app,inference,dev]")
    _cli_log(f"  2) python -m music_lyrics_ai.cli preload-model --preset {settings.model.key}")
    _cli_log(f"  3) python -m music_lyrics_ai.cli launch --preset {settings.model.key}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create-key":
        settings = AppSettings(
            model_preset=args.preset,
            device=args.device,
            output_dir=args.output_dir,
            api_key_store=args.api_key_store,
        )
        key_store = APIKeyStore(settings.api_key_store)
        _cli_log(f"Creating API key in {settings.api_key_store} ...")
        print(key_store.create_key(args.label))
        return

    if args.command == "doctor":
        settings = _settings_from_args(args)
        _run_doctor(settings)
        return

    if args.command == "preload-model":
        settings = _settings_from_args(args)
        _cli_log(
            f"Preloading model preset={settings.model.key} ({settings.model.repo_id}). This step is специально for the first run so you can wait for the download before opening the UI."
        )
        _cli_log("If PowerShell looks idle, do not press Ctrl+C. Wait for 'Model ready'.")
        service = MusicStudioService(settings)
        service.warmup()
        _cli_log("Preload finished. Now you can run 'lyrics-ai launch --preset %s'." % settings.model.key)
        return

    if args.command == "launch":
        settings = _settings_from_args(args)
        _cli_log(
            f"Starting Gradio studio on http://{args.host}:{args.port} with preset={settings.model.key} ({settings.model.repo_id})."
        )
        _cli_log(
            "The UI should open quickly, but the first actual generation may download model files for several minutes. If PowerShell seems idle, wait and do not press Ctrl+C."
        )
        _cli_log(f"If you want to download the model first, run: lyrics-ai preload-model --preset {settings.model.key}")
        key_store = APIKeyStore(settings.api_key_store)
        service = MusicStudioService(settings)
        demo = build_demo(service, key_store, settings)
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
        return

    if args.command == "serve-api":
        settings = _settings_from_args(args, api_mode=True)
        _cli_log(
            f"Starting REST API on http://{args.host}:{args.port} with preset={settings.model.key} ({settings.model.repo_id})."
        )
        _cli_log(
            "The first generation request may download model files for several minutes. Keep the terminal open and wait until the request finishes."
        )
        _cli_log(f"If you want to download the model first, run: lyrics-ai preload-model --preset {settings.model.key}")
        key_store = APIKeyStore(settings.api_key_store)
        service = MusicStudioService(settings)
        app = create_api_app(service, key_store, settings)

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
