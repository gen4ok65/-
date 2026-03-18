from __future__ import annotations

import argparse

from .api import create_api_app
from .api_keys import APIKeyStore
from .config import AppSettings, MODEL_PRESETS
from .gradio_app import build_demo
from .service import MusicStudioService


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
        print(key_store.create_key(args.label))
        return

    if args.command == "launch":
        settings = _settings_from_args(args)
        key_store = APIKeyStore(settings.api_key_store)
        service = MusicStudioService(settings)
        demo = build_demo(service, key_store, settings)
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
        return

    if args.command == "serve-api":
        settings = _settings_from_args(args, api_mode=True)
        key_store = APIKeyStore(settings.api_key_store)
        service = MusicStudioService(settings)
        app = create_api_app(service, key_store, settings)

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
