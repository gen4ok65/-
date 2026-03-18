from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import GenerationConfig, LyricsModelConfig, TrainingConfig
from .inference import LyricsGenerator
from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run a multi-style lyrics generation model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model on a JSONL lyrics dataset")
    train_parser.add_argument("--dataset", required=True, help="Path to JSONL file with {style, text}")
    train_parser.add_argument("--output-dir", required=True, help="Directory for model artifacts")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--max-seq-len", type=int, default=256)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--n-heads", type=int, default=8)
    train_parser.add_argument("--n-layers", type=int, default=6)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--device", default="cpu")

    generate_parser = subparsers.add_parser("generate", help="Generate lyrics from a trained model")
    generate_parser.add_argument("--model-dir", required=True)
    generate_parser.add_argument("--style", required=True)
    generate_parser.add_argument("--prompt", default="")
    generate_parser.add_argument("--max-new-tokens", type=int, default=120)
    generate_parser.add_argument("--temperature", type=float, default=0.9)
    generate_parser.add_argument("--top-k", type=int, default=40)
    generate_parser.add_argument("--top-p", type=float, default=0.92)
    generate_parser.add_argument("--device", default="cpu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        model_config = LyricsModelConfig(
            vocab_size=0,
            style_vocab_size=0,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        metrics = train_model(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            model_config=model_config,
            training_config=training_config,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "generate":
        generator = LyricsGenerator(args.model_dir, device=args.device)
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(generator.generate(style=args.style, prompt=args.prompt, generation_config=generation_config))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
