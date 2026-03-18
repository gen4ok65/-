from __future__ import annotations

import json
import random
from pathlib import Path

from .config import LyricsModelConfig, TrainingConfig
from .data import StyleVocabulary, build_sequences, load_lyrics_dataset
from .model import StyleConditionedTransformer, count_parameters
from .tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    model_config: LyricsModelConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict[str, float | int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_config = training_config or TrainingConfig()
    set_seed(training_config.seed)

    samples = load_lyrics_dataset(dataset_path)
    tokenizer = CharTokenizer.build(sample.text for sample in samples)
    style_vocab = StyleVocabulary(sample.style for sample in samples)

    if model_config is None:
        model_config = LyricsModelConfig(
            vocab_size=tokenizer.vocab_size,
            style_vocab_size=len(style_vocab),
        )
    else:
        model_config.vocab_size = tokenizer.vocab_size
        model_config.style_vocab_size = len(style_vocab)

    sequences = build_sequences(samples, tokenizer, style_vocab, model_config.max_seq_len)
    model = StyleConditionedTransformer(model_config)

    final_loss = 0.0
    step_count = 0
    for epoch in range(training_config.epochs):
        random.shuffle(sequences)
        epoch_loss = 0.0
        for sequence in sequences:
            loss = model.train_sequence(
                input_ids=sequence["input_ids"],
                labels=sequence["labels"],
                style_id=sequence["style_id"],
                learning_rate=training_config.learning_rate,
            )
            epoch_loss += loss
            step_count += 1
        final_loss = epoch_loss / max(len(sequences), 1)
        print(f"epoch={epoch + 1} loss={final_loss:.4f}")

    model.save(output_path / "model.json")
    tokenizer.save(output_path / "tokenizer.json")
    style_vocab.save(output_path / "styles.json")
    metrics = {
        "final_loss": round(final_loss, 4),
        "steps": step_count,
        "parameters": count_parameters(model),
        "num_samples": len(samples),
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
