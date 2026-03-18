from __future__ import annotations

from pathlib import Path

from music_lyrics_ai.config import GenerationConfig, LyricsModelConfig, TrainingConfig
from music_lyrics_ai.inference import LyricsGenerator
from music_lyrics_ai.training import train_model


def test_training_and_generation(tmp_path: Path) -> None:
    dataset_path = Path("data/sample_lyrics.jsonl")
    output_dir = tmp_path / "artifacts"

    metrics = train_model(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_config=LyricsModelConfig(
            vocab_size=0,
            style_vocab_size=0,
            max_seq_len=48,
            d_model=24,
            n_heads=1,
            n_layers=1,
            dropout=0.0,
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=0.03,
            warmup_steps=1,
        ),
    )

    assert metrics["num_samples"] == 10
    assert metrics["parameters"] > 0
    assert (output_dir / "model.json").exists()

    generator = LyricsGenerator(output_dir)
    text = generator.generate("rock", prompt="Tonight", generation_config=GenerationConfig(max_new_tokens=20))
    assert isinstance(text, str)
    assert len(text) > 0
