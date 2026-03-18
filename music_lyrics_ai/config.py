from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LyricsModelConfig:
    vocab_size: int
    style_vocab_size: int
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    ff_mult: int = 4


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_steps: int = 100
    seed: int = 42
    device: str = "cpu"


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 120
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.92
