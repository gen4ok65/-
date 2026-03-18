from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .tokenizer import CharTokenizer


@dataclass(slots=True)
class LyricsSample:
    style: str
    text: str


class StyleVocabulary:
    def __init__(self, styles: Iterable[str]):
        unique_styles = sorted(set(styles))
        self.itos = unique_styles
        self.stoi = {style: idx for idx, style in enumerate(unique_styles)}

    def encode(self, style: str) -> int:
        return self.stoi[style]

    def decode(self, style_id: int) -> str:
        return self.itos[style_id]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"styles": self.itos}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StyleVocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["styles"])

    def __len__(self) -> int:
        return len(self.itos)


def load_lyrics_dataset(path: str | Path) -> list[LyricsSample]:
    samples: list[LyricsSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        samples.append(LyricsSample(style=record["style"], text=record["text"].strip()))
    if not samples:
        raise ValueError(f"Dataset at {path} is empty")
    return samples


def build_sequences(
    samples: list[LyricsSample],
    tokenizer: CharTokenizer,
    style_vocab: StyleVocabulary,
    max_seq_len: int,
) -> list[dict[str, list[int] | int]]:
    sequences: list[dict[str, list[int] | int]] = []
    for sample in samples:
        encoded = tokenizer.encode(sample.text)
        trimmed = encoded[: max_seq_len + 1]
        if len(trimmed) < 2:
            trimmed = [tokenizer.bos_id, tokenizer.eos_id]
        sequences.append(
            {
                "input_ids": trimmed[:-1],
                "labels": trimmed[1:],
                "style_id": style_vocab.encode(sample.style),
            }
        )
    return sequences
