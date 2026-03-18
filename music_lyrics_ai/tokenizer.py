from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass(slots=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        alphabet = sorted({char for text in texts for char in text})
        itos = SPECIAL_TOKENS + alphabet
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        tokens = [self.stoi.get(char, self.unk_id) for char in text]
        if add_special_tokens:
            return [self.bos_id, *tokens, self.eos_id]
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        specials = {self.pad_token, self.bos_token, self.eos_token}
        chars: list[str] = []
        for token_id in token_ids:
            if not 0 <= token_id < len(self.itos):
                continue
            token = self.itos[token_id]
            if skip_special_tokens and token in specials:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        data = {"itos": self.itos}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = data["itos"]
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
