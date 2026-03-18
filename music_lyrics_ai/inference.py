from __future__ import annotations

from pathlib import Path

from .config import GenerationConfig
from .data import StyleVocabulary
from .model import StyleConditionedTransformer
from .tokenizer import CharTokenizer


class LyricsGenerator:
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        del device
        model_dir = Path(model_dir)
        self.tokenizer = CharTokenizer.load(model_dir / "tokenizer.json")
        self.style_vocab = StyleVocabulary.load(model_dir / "styles.json")
        self.model = StyleConditionedTransformer.load(model_dir / "model.json")

    def generate(self, style: str, prompt: str = "", generation_config: GenerationConfig | None = None) -> str:
        if style not in self.style_vocab.stoi:
            raise ValueError(f"Unknown style '{style}'. Available: {', '.join(self.style_vocab.itos)}")
        generation_config = generation_config or GenerationConfig()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        generated = self.model.generate(
            input_ids=prompt_ids,
            style_id=self.style_vocab.encode(style),
            generation_config=generation_config,
            eos_token_id=self.tokenizer.eos_id,
        )
        return self.tokenizer.decode(generated)
