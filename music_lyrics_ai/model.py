from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import GenerationConfig, LyricsModelConfig


@dataclass(slots=True)
class RNNState:
    hidden: list[float]


class StyleConditionedTransformer:
    """A compact style-conditioned recurrent language model.

    The class name is preserved for API compatibility, but the implementation is a
    self-contained character-level neural network that does not require external ML libraries.
    """

    def __init__(self, config: LyricsModelConfig) -> None:
        self.config = config
        self.hidden_size = config.d_model
        scale = 0.08
        self.token_embed = self._rand_matrix(config.vocab_size, config.d_model, scale)
        self.style_embed = self._rand_matrix(config.style_vocab_size, config.d_model, scale)
        self.w_xh = self._rand_matrix(config.d_model, self.hidden_size, scale)
        self.w_hh = self._rand_matrix(self.hidden_size, self.hidden_size, scale)
        self.w_sh = self._rand_matrix(config.d_model, self.hidden_size, scale)
        self.b_h = [0.0 for _ in range(self.hidden_size)]
        self.w_hy = self._rand_matrix(self.hidden_size, config.vocab_size, scale)
        self.b_y = [0.0 for _ in range(config.vocab_size)]

    @staticmethod
    def _rand_matrix(rows: int, cols: int, scale: float) -> list[list[float]]:
        return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _zeros_matrix(rows: int, cols: int) -> list[list[float]]:
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _zeros_vector(size: int) -> list[float]:
        return [0.0 for _ in range(size)]

    @staticmethod
    def _matvec(vector: list[float], matrix: list[list[float]]) -> list[float]:
        cols = len(matrix[0])
        output = [0.0 for _ in range(cols)]
        for i, value in enumerate(vector):
            row = matrix[i]
            for j in range(cols):
                output[j] += value * row[j]
        return output

    @staticmethod
    def _vec_add(*vectors: list[float]) -> list[float]:
        total = [0.0 for _ in range(len(vectors[0]))]
        for vector in vectors:
            for i, value in enumerate(vector):
                total[i] += value
        return total

    @staticmethod
    def _tanh(vector: list[float]) -> list[float]:
        return [math.tanh(value) for value in vector]

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        max_logit = max(logits)
        exps = [math.exp(value - max_logit) for value in logits]
        total = sum(exps)
        return [value / total for value in exps]

    @staticmethod
    def _sample_from_probs(probs: list[float]) -> int:
        threshold = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if cumulative >= threshold:
                return idx
        return len(probs) - 1

    def init_state(self) -> RNNState:
        return RNNState(hidden=[0.0 for _ in range(self.hidden_size)])

    def forward_step(self, token_id: int, style_id: int, prev_hidden: list[float]) -> tuple[list[float], list[float], dict[str, list[float]]]:
        x = self.token_embed[token_id]
        s = self.style_embed[style_id]
        preact = self._vec_add(
            self._matvec(x, self.w_xh),
            self._matvec(prev_hidden, self.w_hh),
            self._matvec(s, self.w_sh),
            self.b_h,
        )
        hidden = self._tanh(preact)
        logits = self._vec_add(self._matvec(hidden, self.w_hy), self.b_y)
        cache = {"x": x, "s": s, "prev_hidden": prev_hidden, "preact": preact, "hidden": hidden}
        return hidden, logits, cache

    def train_sequence(self, input_ids: list[int], labels: list[int], style_id: int, learning_rate: float) -> float:
        caches: list[dict[str, list[float]]] = []
        logits_steps: list[list[float]] = []
        hidden = self.init_state().hidden
        loss = 0.0

        for token_id, label in zip(input_ids, labels):
            hidden, logits, cache = self.forward_step(token_id, style_id, hidden)
            caches.append(cache)
            logits_steps.append(logits)
            probs = self._softmax(logits)
            loss -= math.log(max(probs[label], 1e-12))

        d_token_embed = self._zeros_matrix(self.config.vocab_size, self.config.d_model)
        d_style_embed = self._zeros_matrix(self.config.style_vocab_size, self.config.d_model)
        d_w_xh = self._zeros_matrix(self.config.d_model, self.hidden_size)
        d_w_hh = self._zeros_matrix(self.hidden_size, self.hidden_size)
        d_w_sh = self._zeros_matrix(self.config.d_model, self.hidden_size)
        d_b_h = self._zeros_vector(self.hidden_size)
        d_w_hy = self._zeros_matrix(self.hidden_size, self.config.vocab_size)
        d_b_y = self._zeros_vector(self.config.vocab_size)
        d_hidden_next = self._zeros_vector(self.hidden_size)

        for step in reversed(range(len(input_ids))):
            cache = caches[step]
            probs = self._softmax(logits_steps[step])
            probs[labels[step]] -= 1.0

            for i in range(self.hidden_size):
                hidden_value = cache["hidden"][i]
                for j in range(self.config.vocab_size):
                    d_w_hy[i][j] += hidden_value * probs[j]
            for j in range(self.config.vocab_size):
                d_b_y[j] += probs[j]

            d_hidden = d_hidden_next[:]
            for i in range(self.hidden_size):
                row = self.w_hy[i]
                d_hidden[i] += sum(row[j] * probs[j] for j in range(self.config.vocab_size))

            d_preact = [d_hidden[i] * (1.0 - cache["hidden"][i] ** 2) for i in range(self.hidden_size)]

            for i in range(self.config.d_model):
                for j in range(self.hidden_size):
                    d_w_xh[i][j] += cache["x"][i] * d_preact[j]
                    d_w_sh[i][j] += cache["s"][i] * d_preact[j]
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    d_w_hh[i][j] += cache["prev_hidden"][i] * d_preact[j]
            for j in range(self.hidden_size):
                d_b_h[j] += d_preact[j]

            d_x = [sum(self.w_xh[i][j] * d_preact[j] for j in range(self.hidden_size)) for i in range(self.config.d_model)]
            d_s = [sum(self.w_sh[i][j] * d_preact[j] for j in range(self.hidden_size)) for i in range(self.config.d_model)]
            d_hidden_next = [sum(self.w_hh[i][j] * d_preact[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]

            token_id = input_ids[step]
            for i in range(self.config.d_model):
                d_token_embed[token_id][i] += d_x[i]
                d_style_embed[style_id][i] += d_s[i]

        self._apply_update(self.token_embed, d_token_embed, learning_rate)
        self._apply_update(self.style_embed, d_style_embed, learning_rate)
        self._apply_update(self.w_xh, d_w_xh, learning_rate)
        self._apply_update(self.w_hh, d_w_hh, learning_rate)
        self._apply_update(self.w_sh, d_w_sh, learning_rate)
        self._apply_update(self.w_hy, d_w_hy, learning_rate)
        self._apply_vector_update(self.b_h, d_b_h, learning_rate)
        self._apply_vector_update(self.b_y, d_b_y, learning_rate)

        return loss / max(len(input_ids), 1)

    @staticmethod
    def _apply_update(weights: list[list[float]], grads: list[list[float]], learning_rate: float) -> None:
        for i, row in enumerate(weights):
            for j in range(len(row)):
                row[j] -= learning_rate * grads[i][j]

    @staticmethod
    def _apply_vector_update(weights: list[float], grads: list[float], learning_rate: float) -> None:
        for i in range(len(weights)):
            weights[i] -= learning_rate * grads[i]

    def generate(self, input_ids: list[int], style_id: int, generation_config: GenerationConfig, eos_token_id: int) -> list[int]:
        generated = list(input_ids)
        hidden = self.init_state().hidden
        for token_id in input_ids:
            hidden, _, _ = self.forward_step(token_id, style_id, hidden)

        current_token = input_ids[-1]
        for _ in range(generation_config.max_new_tokens):
            hidden, logits, _ = self.forward_step(current_token, style_id, hidden)
            adjusted = [value / max(generation_config.temperature, 1e-5) for value in logits]
            filtered = top_k_top_p_filtering(adjusted, generation_config.top_k, generation_config.top_p)
            probs = self._softmax(filtered)
            current_token = self._sample_from_probs(probs)
            generated.append(current_token)
            if current_token == eos_token_id:
                break
        return generated

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "config": asdict(self.config),
            "weights": {
                "token_embed": self.token_embed,
                "style_embed": self.style_embed,
                "w_xh": self.w_xh,
                "w_hh": self.w_hh,
                "w_sh": self.w_sh,
                "b_h": self.b_h,
                "w_hy": self.w_hy,
                "b_y": self.b_y,
            },
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StyleConditionedTransformer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(LyricsModelConfig(**payload["config"]))
        weights = payload["weights"]
        model.token_embed = weights["token_embed"]
        model.style_embed = weights["style_embed"]
        model.w_xh = weights["w_xh"]
        model.w_hh = weights["w_hh"]
        model.w_sh = weights["w_sh"]
        model.b_h = weights["b_h"]
        model.w_hy = weights["w_hy"]
        model.b_y = weights["b_y"]
        return model


def top_k_top_p_filtering(logits: list[float], top_k: int, top_p: float) -> list[float]:
    filtered = logits[:]
    if top_k > 0 and top_k < len(filtered):
        threshold = sorted(filtered, reverse=True)[top_k - 1]
        filtered = [value if value >= threshold else -1e9 for value in filtered]

    if 0.0 < top_p < 1.0:
        indexed = sorted(enumerate(filtered), key=lambda item: item[1], reverse=True)
        stable_logits = [item[1] for item in indexed]
        max_logit = max(stable_logits)
        exp_values = [math.exp(value - max_logit) if value > -1e8 else 0.0 for value in stable_logits]
        total = sum(exp_values) or 1.0
        cumulative = 0.0
        keep = set()
        for (idx, _), exp_value in zip(indexed, exp_values):
            prob = exp_value / total
            cumulative += prob
            keep.add(idx)
            if cumulative >= top_p:
                break
        filtered = [value if i in keep else -1e9 for i, value in enumerate(filtered)]
    return filtered


def count_parameters(model: StyleConditionedTransformer) -> int:
    matrices = [
        model.token_embed,
        model.style_embed,
        model.w_xh,
        model.w_hh,
        model.w_sh,
        model.w_hy,
    ]
    vectors = [model.b_h, model.b_y]
    return sum(len(row) for matrix in matrices for row in matrix) + sum(len(vector) for vector in vectors)
