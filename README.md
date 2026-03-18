# Music Lyrics AI

Полноценный стартовый проект нейросети для генерации текстов песен в разных стилях: rock, pop, rap, jazz, folk и любых других, которые вы добавите в датасет.

## Что внутри

- **Style-conditioned neural network** — самодостаточная символьная рекуррентная сеть на чистом Python.
- **Условная генерация по стилю** — отдельный embedding музыкального стиля влияет на скрытое состояние в каждом шаге.
- **Полный pipeline** — загрузка датасета, токенизация, обучение, сохранение артефактов, инференс и CLI.
- **Сэмпловый датасет** — `data/sample_lyrics.jsonl` для быстрого запуска и smoke-test.
- **Тесты** — проверка, что модель обучается и генерирует текст без внешних ML-библиотек.

## Формат датасета

Нужен JSONL-файл, где каждая строка выглядит так:

```json
{"style": "rock", "text": "Your lyric text here"}
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Обучение

```bash
lyrics-ai train \
  --dataset data/sample_lyrics.jsonl \
  --output-dir artifacts/demo \
  --epochs 30 \
  --learning-rate 0.03 \
  --max-seq-len 128 \
  --d-model 64
```

После обучения в `artifacts/demo` будут:

- `model.json` — веса и конфигурация сети.
- `tokenizer.json` — словарь символов.
- `styles.json` — словарь стилей.
- `metrics.json` — итоговые метрики обучения.

## Генерация

```bash
lyrics-ai generate \
  --model-dir artifacts/demo \
  --style rap \
  --prompt "Midnight in the city" \
  --max-new-tokens 120 \
  --temperature 0.95 \
  --top-k 20 \
  --top-p 0.9
```

## Архитектура

1. Символьный токенизатор строится по корпусу текстов.
2. Каждый пример преобразуется в `input_ids`, `labels` и `style_id`.
3. Нейросеть на каждом шаге использует:
   - embedding текущего символа,
   - embedding выбранного музыкального стиля,
   - предыдущее скрытое состояние,
   - нелинейность `tanh`,
   - выходной softmax по словарю символов.
4. Обучение идёт через backpropagation through time.
5. Генерация выполняется autoregressive-циклом с `temperature`, `top-k`, `top-p`.

## Как улучшить качество

Чтобы проект стал production-grade, добавьте:

- большой корпус текстов песен по стилям;
- субсловную токенизацию;
- мини-батчи и векторизацию через NumPy/PyTorch;
- GPU-обучение;
- валидационный сплит и early stopping;
- фильтрацию токсичного контента;
- перенос обучения на предобученную языковую модель.

## Структура проекта

```text
music_lyrics_ai/
  __init__.py
  cli.py
  config.py
  data.py
  inference.py
  model.py
  tokenizer.py
  training.py
data/
  sample_lyrics.jsonl
tests/
  test_smoke.py
```
