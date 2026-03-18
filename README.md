# Music Lyrics AI

Полноценный стартовый проект нейросети для генерации текстов песен в разных стилях: rock, pop, rap, jazz, folk и любых других, которые вы добавите в датасет.

> Важно: текущая версия проекта — это **учебная локальная нейросеть на чистом Python**, которую можно запустить даже без PyTorch. Она подходит для экспериментов, понимания пайплайна и генерации простых текстов, но это не промышленная большая языковая модель уровня ChatGPT/Suno.

---

## Что внутри

- **Style-conditioned neural network** — самодостаточная символьная рекуррентная сеть на чистом Python.
- **Условная генерация по стилю** — отдельный embedding музыкального стиля влияет на скрытое состояние в каждом шаге.
- **Полный pipeline** — загрузка датасета, токенизация, обучение, сохранение артефактов, инференс и CLI.
- **Сэмпловый датасет** — `data/sample_lyrics.jsonl` для быстрого запуска и smoke-test.
- **Тесты** — проверка, что модель обучается и генерирует текст без внешних ML-библиотек.

---

## Что нужно для запуска на ноутбуке

Минимально нужно:

1. **Ноутбук с Windows / macOS / Linux**.
2. **Python 3.10 или новее**.
3. **Интернет только для первого скачивания Python и VS Code**.
4. **Архив проекта** или папка с этим репозиторием.
5. Желательно **VS Code** — так будет проще понять, куда нажимать.

Если Python ещё не установлен:

- **Windows**: зайдите на https://www.python.org/downloads/ , скачайте установщик, при установке **обязательно поставьте галочку `Add Python to PATH`**, потом нажмите `Install Now`.
- **macOS**: можно установить Python с python.org или через Homebrew.
- **Linux**: обычно Python уже есть; если нет — установите через менеджер пакетов.

Проверка, что Python установлен:

```bash
python --version
```

или, если не сработало:

```bash
python3 --version
```

---

## Самый простой способ запуска через VS Code

Ниже инструкция буквально "куда нажимать".

### Шаг 1. Скачайте и откройте проект

1. Скачайте папку проекта к себе на ноутбук.
2. Откройте **Visual Studio Code**.
3. В верхнем меню нажмите:
   - **File** → **Open Folder...**
4. Выберите папку проекта `music-lyrics-ai`.
5. Нажмите **Open / Открыть**.

После этого слева появится дерево файлов:

- `README.md`
- папка `music_lyrics_ai`
- папка `data`
- папка `tests`
- файл `pyproject.toml`

### Шаг 2. Откройте встроенный терминал

В VS Code нажмите:

- **Terminal** → **New Terminal**

Откроется нижняя панель с командной строкой.
Именно туда нужно вставлять команды ниже.

### Шаг 3. Создайте виртуальное окружение

#### Windows

Вставьте в терминал:

```powershell
python -m venv .venv
```

#### macOS / Linux

```bash
python3 -m venv .venv
```

После этого в папке проекта появится новая папка `.venv`.

### Шаг 4. Активируйте виртуальное окружение

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Если PowerShell ругается на права, откройте PowerShell **от имени пользователя** и выполните один раз:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Потом снова:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

Когда окружение активировано, слева в терминале обычно появляется префикс вроде:

```text
(.venv)
```

### Шаг 5. Установите проект

Если вы на Windows и команда `python` работает:

```bash
pip install -e .[dev]
```

Если вы на macOS/Linux и используете `python3`, то безопаснее так:

```bash
python -m pip install -e .[dev]
```

или:

```bash
python3 -m pip install -e .[dev]
```

### Шаг 6. Проверьте, что всё установилось

Запустите:

```bash
python -m pytest -q
```

Если всё хорошо, вы увидите что-то вроде:

```text
1 passed
```

---

## Быстрый запуск: обучить нейросеть и получить первый текст

Это самый короткий сценарий, чтобы убедиться, что проект работает.

### Шаг 1. Запустите обучение

В терминале, находясь в корне проекта, выполните:

```bash
python -m music_lyrics_ai.cli train \
  --dataset data/sample_lyrics.jsonl \
  --output-dir artifacts/demo \
  --epochs 30 \
  --learning-rate 0.03 \
  --max-seq-len 128 \
  --d-model 64
```

#### Что делает эта команда

- `--dataset data/sample_lyrics.jsonl` — берёт готовый датасет из папки `data`
- `--output-dir artifacts/demo` — складывает обученную модель в папку `artifacts/demo`
- `--epochs 30` — 30 проходов по датасету
- `--learning-rate 0.03` — скорость обучения
- `--max-seq-len 128` — максимальная длина текста в символах
- `--d-model 64` — размер скрытого слоя сети

### Шаг 2. Дождитесь окончания обучения

Во время работы вы увидите строки типа:

```text
epoch=1 loss=...
epoch=2 loss=...
```

Когда обучение закончится, появится папка:

```text
artifacts/demo
```

Внутри будут файлы:

- `model.json` — веса нейросети
- `tokenizer.json` — токенизатор
- `styles.json` — список стилей
- `metrics.json` — метрики обучения

### Шаг 3. Сгенерируйте текст

Теперь вставьте в терминал:

```bash
python -m music_lyrics_ai.cli generate \
  --model-dir artifacts/demo \
  --style rap \
  --prompt "Midnight in the city" \
  --max-new-tokens 120 \
  --temperature 0.95 \
  --top-k 20 \
  --top-p 0.9
```

После этого в терминале появится сгенерированный текст песни.

---

## Как запускать без терминальной магии: буквально по шагам в VS Code

Если вы совсем новичок, делайте так:

1. Откройте проект в VS Code.
2. Слева нажмите на файл **`README.md`**, чтобы держать инструкцию перед глазами.
3. В верхнем меню нажмите **Terminal** → **New Terminal**.
4. По очереди копируйте команды из README и вставляйте в терминал.
5. После каждой команды нажимайте **Enter**.
6. Сначала создайте `.venv`, потом активируйте её, потом установите проект, потом запускайте обучение.
7. После обучения откройте слева папку **`artifacts`** → **`demo`** и проверьте, что файлы модели появились.

---

## Как поменять стили и свои тексты

Файл датасета лежит здесь:

```text
data/sample_lyrics.jsonl
```

Вы можете открыть его в VS Code и заменить содержимое своими песнями.

### Формат одной строки

```json
{"style": "rock", "text": "Your lyric text here"}
```

### Важно

- **Одна строка = один JSON-объект**.
- Поле `style` — название стиля.
- Поле `text` — текст песни.
- Можно делать сколько угодно стилей: `rock`, `trap`, `drill`, `edm`, `indie`, `metal` и т.д.

### Пример своего файла

```json
{"style": "trap", "text": "Night city, loud bass, cold light"}
{"style": "trap", "text": "Money talks, skyline burns tonight"}
{"style": "indie", "text": "We kept summer in a paper cup"}
{"style": "indie", "text": "Blue guitars and roads that wake us up"}
```

После этого просто снова запускаете обучение, но уже на своём файле.

Например:

```bash
python -m music_lyrics_ai.cli train \
  --dataset data/my_lyrics.jsonl \
  --output-dir artifacts/my_model \
  --epochs 50 \
  --learning-rate 0.02 \
  --max-seq-len 160 \
  --d-model 96
```

А затем генерацию:

```bash
python -m music_lyrics_ai.cli generate \
  --model-dir artifacts/my_model \
  --style trap \
  --prompt "Neon rain on my hoodie" \
  --max-new-tokens 120
```

---

## Что означают основные параметры

### При обучении

- `--dataset` — путь к вашему датасету
- `--output-dir` — папка, куда сохранить модель
- `--epochs` — сколько раз сеть увидит весь датасет
- `--learning-rate` — насколько быстро меняются веса
- `--max-seq-len` — максимальная длина текста в символах
- `--d-model` — размер скрытого представления; чем больше, тем потенциально лучше, но медленнее

### При генерации

- `--model-dir` — папка с обученной моделью
- `--style` — стиль, который есть в датасете
- `--prompt` — начальная фраза
- `--max-new-tokens` — сколько символов дописывать
- `--temperature` — степень случайности
- `--top-k` — ограничение на число кандидатов
- `--top-p` — ограничение по суммарной вероятности

---

## Если хотите запускать кликом мышки, а не командами

Можно сделать так:

1. Откройте проект в VS Code.
2. Создайте файл `run_train.bat` (Windows) или `run_train.sh` (macOS/Linux).
3. Запишите туда команду обучения.
4. Запускайте этот файл двойным кликом.

### Пример для Windows: `run_train.bat`

```bat
call .venv\Scripts\activate.bat
python -m music_lyrics_ai.cli train --dataset data/sample_lyrics.jsonl --output-dir artifacts/demo --epochs 30 --learning-rate 0.03 --max-seq-len 128 --d-model 64
pause
```

### Пример для macOS/Linux: `run_train.sh`

```bash
#!/usr/bin/env bash
source .venv/bin/activate
python -m music_lyrics_ai.cli train --dataset data/sample_lyrics.jsonl --output-dir artifacts/demo --epochs 30 --learning-rate 0.03 --max-seq-len 128 --d-model 64
```

Тогда вам не придётся каждый раз вручную вводить команду.

---

## Частые ошибки и что делать

### Ошибка: `python is not recognized`

Значит Python не установлен или не добавлен в PATH.
Переустановите Python и поставьте галочку **Add Python to PATH**.

### Ошибка: `No module named pip`

Попробуйте:

```bash
python -m ensurepip --upgrade
```

### Ошибка: `No module named music_lyrics_ai`

Скорее всего, вы забыли выполнить:

```bash
pip install -e .[dev]
```

или делаете запуск не из корня проекта.

### Ошибка: стиль не найден

Если передали `--style trap`, а в датасете такого стиля нет, генерация не запустится.
Убедитесь, что этот стиль реально есть в вашем JSONL-файле.

### Модель пишет странный текст

Это нормально для маленького датасета.
Чтобы качество было лучше, нужно:

- больше примеров,
- чище тексты,
- больше эпох,
- больше `d-model`,
- больше данных на каждый стиль.

---

## Формат датасета

Нужен JSONL-файл, где каждая строка выглядит так:

```json
{"style": "rock", "text": "Your lyric text here"}
```

---

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

---

## Как улучшить качество

Чтобы проект стал production-grade, добавьте:

- большой корпус текстов песен по стилям;
- субсловную токенизацию;
- мини-батчи и векторизацию через NumPy/PyTorch;
- GPU-обучение;
- валидационный сплит и early stopping;
- фильтрацию токсичного контента;
- перенос обучения на предобученную языковую модель.

---

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
