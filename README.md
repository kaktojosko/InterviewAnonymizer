# Interview Anonymizer 🎭

**Текущая версия: v5.0.0 Stable.**

Interview Anonymizer — это высокопроизводительный локальный сервис для автоматической анонимизации видео-интервью и создания текстовой расшифровки с разделением по ролям говорящих. Инструмент скрывает лица (сверхбыстрый детектор YuNet), изменяет голос (Pitch Shifting) и формирует отдельный `.txt`-транскрипт диалога (`Спикер 1`, `Спикер 2`, ...), либо работает в режиме **«Только транскрибация»** для любых видео и аудиозаписей. По умолчанию все нейросети работают **100% локально без облачных вызовов** (опционально можно подключить как локальную, так и облачную OpenAI-совместимую LLM для Шага 4).

Специально оптимизирован для работы с длинными записями (часовые интервью на русском, английском и смешанной IT-речи) на обычных процессорах (CPU) без тяжёлого PyTorch-рантайма, а при наличии видеокарты NVIDIA (вкл. RTX 50-серию Blackwell) обеспечивает **полноценное GPU-ускорение и для GigaAM v3 FP32/CTC / Canary 1B v2 (`onnxruntime-gpu`), и для Whisper / Breeze-ASR-25 (`CTranslate2`)**.

## ✨ Ключевые особенности

* **Полноточный GigaAM v3 FP32 на GPU (CUDA) и CPU (v5.0)**: Основной профиль `gigaam-v3` переведён на полноточную модель **Sber GigaAM v3 E2E RNN-T FP32** (892 МБ) через `onnx-asr` + `ONNX Runtime` — без потерь от INT8-квантизации на латинских подсловах и тихой речи, с ускорением до **~60× на NVIDIA GPU** (и параллельным выполнением на GPU одновременно с размытием лиц на CPU!). Для сверхбыстрой пакетной обработки добавлен **`gigaam-v3-ctc`** (до **240× на GPU**), а для слабых машин без GPU сохранён лёгкий **`gigaam-v3-int8`** (232 МБ).
* **Breeze-ASR-25, NVIDIA Canary 1B v2 и Canary-Qwen 2.5B (v5.0)**: Добавлены три топовых профиля распознавания:
  * **`breeze-asr-25`** (`MediaTek Breeze-ASR-25`, CTranslate2) — лучшая открытая модель для смешанной RU+EN IT-речи и нативного вывода англоязычных терминов (`gRPC`, `OAuth 2`, `JWT`, `PostgreSQL`, `Kubernetes`, `Camunda`) без галлюцинаций Whisper v3;
  * **`canary-1b-v2`** (`NVIDIA Canary 1B v2`, `onnx-asr`, 3.9 ГБ, GPU CUDA / CPU) — 1-миллиардная мультиязычная модель на 25 европейских языков (RU/EN/DE/FR/ES/UK и др.) с нативной пунктуацией и регистром;
  * **`canary-qwen-2.5b`** (`NVIDIA Canary-Qwen 2.5B`, чистый `ONNX Runtime` INT8, 2.9 ГБ, `FastConformer + Qwen3-1.7B`, без PyTorch/NeMo) — мировой **#1 Open ASR Leaderboard (5.63% WER)** для **англоязычных** интервью (**только English**).
* **Опциональное шумоподавление речи GTCRN ONNX (v5.0)**: Встроенный ультралёгкий нейросетевой шумодав `GTCRN` (~535 КБ, `sherpa-onnx`, выключен по умолчанию) очищает шум кулера, трафика и эхо офиса перед подачей в ASR, при этом диаризация `CAM++` всегда получает **оригинальный тембр голоса** для максимальной точности разделения спикеров.
* **Опциональный Шаг 4: LLM-корректура через OpenAI-compatible API — локально или в облаке (v5.0)**: Подключается к **локальному** серверу (**Ollama**, **LM Studio**, **vLLM** с Qwen / DeepSeek — 100% данных на ПК) **либо к облачному API** (**DeepSeek API** `api.deepseek.com/v1` / **Qwen Cloud DashScope** `dashscope-intl.aliyuncs.com/compatible-mode/v1`) с быстрыми пресетами, полем `API-ключ` и индикатором приватности (`локальный сервер` / `облачный API`). Ключ API никогда не логируется, не возвращается в ответах API и не сохраняется в `localStorage`. Корректура восстанавливает искажённые IT-термины и пунктуацию, гарантированно сохраняя исходные таймкоды и роли (с валидацией длин/мета-утечек и автоматическим сохранением «сырого» бэкапа `*_transcript_raw.txt`).
* **Длинная речь без потерь (Longform ≤ 20 с + Silero VAD, v4.6+)**: Любые непрерывные монологи (1–3+ минуты) автоматически дробятся строго до `≤ 20 секунд` (надёжно ниже 25-секундного лимита энкодера `GigaAM v3`) по микропаузам `Silero VAD` и акустическим долинам, а при полном отсутствии пауз — скользящим окном с перекрытием `1.5 с` и дедупликацией границ. Объединение `Silero VAD` с сегментами диаризации и поканальное выравнивание громкости (`normalize_chunk_level`) спасают тихих удалённых собеседников от пропуска.
* **Опциональный словарь IT-терминов и англицизмов (v4.6+)**: Пустое по умолчанию поле словаря/контекста (`--prompt` в CLI или быстрые пресеты **«Системный аналитик»**, **«Backend / DevOps»**, **«Frontend / QA»** в UI) передаётся в `initial_prompt` моделей `Whisper`/`Breeze`, используется в LLM-корректуре и включает опциональную нормализацию кириллических транслитераций указанных терминов (`gRPC`, `OAuth 2`, `JWT`, `PostgreSQL`, `Sequence`, `Camunda`, `Swagger`, `Docs as Code` и др.) — ничего не подставляется без вашего выбора.
* **Усиленная диаризация схожих голосов (UPGMA + чувствительность, v4.6+)**: Точный выбор числа участников (`Авто`, `2`, `3`, `4`, `5`, `6`, `4+`), иерархическая кластеризация Average-Linkage (`UPGMA`) с нижнеквантильной защитой от «цепной» склейки и пресеты **«Разделение голосов»** (`--sensitivity normal | high | low`) — режим `Высокая` предотвращает склеивание схожих однополых голосов (например, 4 женщин-участниц в 2 кластера).
* **Нативная поддержка многодорожечных записей OBS (v4.6+)**: Настройка **«Аудиодорожки»** (`--audio-tracks auto | separate | mix | first`). Если файл содержит 2 отдельные аудиодорожки (например, OBS: Дорожка 1 — микрофон кандидата, Дорожка 2 — системный звук / интервьюеры), Дорожка 1 расшифровывается детерминированно как `Спикер 1` со 100% точностью без диаризации, диаризация запускается только на Дорожке 2 (`Спикер 2..N`), а реплики сводятся по таймкодам в единый транскрипт. При анонимизации `.mp4` все дорожки автоматически микшируются, чтобы звук интервьюеров не пропадал.
* **Отдельное создание и пересоздание транскрипта после записи (v4.5+)**: Можно сгенерировать транскрипт уже *после* завершения анонимизации или пересоздать его другой моделью/настройками — прямо на экране результата или из блока **«Созданные записи»** (`output/`). Для недавних задач используется кэш оригинального 16 кГц аудио *до* питч-шифтинга (включая обе дорожки для OBS-записей, с автоматическим fallback на готовый `.mp4`). При отмене пересоздания видео и предыдущий транскрипт сохраняются.
* **Режим «Только транскрибация» (v4.5+)**: Отдельная вкладка в веб-интерфейсе и флаг `--transcribe-only` в CLI полностью пропускают нарезку видео, детекцию/размытие лиц, изменение голоса и сборку `.mp4`. Поддерживаются видео и аудио форматы: `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, `.m4v`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.opus`, `.flac`, `.aac`, `.wma`.
* **Точная атрибуция живой речи и перебиваний (v4.4+)**: Фразово-синтаксическое выравнивание (Clause-Aware Alignment) не разрывает предложения на стыках перебиваний, а акустическая перепроверка реплик по голосовым отпечаткам `CAM++` точно разделяет быстрые диалоги «вопрос — короткий ответ».
* **Ускоренный CPU/GPU-конвейер (v4.3+)**: Нативная планарная обработка видео в `YUV420p` без полнокадровых конвертаций `YUV↔BGR`, фоновое кодирование питч-аудио в `AAC` с однопроходной stream-copy сборкой `.mp4`, аппаратная развязка GPU ASR (Phase A параллельно с CPU блюром лиц) и CPU-диаризации (Phase B после склейки видео), а также мгновенная отмена («Прервать») с очисткой всех временных файлов.
* **Стойкая анонимизация лиц без протечек**: Нейросеть **YuNet (OpenCV)** на `576p`-копии каждого **18-го кадра** (базовый интервал) с **адаптивной бисекцией до каждых 2 кадров** при перетаскивании/ресайзе окна браузера, смене вкладок или появлении новых лиц, глобально-оптимальным матчингом, заполнением 1-кадровых провалов и динамическим расширением мозаики в движении.
* **Высококонтрастный Claymorphism UI**: Переключатель режимов («Анонимизация» / «Только транскрибация»), список созданных записей, кастомные селекты с многострочным отображением, пресеты словаря, сворачиваемый блок «Улучшение качества» (GTCRN + LLM-корректура с пресетами Ollama / LM Studio / DeepSeek API / Qwen Cloud), Светлая/Тёмная тема (`localStorage`), выбор CPU/GPU и живой лог.

## 🧠 Локальные модели транскрибации и диаризации

Все ASR-модели и диаризация работают **100% локально** через легковесные движки `ONNX Runtime` (`onnx-asr`, `sherpa-onnx`, встроенный `CanaryQwenSpeechLlm`) и `CTranslate2` (`faster-whisper`) без тяжёлого PyTorch. В интерфейсе и CLI доступны **11 профилей** распознавания речи:

| Профиль (`id`) | Модель | Движок | Ускорение | Языки | Диск | Пик RAM / VRAM | Назначение |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gigaam-v3`** *(по умолч.)* | `Sber GigaAM v3 E2E RNN-T (FP32)` | `onnx-asr` | `CPU` / **`GPU CUDA`** | **Русский (RU)** с нативной пунктуацией и «ё» | ~892 МБ | ~1.5 ГБ | Эталон качества на русской речи (до ~60× на GPU, longform `≤20 с`) |
| **`gigaam-v3-ctc`** | `Sber GigaAM v3 E2E CTC (FP32)` | `onnx-asr` | `CPU` / **`GPU CUDA`** | **Русский (RU)** с нативной пунктуацией и «ё» | ~886 МБ | ~1.5 ГБ | Турбо-режим для русского (до **240× на GPU**, часовое интервью за ~15–20 с) |
| **`breeze-asr-25`** | `MediaTek Breeze-ASR-25` (Whisper v2 base) | `faster-whisper` | `CPU` / **`GPU CUDA`** | **RU + EN code-switching** | ~3.1 ГБ | ~3.2 ГБ | Лучший выбор для смешанной RU+EN IT-речи и латинских терминов без галлюцинаций v3 |
| **`canary-1b-v2`** | `NVIDIA Canary 1B v2 (FP32)` | `onnx-asr` | `CPU` / **`GPU CUDA`** | **25 языков Европы** (вкл. RU, EN, DE, FR, ES) | ~3.9 ГБ | ~4.2 ГБ | Мощная 1B-модель с пунктуацией и регистром для EN / европейских / мультиязычных интервью |
| **`canary-qwen-2.5b`** | `NVIDIA Canary-Qwen 2.5B (SALM INT8)` | `onnxruntime` | `CPU` (`INT8`, ~3.5× RT) | **English ONLY (EN)** с пунктуацией и регистром | ~2.9 ГБ | ~4.2 ГБ | Мировой **#1 Open ASR Leaderboard (5.63% WER)** для чисто английских интервью (`FastConformer + Qwen3-1.7B`) |
| **`large-v3-ru`** | `Whisper Large v3 Russian` (русский файнтюн) | `faster-whisper` | `CPU` / **`GPU CUDA`** | **Русский + RU/EN** | ~3.1 ГБ | ~3.2 ГБ | Русский файнтюн Whisper v3 с поддержкой `initial_prompt` словаря |
| **`turbo`** | `Whisper Large v3 Turbo` | `faster-whisper` | `CPU` / **`GPU CUDA`** | **99 языков** | ~810 МБ | ~1.5 ГБ | Универсальный баланс скорости и мультиязычности на CPU и GPU |
| **`large-v3`** | `Whisper Large v3` (полный, 32 слоя декодера) | `faster-whisper` | `CPU` / **`GPU CUDA`** | **99 языков** | ~3.1 ГБ | ~3.2 ГБ | Мультиязычный эталон для нерусских или сложных многоязычных интервью |
| **`gigaam-v3-int8`** | `Sber GigaAM v3 E2E RNN-T (INT8)` | `onnx-asr` | `CPU` (`INT8`) | **Русский (RU)** с пунктуацией | ~232 МБ | ~0.8 ГБ | Лёгкая CPU-версия для машин без GPU или с малым объёмом ОЗУ |
| **`parakeet`** | `NVIDIA Parakeet TDT 0.6B v3` | `sherpa-onnx` | `CPU` (`INT8`) | **25 языков Европы** (вкл. Ru, En, De, Fr, Es, Uk) | ~600 МБ | ~1.1 ГБ | Быстрая пакетная обработка европейских языков на CPU |
| **`small`** | `Whisper Small` | `faster-whisper` | `CPU` / **`GPU CUDA`** | **99 языков** | ~245 МБ | ~0.6 ГБ | Ультра-лёгкий режим для ноутбуков и мини-ПК |

> **Как получить качество выше приложения Салют:**
> 1. Для чистой русской речи: **`gigaam-v3` (FP32 на GPU)** + при плотных IT-терминах включите **LLM-корректуру** (локально `qwen3:8b` / `deepseek-r1:8b` в Ollama **или** облачно `DeepSeek API` / `Qwen Cloud`) + пресет словаря.
> 2. Для плотной двуязычной RU+EN IT-речи без LLM: **`breeze-asr-25`** + словарь терминов.
> 3. Для чисто английских интервью (максимум точности): **`canary-qwen-2.5b`** (EN ONLY, 5.63% WER) или **`canary-1b-v2`** (25 языков, GPU).
> 4. Для 4+ похожих голосов: включите **Разделение голосов → Высокая** (или записывайте в OBS на 2 дорожки: микрофон + системный звук).
> 5. При шуме кулера/улицы: включите **Шумоподавление речи (GTCRN ONNX)**.

Для разделения по ролям используется двухэтапный алгоритм **Anchor UPGMA Diarization + Acoustic Clause Verification** (а для 2-дорожечных OBS-записей — детерминированное отделение Дорожки 1 как `Спикер 1` + независимая диаризация Дорожки 2):
* **Сегментация речи и фильтр гудков**: `pyannote-segmentation-3.0` (`INT8 ONNX`, ~6 МБ, шаг `0.18s`) + `Silero VAD` (защита от пропуска тихих реплик) + спектральный фильтр узкополосных гудков ожидания вызова (`Zoom` / `Teams` / `Телемост`).
* **Голосовые эмбеддинги (Voiceprints) и фразовое выравнивание**: `3D-Speaker CAM++` (`ONNX`, ~28 МБ) + взвешенная кластеризация Average-Linkage (`UPGMA`) с настраиваемой чувствительностью (`normal` / `high` / `low`), синтаксическая группировка фраз при перебиваниях и акустическая перепроверка коротких ответов по тембру говорящего.

### Пример выходного файла транскрипта (`.txt`)

```text
# Транскрибация интервью (Interview Anonymizer v5.0.0)
# Длительность: 00:42:18 | Модель: GigaAM v3 E2E RNN-T (FP32) | Обнаружено спикеров: 2
# Постобработка: LLM-корректура (qwen3:8b, исправлено реплик: 14)
# ========================================================================

[00:00:01 - 00:00:09] Спикер 1: Привет! Расскажи немного про ваш опыт с проектом и как устроена команда.

[00:00:10 - 00:00:24] Спикер 2: Привет, давай начнём с архитектуры. Мы перенесли основной сервис с REST на gRPC и подключили OAuth 2...
```


## 🛠 Технологический стек

* **Backend**: Python 3.11+, FastAPI, Uvicorn, `httpx`
* **Computer Vision**: OpenCV (`cv2`), YuNet (ONNX)
* **Speech Recognition, Denoising & Diarization**: `onnx-asr` (`ONNX Runtime` CPU / `onnxruntime-gpu` CUDA 12.8), встроенный ONNX Speech-LLM раннер (`Canary-Qwen 2.5B`), `sherpa-onnx` (`GTCRN`, `pyannote`, `3D-Speaker CAM++`), `faster-whisper` (`CTranslate2` INT8 / CUDA FP16), `huggingface-hub`
* **Optional LLM Post-Editing**: Любой OpenAI-совместимый API — локальный (**Ollama**, **LM Studio**, **vLLM**: Qwen / DeepSeek) или облачный (**DeepSeek API**, **Qwen Cloud DashScope**)
* **Video/Audio Manipulation**: FFmpeg, `ffmpeg-python`
* **Frontend**: HTML5, Vanilla JavaScript, CSS3 (CSS Variables, Claymorphism)
* **DevOps**: Docker, Docker Compose (лёгкий CPU-образ по умолчанию + GPU-оверлей `ENABLE_GPU=1`)

## 🚀 Установка и запуск (через Docker)

Проект полностью контейнеризирован. Вам не нужно устанавливать Python или FFmpeg локально, достаточно установленного [Docker](https://www.docker.com/) и [Docker Compose](https://docs.docker.com/compose/).

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/kaktojosko/InterviewAnonymizer.git
   cd InterviewAnonymizer
   ```

2. Соберите и запустите контейнер в стандартном режиме (CPU):
   ```bash
   docker compose up --build -d
   ```

3. *(Рекомендуется при наличии NVIDIA GPU)* Если на хосте установлена видеокарта NVIDIA и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (или Docker Desktop с WSL2 GPU на Windows), запустите с GPU-оверлеем — это соберёт образ с `onnxruntime-gpu` (CUDA 12.8 + cuDNN 9) и откроет GPU-ускорение для **GigaAM v3 FP32 / CTC**, **Canary 1B v2**, **Breeze-ASR-25** и **Whisper**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
   ```

4. *(Опционально для Шага 4 — LLM-корректуры)*
   * **Вариант А (100% локально — Ollama / LM Studio):**
     ```bash
     ollama pull qwen3:8b
     ollama serve
     ```
     В Docker-контейнере уже настроен `host.docker.internal`, поэтому URL `http://host.docker.internal:11434/v1` подставится автоматически (при запуске без Docker — `http://localhost:11434/v1`). API-ключ оставляйте пустым.
   * **Вариант Б (облачный OpenAI-compatible API — DeepSeek / Qwen Cloud):**
     В блоке **«Улучшение качества → LLM-корректура»** нажмите пресет **«DeepSeek API»** (`https://api.deepseek.com/v1`, `deepseek-chat`) или **«Qwen Cloud»** (`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, `qwen-plus`) и вставьте ваш `API-ключ` (`sk-...`). Ключ хранится только в оперативной памяти вкладки (`sessionStorage`), никогда не сохраняется в `localStorage` и не попадает в логи.

5. Откройте интерфейс в браузере:
   ```text
   http://localhost:8000
   ```

> **Примечание о моделях**: При первом запуске задачи с включённой транскрибацией выбранная модель автоматически скачается в примонтированную папку `./models` и в дальнейшем будет загружаться мгновенно с диска без доступа в интернет.

### Полностью автономная (Air-Gapped) сборка образа

Если контейнер планируется запускать в закрытом контуре без доступа к сети, запеките нужные модели прямо в Docker-образ на этапе сборки (`gigaam-v3`, `gigaam-v3-ctc`, `breeze-asr-25`, `canary-1b-v2`, `canary-qwen-2.5b`, `large-v3-ru`, `turbo`, `large-v3`, `gigaam-v3-int8`, `parakeet`, `small`, `denoiser` или `all`):

```bash
docker build --build-arg PRELOAD_MODELS=gigaam-v3,breeze-asr-25,denoiser -t interview-anonymizer:offline .
```

Модели сохраняются внутри образа в `/opt/preloaded_models`, поэтому монтирование пустой папки `./models:/app/models` не затирает предзагруженные веса.

## 💻 Использование из командной строки (CLI)

Помимо веб-интерфейса, сервис можно запускать напрямую через `main.py`:

```bash
# Стандартный запуск на CPU (анонимизация видео + автотранскрибация GigaAM v3 FP32)
python main.py -i interview.mp4 -o output/interview_anon.mp4

# Максимальное качество на GPU: GigaAM v3 FP32 (CUDA) + шумоподавление GTCRN + локальная LLM-корректура (Ollama)
python main.py -i interview.mp4 --transcribe-only --device cuda --asr-model gigaam-v3 --denoise --llm-polish --llm-model qwen3:8b --prompt "REST, gRPC, HTTP 2.0, OAuth 2, JWT, PostgreSQL, Kafka, Camunda"

# То же самое с облачным DeepSeek API (или Qwen Cloud)
python main.py -i interview.mp4 --transcribe-only --device cuda --asr-model gigaam-v3 --llm-polish --llm-url https://api.deepseek.com/v1 --llm-model deepseek-chat --llm-api-key "$DEEPSEEK_API_KEY"

# Сверхбыстрый GPU-режим (до 240× на RTX): GigaAM v3 CTC FP32
python main.py -i recording.m4a --transcribe-only --device cuda --asr-model gigaam-v3-ctc

# Англоязычное интервью (#1 Open ASR Leaderboard: NVIDIA Canary-Qwen 2.5B INT8, EN ONLY)
python main.py -i english_interview.mp4 --transcribe-only --asr-model canary-qwen-2.5b --language en

# Мультиязычное / европейское интервью (NVIDIA Canary 1B v2 на GPU)
python main.py -i eu_interview.mp4 --transcribe-only --device cuda --asr-model canary-1b-v2 --language en

# Плотная RU+EN IT-речь с латиницей (Breeze-ASR-25) + 4 участницы со схожими голосами
python main.py -i sa_interview.mp4 --transcribe-only --device cuda --asr-model breeze-asr-25 --speakers 4 --sensitivity high --prompt "Собеседование системного аналитика: REST, gRPC, HTTP 2.0, SOAP, JWT, Bearer, OAuth 2, PostgreSQL, Redis, MongoDB, Kafka, C4, Sequence, Activity, ERD, BPMN, Camunda, Swagger, Postman, Confluence, Jira, Git, Jenkins, Docs as Code"

# Двухдорожечная OBS-запись (Дорожка 1 = Спикер 1, Дорожка 2 = диаризация собеседников)
python main.py -i obs_2tracks.mp4 -o output/obs_anon.mp4 --audio-tracks separate --speakers 4 --sensitivity high

# Только анонимизация видео и голоса (без транскрибации)
python main.py -i interview.mp4 -o output/interview_anon.mp4 --no-transcribe
```

Предзагрузка локальных моделей вручную в `./models`:

```bash
python -m src.model_manager --preload gigaam-v3,denoiser
```

## ⚠️ Хранение файлов и освобождение места

Приложение работает локально и хранит рабочие файлы на вашем компьютере:

* `uploads/` — загруженные исходные файлы. Удаляются автоматически после завершения задачи, но могут остаться после аварийной остановки.
* `output/` — готовые анонимизированные видео (`*_anonymized.mp4`), файлы расшифровки (`*_transcript.txt`) и при включённой LLM-корректуре — исходный pre-LLM бэкап (`*_transcript_raw.txt`). Остаются на диске, чтобы их можно было скачать повторно или досоздать/пересоздать транскрипт через интерфейс.
* `tmp/` — временные чанки видео, рабочие WAV-дорожки и ограниченный кэш оригинального 16 кГц аудио (`tmp/audio_cache/`, до 8 последних задач для отложенной транскрибации; в Docker хранится внутри быстрой нативной ФС контейнера и очищается при перезапуске).
* `models/` — локальный кэш скачанных нейросетевых моделей распознавания речи, шумоподавления (`gtcrn_simple.onnx`) и диаризации (~260 МБ – 3.9 ГБ в зависимости от выбранного профиля). **Не очищайте эту папку**, если не хотите скачивать модели заново.

Длинные видео могут быстро занимать десятки гигабайт. Периодически проверяйте размер папок `uploads/`, `output/` и `tmp/`, особенно после тестовых прогонов.

### Очистка рабочих файлов (с сохранением кэша моделей)

Перед удалением убедитесь, что нужные готовые видео и `.txt`-транскрипты уже скачаны: очистка `output/` удалит результаты обработки.

PowerShell:

```powershell
docker compose down
Remove-Item -Recurse -Force .\uploads\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\output\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\tmp\* -ErrorAction SilentlyContinue
docker compose up -d
```

Bash:

```bash
docker compose down
rm -rf uploads/* output/* tmp/*
docker compose up -d
```

## 🧹 Очистка Docker-кэша

Docker хранит build cache, старые слои образов и остановленные контейнеры. После частых пересборок или больших видео кэш может занимать много места. Текущий `Dockerfile` копирует в образ только код приложения, а `.dockerignore` исключает `uploads/`, `output/`, `tmp/` и `models/`, но build cache Docker всё равно нужно иногда чистить.

Посмотреть, сколько места занимает Docker:

```bash
docker system df
```

Безопасная очистка build cache:

```bash
docker builder prune
```

Более агрессивная очистка build cache:

```bash
docker builder prune --all --force
```

Если сборка падает с ошибкой вида `parent snapshot ... does not exist` или `failed to prepare extraction snapshot`, очистите BuildKit-кэш и пересоберите образ:

```bash
docker compose down
docker builder prune --all --force
docker compose build --no-cache web
docker compose up -d web
```

Осторожно с глобальной очисткой:

```bash
docker system prune --all
```

Эта команда удаляет неиспользуемые образы, остановленные контейнеры и сети Docker во всей системе, а не только в этом проекте. Не используйте `--volumes`, если не уверены, что вам не нужны Docker volumes других проектов.

## ⚙️ Конфигурация системы

Все ключевые настройки вынесены в файл `src/config.py` и поддерживают переопределение через переменные окружения:

```python
# Настройки отступа блюра от краев лица
FACE_PADDING_PERCENT_X = 0.40
FACE_PADDING_PERCENT_Y = 0.40

# Порог уверенности и разрешение входа нейросети YuNet (576p для мелких лиц в боковой галерее)
YUNET_MIN_CONFIDENCE = 0.25
YUNET_TARGET_HEIGHT = 576

# Базовый интервал детекции: каждый 18-й кадр (~0.6 с при 30 FPS).
# При перетаскивании/ресайзе окна или резком движении интервал автоматически
# дробится адаптивной бисекцией вплоть до ADAPTIVE_MIN_DETECT_STEP (каждые 2 кадра).
DETECT_EVERY_N_FRAMES = 18
ADAPTIVE_MIN_DETECT_STEP = 2

# Ограничение FPS обработки. 30 FPS снижает нагрузку на CPU для VFR/60 FPS видео.
MAX_PROCESSING_FPS = 30.0

# Параметры H.264 кодирования обработанных чанков
OUTPUT_VIDEO_PRESET = os.environ.get("OUTPUT_VIDEO_PRESET", "ultrafast")
OUTPUT_VIDEO_CRF = os.environ.get("OUTPUT_VIDEO_CRF", "26")

# Количество параллельных воркеров по умолчанию (ограничено CPU_COUNT - 1)
CPU_COUNT = os.cpu_count() or multiprocessing.cpu_count() or 1
MAX_ALLOWED_WORKERS = max(1, CPU_COUNT - 1)
MAX_WORKERS = max(1, min(MAX_ALLOWED_WORKERS, int(os.environ.get("VIDEO_MAX_WORKERS", CPU_COUNT - 3))))

# Смещение тональности голоса (в полутонах. Отрицательные значения делают голос ниже)
PITCH_SHIFT_SEMITONES = -2.0

# Локальная автотранскрибация, шумоподавление, диаризация и опциональная LLM-корректура
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
MAX_CACHED_AUDIO_FILES = max(1, int(os.environ.get("MAX_CACHED_AUDIO_FILES", "8")))
TRANSCRIPTION_ENABLED_DEFAULT = os.environ.get("TRANSCRIPTION_ENABLED", "true").lower() not in ("0", "false", "no", "off")
DEFAULT_ASR_MODEL = os.environ.get("DEFAULT_ASR_MODEL", "gigaam-v3").lower()  # gigaam-v3 | gigaam-v3-ctc | breeze-asr-25 | canary-1b-v2 | canary-qwen-2.5b | large-v3-ru | turbo | large-v3 | gigaam-v3-int8 | parakeet | small
DEFAULT_ASR_LANGUAGE = os.environ.get("DEFAULT_ASR_LANGUAGE", "ru").lower()   # ru | auto | en
DEFAULT_COMPUTE_DEVICE = os.environ.get("DEFAULT_COMPUTE_DEVICE", "cpu").lower()  # cpu | cuda
DENOISE_ENABLED_DEFAULT = os.environ.get("DENOISE_ENABLED", "false").lower() in ("1", "true", "yes", "on")
LLM_POLISH_ENABLED_DEFAULT = os.environ.get("LLM_POLISH_ENABLED", "false").lower() in ("1", "true", "yes", "on")
DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # Для облачных DeepSeek / Qwen Cloud; пусто для Ollama / LM Studio
DIARIZATION_CENTROID_SIM_THRESHOLD = float(os.environ.get("DIARIZATION_SIM_THRESHOLD", "0.60"))
DEFAULT_DIARIZATION_SENSITIVITY = os.environ.get("DIARIZATION_SENSITIVITY", "normal").lower()  # normal | high | low
DEFAULT_AUDIO_TRACK_MODE = os.environ.get("AUDIO_TRACK_MODE", "auto").lower()  # auto | separate | mix | first
```

### Подбор количества воркеров

Начинайте с `4-6` воркеров на обычных 6-ядерных CPU и сравнивайте общее время обработки одного и того же видео. Важен не пик загрузки CPU, а строка `Total time` в логах. Если первые чанки идут значительно дольше последних, значит система перегружена и количество воркеров стоит уменьшить.

Приложение не позволит выбрать больше `(количество логических CPU - 1)`, чтобы оставить системе хотя бы один поток для ОС, Docker и браузера. В режиме `CPU` выбранное число потоков также передаётся движкам транскрибации после склейки видео; в режиме `GPU CUDA` транскрибация (`GigaAM v3 FP32/CTC`, `Canary 1B v2`, `Breeze-ASR-25` или `Whisper`) работает параллельно с видеообработкой на видеокарте, не отнимая потоки у воркеров размытия лиц.

## 📁 Структура проекта

```text
├── src/
│   ├── models/               # Встроенная ONNX модель детекции лиц (YuNet)
│   ├── audio_processor.py    # Анонимизация и pitch-shifting звука
│   ├── canary_qwen_asr.py    # Чистый ONNX Speech-LLM раннер для NVIDIA Canary-Qwen 2.5B (EN)
│   ├── cancellation.py       # Потокобезопасный токен отмены задач и остановки процессов
│   ├── config.py             # Конфигурация видео, воркеров, 11 профилей ASR, GTCRN и LLM-пресетов
│   ├── ffmpeg_utils.py       # Работа с FFmpeg: метаданные, чанкинг, склейка, WAV 16 kHz
│   ├── io_manager.py         # Управление временными папками задач
│   ├── llm_postprocessor.py  # Опциональная LLM-корректура (локально Ollama/LM Studio или облачно DeepSeek/Qwen)
│   ├── model_manager.py      # Локальный кэш, автозагрузка и валидация весов моделей
│   ├── orchestrator.py       # Оркестратор конвейера (видео + питч + CPU/GPU транскрибация)
│   ├── transcriber.py        # GTCRN шумодав, диаризация CAM++, onnx-asr/Whisper ASR и сборка .txt
│   └── video_processor.py    # Ядро анонимизации лиц: YuNet, интерполяция, блюр
├── static/
│   ├── index.html            # Главная страница (Claymorphism UI)
│   ├── script.js             # Логика UI (кастомные селекты, CPU/GPU, GTCRN, LLM-пресеты, поллинг)
│   └── style.css             # Стили тем, дроп-даунов, тумблеров и анимаций
├── tests/                    # Модульные, интеграционные и Node.js UI автотесты (pytest)
├── app.py                    # FastAPI сервер и HTTP API
├── main.py                   # CLI-интерфейс запуска обработки и транскрибации
├── docker-compose.yml        # Стандартный запуск контейнера (CPU + host.docker.internal)
├── docker-compose.gpu.yml    # GPU-оверлей (ENABLE_GPU=1 -> onnxruntime-gpu CUDA 12.8 + NVIDIA GPU)
├── Dockerfile                # Сборка образа (Python 3.11 + ARG ENABLE_GPU + ARG PRELOAD_MODELS)
└── requirements.txt          # Python зависимости
```

## 📜 Лицензия

Распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).
