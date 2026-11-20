# Interview Anonymizer 🎭

**Текущая версия: v5.2.1 Stable.**

Interview Anonymizer — это высокопроизводительный локальный сервис для автоматической анонимизации видео-интервью и создания текстовой расшифровки с разделением по ролям говорящих. Инструмент скрывает лица (сверхбыстрый детектор YuNet), изменяет голос (Pitch Shifting) и формирует отдельный `.txt`-транскрипт диалога (`Спикер 1`, `Спикер 2`, ...), либо работает в режиме **«Только транскрибация»** для любых видео и аудиозаписей. По умолчанию все нейросети работают **100% локально без облачных вызовов** (опционально можно выбрать облачную ASR-модель **Deepgram Nova-3** или подключить локальную / облачную OpenAI-совместимую LLM для Шага 4).

Специально оптимизирован для работы с длинными записями (часовые интервью на русском, английском и смешанной IT-речи) на обычных процессорах (CPU) без тяжёлого PyTorch-рантайма, а при наличии видеокарты NVIDIA (вкл. RTX 50-серию Blackwell) обеспечивает **полноценное GPU-ускорение и для GigaAM v3 FP32/CTC (`onnxruntime-gpu`), и для Breeze-ASR-25 / Whisper Turbo (`CTranslate2`)**.

## ✨ Ключевые особенности

* **Очищенное ядро из 5 лучших ASR-моделей (v5.1)**: В списке оставлены только 5 проверенных лидеров без дублирования и слабых моделей:
  * **`gigaam-v3`** *(по умолчанию)* — полноточная **Sber GigaAM v3 E2E RNN-T FP32** (892 МБ, `onnx-asr`, GPU CUDA / CPU, до ~60× на GPU) — эталон точности для русской речи;
  * **`gigaam-v3-ctc`** — **Sber GigaAM v3 E2E CTC FP32** (886 МБ, до **240× на GPU**) — сверхбыстрая пакетная транскрибация длинных русских записей;
  * **`breeze-asr-25`** — **MediaTek Breeze-ASR-25** (3.1 ГБ, `faster-whisper` CT2, GPU/CPU) — лучший локальный профиль для смешанной RU+EN IT-речи с латиницей (`gRPC`, `OAuth 2`, `JWT`, `PostgreSQL`, `Camunda`) без галлюцинаций Whisper v3;
  * **`turbo`** — **Whisper Large v3 Turbo** (810 МБ, 99 языков, GPU/CPU) — универсальная лёгкая мультиязычная модель;
  * **`deepgram-nova-3`** *(опционально, облако)* — прямой клиент официального REST API **Deepgram Nova-3** (`https://api.deepgram.com/v1/listen`, без сторонних SDK) с RU/EN/`multi`, `keyterm`-словарём, сжатием `FLAC` и нативной облачной диаризацией. Deepgram API-ключ хранится только в `sessionStorage` вкладки (или берётся из `DEEPGRAM_API_KEY`), не сохраняется на диск и не попадает в логи.
* **Опциональное шумоподавление речи GTCRN ONNX (v5.0)**: Встроенный ультралёгкий нейросетевой шумодав `GTCRN` (~535 КБ, `sherpa-onnx`, выключен по умолчанию) очищает шум кулера, трафика и эхо офиса перед подачей в ASR, при этом диаризация `CAM++` всегда получает **оригинальный тембр голоса** для максимальной точности разделения спикеров.
* **Продвинутый Шаг 4: LLM-корректура v5.2.1 (`deepseek-flash` / Qwen Cloud / локальный Ollama)**: Подключается к **облачному API** (**DeepSeek API** `https://api.deepseek.com/v1`, модель по умолчанию **`deepseek-flash`** с non-thinking режимом и 3-поточной отправкой, / **Qwen Cloud DashScope** `qwen-plus`) **либо к локальному серверу** (**Ollama**, **LM Studio**, **vLLM** — 100% данных на ПК). В **v5.2–v5.2.1** добавлены:
  * **Дробление длинных монологов** (5–7-минутные ответы автоматически режутся по границам предложений на фрагменты ≤ 155 слов перед отправкой в LLM и бесшовно собираются обратно под исходные таймкоды — больше нет «усталости внимания» в середине монолога);
  * **Pass-0 автоглоссарий записи с защитой от «отравления»** (предварительный проход по всему файлу извлекает канонические названия продуктов, компаний и латинских терминов без 2–3-буквенных кириллических омофонов вроде `ПИО`/`ЕБС`);
  * **Программный пословный фонетический стоп-кран + защита стиля и регистра**: проверяет согласные/слоги каждой замены и точечно откатывает ложные межтехнологические подмены (`Debiver → Kibana`, `свапу → Swagger`, `соSкодо → Confluence`, `АБС → ЕБС`, `ТЛП → ПИО`), сохраняет живые звательные формы имён (`Алин,`, `Юль,`) и фиксирует точный регистр терминов (`GRPC` → `gRPC`), при этом пропуская реальные «русские обманки» RNN-T (`поджирописью` → `по gRPC`, `по листам` → `по RESTам`);
  * **Сырой бэкап `*_transcript_raw.txt`** и кнопка его скачивания прямо в UI. Ключ API никогда не логируется и не сохраняется в `localStorage`.
* **Длинная речь без потерь (Longform ≤ 20 с + Silero VAD + нахлёст ±0.5 с, v5.1+)**: Любые непрерывные монологи автоматически объединяются по коротким паузам (≤ 0.55 с) до нарезки и дробятся строго до `≤ 20 секунд` (ниже 25-секундного лимита энкодера `GigaAM v3`) по микропаузам `Silero VAD` и акустическим долинам с мягким контекстным нахлёстом ±0.5–1.6 с (без «съеденных» слогов и 4-секундных обрывов на стыках).
* **Безопасный опциональный словарь терминов (v5.1–v5.2.1)**: Пустое по умолчанию поле словаря/контекста (`--prompt` в CLI или быстрые пресеты **«Системный аналитик»**, **«Backend / DevOps»**, **«Frontend / QA»** в UI) передаётся в `initial_prompt` моделей `Whisper`/`Breeze` (без вредного `hotwords` на Breeze), в `keyterm` Deepgram Nova-3 (кроме `multi`) и в LLM-корректуру; regex-нормализация срабатывает только по целым словам и только когда LLM-корректура выключена (или на отдельных упавших батчах/фрагментах), не искажая омонимы (`GPT-токены`, `жиры`, `редис`).
* **Ускоренная однопроходная диаризация без «взрыва спикеров» (v5.1–v5.2.1)**: Батчевая сегментация `pyannote-segmentation-3.0` напрямую через `ONNX Runtime` (CPU / GPU CUDA с прямым CPU-fallback, шаг 5 с, **без дублирующего прогона `CAM++`**), откалиброванные пороги `UPGMA` + когерентная пересборка центроидов + канально-инвариантный SVD-сплит, lookahead-атрибуция стартовых реплик (`00:00:00`), проверка крайних реплик по `CAM++` и аккуратное отделение коротких «прилипших хвостов» на стыках смены говорящего.
* **Нативная поддержка многодорожечных записей OBS (v4.6+)**: Настройка **«Аудиодорожки»** (`--audio-tracks auto | separate | mix | first`). Если файл содержит 2 отдельные аудиодорожки (например, OBS: Дорожка 1 — микрофон кандидата, Дорожка 2 — системный звук / интервьюеры), Дорожка 1 расшифровывается детерминированно как `Спикер 1` со 100% точностью без диаризации, диаризация запускается только на Дорожке 2 (`Спикер 2..N`), а реплики сводятся по таймкодам в единый транскрипт. При анонимизации `.mp4` все дорожки автоматически микшируются, чтобы звук интервьюеров не пропадал.
* **Отдельное создание и пересоздание транскрипта после записи (v4.5+)**: Можно сгенерировать транскрипт уже *после* завершения анонимизации или пересоздать его другой моделью/настройками — прямо на экране результата или из блока **«Созданные записи»** (`output/`). Для недавних задач используется кэш оригинального 16 кГц аудио *до* питч-шифтинга (включая обе дорожки для OBS-записей, с автоматическим fallback на готовый `.mp4`). При отмене пересоздания видео и предыдущий транскрипт сохраняются.
* **Режим «Только транскрибация» (v4.5+)**: Отдельная вкладка в веб-интерфейсе и флаг `--transcribe-only` в CLI полностью пропускают нарезку видео, детекцию/размытие лиц, изменение голоса и сборку `.mp4`. Поддерживаются видео и аудио форматы: `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, `.m4v`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.opus`, `.flac`, `.aac`, `.wma`.
* **Точная атрибуция живой речи и перебиваний (v4.4+)**: Фразово-синтаксическое выравнивание (Clause-Aware Alignment) не разрывает предложения на стыках перебиваний, а акустическая перепроверка реплик по голосовым отпечаткам `CAM++` точно разделяет быстрые диалоги «вопрос — короткий ответ».
* **Ускоренный CPU/GPU-конвейер (v4.3+)**: Нативная планарная обработка видео в `YUV420p` без полнокадровых конвертаций `YUV↔BGR`, фоновое кодирование питч-аудио в `AAC` с однопроходной stream-copy сборкой `.mp4`, аппаратная развязка GPU ASR (Phase A параллельно с CPU блюром лиц) и диаризации (Phase B после склейки видео), а также мгновенная отмена («Прервать») с очисткой всех временных файлов.
* **Стойкая анонимизация лиц без протечек**: Нейросеть **YuNet (OpenCV)** на `576p`-копии каждого **18-го кадра** (базовый интервал) с **адаптивной бисекцией до каждых 2 кадров** при перетаскивании/ресайзе окна браузера, смене вкладок или появлении новых лиц, глобально-оптимальным матчингом, заполнением 1-кадровых провалов и динамическим расширением мозаики в движении.
* **Двухшаговый запуск и авто-обновление статики (v5.1–v5.2)**: Сначала выберите или перетащите запись (аудиофайлы автоматически переключают режим в «Только транскрибация»), спокойно настройте модель/словарь/LLM и нажмите **«Начать»** (доступны кнопки «Заменить» и «Убрать»). Сервер отдаёт `/` и `/api/config` с `Cache-Control: no-store` и `APP_BUILD_ID` (`git HEAD` + хэш статики) без застревания старого кэша браузера.
* **Высококонтрастный Claymorphism UI**: Переключатель режимов («Анонимизация» / «Только транскрибация»), список созданных записей, кастомные селекты с многострочным отображением, пресеты словаря, сворачиваемый блок «Улучшение качества» (GTCRN + LLM-корректура с пресетами DeepSeek API / Qwen Cloud / Ollama / LM Studio), Светлая/Тёмная тема (`localStorage`), выбор CPU/GPU и живой лог.

## 🧠 Модели транскрибации и диаризации

4 локальные ASR-модели и диаризация работают **100% офлайн** через легковесные движки `ONNX Runtime` (`onnx-asr`, `sherpa-onnx`) и `CTranslate2` (`faster-whisper`) без тяжёлого PyTorch; дополнительно доступен опциональный облачный профиль **`deepgram-nova-3`**:

| Профиль (`id`) | Модель | Движок | Ускорение | Языки | Диск | Пик RAM / VRAM | Назначение |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gigaam-v3`** *(по умолч.)* | `Sber GigaAM v3 E2E RNN-T (FP32)` | `onnx-asr` | `CPU` / **`GPU CUDA`** | **Русский (RU)** с нативной пунктуацией и «ё» | ~892 МБ | ~1.5 ГБ | Эталон качества на русской речи (до ~60× на GPU, longform `≤20 с`) |
| **`gigaam-v3-ctc`** | `Sber GigaAM v3 E2E CTC (FP32)` | `onnx-asr` | `CPU` / **`GPU CUDA`** | **Русский (RU)** с нативной пунктуацией и «ё» | ~886 МБ | ~1.5 ГБ | Турбо-режим для русского (до **240× на GPU**, часовое интервью за ~15–20 с) |
| **`breeze-asr-25`** | `MediaTek Breeze-ASR-25` (Whisper v2 base) | `faster-whisper` | `CPU` / **`GPU CUDA`** | **RU + EN code-switching** | ~3.1 ГБ | ~3.2 ГБ | Лучший локальный выбор для смешанной RU+EN IT-речи и латинских терминов без галлюцинаций v3 |
| **`turbo`** | `Whisper Large v3 Turbo` | `faster-whisper` | `CPU` / **`GPU CUDA`** | **99 языков** | ~810 МБ | ~1.5 ГБ | Универсальная быстрая мультиязычная модель для EN и других языков на CPU и GPU |
| **`deepgram-nova-3`** *(облако)* | `Deepgram Nova-3` (REST API) | `deepgram_cloud` | Облако (`api.deepgram.com`) | **RU / EN / multi** + `keyterm` + диаризация | 0 МБ | ~0.1 ГБ | Опциональный облачный API (нужен `DEEPGRAM_API_KEY`; нативная диаризация и точные IT-термины) |

> **Как получить максимальное качество:**
> 1. Для чистой русской речи (100% локально + LLM): **`gigaam-v3` (RNN-T FP32 на GPU)** + при плотных IT-терминах включите **LLM-корректуру** (`DeepSeek API` с моделью **`deepseek-flash`** / `Qwen Cloud` **или** локально `qwen3:8b` в Ollama) + пресет словаря.
> 2. Для облачного распознавания без локального GPU (RU+EN IT-термины + встроенная диаризация): **`deepgram-nova-3`** + Deepgram API-ключ + пресет словаря (`keyterms`).
> 3. Для плотной двуязычной RU+EN IT-речи 100% локально без LLM: **`breeze-asr-25`** + словарь терминов.
> 4. Для чисто английских или мультиязычных интервью локально: **`turbo`** (99 языков) или **`breeze-asr-25`**.
> 5. Для 4+ похожих голосов: укажите точное число спикеров или включите **Разделение голосов → Высокая** (или записывайте в OBS на 2 дорожки: микрофон + системный звук).
> 6. При шуме кулера/улицы: включите **Шумоподавление речи (GTCRN ONNX)**.

Для разделения по ролям используется однопроходный алгоритм **Direct ONNX Pyannote 3.0 + CAM++ Anchor UPGMA + SVD Coherence Split + Acoustic Clause Verification** (для `deepgram-nova-3` используется нативная облачная диаризация Deepgram, а для 2-дорожечных OBS-записей — детерминированное отделение Дорожки 1 как `Спикер 1` + независимая диаризация Дорожки 2):
* **Сегментация речи и фильтр гудков**: `pyannote-segmentation-3.0` (`ONNX Runtime` CPU/GPU CUDA, шаг 5 с со сшивкой окон, `min_duration_on/off = 0.18s`) + `Silero VAD` (защита от пропуска тихих реплик) + спектральный фильтр узкополосных гудков ожидания вызова (`Zoom` / `Teams` / `Телемост`).
* **Голосовые эмбеддинги (Voiceprints) и фразовое выравнивание**: однократный прогон `3D-Speaker CAM++` (`ONNX`, ~28 МБ) + откалиброванная кластеризация Average-Linkage (`UPGMA`) с когерентной пересборкой центроидов и настраиваемой чувствительностью (`normal` / `high` / `low`), синтаксическая группировка фраз при перебиваниях и акустическая перепроверка коротких и крайних реплик по тембру говорящего.

### Пример выходного файла транскрипта (`.txt`)

```text
# Транскрибация интервью (Interview Anonymizer v5.2.1)
# Длительность: 00:42:18 | Модель: GigaAM v3 E2E RNN-T (FP32) | Обнаружено спикеров: 2
# Постобработка: LLM-корректура (deepseek-flash, исправлено реплик: 14)
# ========================================================================

[00:00:01 - 00:00:09] Спикер 1: Привет! Расскажи немного про ваш опыт с проектом и как устроена команда.

[00:00:10 - 00:00:24] Спикер 2: Привет, давай начнём с архитектуры. Мы перенесли основной сервис с REST на gRPC и подключили OAuth 2...
```


## 🛠 Технологический стек

* **Backend**: Python 3.11+, FastAPI, Uvicorn, `httpx`
* **Computer Vision**: OpenCV (`cv2`), YuNet (ONNX)
* **Speech Recognition, Denoising & Diarization**: `onnx-asr` (`ONNX Runtime` CPU / `onnxruntime-gpu` CUDA 12.8), `sherpa-onnx` (`GTCRN`, `3D-Speaker CAM++`), прямой `onnxruntime` раннер `pyannote-segmentation-3.0`, `faster-whisper` (`CTranslate2` INT8 / CUDA FP16), `huggingface-hub`, опциональный клиент **Deepgram Nova-3 REST API** (`httpx`)
* **Optional LLM Post-Editing**: Любой OpenAI-совместимый API — облачный (**DeepSeek API `deepseek-flash`**, **Qwen Cloud DashScope**) или локальный (**Ollama**, **LM Studio**, **vLLM**: Qwen / DeepSeek)
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

3. *(Рекомендуется при наличии NVIDIA GPU)* Если на хосте установлена видеокарта NVIDIA и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (или Docker Desktop с WSL2 GPU на Windows), запустите с GPU-оверлеем — это соберёт образ с `onnxruntime-gpu` (CUDA 12.8 + cuDNN 9) и откроет GPU-ускорение для **GigaAM v3 FP32 / CTC**, **pyannote**, **Breeze-ASR-25** и **Whisper Turbo**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
   ```

4. *(Опционально для Шага 4 — LLM-корректуры)*
   * **Вариант А (облачный OpenAI-compatible API — DeepSeek / Qwen Cloud, по умолчанию):**
     В блоке **«Улучшение качества → LLM-корректура»** по умолчанию выбрана модель **`deepseek-flash`** (`https://api.deepseek.com/v1`, также доступен пресет **«Qwen Cloud»** `qwen-plus`) — достаточно вставить ваш `API-ключ` (`sk-...`). Ключ хранится только в оперативной памяти вкладки (`sessionStorage`), никогда не сохраняется в `localStorage` и не попадает в логи.
   * **Вариант Б (100% локально — Ollama / LM Studio):**
     ```bash
     ollama pull qwen3:8b
     ollama serve
     ```
     Нажмите пресет **«Ollama (локально)»**: в Docker-контейнере уже настроен `host.docker.internal`, поэтому URL `http://host.docker.internal:11434/v1` подставится автоматически (при запуске без Docker — `http://localhost:11434/v1`). API-ключ оставляйте пустым.

5. Откройте интерфейс в браузере:
   ```text
   http://localhost:8000
   ```

### 🔄 Обновление после `git pull`

Фронтенд и версия в интерфейсе теперь отдаются с `Cache-Control: no-store` и автоматическим `APP_BUILD_ID` (`git HEAD` + хэш статики), но в Docker код копируется внутрь образа на этапе сборки — поэтому после `git pull` обязательно пересоберите контейнер:

```bash
git pull
# CPU:
docker compose up --build -d
# или GPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

При локальном запуске без Docker достаточно перезапустить `python app.py` / `uvicorn app:app` и обновить вкладку браузера.

> **Примечание о моделях**: При первом запуске задачи с включённой локальной транскрибацией выбранная модель автоматически скачается в примонтированную папку `./models` и в дальнейшем будет загружаться мгновенно с диска без доступа в интернет.

### Полностью автономная (Air-Gapped) сборка образа

Если контейнер планируется запускать в закрытом контуре без доступа к сети, запеките нужные модели прямо в Docker-образ на этапе сборки (`gigaam-v3`, `gigaam-v3-ctc`, `breeze-asr-25`, `turbo`, `denoiser` или `all`):

```bash
docker build --build-arg PRELOAD_MODELS=gigaam-v3,breeze-asr-25,denoiser -t interview-anonymizer:offline .
```

Модели сохраняются внутри образа в `/opt/preloaded_models`, поэтому монтирование пустой папки `./models:/app/models` не затирает предзагруженные веса.

## 💻 Использование из командной строки (CLI)

Помимо веб-интерфейса, сервис можно запускать напрямую через `main.py`:

```bash
# Стандартный запуск на CPU (анонимизация видео + автотранскрибация GigaAM v3 FP32)
python main.py -i interview.mp4 -o output/interview_anon.mp4

# Максимальное качество на GPU: GigaAM v3 FP32 (CUDA) + шумоподавление GTCRN + облачная LLM-корректура DeepSeek (deepseek-flash по умолчанию)
python main.py -i interview.mp4 --transcribe-only --device cuda --asr-model gigaam-v3 --denoise --llm-polish --llm-api-key "$DEEPSEEK_API_KEY" --prompt "REST, gRPC, HTTP 2.0, OAuth 2, JWT, PostgreSQL, Kafka, Camunda"

# То же самое 100% локально через Ollama
python main.py -i interview.mp4 --transcribe-only --device cuda --asr-model gigaam-v3 --denoise --llm-polish --llm-url http://localhost:11434/v1 --llm-model qwen3:8b

# Облачное распознавание Deepgram Nova-3 (с нативной диаризацией и словарём keyterms)
python main.py -i interview.mp4 --transcribe-only --asr-model deepgram-nova-3 --asr-api-key "$DEEPGRAM_API_KEY" --prompt "REST, gRPC, HTTP 2.0, OAuth 2, JWT, PostgreSQL, Camunda"

# Сверхбыстрый GPU-режим (до 240× на RTX): GigaAM v3 CTC FP32
python main.py -i recording.m4a --transcribe-only --device cuda --asr-model gigaam-v3-ctc

# Мультиязычное / английское интервью (Whisper Turbo, 99 языков)
python main.py -i english_interview.mp4 --transcribe-only --device cuda --asr-model turbo --language en

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

# Локальная автотранскрибация, облачный Deepgram Nova-3, шумоподавление, диаризация и опциональная LLM-корректура
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
MAX_CACHED_AUDIO_FILES = max(1, int(os.environ.get("MAX_CACHED_AUDIO_FILES", "8")))
TRANSCRIPTION_ENABLED_DEFAULT = os.environ.get("TRANSCRIPTION_ENABLED", "true").lower() not in ("0", "false", "no", "off")
DEFAULT_ASR_MODEL = os.environ.get("DEFAULT_ASR_MODEL", "gigaam-v3").lower()  # gigaam-v3 | gigaam-v3-ctc | breeze-asr-25 | turbo | deepgram-nova-3
DEFAULT_DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")  # Только при выборе deepgram-nova-3
DEFAULT_ASR_LANGUAGE = os.environ.get("DEFAULT_ASR_LANGUAGE", "ru").lower()   # ru | auto | en
DEFAULT_COMPUTE_DEVICE = os.environ.get("DEFAULT_COMPUTE_DEVICE", "cpu").lower()  # cpu | cuda
DENOISE_ENABLED_DEFAULT = os.environ.get("DENOISE_ENABLED", "false").lower() in ("1", "true", "yes", "on")
LLM_POLISH_ENABLED_DEFAULT = os.environ.get("LLM_POLISH_ENABLED", "false").lower() in ("1", "true", "yes", "on")
DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-flash")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # Для облачных DeepSeek / Qwen Cloud; пусто для Ollama / LM Studio
DIARIZATION_CENTROID_SIM_THRESHOLD = float(os.environ.get("DIARIZATION_SIM_THRESHOLD", "0.56"))
DEFAULT_DIARIZATION_SENSITIVITY = os.environ.get("DIARIZATION_SENSITIVITY", "normal").lower()  # normal | high | low
DEFAULT_AUDIO_TRACK_MODE = os.environ.get("AUDIO_TRACK_MODE", "auto").lower()  # auto | separate | mix | first
```

### Подбор количества воркеров

Начинайте с `4-6` воркеров на обычных 6-ядерных CPU и сравнивайте общее время обработки одного и того же видео. Важен не пик загрузки CPU, а строка `Total time` в логах. Если первые чанки идут значительно дольше последних, значит система перегружена и количество воркеров стоит уменьшить.

Приложение не позволит выбрать больше `(количество логических CPU - 1)`, чтобы оставить системе хотя бы один поток для ОС, Docker и браузера. В режиме `CPU` выбранное число потоков также передаётся движкам транскрибации после склейки видео; в режиме `GPU CUDA` транскрибация (`GigaAM v3 FP32/CTC`, `Breeze-ASR-25` или `Whisper Turbo`) работает параллельно с видеообработкой на видеокарте, не отнимая потоки у воркеров размытия лиц.

## 📁 Структура проекта

```text
├── src/
│   ├── models/               # Встроенная ONNX модель детекции лиц (YuNet)
│   ├── audio_processor.py    # Анонимизация и pitch-shifting звука
│   ├── cancellation.py       # Потокобезопасный токен отмены задач и остановки процессов
│   ├── config.py             # Конфигурация видео, воркеров, 5 профилей ASR, GTCRN и LLM-пресетов
│   ├── deepgram_asr.py       # Опциональный клиент Deepgram Nova-3 REST API (keyterms + диаризация)
│   ├── ffmpeg_utils.py       # Работа с FFmpeg: метаданные, чанкинг, склейка, WAV 16 kHz
│   ├── io_manager.py         # Управление временными папками задач
│   ├── llm_postprocessor.py  # Опциональная LLM-корректура (DeepSeek deepseek-flash / Qwen / Ollama / LM Studio)
│   ├── model_manager.py      # Локальный кэш, автозагрузка и валидация весов моделей
│   ├── orchestrator.py       # Оркестратор конвейера (видео + питч + CPU/GPU/Cloud транскрибация)
│   ├── transcriber.py        # GTCRN шумодав, 1-pass pyannote + CAM++, onnx-asr/Whisper/Deepgram ASR и .txt
│   └── video_processor.py    # Ядро анонимизации лиц: YuNet, адаптивная бисекция, блюр
├── static/
│   ├── index.html            # Главная страница (Claymorphism UI)
│   ├── script.js             # Логика UI (кастомные селекты, CPU/GPU, GTCRN, LLM-пресеты, поллинг)
│   └── style.css             # Стили тем, дроп-даунов, тумблеров и анимаций
├── tests/                    # Модульные, интеграционные и Node.js UI автотесты (pytest)
├── app.py                    # FastAPI сервер, cache-busting (APP_BUILD_ID) и HTTP API
├── main.py                   # CLI-интерфейс запуска обработки и транскрибации
├── docker-compose.yml        # Стандартный запуск контейнера (CPU + host.docker.internal)
├── docker-compose.gpu.yml    # GPU-оверлей (ENABLE_GPU=1 -> onnxruntime-gpu CUDA 12.8 + NVIDIA GPU)
├── Dockerfile                # Сборка образа (Python 3.11 + ARG ENABLE_GPU + ARG PRELOAD_MODELS)
└── requirements.txt          # Python зависимости
```

## 📜 Лицензия

Распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).
