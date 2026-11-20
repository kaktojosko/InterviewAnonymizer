import os
import multiprocessing

# Base paths
APP_VERSION = "5.0.0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_BASE_DIR = os.path.join(BASE_DIR, "tmp")
AUDIO_CACHE_DIR = os.path.join(TMP_BASE_DIR, "audio_cache")
MAX_CACHED_AUDIO_FILES = max(1, int(os.environ.get("MAX_CACHED_AUDIO_FILES", "8")))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
PRELOADED_MODELS_DIR = os.environ.get("PRELOADED_MODELS_DIR", "/opt/preloaded_models")

# Video Processing Settings
FACE_PADDING_PERCENT_X = 0.40
FACE_PADDING_PERCENT_Y = 0.40
YUNET_MIN_CONFIDENCE = 0.25
YUNET_TARGET_HEIGHT = 576

# Base face detection cadence: detect every 18 frames (~0.6s at 30fps).
# When window drags, layout switches, fast face motion, or face count changes occur
# within a batch, VideoProcessor recursively bisects the buffer down to
# ADAPTIVE_MIN_DETECT_STEP (2 frames) so faces stay 100% anonymized in motion.
DETECT_EVERY_N_FRAMES = 18
ADAPTIVE_MIN_DETECT_STEP = 2
MOTION_DIFF_PIXEL_THRESHOLD = 18
MOTION_CHANGED_RATIO_TRIGGER = 0.06

# Cap the normalized output stream used by the processor.
# 60 FPS doubles CPU work for little benefit on interview/webcam footage.
# Set to 0 to preserve the source frame rate.
MAX_PROCESSING_FPS = 30.0

# H.264 chunk encoding. CRF 26 keeps the ultrafast path but avoids
# oversized output on long, low-bitrate interview recordings.
OUTPUT_VIDEO_PRESET = os.environ.get("OUTPUT_VIDEO_PRESET", "ultrafast")
OUTPUT_VIDEO_CRF = os.environ.get("OUTPUT_VIDEO_CRF", "26")

# Parallelization and Chunking Settings
# Break video into 1-minute chunks for MAXIMUM parallel processing
CHUNK_DURATION_SECONDS = 60 
# Leave 3 logical CPUs free for OS and UI responsiveness by default.
# Per-upload values are capped at logical CPU count minus one.
CPU_COUNT = os.cpu_count() or multiprocessing.cpu_count() or 1
MAX_ALLOWED_WORKERS = max(1, CPU_COUNT - 1)
MAX_WORKERS = max(1, min(MAX_ALLOWED_WORKERS, int(os.environ.get("VIDEO_MAX_WORKERS", CPU_COUNT - 3))))

# Audio Settings
PITCH_SHIFT_SEMITONES = -2.0

# Job Modes & Allowed File Extensions
JOB_MODE_OPTIONS = ["anonymize", "transcribe_only"]
DEFAULT_JOB_MODE = "anonymize"
ANONYMIZE_ALLOWED_EXTENSIONS = (".mp4",)
TRANSCRIBE_ONLY_ALLOWED_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".m4v",
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".opus",
    ".flac",
    ".aac",
    ".wma",
)

# Local Auto-Transcription & Speaker Diarization Settings
TRANSCRIPTION_ENABLED_DEFAULT = os.environ.get("TRANSCRIPTION_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")
DEFAULT_ASR_MODEL = os.environ.get("DEFAULT_ASR_MODEL", "gigaam-v3").strip().lower()

ASR_LANGUAGE_OPTIONS = ["ru", "auto", "en"]
ASR_LANGUAGE_LABELS = {
    "ru": "Русский (RU)",
    "auto": "Авто (мультиязычный)",
    "en": "English (EN)",
}
DEFAULT_ASR_LANGUAGE = os.environ.get("DEFAULT_ASR_LANGUAGE", "ru").strip().lower()
if DEFAULT_ASR_LANGUAGE not in ASR_LANGUAGE_OPTIONS:
    DEFAULT_ASR_LANGUAGE = "ru"

ASR_MODEL_PROFILES = {
    "gigaam-v3": {
        "id": "gigaam-v3",
        "label": "GigaAM v3 FP32 — эталон для русского (RU, GPU/CPU, 892 МБ)",
        "short_name": "GigaAM v3 E2E RNN-T (FP32)",
        "engine": "onnx_asr",
        "onnx_asr_model": "gigaam-v3-e2e-rnnt",
        "quantization": None,
        "repo_id": "istupakov/gigaam-v3-onnx",
        "max_chunk_sec": 20.0,
        "overlap_sec": 1.6,
        "batch_size_cuda": 8,
        "batch_size_cpu": 4,
        "required_files": [
            "config.json",
            "v3_e2e_rnnt_encoder.onnx",
            "v3_e2e_rnnt_decoder.onnx",
            "v3_e2e_rnnt_joint.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ],
        "allow_patterns": [
            "config.json",
            "v3_e2e_rnnt.yaml",
            "v3_e2e_rnnt_encoder.onnx",
            "v3_e2e_rnnt_decoder.onnx",
            "v3_e2e_rnnt_joint.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ],
        "min_total_bytes": 750_000_000,
        "supports_cuda": True,
        "fixed_language": "ru",
    },
    "gigaam-v3-ctc": {
        "id": "gigaam-v3-ctc",
        "label": "GigaAM v3 CTC FP32 — турбо для русского (до 240× GPU, 886 МБ)",
        "short_name": "GigaAM v3 E2E CTC (FP32)",
        "engine": "onnx_asr",
        "onnx_asr_model": "gigaam-v3-e2e-ctc",
        "quantization": None,
        "repo_id": "istupakov/gigaam-v3-onnx",
        "max_chunk_sec": 20.0,
        "overlap_sec": 1.6,
        "batch_size_cuda": 16,
        "batch_size_cpu": 6,
        "required_files": [
            "config.json",
            "v3_e2e_ctc.onnx",
            "v3_e2e_ctc_vocab.txt",
        ],
        "allow_patterns": [
            "config.json",
            "v3_e2e_ctc.yaml",
            "v3_e2e_ctc.onnx",
            "v3_e2e_ctc_vocab.txt",
        ],
        "min_total_bytes": 750_000_000,
        "supports_cuda": True,
        "fixed_language": "ru",
    },
    "breeze-asr-25": {
        "id": "breeze-asr-25",
        "label": "Breeze-ASR-25 — лучший для RU+EN IT-речи с латиницей (3.1 ГБ)",
        "short_name": "MediaTek Breeze-ASR-25 (CT2)",
        "engine": "faster_whisper",
        "repo_id": "SoybeanMilk/faster-whisper-Breeze-ASR-25",
        "decode_params": {
            "beam_size_cpu": 3,
            "beam_size_cuda": 5,
            "patience": 1.2,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 4,
            "no_speech_threshold": 0.58,
            "compression_ratio_threshold": 2.3,
            "log_prob_threshold": -0.95,
            "hallucination_silence_threshold": 1.5,
            "temperatures": (0.0, 0.2, 0.4),
            "max_window_sec": 28.0,
            "use_hotwords": True,
        },
        "required_files": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 2_500_000_000,
        "supports_cuda": True,
        "default_language": "ru",
    },
    "canary-1b-v2": {
        "id": "canary-1b-v2",
        "label": "Canary 1B v2 — 25 языков + пунктуация (NVIDIA, GPU/CPU, 3.9 ГБ)",
        "short_name": "NVIDIA Canary 1B v2 (FP32)",
        "engine": "onnx_asr",
        "onnx_asr_model": "nemo-canary-1b-v2",
        "quantization": None,
        "repo_id": "istupakov/canary-1b-v2-onnx",
        "max_chunk_sec": 26.0,
        "overlap_sec": 0.0,
        "batch_size_cuda": 4,
        "batch_size_cpu": 1,
        "has_word_timestamps": False,
        "passes_language": True,
        "required_files": [
            "config.json",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder-model.onnx",
            "vocab.txt",
        ],
        "allow_patterns": [
            "config.json",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder-model.onnx",
            "vocab.txt",
        ],
        "min_total_bytes": 3_500_000_000,
        "supports_cuda": True,
        "default_language": "ru",
    },
    "canary-qwen-2.5b": {
        "id": "canary-qwen-2.5b",
        "label": "Canary-Qwen 2.5B — топ-1 для English (EN ONLY, INT8 CPU, 2.9 ГБ)",
        "short_name": "NVIDIA Canary-Qwen 2.5B (INT8, EN)",
        "engine": "onnx_canary_qwen",
        "repo_id": "OpenVoiceOS/canary-qwen-2.5b-onnx",
        "max_chunk_sec": 28.0,
        "overlap_sec": 0.0,
        "has_word_timestamps": False,
        "repetition_penalty": 1.05,
        "edge_pad_sec": 0.12,
        "required_files": [
            "config.json",
            "vocab.json",
            "encoder_int8.onnx",
            "embed_tokens_int8.onnx",
            "decoder_int8.onnx",
        ],
        "allow_patterns": [
            "config.json",
            "vocab.json",
            "encoder_int8.onnx",
            "embed_tokens_int8.onnx",
            "decoder_int8.onnx",
        ],
        "min_total_bytes": 2_500_000_000,
        "supports_cuda": False,
        "fixed_language": "en",
        "default_language": "en",
    },
    "large-v3-ru": {
        "id": "large-v3-ru",
        "label": "Whisper Large v3 RU — русский файнтюн (RU/EN, 3.1 ГБ)",
        "short_name": "Whisper Large v3 Russian",
        "engine": "faster_whisper",
        "repo_id": "bzikst/faster-whisper-large-v3-russian",
        "decode_params": {
            "beam_size_cpu": 3,
            "beam_size_cuda": 5,
            "patience": 1.1,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 4,
            "no_speech_threshold": 0.62,
            "compression_ratio_threshold": 2.2,
            "log_prob_threshold": -0.90,
            "hallucination_silence_threshold": 1.2,
            "temperatures": (0.0, 0.2, 0.4),
            "max_window_sec": 28.0,
            "use_hotwords": True,
        },
        "required_files": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 2_500_000_000,
        "supports_cuda": True,
        "default_language": "ru",
    },
    "turbo": {
        "id": "turbo",
        "label": "Whisper Turbo — универсальная (99 яз., 810 МБ)",
        "short_name": "Whisper Large v3 Turbo (INT8)",
        "engine": "faster_whisper",
        "repo_id": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "decode_params": {
            "beam_size_cpu": 2,
            "beam_size_cuda": 5,
            "patience": 1.0,
            "repetition_penalty": 1.08,
            "no_repeat_ngram_size": 4,
            "no_speech_threshold": 0.60,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "hallucination_silence_threshold": 1.5,
            "temperatures": (0.0, 0.2, 0.4),
            "max_window_sec": 28.5,
            "use_hotwords": True,
        },
        "required_files": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 600_000_000,
        "supports_cuda": True,
    },
    "large-v3": {
        "id": "large-v3",
        "label": "Whisper Large v3 — мультиязычная (99 яз., 3.1 ГБ)",
        "short_name": "Whisper Large v3",
        "engine": "faster_whisper",
        "repo_id": "Systran/faster-whisper-large-v3",
        "decode_params": {
            "beam_size_cpu": 3,
            "beam_size_cuda": 5,
            "patience": 1.1,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 4,
            "no_speech_threshold": 0.62,
            "compression_ratio_threshold": 2.2,
            "log_prob_threshold": -0.90,
            "hallucination_silence_threshold": 1.2,
            "temperatures": (0.0, 0.2, 0.4),
            "max_window_sec": 28.0,
            "use_hotwords": True,
        },
        "required_files": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 2_500_000_000,
        "supports_cuda": True,
    },
    "gigaam-v3-int8": {
        "id": "gigaam-v3-int8",
        "label": "GigaAM v3 INT8 — лёгкая для русского на CPU (232 МБ)",
        "short_name": "GigaAM v3 E2E RNN-T (INT8, CPU)",
        "engine": "onnx_asr",
        "onnx_asr_model": "gigaam-v3-e2e-rnnt",
        "quantization": "int8",
        "repo_id": "istupakov/gigaam-v3-onnx",
        "max_chunk_sec": 18.0,
        "overlap_sec": 1.6,
        "batch_size_cpu": 4,
        "required_files": [
            "config.json",
            "v3_e2e_rnnt_encoder.int8.onnx",
            "v3_e2e_rnnt_decoder.int8.onnx",
            "v3_e2e_rnnt_joint.int8.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ],
        "allow_patterns": [
            "config.json",
            "v3_e2e_rnnt.yaml",
            "v3_e2e_rnnt_encoder.int8.onnx",
            "v3_e2e_rnnt_decoder.int8.onnx",
            "v3_e2e_rnnt_joint.int8.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ],
        "min_total_bytes": 180_000_000,
        "supports_cuda": False,
        "fixed_language": "ru",
    },
    "parakeet": {
        "id": "parakeet",
        "label": "Parakeet v3 — быстрая (25 яз., 600 МБ)",
        "short_name": "NVIDIA Parakeet TDT 0.6B v3 (INT8)",
        "engine": "sherpa_parakeet",
        "repo_id": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        "encoder_file": "encoder.int8.onnx",
        "decoder_file": "decoder.int8.onnx",
        "joiner_file": "joiner.int8.onnx",
        "tokens_file": "tokens.txt",
        "feature_dim": 80,
        "max_chunk_sec": 20.0,
        "overlap_sec": 1.5,
        "batch_size_cpu": 4,
        "required_files": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
        "allow_patterns": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
        "min_total_bytes": 400_000_000,
        "supports_cuda": False,
    },
    "small": {
        "id": "small",
        "label": "Whisper Small — лёгкая (99 яз., 245 МБ)",
        "short_name": "Whisper Small (INT8)",
        "engine": "faster_whisper",
        "repo_id": "Systran/faster-whisper-small",
        "decode_params": {
            "beam_size_cpu": 2,
            "beam_size_cuda": 5,
            "patience": 1.0,
            "repetition_penalty": 1.06,
            "no_repeat_ngram_size": 4,
            "no_speech_threshold": 0.60,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.05,
            "hallucination_silence_threshold": 1.5,
            "temperatures": (0.0, 0.2, 0.4),
            "max_window_sec": 28.5,
            "use_hotwords": True,
        },
        "required_files": ["model.bin", "config.json", "tokenizer.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 180_000_000,
        "supports_cuda": True,
    },
}

if DEFAULT_ASR_MODEL not in ASR_MODEL_PROFILES:
    DEFAULT_ASR_MODEL = "gigaam-v3"

# Optional lightweight GTCRN neural speech denoiser (~535 KB ONNX via sherpa-onnx)
DENOISER_MODEL = {
    "filename": "gtcrn_simple.onnx",
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
    "min_bytes": 400_000,
}
DENOISE_ENABLED_DEFAULT = os.environ.get("DENOISE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

# Optional Stage 4 OpenAI-compatible LLM transcript post-editing (local Ollama / LM Studio OR cloud DeepSeek / Qwen)
_LOCAL_LLM_HOST = "host.docker.internal" if os.path.exists("/.dockerenv") else "localhost"
LLM_POLISH_ENABLED_DEFAULT = os.environ.get("LLM_POLISH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
DEFAULT_LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    f"http://{_LOCAL_LLM_HOST}:11434/v1",
).strip()
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b").strip()
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()

LLM_PROVIDER_PRESETS = {
    "ollama_local": {
        "id": "ollama_local",
        "label": "Ollama (локально)",
        "base_url": f"http://{_LOCAL_LLM_HOST}:11434/v1",
        "model": "qwen3:8b",
        "requires_api_key": False,
        "is_cloud": False,
    },
    "lmstudio_local": {
        "id": "lmstudio_local",
        "label": "LM Studio (локально)",
        "base_url": f"http://{_LOCAL_LLM_HOST}:1234/v1",
        "model": "qwen3-8b",
        "requires_api_key": False,
        "is_cloud": False,
    },
    "deepseek_cloud": {
        "id": "deepseek_cloud",
        "label": "DeepSeek API (облако)",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "requires_api_key": True,
        "is_cloud": True,
    },
    "qwen_cloud": {
        "id": "qwen_cloud",
        "label": "Qwen Cloud (DashScope)",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "requires_api_key": True,
        "is_cloud": True,
    },
}

COMPUTE_DEVICE_OPTIONS = ["cpu", "cuda"]
DEFAULT_COMPUTE_DEVICE = os.environ.get("DEFAULT_COMPUTE_DEVICE", "cpu").strip().lower()
if DEFAULT_COMPUTE_DEVICE not in COMPUTE_DEVICE_OPTIONS:
    DEFAULT_COMPUTE_DEVICE = "cpu"

DIARIZATION_MODELS = {
    "segmentation": {
        "repo_id": "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        "filename": "model.onnx",
        "min_bytes": 4_000_000,
    },
    "embedding": {
        "repo_id": "csukuangfj/speaker-embedding-models",
        "filename": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "min_bytes": 20_000_000,
    },
}

SPEAKER_COUNT_OPTIONS = ["auto", "2", "3", "4", "5", "6", "4+"]

# Diarization sensitivity presets for similar voices (e.g. multiple same-gender speakers)
DIARIZATION_SENSITIVITY_OPTIONS = ["normal", "high", "low"]
DEFAULT_DIARIZATION_SENSITIVITY = os.environ.get("DIARIZATION_SENSITIVITY", "normal").strip().lower()
if DEFAULT_DIARIZATION_SENSITIVITY not in DIARIZATION_SENSITIVITY_OPTIONS:
    DEFAULT_DIARIZATION_SENSITIVITY = "normal"

# Fine-grained turn boundary threshold (cosine distance) for initial pyannote segmentation pass.
# Tighter 0.25 prevents Pass 1 FastClustering from fusing back-to-back similar female voices.
DIARIZATION_SEGMENTATION_THRESHOLD = float(os.environ.get("DIARIZATION_SEG_THRESHOLD", "0.25"))
# Cosine similarity thresholds for Anchor-Linkage voiceprint clustering (CAM++).
# Tuned for same-gender / compressed Zoom-channel audio so similar speakers are not collapsed to 2.
DIARIZATION_CENTROID_SIM_THRESHOLD = float(os.environ.get("DIARIZATION_SIM_THRESHOLD", "0.64"))
DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD = float(os.environ.get("DIARIZATION_MULTI_SIM_THRESHOLD", "0.68"))

DIARIZATION_SENSITIVITY_PROFILES = {
    "normal": {
        "id": "normal",
        "label": "Обычная (баланс)",
        "sim_threshold": DIARIZATION_CENTROID_SIM_THRESHOLD,
        "multi_sim_threshold": DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD,
        "seg_threshold": DIARIZATION_SEGMENTATION_THRESHOLD,
        "svd_gap_threshold": 0.085,
        "svd_max_cent_sim": 0.84,
        "minor_min_sec": 4.0,
    },
    "high": {
        "id": "high",
        "label": "Высокая (похожие голоса / один пол)",
        "sim_threshold": max(DIARIZATION_CENTROID_SIM_THRESHOLD + 0.05, 0.69),
        "multi_sim_threshold": max(DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD + 0.05, 0.73),
        "seg_threshold": min(DIARIZATION_SEGMENTATION_THRESHOLD, 0.20),
        "svd_gap_threshold": 0.065,
        "svd_max_cent_sim": 0.87,
        "minor_min_sec": 2.8,
    },
    "low": {
        "id": "low",
        "label": "Низкая (строже объединять)",
        "sim_threshold": min(DIARIZATION_CENTROID_SIM_THRESHOLD - 0.08, 0.56),
        "multi_sim_threshold": min(DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD - 0.08, 0.60),
        "seg_threshold": max(DIARIZATION_SEGMENTATION_THRESHOLD, 0.34),
        "svd_gap_threshold": 0.115,
        "svd_max_cent_sim": 0.78,
        "minor_min_sec": 6.0,
    },
}

# Multi-track audio handling (e.g. OBS recordings with separate Mic + System Audio tracks)
AUDIO_TRACK_MODE_OPTIONS = ["auto", "separate", "mix", "first"]
AUDIO_TRACK_MODE_LABELS = {
    "auto": "Авто (2 дорожки → раздельно, иначе 1)",
    "separate": "Раздельно (1-я = Спикер 1, 2-я = собеседники)",
    "mix": "Смешать все аудиодорожки",
    "first": "Только 1-я аудиодорожка",
}
DEFAULT_AUDIO_TRACK_MODE = os.environ.get("AUDIO_TRACK_MODE", "auto").strip().lower()
if DEFAULT_AUDIO_TRACK_MODE not in AUDIO_TRACK_MODE_OPTIONS:
    DEFAULT_AUDIO_TRACK_MODE = "auto"

# Optional user-selectable domain glossary presets (EMPTY by default — never auto-injected!)
ASR_PROMPT_PRESETS = {
    "system_analyst": {
        "id": "system_analyst",
        "label": "Системный аналитик",
        "prompt": (
            "Собеседование системного аналитика: REST, gRPC, HTTP 2.0, HTTP/1.1, SOAP, JSON, "
            "JWT, Bearer, OAuth 2, API-шлюз, PostgreSQL, Redis, MongoDB, Kafka, C4, Sequence, "
            "Activity, ERD, BPMN, Camunda, PlantUML, draw.io, UI-kit, use cases, Swagger, "
            "Postman, Confluence, Jira, Git, Jenkins, Docs as Code, Spec-Driven Development, "
            "CRM, SFA, OTLP, C++"
        ),
    },
    "backend_devops": {
        "id": "backend_devops",
        "label": "Backend / DevOps",
        "prompt": (
            "Техническое собеседование: Python, Go, Java, Docker, Kubernetes, CI/CD, GitLab, "
            "PostgreSQL, Redis, Kafka, RabbitMQ, gRPC, REST API, GraphQL, OpenAPI, Prometheus, "
            "Grafana, Terraform, Linux"
        ),
    },
    "frontend_qa": {
        "id": "frontend_qa",
        "label": "Frontend / QA",
        "prompt": (
            "Техническое собеседование: JavaScript, TypeScript, React, Vue, Next.js, REST, "
            "GraphQL, WebSocket, Playwright, Cypress, Selenium, Postman, Swagger, Allure, Jira, Git"
        ),
    },
}

# Backwards-compatible alias
DIARIZATION_CLUSTER_THRESHOLD = DIARIZATION_CENTROID_SIM_THRESHOLD
DIARIZATION_MERGE_GAP_SEC = 0.45
DIARIZATION_MIN_SEGMENT_SEC = 0.18
TRANSCRIPT_PAUSE_BREAK_SEC = 2.5




