import os
import multiprocessing

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_BASE_DIR = os.path.join(BASE_DIR, "tmp")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
PRELOADED_MODELS_DIR = os.environ.get("PRELOADED_MODELS_DIR", "/opt/preloaded_models")

# Video Processing Settings
FACE_PADDING_PERCENT_X = 0.35
FACE_PADDING_PERCENT_Y = 0.35
YUNET_MIN_CONFIDENCE = 0.3

# Detect faces every N frames. In between, use interpolation.
# Higher = faster, but less accurate tracking for fast movements.
DETECT_EVERY_N_FRAMES = 28

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

# Local Auto-Transcription & Speaker Diarization Settings
TRANSCRIPTION_ENABLED_DEFAULT = os.environ.get("TRANSCRIPTION_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")
DEFAULT_ASR_MODEL = os.environ.get("DEFAULT_ASR_MODEL", "turbo").strip().lower()

ASR_MODEL_PROFILES = {
    "turbo": {
        "id": "turbo",
        "label": "Whisper Turbo — точная (99 яз., 810 МБ)",
        "short_name": "Whisper Large v3 Turbo (INT8)",
        "engine": "faster_whisper",
        "repo_id": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "required_files": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 600_000_000,
    },
    "large-v3": {
        "id": "large-v3",
        "label": "Whisper Large v3 — макс. точность (99 яз., 3.1 ГБ)",
        "short_name": "Whisper Large v3",
        "engine": "faster_whisper",
        "repo_id": "Systran/faster-whisper-large-v3",
        "required_files": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 2_500_000_000,
    },
    "parakeet": {
        "id": "parakeet",
        "label": "Parakeet v3 — быстрая (25 яз., 600 МБ)",
        "short_name": "NVIDIA Parakeet TDT 0.6B v3 (INT8)",
        "engine": "sherpa_parakeet",
        "repo_id": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        "required_files": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
        "allow_patterns": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
        "min_total_bytes": 400_000_000,
    },
    "small": {
        "id": "small",
        "label": "Whisper Small — лёгкая (99 яз., 245 МБ)",
        "short_name": "Whisper Small (INT8)",
        "engine": "faster_whisper",
        "repo_id": "Systran/faster-whisper-small",
        "required_files": ["model.bin", "config.json", "tokenizer.json"],
        "allow_patterns": ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"],
        "min_total_bytes": 180_000_000,
    },
}

if DEFAULT_ASR_MODEL not in ASR_MODEL_PROFILES:
    DEFAULT_ASR_MODEL = "turbo"

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

SPEAKER_COUNT_OPTIONS = ["auto", "2", "3", "4+"]
# Fine-grained turn boundary threshold for initial pyannote segmentation pass
DIARIZATION_SEGMENTATION_THRESHOLD = float(os.environ.get("DIARIZATION_SEG_THRESHOLD", "0.35"))
# Cosine similarity thresholds for Anchor-Centroid voiceprint clustering (CAM++)
DIARIZATION_CENTROID_SIM_THRESHOLD = float(os.environ.get("DIARIZATION_SIM_THRESHOLD", "0.54"))
DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD = float(os.environ.get("DIARIZATION_MULTI_SIM_THRESHOLD", "0.58"))
# Backwards-compatible alias
DIARIZATION_CLUSTER_THRESHOLD = DIARIZATION_CENTROID_SIM_THRESHOLD
DIARIZATION_MERGE_GAP_SEC = 0.45
DIARIZATION_MIN_SEGMENT_SEC = 0.18
TRANSCRIPT_PAUSE_BREAK_SEC = 2.5


