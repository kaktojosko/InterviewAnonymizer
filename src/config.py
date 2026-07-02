import os
import multiprocessing

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_BASE_DIR = os.path.join(BASE_DIR, "tmp")

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
