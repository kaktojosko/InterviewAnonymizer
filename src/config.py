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

# Parallelization and Chunking Settings
# Break video into 1-minute chunks for MAXIMUM parallel processing
CHUNK_DURATION_SECONDS = 60 
# Limit to 7 workers maximum to avoid memory bandwidth bottleneck,
# while leaving at least 2 cores for OS and UI responsiveness.
MAX_WORKERS = min(7, max(1, os.cpu_count() - 2))

# Audio Settings
PITCH_SHIFT_SEMITONES = -2.0
