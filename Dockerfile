FROM python:3.11-slim-bookworm

# Install system dependencies (ffmpeg and audio requirements)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optional GPU overlay (set ENABLE_GPU=1 via docker-compose.gpu.yml to enable
# CUDA acceleration for both faster-whisper AND onnx-asr / GigaAM v3 FP32)
ARG ENABLE_GPU=0

# Install Python dependencies (includes lightweight cuBLAS runtime for optional NVIDIA GPU mode)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$ENABLE_GPU" = "1" ]; then \
        pip uninstall -y onnxruntime || true; \
        pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]<1.27"; \
    fi && \
    python -c "import glob, site; paths = sorted({p for sp in site.getsitepackages() for p in glob.glob(sp + '/nvidia/*/lib')}); open('/etc/ld.so.conf.d/nvidia-pip.conf', 'w').write('\n'.join(paths) + '\n') if paths else None" && \
    ldconfig

# Copy only runtime files. Runtime data directories are bind-mounted by compose.
COPY app.py main.py ./
COPY src ./src
COPY static ./static

ENV MODELS_DIR=/app/models \
    PRELOADED_MODELS_DIR=/opt/preloaded_models

RUN mkdir -p /app/models /opt/preloaded_models

# Optional: pre-bake ASR + diarization models during build for air-gapped environments
# Example: docker build --build-arg PRELOAD_MODELS=turbo .
ARG PRELOAD_MODELS=""
RUN if [ -n "$PRELOAD_MODELS" ]; then \
        MODELS_DIR=/opt/preloaded_models python -m src.model_manager --preload "$PRELOAD_MODELS"; \
    fi

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
