FROM python:3.10-slim

# Install system dependencies (ffmpeg and audio requirements)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime files. Runtime data directories are bind-mounted by compose.
COPY app.py main.py ./
COPY src ./src
COPY static ./static

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
