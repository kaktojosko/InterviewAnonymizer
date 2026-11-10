import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.orchestrator import Orchestrator
from src.io_manager import IOManager
from src.config import (
    ASR_MODEL_PROFILES,
    COMPUTE_DEVICE_OPTIONS,
    CPU_COUNT,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    MAX_ALLOWED_WORKERS,
    MAX_WORKERS,
    MODELS_DIR,
    SPEAKER_COUNT_OPTIONS,
    TMP_BASE_DIR,
    TRANSCRIPTION_ENABLED_DEFAULT,
)
from src.transcriber import probe_cuda_available

app = FastAPI(title="Interview Anonymizer", version="4.1.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory status store
tasks = {}

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(TMP_BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def cleanup_tmp_on_start():
    tmp_root = os.path.abspath(TMP_BASE_DIR)
    if os.path.basename(tmp_root) != "tmp" or not os.path.isdir(tmp_root):
        return

    for entry in os.listdir(tmp_root):
        path = os.path.join(tmp_root, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass

cleanup_tmp_on_start()

def parse_worker_count(raw_workers: str | None) -> int:
    raw_workers = (raw_workers or "auto").strip().lower()
    if raw_workers == "" or raw_workers == "auto":
        return MAX_WORKERS

    try:
        workers = int(raw_workers)
    except ValueError:
        raise HTTPException(status_code=400, detail="Workers must be a number or auto.")

    if workers < 1 or workers > MAX_ALLOWED_WORKERS:
        raise HTTPException(status_code=400, detail=f"Workers must be between 1 and {MAX_ALLOWED_WORKERS}.")

    return workers

def parse_bool_form(raw_val: str | bool | None, default: bool = TRANSCRIPTION_ENABLED_DEFAULT) -> bool:
    if raw_val is None:
        return default
    if isinstance(raw_val, bool):
        return raw_val
    normalized = str(raw_val).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise HTTPException(status_code=400, detail="Invalid boolean value for transcribe.")

def parse_asr_model(raw_model: str | None) -> str:
    model_id = (raw_model or DEFAULT_ASR_MODEL).strip().lower()
    if model_id not in ASR_MODEL_PROFILES:
        valid = ", ".join(ASR_MODEL_PROFILES.keys())
        raise HTTPException(status_code=400, detail=f"Unsupported ASR model '{model_id}'. Valid options: {valid}.")
    return model_id

def parse_speaker_count(raw_speakers: str | None) -> str:
    mode = (raw_speakers or "auto").strip().lower()
    if mode not in SPEAKER_COUNT_OPTIONS:
        valid = ", ".join(SPEAKER_COUNT_OPTIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported speaker count '{mode}'. Valid options: {valid}.")
    return mode

def parse_compute_device(raw_device: str | None) -> str:
    device = (raw_device or DEFAULT_COMPUTE_DEVICE).strip().lower()
    if device not in COMPUTE_DEVICE_OPTIONS:
        valid = ", ".join(COMPUTE_DEVICE_OPTIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported compute device '{device}'. Valid options: {valid}.")
    return device

def background_process(
    task_id: str,
    input_path: str,
    output_path: str,
    worker_count: int,
    transcribe_enabled: bool,
    transcript_path: str | None,
    asr_model: str,
    speakers: str,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
):
    init_logs = ["Initializing...", f"Worker count: {worker_count}"]
    if transcribe_enabled:
        init_logs.append(
            f"Auto-transcription: ON (model={asr_model}, device={compute_device}, speakers={speakers})"
        )
    else:
        init_logs.append("Auto-transcription: OFF")

    tasks[task_id] = {
        "status": "processing",
        "message": "Initializing...",
        "logs": init_logs,
        "worker_count": worker_count,
        "transcribe_enabled": transcribe_enabled,
        "transcript_ready": False,
        "asr_model": asr_model,
        "speakers": speakers,
        "compute_device": compute_device,
    }
    
    def status_callback(msg: str):
        tasks[task_id]["message"] = msg
        tasks[task_id]["logs"].append(msg)
        
    io_manager = IOManager()
    job_dir = io_manager.create_job_dir()
    
    try:
        result = Orchestrator.process_video_job(
            input_path=input_path,
            output_path=output_path,
            job_dir=job_dir,
            status_cb=status_callback,
            worker_count=worker_count,
            transcribe=transcribe_enabled,
            transcript_path=transcript_path,
            asr_model=asr_model,
            speaker_count=speakers,
            compute_device=compute_device,
        )
        tasks[task_id]["transcript_ready"] = bool(result.get("transcript_ready"))
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Done!"
        tasks[task_id]["logs"].append("Done!")
    except Exception as e:
        print(f"Error in background_process: {e}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = str(e)
        tasks[task_id]["logs"].append(str(e))
    finally:
        io_manager.cleanup_job_dir(job_dir)
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/config")
async def get_config():
    return {
        "version": "4.1.0",
        "cpu_count": CPU_COUNT,
        "default_workers": MAX_WORKERS,
        "max_workers": MAX_ALLOWED_WORKERS,
        "worker_options": ["1", "2", "4", "6", "12", "16", "custom"],
        "transcription_default": TRANSCRIPTION_ENABLED_DEFAULT,
        "default_asr_model": DEFAULT_ASR_MODEL,
        "asr_models": [
            {"id": p["id"], "label": p["label"], "short_name": p["short_name"]}
            for p in ASR_MODEL_PROFILES.values()
        ],
        "speaker_options": SPEAKER_COUNT_OPTIONS,
        "cuda_available": probe_cuda_available(),
        "default_compute_device": DEFAULT_COMPUTE_DEVICE,
        "compute_devices": [
            {"id": "cpu", "label": "CPU"},
            {"id": "cuda", "label": "GPU (CUDA)"},
        ],
    }

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    workers: str = Form("4"),
    transcribe: str = Form("true" if TRANSCRIPTION_ENABLED_DEFAULT else "false"),
    asr_model: str = Form(DEFAULT_ASR_MODEL),
    speakers: str = Form("auto"),
    compute_device: str = Form(DEFAULT_COMPUTE_DEVICE),
):
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    worker_count = parse_worker_count(workers)
    transcribe_enabled = parse_bool_form(transcribe, default=TRANSCRIPTION_ENABLED_DEFAULT)
    asr_model_id = parse_asr_model(asr_model)
    speaker_mode = parse_speaker_count(speakers)
    device_mode = parse_compute_device(compute_device)
        
    task_id = str(uuid.uuid4())
    input_path = os.path.join("uploads", f"{task_id}.mp4")
    output_path = os.path.join("output", f"{task_id}_anonymized.mp4")
    transcript_path = (
        os.path.join("output", f"{task_id}_transcript.txt") if transcribe_enabled else None
    )
    
    with open(input_path, "wb") as buffer:
        while True:
            chunk = await file.read(4 * 1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
        
    tasks[task_id] = {
        "status": "queued",
        "message": "Waiting in queue...",
        "logs": ["Waiting in queue...", f"Worker count: {worker_count}"],
        "worker_count": worker_count,
        "transcribe_enabled": transcribe_enabled,
        "transcript_ready": False,
        "asr_model": asr_model_id,
        "speakers": speaker_mode,
        "compute_device": device_mode,
    }
    
    background_tasks.add_task(
        background_process,
        task_id,
        input_path,
        output_path,
        worker_count,
        transcribe_enabled,
        transcript_path,
        asr_model_id,
        speaker_mode,
        device_mode,
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Waiting in queue...",
        "worker_count": worker_count,
        "transcribe_enabled": transcribe_enabled,
        "asr_model": asr_model_id,
        "speakers": speaker_mode,
        "compute_device": device_mode,
    }

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **tasks[task_id]}

@app.get("/api/download/{task_id}")
async def download_video(task_id: str):
    if task_id not in tasks or tasks[task_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="File not ready or task failed")
        
    output_path = os.path.join("output", f"{task_id}_anonymized.mp4")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        output_path, 
        media_type="video/mp4", 
        filename=f"anonymized_{task_id}.mp4"
    )

@app.get("/api/download-transcript/{task_id}")
async def download_transcript(task_id: str):
    if task_id not in tasks or tasks[task_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="Transcript not ready or task failed")

    if not tasks[task_id].get("transcribe_enabled") or not tasks[task_id].get("transcript_ready"):
        raise HTTPException(status_code=404, detail="Transcript was disabled or not generated for this task")

    transcript_path = os.path.join("output", f"{task_id}_transcript.txt")
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Transcript file not found")

    return FileResponse(
        transcript_path,
        media_type="text/plain; charset=utf-8",
        filename=f"transcript_{task_id}.txt",
    )

if __name__ == "__main__":
    import uvicorn
    # Important on Windows for multiprocessing
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
