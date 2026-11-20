from datetime import datetime, timezone
import os
import re
import shutil
import uuid
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from src.orchestrator import Orchestrator
from src.io_manager import IOManager
from src.config import (
    ANONYMIZE_ALLOWED_EXTENSIONS,
    APP_BUILD_ID,
    APP_VERSION,
    ASR_LANGUAGE_LABELS,
    ASR_LANGUAGE_OPTIONS,
    ASR_MODEL_PROFILES,
    ASR_PROMPT_PRESETS,
    AUDIO_CACHE_DIR,
    AUDIO_TRACK_MODE_LABELS,
    AUDIO_TRACK_MODE_OPTIONS,
    BASE_DIR,
    COMPUTE_DEVICE_OPTIONS,
    CPU_COUNT,
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_MODEL,
    DEFAULT_AUDIO_TRACK_MODE,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_DEEPGRAM_API_KEY,
    DEFAULT_DIARIZATION_SENSITIVITY,
    DEFAULT_JOB_MODE,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DENOISE_ENABLED_DEFAULT,
    DIARIZATION_SENSITIVITY_OPTIONS,
    DIARIZATION_SENSITIVITY_PROFILES,
    JOB_MODE_OPTIONS,
    LLM_POLISH_ENABLED_DEFAULT,
    LLM_PROVIDER_PRESETS,
    MAX_ALLOWED_WORKERS,
    MAX_CACHED_AUDIO_FILES,
    MAX_WORKERS,
    MODELS_DIR,
    SPEAKER_COUNT_OPTIONS,
    TMP_BASE_DIR,
    TRANSCRIBE_ONLY_ALLOWED_EXTENSIONS,
    TRANSCRIPTION_ENABLED_DEFAULT,
    _resolve_build_id,
)
from src.cancellation import CancellationToken, TaskCancelledError
from src.transcriber import probe_cuda_available, probe_onnx_cuda_available

app = FastAPI(title="Interview Anonymizer", version=APP_VERSION)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if (
        path == "/"
        or path == "/api/config"
        or path.startswith("/api/recordings")
        or path.startswith("/api/status/")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


# In-memory status store & cancellation tokens
tasks = {}
cancel_tokens: dict[str, CancellationToken] = {}

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(TMP_BASE_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
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
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


cleanup_tmp_on_start()


def _validate_task_id(task_id: str) -> str:
    if not task_id or not _TASK_ID_RE.match(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id format.")
    return task_id


def _audio_cache_path(task_id: str) -> str:
    return os.path.join(AUDIO_CACHE_DIR, f"{task_id}.wav")


def _audio_cache_track2_path(task_id: str) -> str:
    return os.path.join(AUDIO_CACHE_DIR, f"{task_id}.track2.wav")


def _raw_transcript_path(task_id: str) -> str:
    return os.path.join("output", f"{task_id}_transcript_raw.txt")


def _prune_audio_cache(keep_task_id: str | None = None) -> None:
    if not os.path.isdir(AUDIO_CACHE_DIR):
        return
    # Group cached audio files by task_id so track1 + track2 are pruned together
    by_task: dict[str, tuple[float, list[str]]] = {}
    for fname in os.listdir(AUDIO_CACHE_DIR):
        if not fname.endswith(".wav"):
            continue
        full = os.path.join(AUDIO_CACHE_DIR, fname)
        if not os.path.isfile(full):
            continue
        if fname.endswith(".track2.wav"):
            tid = fname[: -len(".track2.wav")]
        else:
            tid = fname[: -len(".wav")]
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        prev_mtime, paths = by_task.get(tid, (0.0, []))
        paths.append(full)
        by_task[tid] = (max(prev_mtime, mtime), paths)

    if len(by_task) <= MAX_CACHED_AUDIO_FILES:
        return

    ordered = sorted(by_task.items(), key=lambda item: item[1][0], reverse=True)
    kept = 0
    for tid, (_, paths) in ordered:
        if keep_task_id and tid == keep_task_id:
            kept += 1
            continue
        if kept < MAX_CACHED_AUDIO_FILES:
            kept += 1
            continue
        for p in paths:
            _safe_remove(p)


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


def parse_bool_form(
    raw_val: str | bool | None,
    default: bool = TRANSCRIPTION_ENABLED_DEFAULT,
    field_name: str = "transcribe",
) -> bool:
    if raw_val is None:
        return default
    if isinstance(raw_val, bool):
        return raw_val
    normalized = str(raw_val).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise HTTPException(status_code=400, detail=f"Invalid boolean value for {field_name}.")


def parse_asr_model(raw_model: str | None) -> str:
    model_id = (raw_model or DEFAULT_ASR_MODEL).strip().lower()
    if model_id not in ASR_MODEL_PROFILES:
        valid = ", ".join(ASR_MODEL_PROFILES.keys())
        raise HTTPException(status_code=400, detail=f"Unsupported ASR model '{model_id}'. Valid options: {valid}.")
    return model_id


def parse_language(raw_language: str | None) -> str:
    lang = (raw_language or DEFAULT_ASR_LANGUAGE).strip().lower()
    if lang not in ASR_LANGUAGE_OPTIONS:
        valid = ", ".join(ASR_LANGUAGE_OPTIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'. Valid options: {valid}.")
    return lang


def parse_job_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or DEFAULT_JOB_MODE).strip().lower()
    if mode not in JOB_MODE_OPTIONS:
        valid = ", ".join(JOB_MODE_OPTIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported mode '{mode}'. Valid options: {valid}.")
    return mode


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


def parse_diarization_sensitivity(raw_sens: str | None) -> str:
    sens = (raw_sens or DEFAULT_DIARIZATION_SENSITIVITY).strip().lower()
    if sens not in DIARIZATION_SENSITIVITY_OPTIONS:
        valid = ", ".join(DIARIZATION_SENSITIVITY_OPTIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported sensitivity '{sens}'. Valid options: {valid}.")
    return sens


def parse_audio_tracks(raw_tracks: str | None) -> str:
    mode = (raw_tracks or DEFAULT_AUDIO_TRACK_MODE).strip().lower()
    if mode not in AUDIO_TRACK_MODE_OPTIONS:
        valid = ", ".join(AUDIO_TRACK_MODE_OPTIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported audio_tracks '{mode}'. Valid options: {valid}.")
    return mode


def parse_custom_prompt(raw_prompt: str | None) -> str:
    if not raw_prompt:
        return ""
    return " ".join(str(raw_prompt).split())[:600]


def parse_llm_base_url(raw_url: str | None) -> str:
    cleaned = (raw_url or "").strip()[:300]
    if not cleaned:
        return DEFAULT_LLM_BASE_URL
    lower = cleaned.lower()
    if not (lower.startswith("http://") or lower.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="llm_base_url must start with http:// or https://.",
        )
    return cleaned


def parse_llm_model(raw_model: str | None) -> str:
    cleaned = (raw_model or "").strip()[:120]
    return cleaned or DEFAULT_LLM_MODEL


def parse_llm_api_key(raw_key: str | None) -> str:
    cleaned = (raw_key or "").strip()[:300]
    return cleaned or DEFAULT_LLM_API_KEY


def parse_asr_api_key(raw_key: str | None) -> str:
    cleaned = (raw_key or "").strip()[:300]
    return cleaned or DEFAULT_DEEPGRAM_API_KEY


def _safe_remove(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


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
    language: str = DEFAULT_ASR_LANGUAGE,
    original_filename: str = "",
    custom_prompt: str = "",
    diarization_sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
    audio_tracks: str = DEFAULT_AUDIO_TRACK_MODE,
    denoise: bool = DENOISE_ENABLED_DEFAULT,
    llm_polish: bool = LLM_POLISH_ENABLED_DEFAULT,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_api_key: str = "",
    asr_api_key: str = "",
):
    cancel_token = cancel_tokens.get(task_id)
    if cancel_token is None:
        cancel_token = CancellationToken()
        cancel_tokens[task_id] = cancel_token

    if cancel_token.is_cancelled():
        cancel_tokens.pop(task_id, None)
        _safe_remove(input_path)
        return

    init_logs = ["Initializing...", f"Worker count: {worker_count}"]
    if transcribe_enabled:
        extras = []
        if denoise:
            extras.append("denoise=GTCRN")
        if llm_polish:
            extras.append(f"llm={llm_model}")
        extra_str = f", {', '.join(extras)}" if extras else ""
        init_logs.append(
            f"Auto-transcription: ON (model={asr_model}, lang={language}, device={compute_device}, "
            f"speakers={speakers}, sensitivity={diarization_sensitivity}, tracks={audio_tracks}{extra_str})"
        )
    else:
        init_logs.append("Auto-transcription: OFF")

    tasks[task_id] = {
        "status": "processing",
        "mode": "anonymize",
        "message": "Initializing...",
        "logs": init_logs,
        "worker_count": worker_count,
        "transcribe_enabled": transcribe_enabled,
        "video_ready": False,
        "transcript_ready": False,
        "has_original_audio": False,
        "can_transcribe": False,
        "is_post_transcribe": False,
        "asr_model": asr_model,
        "language": language,
        "speakers": speakers,
        "compute_device": compute_device,
        "custom_prompt": custom_prompt,
        "diarization_sensitivity": diarization_sensitivity,
        "audio_tracks": audio_tracks,
        "denoise": denoise,
        "llm_polish": llm_polish,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "original_filename": original_filename or tasks.get(task_id, {}).get("original_filename", ""),
    }

    def status_callback(msg: str):
        if cancel_token.is_cancelled():
            return
        entry = tasks.get(task_id)
        if entry is None or entry.get("status") == "cancelled":
            return
        entry["message"] = msg
        entry["logs"].append(msg)

    io_manager = IOManager()
    job_dir = io_manager.create_job_dir()
    cached_audio_path = _audio_cache_path(task_id)
    cached_track2_path = _audio_cache_track2_path(task_id)
    eff_api_key = (llm_api_key or "").strip() or DEFAULT_LLM_API_KEY
    eff_asr_api_key = (asr_api_key or "").strip() or DEFAULT_DEEPGRAM_API_KEY

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
            cancel_token=cancel_token,
            language=language,
            cached_audio_path=cached_audio_path,
            cached_track2_path=cached_track2_path,
            custom_prompt=custom_prompt,
            diarization_sensitivity=diarization_sensitivity,
            audio_tracks=audio_tracks,
            denoise=denoise,
            llm_polish=llm_polish,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=eff_api_key,
            asr_api_key=eff_asr_api_key,
        )
        cancel_token.check()
        video_ready = bool(result.get("video_ready"))
        audio_cached = bool(result.get("audio_cached"))
        tasks[task_id]["video_ready"] = video_ready
        tasks[task_id]["transcript_ready"] = bool(result.get("transcript_ready"))
        tasks[task_id]["raw_transcript_ready"] = os.path.isfile(_raw_transcript_path(task_id))
        tasks[task_id]["has_original_audio"] = audio_cached
        tasks[task_id]["can_transcribe"] = audio_cached or video_ready
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Done!"
        tasks[task_id]["logs"].append("Done!")
        if audio_cached:
            _prune_audio_cache(keep_task_id=task_id)
    except TaskCancelledError:
        entry = tasks.get(task_id)
        if entry is not None:
            entry["status"] = "cancelled"
            entry["message"] = "Обработка прервана."
            entry["video_ready"] = False
            entry["transcript_ready"] = False
            entry["raw_transcript_ready"] = False
            entry["has_original_audio"] = False
            entry["can_transcribe"] = False
            cancel_note = "Обработка прервана пользователем."
            if not entry["logs"] or entry["logs"][-1] != cancel_note:
                entry["logs"].append(cancel_note)
    except Exception as e:
        if cancel_token.is_cancelled():
            entry = tasks.get(task_id)
            if entry is not None:
                entry["status"] = "cancelled"
                entry["message"] = "Обработка прервана."
                entry["video_ready"] = False
                entry["transcript_ready"] = False
                entry["raw_transcript_ready"] = False
                entry["has_original_audio"] = False
                entry["can_transcribe"] = False
        else:
            print(f"Error in background_process: {e}")
            tasks[task_id]["status"] = "error"
            tasks[task_id]["message"] = str(e)
            tasks[task_id]["logs"].append(str(e))
    finally:
        cancel_tokens.pop(task_id, None)
        io_manager.cleanup_job_dir(job_dir)
        _safe_remove(input_path)
        if cancel_token.is_cancelled() or tasks.get(task_id, {}).get("status") == "cancelled":
            _safe_remove(output_path)
            _safe_remove(transcript_path)
            _safe_remove(_raw_transcript_path(task_id))
            _safe_remove(cached_audio_path)
            _safe_remove(cached_track2_path)


def background_transcribe_only(
    task_id: str,
    input_path: str,
    transcript_path: str,
    worker_count: int,
    asr_model: str,
    language: str,
    speakers: str,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    is_post_transcribe: bool = False,
    input_is_wav_16k: bool = False,
    delete_input_after: bool = True,
    source_note: str | None = None,
    original_filename: str = "",
    custom_prompt: str = "",
    diarization_sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
    audio_tracks: str = DEFAULT_AUDIO_TRACK_MODE,
    input_track2_path: str | None = None,
    denoise: bool = DENOISE_ENABLED_DEFAULT,
    llm_polish: bool = LLM_POLISH_ENABLED_DEFAULT,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_api_key: str = "",
    asr_api_key: str = "",
):
    cancel_token = cancel_tokens.get(task_id)
    if cancel_token is None:
        cancel_token = CancellationToken()
        cancel_tokens[task_id] = cancel_token

    prev_entry = tasks.get(task_id, {})
    prev_video_ready = bool(prev_entry.get("video_ready")) if is_post_transcribe else False
    prev_transcript_ready = bool(prev_entry.get("transcript_ready")) if is_post_transcribe else False
    prev_mode = prev_entry.get("mode", "anonymize" if prev_video_ready else "transcribe_only")
    saved_filename = original_filename or prev_entry.get("original_filename", "")

    if cancel_token.is_cancelled():
        cancel_tokens.pop(task_id, None)
        if delete_input_after:
            _safe_remove(input_path)
        return

    mode_label = "Отдельная транскрибация записи" if is_post_transcribe else "Режим «Только транскрибация»"
    extras = []
    if denoise:
        extras.append("denoise=GTCRN")
    if llm_polish:
        extras.append(f"llm={llm_model}")
    extra_str = f", {', '.join(extras)}" if extras else ""
    init_logs = [
        f"{mode_label}...",
        (
            f"Параметры: model={asr_model}, lang={language}, device={compute_device}, "
            f"speakers={speakers}, sensitivity={diarization_sensitivity}, tracks={audio_tracks}, threads={worker_count}{extra_str}"
        ),
    ]

    tasks[task_id] = {
        "status": "processing",
        "mode": prev_mode if is_post_transcribe else "transcribe_only",
        "message": "Подготовка к транскрибации...",
        "logs": init_logs,
        "worker_count": worker_count,
        "transcribe_enabled": True,
        "video_ready": prev_video_ready,
        "transcript_ready": prev_transcript_ready,
        "has_original_audio": os.path.isfile(_audio_cache_path(task_id)),
        "can_transcribe": True,
        "is_post_transcribe": is_post_transcribe,
        "asr_model": asr_model,
        "language": language,
        "speakers": speakers,
        "compute_device": compute_device,
        "custom_prompt": custom_prompt,
        "diarization_sensitivity": diarization_sensitivity,
        "audio_tracks": audio_tracks,
        "denoise": denoise,
        "llm_polish": llm_polish,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "original_filename": saved_filename,
    }

    def status_callback(msg: str):
        if cancel_token.is_cancelled():
            return
        entry = tasks.get(task_id)
        if entry is None or entry.get("status") == "cancelled":
            return
        entry["message"] = msg
        entry["logs"].append(msg)

    io_manager = IOManager()
    job_dir = io_manager.create_job_dir()
    cached_audio_path = _audio_cache_path(task_id)
    cached_track2_path = _audio_cache_track2_path(task_id)
    # Cache original 16 kHz audio for fresh transcribe_only uploads so user can
    # re-transcribe with another model without re-uploading the media file.
    cache_target = None if (is_post_transcribe and not input_is_wav_16k) else cached_audio_path
    cache_track2_target = None if (is_post_transcribe and not input_is_wav_16k) else cached_track2_path
    eff_api_key = (llm_api_key or "").strip() or DEFAULT_LLM_API_KEY
    eff_asr_api_key = (asr_api_key or "").strip() or DEFAULT_DEEPGRAM_API_KEY

    try:
        result = Orchestrator.process_transcription_only_job(
            input_path=input_path,
            transcript_path=transcript_path,
            job_dir=job_dir,
            status_cb=status_callback,
            worker_count=worker_count,
            asr_model=asr_model,
            language=language,
            speaker_count=speakers,
            compute_device=compute_device,
            cancel_token=cancel_token,
            cached_audio_path=cache_target,
            cached_track2_path=cache_track2_target,
            input_is_wav_16k=input_is_wav_16k,
            input_track2_path=input_track2_path,
            source_note=source_note,
            custom_prompt=custom_prompt,
            diarization_sensitivity=diarization_sensitivity,
            audio_tracks=audio_tracks,
            denoise=denoise,
            llm_polish=llm_polish,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=eff_api_key,
            asr_api_key=eff_asr_api_key,
        )
        cancel_token.check()
        has_orig_audio = os.path.isfile(cached_audio_path)
        tasks[task_id]["video_ready"] = prev_video_ready
        tasks[task_id]["transcript_ready"] = bool(result.get("transcript_ready"))
        tasks[task_id]["raw_transcript_ready"] = os.path.isfile(_raw_transcript_path(task_id))
        tasks[task_id]["has_original_audio"] = has_orig_audio
        tasks[task_id]["can_transcribe"] = has_orig_audio or prev_video_ready
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Done!"
        tasks[task_id]["logs"].append("Done!")
        if has_orig_audio:
            _prune_audio_cache(keep_task_id=task_id)
    except TaskCancelledError:
        entry = tasks.get(task_id)
        if entry is not None:
            entry["status"] = "cancelled"
            entry["message"] = "Транскрибация прервана."
            entry["video_ready"] = prev_video_ready
            entry["transcript_ready"] = prev_transcript_ready and os.path.isfile(transcript_path)
            entry["raw_transcript_ready"] = prev_transcript_ready and os.path.isfile(_raw_transcript_path(task_id))
            entry["has_original_audio"] = os.path.isfile(cached_audio_path)
            entry["can_transcribe"] = entry["has_original_audio"] or prev_video_ready
            cancel_note = "Обработка прервана пользователем."
            if not entry["logs"] or entry["logs"][-1] != cancel_note:
                entry["logs"].append(cancel_note)
    except Exception as e:
        if cancel_token.is_cancelled():
            entry = tasks.get(task_id)
            if entry is not None:
                entry["status"] = "cancelled"
                entry["message"] = "Транскрибация прервана."
                entry["video_ready"] = prev_video_ready
                entry["transcript_ready"] = prev_transcript_ready and os.path.isfile(transcript_path)
                entry["raw_transcript_ready"] = prev_transcript_ready and os.path.isfile(_raw_transcript_path(task_id))
                entry["has_original_audio"] = os.path.isfile(cached_audio_path)
                entry["can_transcribe"] = entry["has_original_audio"] or prev_video_ready
        else:
            print(f"Error in background_transcribe_only: {e}")
            tasks[task_id]["status"] = "error"
            tasks[task_id]["message"] = str(e)
            tasks[task_id]["logs"].append(str(e))
            tasks[task_id]["video_ready"] = prev_video_ready
            tasks[task_id]["transcript_ready"] = prev_transcript_ready and os.path.isfile(transcript_path)
            tasks[task_id]["raw_transcript_ready"] = prev_transcript_ready and os.path.isfile(_raw_transcript_path(task_id))
            tasks[task_id]["has_original_audio"] = os.path.isfile(cached_audio_path)
            tasks[task_id]["can_transcribe"] = tasks[task_id]["has_original_audio"] or prev_video_ready
    finally:
        cancel_tokens.pop(task_id, None)
        io_manager.cleanup_job_dir(job_dir)
        if delete_input_after:
            _safe_remove(input_path)
        if not is_post_transcribe and (
            cancel_token.is_cancelled() or tasks.get(task_id, {}).get("status") == "cancelled"
        ):
            _safe_remove(transcript_path)
            _safe_remove(_raw_transcript_path(task_id))
            _safe_remove(cached_audio_path)
            _safe_remove(cached_track2_path)


_INDEX_HTML_PATH = os.path.join(BASE_DIR, "static", "index.html")
_CSS_HREF_RE = re.compile(r'/static/style\.css(?:\?v=[^"\']+)?')
_JS_SRC_RE = re.compile(r'/static/script\.js(?:\?v=[^"\']+)?')
_FOOTER_VER_RE = re.compile(r'(<footer[^>]*\bid="appVersionFooter"[^>]*>)(.*?)(</footer>)', re.DOTALL)


def _current_build_id() -> str:
    return _resolve_build_id(BASE_DIR, APP_VERSION) or APP_BUILD_ID


def _render_index_html() -> str:
    with open(_INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    build_id = _current_build_id()
    html = _CSS_HREF_RE.sub(f"/static/style.css?v={build_id}", html)
    html = _JS_SRC_RE.sub(f"/static/script.js?v={build_id}", html)
    parts = build_id.split("-")
    short_sha = parts[1] if len(parts) >= 2 and parts[1] else ""
    footer_text = (
        f"v{APP_VERSION} (Stable Edition • {short_sha})"
        if short_sha
        else f"v{APP_VERSION} (Stable Edition)"
    )
    html = _FOOTER_VER_RE.sub(rf"\g<1>\n            {footer_text}\n        \g<3>", html)
    return html


@app.get("/")
async def root():
    return HTMLResponse(
        content=_render_index_html(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/config")
async def get_config():
    ct2_cuda_ok = probe_cuda_available()
    onnx_cuda_ok = probe_onnx_cuda_available()
    return {
        "version": APP_VERSION,
        "build_id": _current_build_id(),
        "cpu_count": CPU_COUNT,
        "default_workers": MAX_WORKERS,
        "max_workers": MAX_ALLOWED_WORKERS,
        "worker_options": ["1", "2", "4", "6", "12", "16", "custom"],
        "default_job_mode": DEFAULT_JOB_MODE,
        "job_modes": [
            {"id": "anonymize", "label": "Анонимизация"},
            {"id": "transcribe_only", "label": "Только транскрибация"},
        ],
        "transcribe_only_extensions": list(TRANSCRIBE_ONLY_ALLOWED_EXTENSIONS),
        "transcription_default": TRANSCRIPTION_ENABLED_DEFAULT,
        "default_asr_model": DEFAULT_ASR_MODEL,
        "asr_models": [
            {
                "id": p["id"],
                "label": p["label"],
                "short_name": p["short_name"],
                "engine": p["engine"],
                "supports_cuda": bool(p.get("supports_cuda", False)),
                "fixed_language": p.get("fixed_language"),
                "is_cloud": bool(p.get("is_cloud", False)),
                "requires_api_key": bool(p.get("requires_api_key", False)),
                "has_native_diarization": bool(p.get("has_native_diarization", False)),
            }
            for p in ASR_MODEL_PROFILES.values()
        ],
        "default_language": DEFAULT_ASR_LANGUAGE,
        "languages": [
            {"id": code, "label": ASR_LANGUAGE_LABELS.get(code, code)}
            for code in ASR_LANGUAGE_OPTIONS
        ],
        "speaker_options": SPEAKER_COUNT_OPTIONS,
        "default_diarization_sensitivity": DEFAULT_DIARIZATION_SENSITIVITY,
        "diarization_sensitivities": [
            {"id": p["id"], "label": p["label"]}
            for p in DIARIZATION_SENSITIVITY_PROFILES.values()
        ],
        "default_audio_track_mode": DEFAULT_AUDIO_TRACK_MODE,
        "audio_track_modes": [
            {"id": code, "label": AUDIO_TRACK_MODE_LABELS.get(code, code)}
            for code in AUDIO_TRACK_MODE_OPTIONS
        ],
        "prompt_presets": [
            {"id": p["id"], "label": p["label"], "prompt": p["prompt"]}
            for p in ASR_PROMPT_PRESETS.values()
        ],
        "cuda_available": bool(ct2_cuda_ok or onnx_cuda_ok),
        "ct2_cuda_available": ct2_cuda_ok,
        "onnx_cuda_available": onnx_cuda_ok,
        "default_compute_device": DEFAULT_COMPUTE_DEVICE,
        "compute_devices": [
            {"id": "cpu", "label": "CPU"},
            {"id": "cuda", "label": "GPU (CUDA)"},
        ],
        "denoise_default": DENOISE_ENABLED_DEFAULT,
        "llm_polish_default": LLM_POLISH_ENABLED_DEFAULT,
        "llm_defaults": {
            "base_url": DEFAULT_LLM_BASE_URL,
            "model": DEFAULT_LLM_MODEL,
        },
        "llm_presets": [dict(p) for p in LLM_PROVIDER_PRESETS.values()],
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    workers: str = Form("4"),
    mode: str = Form(DEFAULT_JOB_MODE),
    transcribe: str = Form("true" if TRANSCRIPTION_ENABLED_DEFAULT else "false"),
    asr_model: str = Form(DEFAULT_ASR_MODEL),
    language: str = Form(DEFAULT_ASR_LANGUAGE),
    speakers: str = Form("auto"),
    compute_device: str = Form(DEFAULT_COMPUTE_DEVICE),
    custom_prompt: str = Form(""),
    diarization_sensitivity: str = Form(DEFAULT_DIARIZATION_SENSITIVITY),
    audio_tracks: str = Form(DEFAULT_AUDIO_TRACK_MODE),
    denoise: str = Form("true" if DENOISE_ENABLED_DEFAULT else "false"),
    llm_polish: str = Form("true" if LLM_POLISH_ENABLED_DEFAULT else "false"),
    llm_base_url: str = Form(DEFAULT_LLM_BASE_URL),
    llm_model: str = Form(DEFAULT_LLM_MODEL),
    llm_api_key: str = Form(""),
    asr_api_key: str = Form(""),
):
    job_mode = parse_job_mode(mode)
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())

    if job_mode == "transcribe_only":
        if not filename or ext not in TRANSCRIBE_ONLY_ALLOWED_EXTENSIONS:
            allowed = ", ".join(TRANSCRIBE_ONLY_ALLOWED_EXTENSIONS)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported media format for transcription-only mode. Allowed: {allowed}.",
            )
    else:
        if not filename or ext not in ANONYMIZE_ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    worker_count = parse_worker_count(workers)
    transcribe_enabled = True if job_mode == "transcribe_only" else parse_bool_form(
        transcribe, default=TRANSCRIPTION_ENABLED_DEFAULT, field_name="transcribe"
    )
    asr_model_id = parse_asr_model(asr_model)
    language_id = parse_language(language)
    speaker_mode = parse_speaker_count(speakers)
    device_mode = parse_compute_device(compute_device)
    prompt_clean = parse_custom_prompt(custom_prompt)
    sensitivity_mode = parse_diarization_sensitivity(diarization_sensitivity)
    track_mode = parse_audio_tracks(audio_tracks)
    denoise_enabled = parse_bool_form(denoise, default=DENOISE_ENABLED_DEFAULT, field_name="denoise")
    llm_polish_enabled = parse_bool_form(
        llm_polish, default=LLM_POLISH_ENABLED_DEFAULT, field_name="llm_polish"
    )
    llm_url_clean = parse_llm_base_url(llm_base_url)
    llm_model_clean = parse_llm_model(llm_model)
    llm_key_clean = parse_llm_api_key(llm_api_key)
    asr_key_clean = parse_asr_api_key(asr_api_key)

    task_id = str(uuid.uuid4())
    input_path = os.path.join("uploads", f"{task_id}{ext}")
    output_path = os.path.join("output", f"{task_id}_anonymized.mp4") if job_mode == "anonymize" else None
    transcript_path = (
        os.path.join("output", f"{task_id}_transcript.txt") if transcribe_enabled else None
    )

    try:
        with open(input_path, "wb") as buffer:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except BaseException:
        _safe_remove(input_path)
        raise

    cancel_tokens[task_id] = CancellationToken()
    tasks[task_id] = {
        "status": "queued",
        "mode": job_mode,
        "message": "Waiting in queue...",
        "logs": ["Waiting in queue...", f"Worker count: {worker_count}"],
        "worker_count": worker_count,
        "transcribe_enabled": transcribe_enabled,
        "video_ready": False,
        "transcript_ready": False,
        "has_original_audio": False,
        "can_transcribe": False,
        "is_post_transcribe": False,
        "asr_model": asr_model_id,
        "language": language_id,
        "speakers": speaker_mode,
        "compute_device": device_mode,
        "custom_prompt": prompt_clean,
        "diarization_sensitivity": sensitivity_mode,
        "audio_tracks": track_mode,
        "denoise": denoise_enabled,
        "llm_polish": llm_polish_enabled,
        "llm_base_url": llm_url_clean,
        "llm_model": llm_model_clean,
        "original_filename": os.path.basename(filename),
    }

    if job_mode == "transcribe_only":
        background_tasks.add_task(
            background_transcribe_only,
            task_id,
            input_path,
            transcript_path,
            worker_count,
            asr_model_id,
            language_id,
            speaker_mode,
            device_mode,
            False,
            False,
            True,
            None,
            os.path.basename(filename),
            prompt_clean,
            sensitivity_mode,
            track_mode,
            None,
            denoise_enabled,
            llm_polish_enabled,
            llm_url_clean,
            llm_model_clean,
            llm_key_clean,
            asr_key_clean,
        )
    else:
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
            language_id,
            os.path.basename(filename),
            prompt_clean,
            sensitivity_mode,
            track_mode,
            denoise_enabled,
            llm_polish_enabled,
            llm_url_clean,
            llm_model_clean,
            llm_key_clean,
            asr_key_clean,
        )

    return {
        "task_id": task_id,
        "status": "queued",
        "mode": job_mode,
        "message": "Waiting in queue...",
        "worker_count": worker_count,
        "transcribe_enabled": transcribe_enabled,
        "asr_model": asr_model_id,
        "language": language_id,
        "speakers": speaker_mode,
        "compute_device": device_mode,
        "custom_prompt": prompt_clean,
        "diarization_sensitivity": sensitivity_mode,
        "audio_tracks": track_mode,
        "denoise": denoise_enabled,
        "llm_polish": llm_polish_enabled,
        "llm_base_url": llm_url_clean,
        "llm_model": llm_model_clean,
    }


@app.post("/api/transcribe/{task_id}")
async def transcribe_existing_recording(
    task_id: str,
    background_tasks: BackgroundTasks,
    workers: str = Form("4"),
    asr_model: str = Form(DEFAULT_ASR_MODEL),
    language: str = Form(DEFAULT_ASR_LANGUAGE),
    speakers: str = Form("auto"),
    compute_device: str = Form(DEFAULT_COMPUTE_DEVICE),
    custom_prompt: str = Form(""),
    diarization_sensitivity: str = Form(DEFAULT_DIARIZATION_SENSITIVITY),
    audio_tracks: str = Form(DEFAULT_AUDIO_TRACK_MODE),
    denoise: str = Form("true" if DENOISE_ENABLED_DEFAULT else "false"),
    llm_polish: str = Form("true" if LLM_POLISH_ENABLED_DEFAULT else "false"),
    llm_base_url: str = Form(DEFAULT_LLM_BASE_URL),
    llm_model: str = Form(DEFAULT_LLM_MODEL),
    llm_api_key: str = Form(""),
    asr_api_key: str = Form(""),
):
    """
    Creates (or re-creates) a speaker-diarized .txt transcript separately for
    an already-created recording without re-uploading or re-blurring video.
    Prefers cached original un-pitched 16 kHz audio (incl. dual-track cache)
    when available, falling back seamlessly to output/<task_id>_anonymized.mp4.
    """
    _validate_task_id(task_id)

    existing_task = tasks.get(task_id)
    if existing_task and existing_task.get("status") in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Task is currently running.")

    worker_count = parse_worker_count(workers)
    asr_model_id = parse_asr_model(asr_model)
    language_id = parse_language(language)
    speaker_mode = parse_speaker_count(speakers)
    device_mode = parse_compute_device(compute_device)
    prompt_clean = parse_custom_prompt(custom_prompt)
    sensitivity_mode = parse_diarization_sensitivity(diarization_sensitivity)
    track_mode = parse_audio_tracks(audio_tracks)
    denoise_enabled = parse_bool_form(denoise, default=DENOISE_ENABLED_DEFAULT, field_name="denoise")
    llm_polish_enabled = parse_bool_form(
        llm_polish, default=LLM_POLISH_ENABLED_DEFAULT, field_name="llm_polish"
    )
    llm_url_clean = parse_llm_base_url(llm_base_url)
    llm_model_clean = parse_llm_model(llm_model)
    llm_key_clean = parse_llm_api_key(llm_api_key)
    asr_key_clean = parse_asr_api_key(asr_api_key)

    cached_wav = _audio_cache_path(task_id)
    cached_track2_wav = _audio_cache_track2_path(task_id)
    video_path = os.path.join("output", f"{task_id}_anonymized.mp4")
    transcript_path = os.path.join("output", f"{task_id}_transcript.txt")

    has_cached_wav = os.path.isfile(cached_wav)
    has_cached_track2 = os.path.isfile(cached_track2_wav)
    has_video = os.path.isfile(video_path)

    if has_cached_wav:
        source_path = cached_wav
        source_track2_path = cached_track2_wav if has_cached_track2 else None
        input_is_wav_16k = True
        source_note = None
    elif has_video:
        source_path = video_path
        source_track2_path = None
        input_is_wav_16k = False
        source_note = (
            "Оригинальное аудио не найдено в кэше сессии — извлечение звука из сохранённой записи..."
        )
    else:
        raise HTTPException(
            status_code=404,
            detail="Recording not found. Upload a media file first.",
        )

    prev_transcript_ready = os.path.isfile(transcript_path)
    prev_filename = (existing_task or {}).get("original_filename", "")
    prev_mode = (existing_task or {}).get("mode", "anonymize" if has_video else "transcribe_only")

    cancel_tokens[task_id] = CancellationToken()
    tasks[task_id] = {
        "status": "queued",
        "mode": prev_mode,
        "message": "Постановка транскрибации в очередь...",
        "logs": ["Постановка отдельной транскрибации в очередь..."],
        "worker_count": worker_count,
        "transcribe_enabled": True,
        "video_ready": has_video,
        "transcript_ready": prev_transcript_ready,
        "has_original_audio": has_cached_wav,
        "can_transcribe": True,
        "is_post_transcribe": True,
        "asr_model": asr_model_id,
        "language": language_id,
        "speakers": speaker_mode,
        "compute_device": device_mode,
        "custom_prompt": prompt_clean,
        "diarization_sensitivity": sensitivity_mode,
        "audio_tracks": track_mode,
        "denoise": denoise_enabled,
        "llm_polish": llm_polish_enabled,
        "llm_base_url": llm_url_clean,
        "llm_model": llm_model_clean,
        "original_filename": prev_filename,
    }

    background_tasks.add_task(
        background_transcribe_only,
        task_id,
        source_path,
        transcript_path,
        worker_count,
        asr_model_id,
        language_id,
        speaker_mode,
        device_mode,
        True,
        input_is_wav_16k,
        False,
        source_note,
        prev_filename,
        prompt_clean,
        sensitivity_mode,
        track_mode,
        source_track2_path,
        denoise_enabled,
        llm_polish_enabled,
        llm_url_clean,
        llm_model_clean,
        llm_key_clean,
        asr_key_clean,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "mode": prev_mode,
        "message": "Постановка транскрибации в очередь...",
        "worker_count": worker_count,
        "asr_model": asr_model_id,
        "language": language_id,
        "speakers": speaker_mode,
        "compute_device": device_mode,
        "custom_prompt": prompt_clean,
        "diarization_sensitivity": sensitivity_mode,
        "audio_tracks": track_mode,
        "denoise": denoise_enabled,
        "llm_polish": llm_polish_enabled,
        "llm_base_url": llm_url_clean,
        "llm_model": llm_model_clean,
        "using_original_audio": has_cached_wav,
    }


@app.get("/api/recordings")
async def list_recordings():
    """Lists existing anonymized videos and transcripts in output/ (newest first)."""
    output_dir = "output"
    if not os.path.isdir(output_dir):
        return {"recordings": []}

    found: dict[str, dict] = {}
    for fname in os.listdir(output_dir):
        full = os.path.join(output_dir, fname)
        if not os.path.isfile(full):
            continue
        tid = None
        kind = None
        if fname.endswith("_anonymized.mp4"):
            tid = fname[: -len("_anonymized.mp4")]
            kind = "video"
        elif fname.endswith("_transcript.txt"):
            tid = fname[: -len("_transcript.txt")]
            kind = "transcript"
        if not tid or not _TASK_ID_RE.match(tid):
            continue

        try:
            st = os.stat(full)
            mtime = st.st_mtime
            size_bytes = st.st_size
        except OSError:
            continue

        rec = found.setdefault(
            tid,
            {
                "task_id": tid,
                "mtime": mtime,
                "video_ready": False,
                "video_size_bytes": 0,
                "transcript_ready": False,
                "transcript_size_bytes": 0,
            },
        )
        rec["mtime"] = max(rec["mtime"], mtime)
        if kind == "video":
            rec["video_ready"] = True
            rec["video_size_bytes"] = size_bytes
        elif kind == "transcript":
            rec["transcript_ready"] = True
            rec["transcript_size_bytes"] = size_bytes

    items = []
    for tid, rec in found.items():
        task_info = tasks.get(tid, {})
        if task_info.get("status") in ("queued", "processing"):
            continue
        has_orig = os.path.isfile(_audio_cache_path(tid))
        has_raw = os.path.isfile(_raw_transcript_path(tid))
        can_transcribe = has_orig or rec["video_ready"]
        created_iso = datetime.fromtimestamp(rec["mtime"], tz=timezone.utc).isoformat()
        items.append(
            {
                "task_id": tid,
                "original_filename": task_info.get("original_filename", ""),
                "created_at": created_iso,
                "mtime": rec["mtime"],
                "video_ready": rec["video_ready"],
                "video_size_bytes": rec["video_size_bytes"],
                "transcript_ready": rec["transcript_ready"],
                "raw_transcript_ready": has_raw,
                "transcript_size_bytes": rec["transcript_size_bytes"],
                "has_original_audio": has_orig,
                "can_transcribe": can_transcribe,
                "asr_model": task_info.get("asr_model"),
                "language": task_info.get("language"),
            }
        )

    items.sort(key=lambda r: r["mtime"], reverse=True)
    for item in items:
        item.pop("mtime", None)
    return {"recordings": items[:30]}


@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    entry = tasks[task_id]
    current_status = entry.get("status")
    is_post = bool(entry.get("is_post_transcribe"))
    if current_status in ("completed", "error", "cancelled"):
        return {
            "task_id": task_id,
            "status": current_status,
            "cancelled": current_status == "cancelled",
            "mode": entry.get("mode", DEFAULT_JOB_MODE),
            "is_post_transcribe": is_post,
            "video_ready": bool(entry.get("video_ready")),
            "transcript_ready": bool(entry.get("transcript_ready")),
            "has_original_audio": bool(entry.get("has_original_audio")),
            "can_transcribe": bool(entry.get("can_transcribe")),
            "asr_model": entry.get("asr_model"),
            "language": entry.get("language"),
            "compute_device": entry.get("compute_device"),
            "speakers": entry.get("speakers"),
        }

    token = cancel_tokens.get(task_id)
    if token is not None:
        token.cancel()

    entry["status"] = "cancelled"
    entry["message"] = "Транскрибация прервана." if is_post else "Обработка прервана."
    if not is_post:
        entry["video_ready"] = False
        entry["transcript_ready"] = False
        entry["has_original_audio"] = False
        entry["can_transcribe"] = False
    cancel_note = "Обработка прервана пользователем."
    if not entry["logs"] or entry["logs"][-1] != cancel_note:
        entry["logs"].append(cancel_note)

    return {
        "task_id": task_id,
        "status": "cancelled",
        "cancelled": True,
        "mode": entry.get("mode", DEFAULT_JOB_MODE),
        "is_post_transcribe": is_post,
        "video_ready": bool(entry.get("video_ready")),
        "transcript_ready": bool(entry.get("transcript_ready")),
        "has_original_audio": bool(entry.get("has_original_audio")),
        "can_transcribe": bool(entry.get("can_transcribe")),
        "asr_model": entry.get("asr_model"),
        "language": entry.get("language"),
        "compute_device": entry.get("compute_device"),
        "speakers": entry.get("speakers"),
    }


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id in tasks:
        return {"task_id": task_id, **tasks[task_id]}

    if _TASK_ID_RE.match(task_id):
        video_path = os.path.join("output", f"{task_id}_anonymized.mp4")
        transcript_path = os.path.join("output", f"{task_id}_transcript.txt")
        has_video = os.path.isfile(video_path)
        has_transcript = os.path.isfile(transcript_path)
        has_raw_transcript = os.path.isfile(_raw_transcript_path(task_id))
        has_orig = os.path.isfile(_audio_cache_path(task_id))
        if has_video or has_transcript:
            return {
                "task_id": task_id,
                "status": "completed",
                "mode": "anonymize" if has_video else "transcribe_only",
                "message": "Done!",
                "logs": [],
                "video_ready": has_video,
                "transcript_ready": has_transcript,
                "raw_transcript_ready": has_raw_transcript,
                "transcribe_enabled": has_transcript,
                "has_original_audio": has_orig,
                "can_transcribe": has_orig or has_video,
                "is_post_transcribe": False,
            }

    raise HTTPException(status_code=404, detail="Task not found")


@app.get("/api/download/{task_id}")
async def download_video(task_id: str):
    _validate_task_id(task_id)
    if task_id in tasks:
        entry = tasks[task_id]
        status = entry.get("status")
        post_video_ok = bool(entry.get("is_post_transcribe") and entry.get("video_ready"))
        if status != "completed" and not post_video_ok:
            raise HTTPException(status_code=400, detail="File not ready or task failed")
        if entry.get("mode") == "transcribe_only" and not entry.get("video_ready"):
            raise HTTPException(status_code=404, detail="No video output for transcription-only task")

    output_path = os.path.join("output", f"{task_id}_anonymized.mp4")
    if not os.path.exists(output_path):
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"anonymized_{task_id}.mp4",
    )


@app.get("/api/download-transcript/{task_id}")
async def download_transcript(task_id: str, raw: bool = False):
    _validate_task_id(task_id)
    if task_id in tasks:
        entry = tasks[task_id]
        status = entry.get("status")
        post_txt_ok = bool(entry.get("is_post_transcribe") and entry.get("transcript_ready"))
        if status != "completed" and not post_txt_ok:
            raise HTTPException(status_code=400, detail="Transcript not ready or task failed")
        if not entry.get("transcript_ready"):
            raise HTTPException(status_code=404, detail="Transcript was disabled or not generated for this task")

    if raw:
        raw_path = _raw_transcript_path(task_id)
        if not os.path.exists(raw_path):
            raise HTTPException(status_code=404, detail="Raw ASR backup transcript not found")
        return FileResponse(
            raw_path,
            media_type="text/plain; charset=utf-8",
            filename=f"transcript_{task_id}_raw.txt",
        )

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

