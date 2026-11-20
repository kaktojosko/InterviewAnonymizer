import asyncio
import os
import subprocess
import wave

import httpx
import pytest

from app import app, tasks
from main import build_arg_parser
from src.ffmpeg_utils import FFmpegUtils
from src.io_manager import IOManager
from src.orchestrator import Orchestrator
from src.transcriber import SpeakerTurn, Transcriber, format_transcript_txt


class _SyncASGIClient:
    def __init__(self, asgi_app):
        self._app = asgi_app

    def get(self, url: str, **kwargs) -> httpx.Response:
        async def _do():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                return await ac.get(url, **kwargs)

        return asyncio.run(_do())

    def post(self, url: str, **kwargs) -> httpx.Response:
        async def _do():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                return await ac.post(url, **kwargs)

        return asyncio.run(_do())


def TestClient(asgi_app):
    return _SyncASGIClient(asgi_app)


def _generate_test_mp4(path: str, duration_sec: float = 2.0, with_audio: bool = True):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=320x240:rate=15:duration={duration_sec}",
    ]
    if with_audio:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=44100:duration={duration_sec}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
            ]
        )
    else:
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-an"])
    cmd.append(path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_cli_arg_parser():
    parser = build_arg_parser()
    args_default = parser.parse_args(["-i", "in.mp4", "-o", "out.mp4"])
    assert args_default.no_transcribe is False
    assert args_default.transcribe_only is False
    assert args_default.asr_model == "gigaam-v3"
    assert args_default.language == "ru"
    assert args_default.speakers == "auto"
    assert args_default.sensitivity == "normal"
    assert args_default.audio_tracks == "auto"
    assert args_default.prompt == ""
    assert args_default.device == "cpu"
    assert args_default.denoise is False
    assert args_default.llm_polish is False
    assert args_default.transcript_output is None

    args_custom = parser.parse_args(
        [
            "-i",
            "in.mp4",
            "-o",
            "out.mp4",
            "-t",
            "dialog.txt",
            "--no-transcribe",
            "--asr-model",
            "breeze-asr-25",
            "--language",
            "auto",
            "--speakers",
            "4",
            "--sensitivity",
            "high",
            "--audio-tracks",
            "separate",
            "--prompt",
            "REST, gRPC, OAuth 2",
            "--device",
            "cuda",
            "--denoise",
            "--llm-polish",
            "--llm-url",
            "http://localhost:1234/v1",
            "--llm-model",
            "qwen3:14b",
            "--llm-api-key",
            "local-key",
            "-w",
            "1",
        ]
    )
    assert args_custom.no_transcribe is True
    assert args_custom.asr_model == "breeze-asr-25"
    assert args_custom.language == "auto"
    assert args_custom.speakers == "4"
    assert args_custom.sensitivity == "high"
    assert args_custom.audio_tracks == "separate"
    assert args_custom.prompt == "REST, gRPC, OAuth 2"
    assert args_custom.device == "cuda"
    assert args_custom.denoise is True
    assert args_custom.llm_polish is True
    assert args_custom.llm_url == "http://localhost:1234/v1"
    assert args_custom.llm_model == "qwen3:14b"
    assert args_custom.llm_api_key == "local-key"
    assert args_custom.transcript_output == "dialog.txt"
    assert args_custom.workers == 1

    args_tr_only = parser.parse_args(["-i", "interview.mp3", "--transcribe-only", "-t", "out.txt"])
    assert args_tr_only.transcribe_only is True
    assert args_tr_only.output is None
    assert args_tr_only.transcript_output == "out.txt"


def test_api_root_and_config():
    client = TestClient(app)
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "modeSwitcher" in root_resp.text
    assert "Только транскрибация" in root_resp.text
    assert "transcribeToggle" in root_resp.text
    assert "languageSelect" in root_resp.text
    assert "computeDeviceSelect" in root_resp.text
    assert "diarizationSensitivitySelect" in root_resp.text
    assert "audioTracksSelect" in root_resp.text
    assert "customPromptInput" in root_resp.text
    assert "promptPresets" in root_resp.text
    assert "denoiseToggle" in root_resp.text
    assert "llmPolishToggle" in root_resp.text
    assert "llmPresets" in root_resp.text
    assert "llmBaseUrlInput" in root_resp.text
    assert "llmModelInput" in root_resp.text
    assert "llmApiKeyInput" in root_resp.text
    assert "llmPrivacyHint" in root_resp.text
    assert "gigaam-v3-ctc" in root_resp.text
    assert "breeze-asr-25" in root_resp.text
    for removed_model in (
        "canary-qwen-2.5b",
        "canary-1b-v2",
        "large-v3-ru",
        'value="large-v3"',
        'value="small"',
        "gigaam-v3-int8",
        "parakeet",
    ):
        assert removed_model not in root_resp.text, f"Removed model '{removed_model}' still in HTML"
    assert "postCustomPromptInput" in root_resp.text
    assert "postDenoiseToggle" in root_resp.text
    assert "postLlmPolishToggle" in root_resp.text
    assert "postLlmPresets" in root_resp.text
    assert "postLlmApiKeyInput" in root_resp.text
    assert "postLlmPrivacyHint" in root_resp.text
    assert "downloadTranscriptBtn" in root_resp.text
    assert "downloadRawTranscriptBtn" in root_resp.text
    assert "selectedFileBar" in root_resp.text
    assert "startProcessingBtn" in root_resp.text
    assert "replaceSelectedFileBtn" in root_resp.text
    assert "clearSelectedFileBtn" in root_resp.text
    assert "postTranscribeBtn" in root_resp.text
    assert "recordingsSection" in root_resp.text
    assert "cancelBtn" in root_resp.text
    assert "Прервать" in root_resp.text
    assert "deepgram-nova-3" in root_resp.text
    assert "asrApiKeyInput" in root_resp.text
    assert "postAsrApiKeyInput" in root_resp.text
    assert "appVersionFooter" in root_resp.text
    assert "v5.1.0" in root_resp.text
    assert "no-store" in root_resp.headers.get("cache-control", "").lower()

    static_resp = client.get("/static/script.js")
    assert static_resp.status_code == 200
    assert "no-cache" in static_resp.headers.get("cache-control", "").lower()

    cfg_resp = client.get("/api/config")
    assert cfg_resp.status_code == 200
    assert "no-store" in cfg_resp.headers.get("cache-control", "").lower()
    data = cfg_resp.json()
    assert data["version"] == "5.1.0"
    assert isinstance(data.get("build_id"), str) and data["build_id"].startswith("5.1.0-")
    assert f"/static/script.js?v={data['build_id']}" in root_resp.text
    assert f"/static/style.css?v={data['build_id']}" in root_resp.text

    assert data["default_job_mode"] == "anonymize"
    assert [m["id"] for m in data["job_modes"]] == ["anonymize", "transcribe_only"]
    assert data["transcription_default"] is True
    assert data["default_asr_model"] == "gigaam-v3"
    assert data["default_language"] == "ru"
    assert [l["id"] for l in data["languages"]] == ["ru", "auto", "en"]
    assert data["default_compute_device"] == "cpu"
    assert isinstance(data["cuda_available"], bool)
    assert isinstance(data["ct2_cuda_available"], bool)
    assert isinstance(data["onnx_cuda_available"], bool)
    assert [d["id"] for d in data["compute_devices"]] == ["cpu", "cuda"]
    model_ids = [m["id"] for m in data["asr_models"]]
    assert model_ids == [
        "gigaam-v3",
        "gigaam-v3-ctc",
        "breeze-asr-25",
        "turbo",
        "deepgram-nova-3",
    ]
    giga_meta = next(m for m in data["asr_models"] if m["id"] == "gigaam-v3")
    assert giga_meta["engine"] == "onnx_asr"
    assert giga_meta["supports_cuda"] is True
    assert giga_meta["is_cloud"] is False
    ctc_meta = next(m for m in data["asr_models"] if m["id"] == "gigaam-v3-ctc")
    assert ctc_meta["engine"] == "onnx_asr"
    assert ctc_meta["supports_cuda"] is True
    breeze_meta = next(m for m in data["asr_models"] if m["id"] == "breeze-asr-25")
    assert breeze_meta["engine"] == "faster_whisper"
    assert breeze_meta["supports_cuda"] is True
    turbo_meta = next(m for m in data["asr_models"] if m["id"] == "turbo")
    assert turbo_meta["engine"] == "faster_whisper"
    assert turbo_meta["supports_cuda"] is True
    dg_meta = next(m for m in data["asr_models"] if m["id"] == "deepgram-nova-3")
    assert dg_meta["engine"] == "deepgram_cloud"
    assert dg_meta["supports_cuda"] is False
    assert dg_meta["is_cloud"] is True
    assert dg_meta["requires_api_key"] is True
    assert dg_meta["has_native_diarization"] is True
    assert data["denoise_default"] is False
    assert data["llm_polish_default"] is False
    assert data["llm_defaults"]["base_url"] == "https://api.deepseek.com/v1"
    assert data["llm_defaults"]["model"] == "deepseek-flash"
    assert "api_key" not in data["llm_defaults"]
    llm_preset_ids = [p["id"] for p in data["llm_presets"]]
    assert llm_preset_ids == ["deepseek_cloud", "qwen_cloud", "ollama_local", "lmstudio_local"]
    ds_preset = next(p for p in data["llm_presets"] if p["id"] == "deepseek_cloud")
    assert ds_preset["base_url"] == "https://api.deepseek.com/v1"
    assert ds_preset["model"] == "deepseek-flash"
    assert ds_preset["is_cloud"] is True
    qw_preset = next(p for p in data["llm_presets"] if p["id"] == "qwen_cloud")
    assert qw_preset["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert qw_preset["model"] == "qwen-plus"
    assert qw_preset["is_cloud"] is True
    assert data["speaker_options"] == ["auto", "2", "3", "4", "5", "6", "4+"]
    assert data["default_diarization_sensitivity"] == "normal"
    assert [s["id"] for s in data["diarization_sensitivities"]] == ["normal", "high", "low"]
    assert data["default_audio_track_mode"] == "auto"
    assert [m["id"] for m in data["audio_track_modes"]] == ["auto", "separate", "mix", "first"]
    preset_ids = [p["id"] for p in data["prompt_presets"]]
    assert "system_analyst" in preset_ids
    assert "backend_devops" in preset_ids


def test_api_upload_validation_errors():
    client = TestClient(app)

    r_ext = client.post(
        "/api/upload",
        files={"file": ("clip.avi", b"dummy", "video/x-msvideo")},
    )
    assert r_ext.status_code == 400

    r_bad_mode = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"mode": "invalid_mode"},
    )
    assert r_bad_mode.status_code == 400

    r_tr_ext = client.post(
        "/api/upload",
        files={"file": ("notes.pdf", b"dummy", "application/pdf")},
        data={"mode": "transcribe_only"},
    )
    assert r_tr_ext.status_code == 400

    r_model = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"asr_model": "nonexistent"},
    )
    assert r_model.status_code == 400

    r_lang = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"language": "klingon"},
    )
    assert r_lang.status_code == 400

    r_spk = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"speakers": "99"},
    )
    assert r_spk.status_code == 400

    r_sens = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"diarization_sensitivity": "ultra_invalid"},
    )
    assert r_sens.status_code == 400

    r_tracks = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"audio_tracks": "stereo_magic"},
    )
    assert r_tracks.status_code == 400

    r_dev = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"compute_device": "tpu"},
    )
    assert r_dev.status_code == 400

    r_bool = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"transcribe": "maybe"},
    )
    assert r_bool.status_code == 400

    r_denoise = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"denoise": "maybe"},
    )
    assert r_denoise.status_code == 400

    r_llm_bool = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"llm_polish": "maybe"},
    )
    assert r_llm_bool.status_code == 400

    r_llm_url = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"llm_base_url": "ftp://evil.local/v1"},
    )
    assert r_llm_url.status_code == 400


def test_extract_transcription_audio_16k_mono(tmp_path):
    mp4_path = str(tmp_path / "sample_with_audio.mp4")
    wav_path = str(tmp_path / "extracted_16k.wav")
    _generate_test_mp4(mp4_path, duration_sec=1.5, with_audio=True)

    ok = FFmpegUtils.extract_transcription_audio(mp4_path, wav_path)
    assert ok is True
    assert os.path.isfile(wav_path)

    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 16000


def test_orchestrator_silent_video_creates_note_transcript(tmp_path):
    in_mp4 = str(tmp_path / "silent_input.mp4")
    out_mp4 = str(tmp_path / "silent_output.mp4")
    out_txt = str(tmp_path / "silent_transcript.txt")
    _generate_test_mp4(in_mp4, duration_sec=1.2, with_audio=False)

    iom = IOManager()
    job_dir = iom.create_job_dir()
    try:
        res = Orchestrator.process_video_job(
            input_path=in_mp4,
            output_path=out_mp4,
            job_dir=job_dir,
            worker_count=1,
            transcribe=True,
            transcript_path=out_txt,
            asr_model="turbo",
            speaker_count="auto",
        )
        assert res["video_ready"] is True
        assert res["transcript_ready"] is True
        assert os.path.isfile(out_mp4)
        assert os.path.isfile(out_txt)
        content = open(out_txt, encoding="utf-8").read()
        assert "[Аудиодорожка в исходном видео отсутствует]" in content
    finally:
        iom.cleanup_job_dir(job_dir)


def test_orchestrator_parallel_gpu_mode(tmp_path, monkeypatch):
    import numpy as np
    import src.transcriber as tr_mod
    from src.transcriber import TimedUnit

    in_mp4 = str(tmp_path / "gpu_input.mp4")
    out_mp4 = str(tmp_path / "gpu_output.mp4")
    out_txt = str(tmp_path / "gpu_transcript.txt")

    # Generate 2.4s MP4 with 1s keyframes (-g 15 at 15fps) so split_video produces >=2 chunks
    # and exercises real 'spawn' ProcessPoolExecutor + Queue inheritance!
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=2.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=2.4",
            "-c:v",
            "libx264",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            in_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    orig_split = FFmpegUtils.split_video
    monkeypatch.setattr(
        FFmpegUtils,
        "split_video",
        staticmethod(lambda inp, pat, _chunk_t, **kw: orig_split(inp, pat, 1, **kw)),
    )

    events = []

    def _mock_gpu_units_only(
        cls,
        wav_path: str,
        asr_model: str = "turbo",
        compute_device: str = "cuda",
        threads: int = 1,
        status_cb=None,
        cancel_token=None,
        language: str = "ru",
        custom_prompt: str = "",
        **kwargs,
    ):
        events.append(("gpu_asr", compute_device, threads))
        if status_cb:
            status_cb("[Profile] GPU ASR завершена за 0.01s (единиц текста: 1)")
        return {
            "samples": np.ones(16000, dtype=np.float32) * 0.1,
            "sample_rate": 16000,
            "duration_sec": 2.4,
            "units": [TimedUnit(0.1, 0.9, " GPU OK")],
            "profile_id": asr_model,
            "custom_prompt": custom_prompt,
            "elapsed_asr": 0.01,
        }

    def _mock_post_mux_diarize(
        cls,
        asr_state: dict,
        output_txt_path: str,
        speaker_count: str = "auto",
        threads: int = 4,
        status_cb=None,
        cancel_token=None,
        diarization_sensitivity: str = "normal",
        custom_prompt: str = "",
        **kwargs,
    ):
        events.append(("cpu_diarize", speaker_count, threads))
        if status_cb:
            status_cb("[Profile] Диаризация завершена за 0.01s")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("[00:00:00 - 00:00:01] Спикер 1: GPU OK\n")
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.02}

    monkeypatch.setattr(tr_mod, "probe_cuda_available", lambda: True)
    monkeypatch.setattr(Transcriber, "transcribe_units_only", classmethod(_mock_gpu_units_only))
    monkeypatch.setattr(
        Transcriber, "diarize_and_write_transcript", classmethod(_mock_post_mux_diarize)
    )

    logs = []
    iom = IOManager()
    job_dir = iom.create_job_dir()
    try:
        res = Orchestrator.process_video_job(
            input_path=in_mp4,
            output_path=out_mp4,
            job_dir=job_dir,
            status_cb=logs.append,
            worker_count=2,
            transcribe=True,
            transcript_path=out_txt,
            asr_model="turbo",
            speaker_count="auto",
            compute_device="cuda",
        )
        assert res["video_ready"] is True
        assert res["transcript_ready"] is True
        assert events[0] == ("gpu_asr", "cuda", 1)
        assert events[1][0] == "cpu_diarize"
        assert events[1][1] == "auto"
        assert events[1][2] >= 2
        # Verify real spawned child workers logged through spawn_ctx.Queue()
        assert sum(1 for m in logs if "[Worker] Started: chunk_" in m) >= 2
        # Verify CPU diarization ran AFTER video muxing finished (no CPU contention!)
        mux_idx = next(i for i, m in enumerate(logs) if "Mux finished" in m)
        diar_idx = next(i for i, m in enumerate(logs) if "Диаризация завершена" in m)
        assert diar_idx > mux_idx
    finally:
        iom.cleanup_job_dir(job_dir)


def test_orchestrator_cancel_aborts_spawn_video_workers(tmp_path, monkeypatch):
    from src.cancellation import CancellationToken, TaskCancelledError

    in_mp4 = str(tmp_path / "cancel_chunks_input.mp4")
    out_mp4 = str(tmp_path / "cancel_chunks_output.mp4")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=5.0",
            "-c:v",
            "libx264",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-an",
            in_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    orig_split = FFmpegUtils.split_video
    monkeypatch.setattr(
        FFmpegUtils,
        "split_video",
        staticmethod(lambda inp, pat, _chunk_t, **kw: orig_split(inp, pat, 1, **kw)),
    )

    token = CancellationToken()
    logs = []

    def _on_status(msg: str):
        logs.append(msg)
        if "[Worker] Started: chunk_" in msg:
            token.cancel()

    iom = IOManager()
    job_dir = iom.create_job_dir()
    try:
        with pytest.raises(TaskCancelledError):
            Orchestrator.process_video_job(
                input_path=in_mp4,
                output_path=out_mp4,
                job_dir=job_dir,
                status_cb=_on_status,
                worker_count=1,
                transcribe=False,
                cancel_token=token,
            )
        assert not os.path.exists(out_mp4)
    finally:
        iom.cleanup_job_dir(job_dir)


def test_api_cancel_running_job_stops_and_cleans_up(tmp_path, monkeypatch):
    import threading
    import time
    from app import background_process, cancel_tokens
    from src.cancellation import CancellationToken

    client = TestClient(app)

    # 404 for unknown task
    assert client.post("/api/cancel/nonexistent-task").status_code == 404

    task_id = "test-live-cancel-task"
    in_mp4 = str(tmp_path / "live_cancel_in.mp4")
    out_mp4 = os.path.join("output", f"{task_id}_anonymized.mp4")
    out_txt = os.path.join("output", f"{task_id}_transcript.txt")
    _generate_test_mp4(in_mp4, duration_sec=1.2, with_audio=True)

    entered_transcribe = threading.Event()

    def _slow_cancellable_transcribe(
        cls,
        wav_path: str,
        output_txt_path: str,
        cancel_token=None,
        **kwargs,
    ):
        # Write a partial transcript before cancellation to verify cleanup deletes it
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("partial transcript")
        entered_transcribe.set()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if cancel_token is not None:
                cancel_token.check()
            time.sleep(0.02)
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 10.0}

    monkeypatch.setattr(
        Transcriber, "transcribe_audio_file", classmethod(_slow_cancellable_transcribe)
    )

    cancel_tokens[task_id] = CancellationToken()
    tasks[task_id] = {
        "status": "queued",
        "message": "Waiting in queue...",
        "logs": [],
        "worker_count": 1,
        "transcribe_enabled": True,
        "transcript_ready": False,
        "asr_model": "turbo",
        "speakers": "auto",
        "compute_device": "cpu",
    }

    worker_thread = threading.Thread(
        target=background_process,
        args=(task_id, in_mp4, out_mp4, 1, True, out_txt, "turbo", "auto", "cpu"),
        daemon=True,
    )
    try:
        worker_thread.start()
        assert entered_transcribe.wait(timeout=15.0), "Job did not reach transcription stage"
        # At this point muxed out_mp4 and partial out_txt both exist on disk
        assert os.path.exists(out_mp4)
        assert os.path.exists(out_txt)

        t0 = time.time()
        cancel_resp = client.post(f"/api/cancel/{task_id}")
        assert cancel_resp.status_code == 200
        cancel_payload = cancel_resp.json()
        assert cancel_payload["task_id"] == task_id
        assert cancel_payload["status"] == "cancelled"
        assert cancel_payload["cancelled"] is True
        assert cancel_payload["is_post_transcribe"] is False
        assert cancel_payload["video_ready"] is False
        assert cancel_payload["transcript_ready"] is False

        worker_thread.join(timeout=3.0)
        assert not worker_thread.is_alive()
        assert (time.time() - t0) < 3.0

        st = client.get(f"/api/status/{task_id}").json()
        assert st["status"] == "cancelled"
        assert st["transcript_ready"] is False
        assert client.get(f"/api/download/{task_id}").status_code == 400
        assert client.get(f"/api/download-transcript/{task_id}").status_code == 400

        # Idempotent second cancel call on already cancelled task
        cancel_again = client.post(f"/api/cancel/{task_id}")
        assert cancel_again.status_code == 200
        assert cancel_again.json()["status"] == "cancelled"

        # Verify input, partial video, partial transcript, and token were cleaned up
        assert task_id not in cancel_tokens
        assert not os.path.exists(in_mp4)
        assert not os.path.exists(out_mp4)
        assert not os.path.exists(out_txt)
    finally:
        tasks.pop(task_id, None)
        cancel_tokens.pop(task_id, None)
        for p in (in_mp4, out_mp4, out_txt):
            if os.path.exists(p):
                os.remove(p)


def test_fastapi_e2e_upload_transcribe_on_and_off(tmp_path, monkeypatch):
    mp4_path = str(tmp_path / "interview_clip.mp4")
    _generate_test_mp4(mp4_path, duration_sec=1.5, with_audio=True)

    def _mock_transcribe_audio_file(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = "turbo",
        speaker_count: str = "auto",
        threads: int = 1,
        diar_threads=None,
        compute_device: str = "cpu",
        status_cb=None,
        cancel_token=None,
        language: str = "ru",
        custom_prompt: str = "",
        diarization_sensitivity: str = "normal",
        **kwargs,
    ):
        assert os.path.isfile(wav_path)
        turns = [
            SpeakerTurn(0.1, 0.7, "Спикер 1", "Привет, как работает ваш backend?"),
            SpeakerTurn(0.8, 1.4, "Спикер 2", "Всё локально на ONNX и CTranslate2."),
        ]
        txt = format_transcript_txt(
            turns=turns,
            duration_sec=1.5,
            model_name="Mock Whisper Turbo",
            num_speakers=2,
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        if status_cb:
            status_cb("Транскрибация завершена")
        return {"num_speakers": 2, "num_turns": 2, "elapsed": 0.01}

    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_mock_transcribe_audio_file))

    client = TestClient(app)
    created_task_ids = []
    try:
        # Case 1: Auto-transcription ON (default CPU)
        with open(mp4_path, "rb") as f:
            up1 = client.post(
                "/api/upload",
                files={"file": ("interview.mp4", f, "video/mp4")},
                data={
                    "workers": "1",
                    "transcribe": "true",
                    "asr_model": "turbo",
                    "speakers": "2",
                    "compute_device": "cpu",
                },
            )
        assert up1.status_code == 200
        assert up1.json()["compute_device"] == "cpu"
        tid1 = up1.json()["task_id"]
        created_task_ids.append(tid1)

        st1 = client.get(f"/api/status/{tid1}").json()
        assert st1["status"] == "completed"
        assert st1["transcribe_enabled"] is True
        assert st1["transcript_ready"] is True
        assert st1["compute_device"] == "cpu"

        dl_vid1 = client.get(f"/api/download/{tid1}")
        assert dl_vid1.status_code == 200
        assert len(dl_vid1.content) > 1000

        dl_txt1 = client.get(f"/api/download-transcript/{tid1}")
        assert dl_txt1.status_code == 200
        assert "Спикер 1: Привет, как работает ваш backend?" in dl_txt1.text
        assert "Спикер 2: Всё локально на ONNX и CTranslate2." in dl_txt1.text

        # Case 2: Auto-transcription OFF
        with open(mp4_path, "rb") as f:
            up2 = client.post(
                "/api/upload",
                files={"file": ("interview.mp4", f, "video/mp4")},
                data={"workers": "1", "transcribe": "false"},
            )
        assert up2.status_code == 200
        tid2 = up2.json()["task_id"]
        created_task_ids.append(tid2)

        st2 = client.get(f"/api/status/{tid2}").json()
        assert st2["status"] == "completed"
        assert st2["transcribe_enabled"] is False
        assert st2["transcript_ready"] is False
        assert st2["can_transcribe"] is True

        dl_vid2 = client.get(f"/api/download/{tid2}")
        assert dl_vid2.status_code == 200

        dl_txt2 = client.get(f"/api/download-transcript/{tid2}")
        assert dl_txt2.status_code == 404
    finally:
        from app import _audio_cache_path

        for tid in created_task_ids:
            tasks.pop(tid, None)
            for suffix in ("_anonymized.mp4", "_transcript.txt"):
                out_file = os.path.join("output", f"{tid}{suffix}")
                if os.path.exists(out_file):
                    os.remove(out_file)
            cached = _audio_cache_path(tid)
            if os.path.exists(cached):
                os.remove(cached)


def test_planar_yuv420p_video_processor_and_single_pass_concat_mux(tmp_path):
    import ffmpeg
    import numpy as np
    from src.audio_processor import AudioProcessor
    from src.video_processor import VideoProcessor

    # 1. Verify planar YUV420p in-place mosaic blur modifies ROI and preserves outside pixels
    vp = VideoProcessor()
    w, h = 160, 120
    vp._init_detector(w, h)
    rng = np.random.RandomState(7)
    yuv_frame = rng.randint(16, 235, size=(w * h * 3) // 2, dtype=np.uint8)
    before = yuv_frame.copy()
    vp._apply_blur(yuv_frame, 50, 30, 40, 40, w, h)
    assert not np.array_equal(yuv_frame, before)
    # Top-left corner outside padded ROI remains untouched
    assert np.array_equal(yuv_frame[:w * 5], before[:w * 5])

    # 2. Verify VideoProcessor.process handles odd input dimensions (e.g. 321x241 -> 320x240 yuv420p)
    #    and single-pass AudioProcessor (.m4a) + FFmpegUtils.concat_and_mux produces valid H.264 + AAC MP4
    odd_in = str(tmp_path / "odd_in.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=322x242:rate=15:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            odd_in,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    chunk_out = str(tmp_path / "chunk_out.mp4")
    # Pass odd orig_width/orig_height (321x241) to verify automatic even rounding (320x240)
    ok_vp = vp.process(odd_in, chunk_out, fps="15/1", orig_width=321, orig_height=241, enc_threads=2)
    assert ok_vp is True
    assert os.path.isfile(chunk_out)

    pitched_m4a = str(tmp_path / "audio_pitched.m4a")
    ok_aud = AudioProcessor.process(odd_in, pitched_m4a, sample_rate=44100)
    assert ok_aud is True
    assert os.path.isfile(pitched_m4a)

    concat_txt = str(tmp_path / "concat_list.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        f.write(f"file '{chunk_out}'\n")

    final_mp4 = str(tmp_path / "final_muxed.mp4")
    FFmpegUtils.concat_and_mux(concat_txt, pitched_m4a, final_mp4)
    assert os.path.isfile(final_mp4)

    probe = ffmpeg.probe(final_mp4)
    streams = {s["codec_type"]: s for s in probe["streams"]}
    assert streams["video"]["codec_name"] == "h264"
    assert streams["video"]["pix_fmt"] == "yuv420p"
    assert int(streams["video"]["width"]) == 320
    assert int(streams["video"]["height"]) == 240
    assert streams["audio"]["codec_name"] == "aac"


def test_frontend_reset_btn_allows_gpu_selection():
    """
    Executes static/script.js in Node.js against a DOM harness to verify that:
      1. After processing a CPU job (including with cloud 'deepgram-nova-3') and clicking
         'Загрузить ещё' (#resetBtn), 'GPU (CUDA)' in #computeDeviceSelect is enabled,
         clickable, auto-switches ASR model from 'deepgram-nova-3' to 'turbo', and submits
         compute_device='cuda' on the next upload.
      2. Clicking <label for="computeDeviceSelect"> opens the dropdown without immediately
         closing via document click bubbling.
      3. When cuda_available=false on the server, 'GPU (CUDA)' is disabled ('нет GPU').
    """
    node_test_js = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function runScenario({ cudaAvailable }) {
    const storage = new Map();
    const localStorage = {
        getItem: (k) => (storage.has(k) ? storage.get(k) : null),
        setItem: (k, v) => storage.set(k, String(v)),
    };

    class ClassList {
        constructor(el, initial = []) {
            this.el = el;
            this.set = new Set(initial.filter(Boolean));
        }
        add(...cls) { cls.forEach((c) => this.set.add(c)); }
        remove(...cls) { cls.forEach((c) => this.set.delete(c)); }
        contains(c) { return this.set.has(c); }
        toggle(c, force) {
            const next = force !== undefined ? Boolean(force) : !this.set.has(c);
            if (next) this.set.add(c); else this.set.delete(c);
            return next;
        }
    }

    class FakeEvent {
        constructor(type, opts = {}) {
            this.type = type;
            this.bubbles = Boolean(opts.bubbles);
            this.defaultPrevented = false;
            this.propagationStopped = false;
        }
        preventDefault() { this.defaultPrevented = true; }
        stopPropagation() { this.propagationStopped = true; }
    }

    const byId = new Map();
    let activeElement = null;

    class FakeElement {
        constructor(tagName, { id = '', classes = [], attrs = {}, text = '' } = {}) {
            this.tagName = tagName.toUpperCase();
            this.id = id;
            if (id) byId.set(id, this);
            this.classList = new ClassList(this, classes);
            this.attrs = { ...attrs };
            this.dataset = {};
            this.children = [];
            this.parentNode = null;
            this.listeners = {};
            this._textContent = text;
            this._innerHTML = '';
            this.disabled = false;
            this.checked = Boolean(attrs.checked);
            this.title = '';
            this.tabIndex = 0;
            this.scrollTop = 0;
            this.scrollHeight = 100;
            this.files = [];
            this._value = attrs.value !== undefined ? String(attrs.value) : '';
            this.selected = Boolean(attrs.selected);
            this.onclick = null;
        }

        get className() { return Array.from(this.classList.set).join(' '); }
        set className(v) { this.classList = new ClassList(this, String(v).split(/\s+/)); }

        get textContent() {
            if (this.children.length === 0) return this._textContent;
            return this._textContent + this.children.map((c) => c.textContent).join('');
        }
        set textContent(v) {
            this._textContent = String(v);
            this.children = [];
        }

        get innerHTML() { return this._innerHTML; }
        set innerHTML(v) {
            this._innerHTML = String(v);
            if (v === '') {
                this.children.forEach((c) => { c.parentNode = null; });
                this.children = [];
            }
        }

        get options() {
            return this.children.filter((c) => c.tagName === 'OPTION');
        }

        get selectedIndex() {
            const opts = this.options;
            const idx = opts.findIndex((o) => o.value === this.value);
            return idx >= 0 ? idx : 0;
        }

        get value() {
            if (this.tagName === 'SELECT') {
                const opts = this.options;
                if (this._value && opts.some((o) => o.value === this._value)) return this._value;
                const sel = opts.find((o) => o.selected) || opts[0];
                return sel ? sel.value : '';
            }
            return this._value;
        }
        set value(v) {
            this._value = String(v);
        }

        setAttribute(k, v) { this.attrs[k] = String(v); }
        getAttribute(k) { return this.attrs[k] ?? null; }

        appendChild(child) {
            if (child.parentNode) {
                const idx = child.parentNode.children.indexOf(child);
                if (idx >= 0) child.parentNode.children.splice(idx, 1);
            }
            child.parentNode = this;
            this.children.push(child);
            return child;
        }

        insertBefore(newNode, refNode) {
            if (newNode.parentNode) {
                const oldIdx = newNode.parentNode.children.indexOf(newNode);
                if (oldIdx >= 0) newNode.parentNode.children.splice(oldIdx, 1);
            }
            newNode.parentNode = this;
            const idx = this.children.indexOf(refNode);
            if (idx >= 0) this.children.splice(idx, 0, newNode);
            else this.children.push(newNode);
            return newNode;
        }

        addEventListener(type, fn) {
            if (!this.listeners[type]) this.listeners[type] = [];
            this.listeners[type].push(fn);
        }

        dispatchEvent(evt) {
            const fns = this.listeners[evt.type] || [];
            fns.forEach((fn) => fn.call(this, evt));
            if (evt.type === 'click' && typeof this.onclick === 'function') {
                this.onclick.call(this, evt);
            }
            if (evt.bubbles && !evt.propagationStopped) {
                if (this.parentNode) this.parentNode.dispatchEvent(evt);
                else documentObj.dispatchEvent(evt);
            }
            return !evt.defaultPrevented;
        }

        click() {
            const evt = new FakeEvent('click', { bubbles: true });
            this.dispatchEvent(evt);
        }

        focus() { activeElement = this; }

        matchesSelector(sel) {
            if (sel === '.sun-icon') return this.classList.contains('sun-icon');
            if (sel === '.moon-icon') return this.classList.contains('moon-icon');
            const labelMatch = sel.match(/^label\[for="(.+)"\]$/);
            if (labelMatch) return this.tagName === 'LABEL' && this.attrs.for === labelMatch[1];
            if (sel === '.clay-select-option:not(:disabled)') {
                return this.classList.contains('clay-select-option') && !this.disabled;
            }
            if (sel === '.clay-select-option.is-selected:not(:disabled)') {
                return (
                    this.classList.contains('clay-select-option') &&
                    this.classList.contains('is-selected') &&
                    !this.disabled
                );
            }
            return false;
        }

        querySelectorAll(sel) {
            const out = [];
            const walk = (node) => {
                node.children.forEach((ch) => {
                    if (ch.matchesSelector(sel)) out.push(ch);
                    walk(ch);
                });
            };
            walk(this);
            return out;
        }

        querySelector(sel) {
            return this.querySelectorAll(sel)[0] || null;
        }
    }

    const root = new FakeElement('div', { id: 'root' });
    const makeSelect = (id, opts) => {
        const container = new FakeElement('div');
        const lbl = new FakeElement('label', { attrs: { for: id }, text: id });
        const sel = new FakeElement('select', { id, classes: ['worker-select'] });
        opts.forEach(([val, text, selected]) => {
            sel.appendChild(new FakeElement('option', { attrs: { value: val, selected }, text }));
        });
        container.appendChild(lbl);
        container.appendChild(sel);
        root.appendChild(container);
        return { sel, lbl, container };
    };

    const themeBtn = root.appendChild(new FakeElement('button', { id: 'themeToggle' }));
    themeBtn.appendChild(new FakeElement('svg', { classes: ['sun-icon', 'hidden'] }));
    themeBtn.appendChild(new FakeElement('svg', { classes: ['moon-icon', 'hidden'] }));

    const uploadPanel = root.appendChild(new FakeElement('div', { id: 'uploadPanel' }));
    uploadPanel.appendChild(new FakeElement('div', { id: 'dropZone' }));
    uploadPanel.appendChild(new FakeElement('input', { id: 'fileInput' }));
    uploadPanel.appendChild(new FakeElement('span', { id: 'workerValue', text: '4' }));
    makeSelect('workerSelect', [['1', '1'], ['2', '2'], ['4', '4', true], ['6', '6'], ['custom', 'Другое']]);
    uploadPanel.appendChild(new FakeElement('div', { id: 'customWorkerWrap', classes: ['hidden'] }));
    uploadPanel.appendChild(new FakeElement('input', { id: 'customWorkerInput' }));
    uploadPanel.appendChild(new FakeElement('p', { id: 'workerLimitText' }));
    uploadPanel.appendChild(new FakeElement('input', { id: 'transcribeToggle', attrs: { checked: true } }));
    uploadPanel.appendChild(new FakeElement('span', { id: 'transcribeStateBadge', text: 'Вкл' }));
    uploadPanel.appendChild(new FakeElement('div', { id: 'transcribeOptionsWrap' }));

    const asrGroup = makeSelect('asrModelSelect', [
        ['turbo', 'Whisper Turbo — универсальная (99 яз., 810 МБ)', true],
        ['breeze-asr-25', 'Breeze-ASR-25 — лучший для RU+EN IT-речи (3.1 ГБ)'],
        ['deepgram-nova-3', 'Deepgram Nova-3 — облачный API'],
    ]);
    const devGroup = makeSelect('computeDeviceSelect', [
        ['cpu', 'CPU', true],
        ['cuda', 'GPU (CUDA)'],
    ]);
    makeSelect('speakerCountSelect', [['auto', 'Авто', true], ['2', '2 спикера']]);

    const statusPanel = root.appendChild(new FakeElement('div', { id: 'statusPanel', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('div', { id: 'spinner' }));
    statusPanel.appendChild(new FakeElement('svg', { id: 'successIcon', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('svg', { id: 'cancelIcon', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('h2', { id: 'statusText' }));
    statusPanel.appendChild(new FakeElement('p', { id: 'statusSubText' }));
    statusPanel.appendChild(new FakeElement('div', { id: 'debugLog', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'cancelBtn', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'downloadBtn', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'downloadTranscriptBtn', classes: ['hidden'] }));
    const resetBtn = statusPanel.appendChild(new FakeElement('button', { id: 'resetBtn', classes: ['hidden'] }));

    const docListeners = {};
    const documentObj = {
        documentElement: new FakeElement('html'),
        get activeElement() { return activeElement; },
        getElementById: (id) => byId.get(id) || null,
        querySelector: (sel) => root.querySelector(sel),
        querySelectorAll: (sel) => root.querySelectorAll(sel),
        createElement: (tag) => new FakeElement(tag),
        createElementNS: (_ns, tag) => new FakeElement(tag),
        addEventListener: (type, fn) => {
            if (!docListeners[type]) docListeners[type] = [];
            docListeners[type].push(fn);
        },
        dispatchEvent: (evt) => {
            (docListeners[evt.type] || []).forEach((fn) => fn(evt));
        },
    };

    const uploadedPayloads = [];
    let taskSeq = 0;

    class FakeFormData {
        constructor() { this.entries = {}; }
        append(k, v) { this.entries[k] = v; }
    }

    const fakeFetch = async (url, opts = {}) => {
        if (url === '/api/config') {
            return {
                ok: true,
                json: async () => ({
                    version: '5.1.0',
                    cpu_count: 8,
                    default_workers: 4,
                    max_workers: 7,
                    transcription_default: true,
                    default_asr_model: 'turbo',
                    asr_models: [
                        { id: 'turbo', label: 'Whisper Turbo — универсальная (99 яз., 810 МБ)', supports_cuda: true },
                        { id: 'breeze-asr-25', label: 'Breeze-ASR-25 — лучший для RU+EN IT-речи (3.1 ГБ)', supports_cuda: true },
                        { id: 'deepgram-nova-3', label: 'Deepgram Nova-3 — облачный API', supports_cuda: false, is_cloud: true },
                    ],
                    cuda_available: cudaAvailable,
                    default_compute_device: 'cpu',
                    compute_devices: [
                        { id: 'cpu', label: 'CPU' },
                        { id: 'cuda', label: 'GPU (CUDA)' },
                    ],
                }),
            };
        }
        if (url === '/api/upload') {
            taskSeq += 1;
            uploadedPayloads.push({ ...opts.body.entries });
            return {
                ok: true,
                json: async () => ({
                    task_id: `task-${taskSeq}`,
                    status: 'queued',
                    worker_count: 4,
                }),
            };
        }
        if (url.startsWith('/api/status/')) {
            return {
                ok: true,
                json: async () => ({
                    status: 'completed',
                    message: 'Done!',
                    logs: ['Initializing...', 'Done!'],
                    transcript_ready: true,
                }),
            };
        }
        throw new Error('Unexpected URL: ' + url);
    };

    const ctx = vm.createContext({
        document: documentObj,
        window: { location: { href: '' }, matchMedia: () => ({ matches: false }) },
        localStorage,
        Event: FakeEvent,
        FormData: FakeFormData,
        AbortController: class { constructor() { this.signal = {}; } abort() {} },
        fetch: fakeFetch,
        setInterval: () => 1,
        clearInterval: () => {},
        alert: () => {},
        console,
        Number,
        String,
        Boolean,
        Array,
        Map,
        Set,
        Date,
    });
    ctx.window.matchMedia = () => ({ matches: false });

    const code = fs.readFileSync('static/script.js', 'utf8');
    vm.runInContext(code, ctx);

    return {
        ctx,
        byId,
        asrGroup,
        devGroup,
        resetBtn,
        uploadPanel,
        statusPanel,
        uploadedPayloads,
        localStorage,
    };
}

(async () => {
    // Scenario 1: Server HAS GPU (cudaAvailable = true)
    const s1 = runScenario({ cudaAvailable: true });
    await new Promise((r) => setTimeout(r, 20));

    const asrWrapper = s1.asrGroup.sel.parentNode;
    const devWrapper = s1.devGroup.sel.parentNode;
    const getMenuButtons = (wrapper) => wrapper.children.find((c) => c.classList.contains('clay-select-menu')).children;

    // Step 1: User selects non-CUDA 'Deepgram Nova-3' for CPU/cloud job
    const dgBtn = getMenuButtons(asrWrapper).find((b) => b.textContent.includes('Deepgram'));
    assert.ok(dgBtn, 'Deepgram option must exist in ASR dropdown');
    dgBtn.click();
    assert.strictEqual(s1.asrGroup.sel.value, 'deepgram-nova-3');
    assert.strictEqual(s1.devGroup.sel.value, 'cpu');

    // Verify GPU (CUDA) button in menu is STILL enabled (not locked out!) with 'Whisper' hint
    const gpuBtnBefore = getMenuButtons(devWrapper)[1];
    assert.strictEqual(gpuBtnBefore.disabled, false, 'GPU option must NOT be disabled when non-CUDA model is selected');
    assert.ok(gpuBtnBefore.textContent.includes('Whisper'), 'GPU option should show Whisper auto-switch hint');

    // Step 2: Upload & complete first video job
    const fileInput = s1.byId.get('fileInput');
    fileInput.files = [{ name: 'interview1.mp4' }];
    fileInput.dispatchEvent(new s1.ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 30));

    assert.strictEqual(s1.uploadedPayloads.length, 1);
    assert.strictEqual(s1.uploadedPayloads[0].asr_model, 'deepgram-nova-3');
    assert.strictEqual(s1.uploadedPayloads[0].compute_device, 'cpu');
    assert.strictEqual(s1.statusPanel.classList.contains('hidden'), false);
    assert.strictEqual(s1.resetBtn.classList.contains('hidden'), false);

    // Step 3: Click 'Загрузить ещё' (#resetBtn)
    s1.resetBtn.click();
    assert.strictEqual(s1.uploadPanel.classList.contains('hidden'), false);
    assert.strictEqual(s1.statusPanel.classList.contains('hidden'), true);

    // Step 4: Click <label for="computeDeviceSelect"> -> menu must open and NOT immediately close!
    s1.devGroup.lbl.click();
    assert.strictEqual(devWrapper.classList.contains('is-open'), true, 'Label click must open Clay select');

    // Step 5: Click 'GPU (CUDA)' option in custom dropdown
    const gpuBtnAfterReset = getMenuButtons(devWrapper)[1];
    assert.strictEqual(gpuBtnAfterReset.disabled, false, 'GPU option must remain enabled after resetBtn');
    gpuBtnAfterReset.click();

    assert.strictEqual(s1.devGroup.sel.value, 'cuda', 'computeDeviceSelect must switch to cuda');
    assert.strictEqual(s1.asrGroup.sel.value, 'turbo', 'asrModelSelect must auto-switch from deepgram-nova-3 to turbo');
    assert.strictEqual(s1.localStorage.getItem('selectedComputeDevice'), 'cuda');
    assert.strictEqual(s1.localStorage.getItem('selectedAsrModel'), 'turbo');

    // Step 5b: User can also switch to 'Breeze-ASR-25' while keeping GPU (CUDA) active
    const breezeBtn = getMenuButtons(asrWrapper).find((b) => b.textContent.includes('Breeze'));
    assert.ok(breezeBtn, 'Breeze-ASR-25 option must exist in ASR dropdown');
    breezeBtn.click();
    assert.strictEqual(s1.asrGroup.sel.value, 'breeze-asr-25');
    assert.strictEqual(s1.devGroup.sel.value, 'cuda', 'GPU must remain active when Breeze-ASR-25 is selected');

    // Step 6: Upload second video -> verify compute_device='cuda' & asr_model='breeze-asr-25' are sent!
    fileInput.files = [{ name: 'interview2.mp4' }];
    fileInput.dispatchEvent(new s1.ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 30));

    assert.strictEqual(s1.uploadedPayloads.length, 2);
    assert.strictEqual(s1.uploadedPayloads[1].compute_device, 'cuda');
    assert.strictEqual(s1.uploadedPayloads[1].asr_model, 'breeze-asr-25');

    // Step 7: Click 'Загрузить ещё' again -> GPU (CUDA) remains selected and enabled
    s1.resetBtn.click();
    assert.strictEqual(s1.devGroup.sel.value, 'cuda');
    assert.strictEqual(getMenuButtons(devWrapper)[1].disabled, false);

    // Scenario 2: Server has NO GPU (cudaAvailable = false) -> GPU option MUST be disabled ('нет GPU')
    const s2 = runScenario({ cudaAvailable: false });
    await new Promise((r) => setTimeout(r, 20));
    const s2GpuBtn = getMenuButtons(s2.devGroup.sel.parentNode)[1];
    assert.strictEqual(s2GpuBtn.disabled, true);
    assert.ok(s2GpuBtn.textContent.includes('нет GPU'));

    console.log('FRONTEND_GPU_RESET_TEST_OK');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_test_js],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node UI test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FRONTEND_GPU_RESET_TEST_OK" in proc.stdout


def test_post_recording_separate_transcription_cache_fallback_and_recordings_api(tmp_path, monkeypatch):
    from app import _audio_cache_path

    mp4_path = str(tmp_path / "post_tr_clip.mp4")
    _generate_test_mp4(mp4_path, duration_sec=1.5, with_audio=True)

    calls = []

    def _mock_transcribe(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = "gigaam-v3",
        speaker_count: str = "auto",
        threads: int = 1,
        diar_threads=None,
        compute_device: str = "cpu",
        status_cb=None,
        cancel_token=None,
        language: str = "ru",
        custom_prompt: str = "",
        diarization_sensitivity: str = "normal",
        **kwargs,
    ):
        calls.append({"wav_path": wav_path, "asr_model": asr_model, "language": language, "speakers": speaker_count})
        turns = [SpeakerTurn(0.1, 1.2, "Спикер 1", f"Модель {asr_model} ({language})")]
        txt = format_transcript_txt(turns, duration_sec=1.5, model_name=asr_model, num_speakers=1)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.01}

    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_mock_transcribe))

    client = TestClient(app)
    tid = None
    try:
        # 1. Anonymize video WITHOUT transcription first
        with open(mp4_path, "rb") as f:
            up = client.post(
                "/api/upload",
                files={"file": ("demo.mp4", f, "video/mp4")},
                data={"workers": "1", "transcribe": "false"},
            )
        assert up.status_code == 200
        tid = up.json()["task_id"]

        st = client.get(f"/api/status/{tid}").json()
        assert st["status"] == "completed"
        assert st["video_ready"] is True
        assert st["transcript_ready"] is False
        assert st["has_original_audio"] is True
        assert st["can_transcribe"] is True
        assert os.path.isfile(_audio_cache_path(tid))

        # Verify GET /api/recordings lists the recording before transcription
        recs = client.get("/api/recordings").json()["recordings"]
        rec_item = next((r for r in recs if r["task_id"] == tid), None)
        assert rec_item is not None
        assert rec_item["video_ready"] is True
        assert rec_item["transcript_ready"] is False
        assert rec_item["has_original_audio"] is True
        assert rec_item["can_transcribe"] is True

        # 2. Generate transcript separately -> must use cached original un-pitched WAV!
        tr1 = client.post(
            f"/api/transcribe/{tid}",
            data={
                "workers": "1",
                "asr_model": "gigaam-v3",
                "language": "ru",
                "speakers": "2",
                "compute_device": "cpu",
            },
        )
        assert tr1.status_code == 200
        assert tr1.json()["using_original_audio"] is True

        st_after1 = client.get(f"/api/status/{tid}").json()
        assert st_after1["status"] == "completed"
        assert st_after1["video_ready"] is True
        assert st_after1["transcript_ready"] is True
        assert len(calls) == 1
        assert os.path.abspath(calls[0]["wav_path"]) == os.path.abspath(_audio_cache_path(tid))
        assert calls[0]["asr_model"] == "gigaam-v3"
        assert calls[0]["language"] == "ru"

        dl_txt1 = client.get(f"/api/download-transcript/{tid}")
        assert dl_txt1.status_code == 200
        assert "Модель gigaam-v3 (ru)" in dl_txt1.text
        assert client.get(f"/api/download/{tid}").status_code == 200

        # 3. Remove cached original audio (simulating container restart) -> falls back to anonymized MP4!
        os.remove(_audio_cache_path(tid))
        tr2 = client.post(
            f"/api/transcribe/{tid}",
            data={
                "workers": "1",
                "asr_model": "breeze-asr-25",
                "language": "ru",
                "speakers": "auto",
                "compute_device": "cpu",
            },
        )
        assert tr2.status_code == 200
        assert tr2.json()["using_original_audio"] is False

        st_after2 = client.get(f"/api/status/{tid}").json()
        assert st_after2["status"] == "completed"
        assert st_after2["video_ready"] is True
        assert st_after2["transcript_ready"] is True
        assert len(calls) == 2
        assert calls[1]["asr_model"] == "breeze-asr-25"

        dl_txt2 = client.get(f"/api/download-transcript/{tid}")
        assert dl_txt2.status_code == 200
        assert "Модель breeze-asr-25 (ru)" in dl_txt2.text
    finally:
        if tid:
            tasks.pop(tid, None)
            for p in (
                os.path.join("output", f"{tid}_anonymized.mp4"),
                os.path.join("output", f"{tid}_transcript.txt"),
                _audio_cache_path(tid),
            ):
                if os.path.exists(p):
                    os.remove(p)


def test_cancel_post_recording_transcription_preserves_video_and_existing_transcript(tmp_path, monkeypatch):
    import threading
    import time
    from app import _audio_cache_path, background_transcribe_only, cancel_tokens
    from src.cancellation import CancellationToken

    client = TestClient(app)
    task_id = "test-cancel-post-transcribe"
    out_mp4 = os.path.join("output", f"{task_id}_anonymized.mp4")
    out_txt = os.path.join("output", f"{task_id}_transcript.txt")
    _generate_test_mp4(out_mp4, duration_sec=1.2, with_audio=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("PREVIOUS_TRANSCRIPT_INTACT\n")

    entered = threading.Event()

    def _slow_retranscribe(cls, wav_path, output_txt_path, cancel_token=None, **kwargs):
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("PARTIAL_NEW_TRANSCRIPT\n")
        entered.set()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if cancel_token is not None:
                cancel_token.check()
            time.sleep(0.02)
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 10.0}

    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_slow_retranscribe))

    cancel_tokens[task_id] = CancellationToken()
    tasks[task_id] = {
        "status": "queued",
        "mode": "anonymize",
        "message": "Queued",
        "logs": [],
        "worker_count": 1,
        "transcribe_enabled": True,
        "video_ready": True,
        "transcript_ready": True,
        "has_original_audio": False,
        "can_transcribe": True,
        "is_post_transcribe": True,
        "asr_model": "gigaam-v3",
        "language": "ru",
        "speakers": "auto",
        "compute_device": "cpu",
    }

    worker = threading.Thread(
        target=background_transcribe_only,
        args=(task_id, out_mp4, out_txt, 1, "gigaam-v3", "ru", "auto", "cpu", True, False, False),
        daemon=True,
    )
    try:
        worker.start()
        assert entered.wait(timeout=10.0)

        cancel_resp = client.post(f"/api/cancel/{task_id}")
        assert cancel_resp.status_code == 200
        cancel_json = cancel_resp.json()
        assert cancel_json["cancelled"] is True
        assert cancel_json["is_post_transcribe"] is True
        assert cancel_json["video_ready"] is True
        assert cancel_json["transcript_ready"] is True
        assert cancel_json["can_transcribe"] is True

        worker.join(timeout=3.0)
        assert not worker.is_alive()

        # Both the original anonymized video AND the previous transcript MUST remain intact & downloadable!
        assert os.path.isfile(out_mp4)
        assert os.path.isfile(out_txt)
        assert open(out_txt, encoding="utf-8").read() == "PREVIOUS_TRANSCRIPT_INTACT\n"

        st = client.get(f"/api/status/{task_id}").json()
        assert st["status"] == "cancelled"
        assert st["video_ready"] is True
        assert st["transcript_ready"] is True
        assert client.get(f"/api/download/{task_id}").status_code == 200
        assert client.get(f"/api/download-transcript/{task_id}").status_code == 200
    finally:
        tasks.pop(task_id, None)
        cancel_tokens.pop(task_id, None)
        for p in (out_mp4, out_txt, _audio_cache_path(task_id)):
            if os.path.exists(p):
                os.remove(p)


def test_transcribe_only_mode_mp4_wav_and_cli_skips_video_processing(tmp_path, monkeypatch):
    from app import _audio_cache_path
    from main import process_video
    from src.video_processor import VideoProcessor

    mp4_path = str(tmp_path / "only_tr.mp4")
    wav_path = str(tmp_path / "only_tr.wav")
    _generate_test_mp4(mp4_path, duration_sec=1.2, with_audio=True)
    FFmpegUtils.extract_transcription_audio(mp4_path, wav_path)

    def _forbid_video_process(*args, **kwargs):
        raise AssertionError("VideoProcessor.process MUST NOT be called in transcribe_only mode")

    def _forbid_split(*args, **kwargs):
        raise AssertionError("FFmpegUtils.split_video MUST NOT be called in transcribe_only mode")

    monkeypatch.setattr(VideoProcessor, "process", _forbid_video_process)
    monkeypatch.setattr(FFmpegUtils, "split_video", staticmethod(_forbid_split))

    seen_models = []

    def _mock_transcribe(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = "gigaam-v3",
        speaker_count: str = "auto",
        threads: int = 1,
        diar_threads=None,
        compute_device: str = "cpu",
        status_cb=None,
        cancel_token=None,
        language: str = "ru",
        custom_prompt: str = "",
        diarization_sensitivity: str = "normal",
        **kwargs,
    ):
        seen_models.append((asr_model, language))
        turns = [SpeakerTurn(0.0, 1.0, "Спикер 1", "Тест только транскрибации.")]
        txt = format_transcript_txt(turns, duration_sec=1.2, model_name=asr_model, num_speakers=1)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.01}

    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_mock_transcribe))

    client = TestClient(app)
    created_ids = []
    try:
        # 1. API transcribe_only with MP4
        with open(mp4_path, "rb") as f:
            r1 = client.post(
                "/api/upload",
                files={"file": ("meeting.mp4", f, "video/mp4")},
                data={"mode": "transcribe_only", "asr_model": "gigaam-v3", "language": "ru", "workers": "1"},
            )
        assert r1.status_code == 200
        assert r1.json()["mode"] == "transcribe_only"
        tid1 = r1.json()["task_id"]
        created_ids.append(tid1)

        st1 = client.get(f"/api/status/{tid1}").json()
        assert st1["status"] == "completed"
        assert st1["mode"] == "transcribe_only"
        assert st1["video_ready"] is False
        assert st1["transcript_ready"] is True
        assert st1["can_transcribe"] is True
        assert client.get(f"/api/download/{tid1}").status_code == 404
        dl1 = client.get(f"/api/download-transcript/{tid1}")
        assert dl1.status_code == 200
        assert "Тест только транскрибации." in dl1.text

        # 2. API transcribe_only with audio-only .wav file
        with open(wav_path, "rb") as f:
            r2 = client.post(
                "/api/upload",
                files={"file": ("voice_note.wav", f, "audio/wav")},
                data={"mode": "transcribe_only", "asr_model": "breeze-asr-25", "language": "ru", "workers": "1"},
            )
        assert r2.status_code == 200
        tid2 = r2.json()["task_id"]
        created_ids.append(tid2)

        st2 = client.get(f"/api/status/{tid2}").json()
        assert st2["status"] == "completed"
        assert st2["transcript_ready"] is True
        assert client.get(f"/api/download-transcript/{tid2}").status_code == 200

        # 3. CLI process_video(transcribe_only=True) without output_path
        cli_txt = str(tmp_path / "cli_only.txt")
        cli_res = process_video(
            input_path=wav_path,
            output_path=None,
            transcript_path=cli_txt,
            asr_model="gigaam-v3",
            language="ru",
            workers=1,
            transcribe_only=True,
        )
        assert cli_res["video_ready"] is False
        assert cli_res["transcript_ready"] is True
        assert os.path.isfile(cli_txt)
        assert seen_models == [("gigaam-v3", "ru"), ("breeze-asr-25", "ru"), ("gigaam-v3", "ru")]
    finally:
        for tid in created_ids:
            tasks.pop(tid, None)
            for p in (
                os.path.join("output", f"{tid}_anonymized.mp4"),
                os.path.join("output", f"{tid}_transcript.txt"),
                _audio_cache_path(tid),
            ):
                if os.path.exists(p):
                    os.remove(p)


def test_frontend_v45_modes_russian_models_and_post_transcribe():
    """
    Executes static/script.js in Node.js to verify v4.5.0 frontend flows:
      1. Default model is GigaAM v3; switching GPU (CUDA) when ONNX CUDA is off auto-switches GigaAM v3 -> Breeze-ASR-25.
      2. Switching to 'Только транскрибация' updates dropzone, accepts .wav/.mp3, sends mode='transcribe_only',
         hides video download button, and shows transcript download button.
      3. Anonymization with transcribe=false displays the 'Создать транскрипт отдельно' panel on completion
         and clicking #postTranscribeBtn calls POST /api/transcribe/{task_id}.
      4. GET /api/recordings populates #recordingsList with 'Создать транскрипт' actions.
    """
    node_test_js = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const storage = new Map();
const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
};

class ClassList {
    constructor(el, initial = []) { this.el = el; this.set = new Set(initial.filter(Boolean)); }
    add(...cls) { cls.forEach((c) => this.set.add(c)); }
    remove(...cls) { cls.forEach((c) => this.set.delete(c)); }
    contains(c) { return this.set.has(c); }
    toggle(c, force) {
        const next = force !== undefined ? Boolean(force) : !this.set.has(c);
        if (next) this.set.add(c); else this.set.delete(c);
        return next;
    }
}

class FakeEvent {
    constructor(type, opts = {}) {
        this.type = type;
        this.bubbles = Boolean(opts.bubbles);
        this.defaultPrevented = false;
        this.propagationStopped = false;
    }
    preventDefault() { this.defaultPrevented = true; }
    stopPropagation() { this.propagationStopped = true; }
}

const byId = new Map();
let activeElement = null;

class FakeElement {
    constructor(tagName, { id = '', classes = [], attrs = {}, text = '' } = {}) {
        this.tagName = tagName.toUpperCase();
        this.id = id;
        if (id) byId.set(id, this);
        this.classList = new ClassList(this, classes);
        this.attrs = { ...attrs };
        this.dataset = {};
        this.children = [];
        this.parentNode = null;
        this.listeners = {};
        this._textContent = text;
        this._innerHTML = '';
        this.disabled = false;
        this.checked = Boolean(attrs.checked);
        this.title = '';
        this.accept = attrs.accept || '';
        this.tabIndex = 0;
        this.scrollTop = 0;
        this.scrollHeight = 100;
        this.files = [];
        this._value = attrs.value !== undefined ? String(attrs.value) : '';
        this.selected = Boolean(attrs.selected);
        this.onclick = null;
    }
    get className() { return Array.from(this.classList.set).join(' '); }
    set className(v) { this.classList = new ClassList(this, String(v).split(/\s+/)); }
    get textContent() {
        if (this.children.length === 0) return this._textContent;
        return this._textContent + this.children.map((c) => c.textContent).join('');
    }
    set textContent(v) { this._textContent = String(v); this.children = []; }
    get innerHTML() { return this._innerHTML; }
    set innerHTML(v) {
        this._innerHTML = String(v);
        if (v === '') {
            this.children.forEach((c) => { c.parentNode = null; });
            this.children = [];
        }
    }
    get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
    get selectedIndex() {
        const opts = this.options;
        const idx = opts.findIndex((o) => o.value === this.value);
        return idx >= 0 ? idx : 0;
    }
    get value() {
        if (this.tagName === 'SELECT') {
            const opts = this.options;
            if (this._value && opts.some((o) => o.value === this._value)) return this._value;
            const sel = opts.find((o) => o.selected) || opts[0];
            return sel ? sel.value : '';
        }
        return this._value;
    }
    set value(v) { this._value = String(v); }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return this.attrs[k] ?? null; }
    appendChild(child) {
        if (child.parentNode) {
            const idx = child.parentNode.children.indexOf(child);
            if (idx >= 0) child.parentNode.children.splice(idx, 1);
        }
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    insertBefore(newNode, refNode) {
        if (newNode.parentNode) {
            const oldIdx = newNode.parentNode.children.indexOf(newNode);
            if (oldIdx >= 0) newNode.parentNode.children.splice(oldIdx, 1);
        }
        newNode.parentNode = this;
        const idx = this.children.indexOf(refNode);
        if (idx >= 0) this.children.splice(idx, 0, newNode);
        else this.children.push(newNode);
        return newNode;
    }
    addEventListener(type, fn) {
        if (!this.listeners[type]) this.listeners[type] = [];
        this.listeners[type].push(fn);
    }
    dispatchEvent(evt) {
        (this.listeners[evt.type] || []).forEach((fn) => fn.call(this, evt));
        if (evt.type === 'click' && typeof this.onclick === 'function') this.onclick.call(this, evt);
        if (evt.bubbles && !evt.propagationStopped) {
            if (this.parentNode) this.parentNode.dispatchEvent(evt);
            else documentObj.dispatchEvent(evt);
        }
        return !evt.defaultPrevented;
    }
    click() { this.dispatchEvent(new FakeEvent('click', { bubbles: true })); }
    focus() { activeElement = this; }
    matchesSelector(sel) {
        if (sel === '.sun-icon') return this.classList.contains('sun-icon');
        if (sel === '.moon-icon') return this.classList.contains('moon-icon');
        const lm = sel.match(/^label\[for="(.+)"\]$/);
        if (lm) return this.tagName === 'LABEL' && this.attrs.for === lm[1];
        if (sel === '.clay-select-option:not(:disabled)') return this.classList.contains('clay-select-option') && !this.disabled;
        if (sel === '.clay-select-option.is-selected:not(:disabled)') {
            return this.classList.contains('clay-select-option') && this.classList.contains('is-selected') && !this.disabled;
        }
        return false;
    }
    querySelectorAll(sel) {
        const out = [];
        const walk = (n) => { n.children.forEach((ch) => { if (ch.matchesSelector(sel)) out.push(ch); walk(ch); }); };
        walk(this);
        return out;
    }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

const root = new FakeElement('div', { id: 'root' });
const makeSelect = (id, opts) => {
    const container = new FakeElement('div');
    const sel = new FakeElement('select', { id, classes: ['worker-select'] });
    opts.forEach(([val, text, selected]) => sel.appendChild(new FakeElement('option', { attrs: { value: val, selected }, text })));
    container.appendChild(sel);
    root.appendChild(container);
    return sel;
};

const themeBtn = root.appendChild(new FakeElement('button', { id: 'themeToggle' }));
themeBtn.appendChild(new FakeElement('svg', { classes: ['sun-icon', 'hidden'] }));
themeBtn.appendChild(new FakeElement('svg', { classes: ['moon-icon', 'hidden'] }));

const uploadPanel = root.appendChild(new FakeElement('div', { id: 'uploadPanel' }));
const modeAnonymizeBtn = uploadPanel.appendChild(new FakeElement('button', { id: 'modeAnonymizeBtn', classes: ['mode-tab', 'is-active'] }));
const modeTranscribeOnlyBtn = uploadPanel.appendChild(new FakeElement('button', { id: 'modeTranscribeOnlyBtn', classes: ['mode-tab'] }));
uploadPanel.appendChild(new FakeElement('div', { id: 'dropZone' }));
const dropZoneTitle = uploadPanel.appendChild(new FakeElement('h2', { id: 'dropZoneTitle' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'dropZoneHint' }));
const fileInput = uploadPanel.appendChild(new FakeElement('input', { id: 'fileInput', attrs: { accept: '.mp4' } }));
uploadPanel.appendChild(new FakeElement('span', { id: 'workerValue', text: '4' }));
makeSelect('workerSelect', [['1', '1'], ['4', '4', true], ['custom', 'Другое']]);
uploadPanel.appendChild(new FakeElement('div', { id: 'customWorkerWrap', classes: ['hidden'] }));
uploadPanel.appendChild(new FakeElement('input', { id: 'customWorkerInput' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'workerLimitText' }));
const transcribeToggle = uploadPanel.appendChild(new FakeElement('input', { id: 'transcribeToggle', attrs: { checked: true } }));
const transcribeSwitchWrap = uploadPanel.appendChild(new FakeElement('label', { id: 'transcribeSwitchWrap' }));
uploadPanel.appendChild(new FakeElement('span', { id: 'transcribeStateBadge', text: 'Вкл' }));
uploadPanel.appendChild(new FakeElement('div', { id: 'transcribeOptionsWrap' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'transcribeHintText' }));

const modelOpts = [
    ['gigaam-v3', 'GigaAM v3 FP32 — эталон для русского (RU, GPU/CPU, 892 МБ)', true],
    ['breeze-asr-25', 'Breeze-ASR-25 — лучший для RU+EN IT-речи (3.1 ГБ)'],
    ['turbo', 'Whisper Turbo — универсальная (99 яз., 810 МБ)'],
    ['deepgram-nova-3', 'Deepgram Nova-3 — облачный API'],
];
const langOpts = [['ru', 'Русский (RU)', true], ['auto', 'Авто (мультиязычный)'], ['en', 'English (EN)']];
const devOpts = [['cpu', 'CPU', true], ['cuda', 'GPU (CUDA)']];
const spkOpts = [['auto', 'Авто', true], ['2', '2 спикера']];

const asrModelSelect = makeSelect('asrModelSelect', modelOpts);
const languageSelect = makeSelect('languageSelect', langOpts);
const computeDeviceSelect = makeSelect('computeDeviceSelect', devOpts);
makeSelect('speakerCountSelect', spkOpts);

const recordingsSection = uploadPanel.appendChild(new FakeElement('div', { id: 'recordingsSection', classes: ['hidden'] }));
uploadPanel.appendChild(new FakeElement('span', { id: 'recordingsCount' }));
const recordingsList = uploadPanel.appendChild(new FakeElement('div', { id: 'recordingsList' }));

const statusPanel = root.appendChild(new FakeElement('div', { id: 'statusPanel', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('div', { id: 'spinner' }));
statusPanel.appendChild(new FakeElement('svg', { id: 'successIcon', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('svg', { id: 'cancelIcon', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('h2', { id: 'statusText' }));
statusPanel.appendChild(new FakeElement('p', { id: 'statusSubText' }));
statusPanel.appendChild(new FakeElement('div', { id: 'debugLog', classes: ['hidden'] }));
const cancelBtn = statusPanel.appendChild(new FakeElement('button', { id: 'cancelBtn', classes: ['hidden'] }));
const downloadBtn = statusPanel.appendChild(new FakeElement('button', { id: 'downloadBtn', classes: ['hidden'] }));
const downloadTranscriptBtn = statusPanel.appendChild(new FakeElement('button', { id: 'downloadTranscriptBtn', classes: ['hidden'] }));
const postTranscribeSection = statusPanel.appendChild(new FakeElement('div', { id: 'postTranscribeSection', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'togglePostTranscribeBtn', classes: ['hidden'] }));
const postTranscribeForm = statusPanel.appendChild(new FakeElement('div', { id: 'postTranscribeForm', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('span', { id: 'postTranscribeTitle' }));
makeSelect('postAsrModelSelect', modelOpts);
makeSelect('postLanguageSelect', langOpts);
makeSelect('postComputeDeviceSelect', devOpts);
makeSelect('postSpeakerCountSelect', spkOpts);
statusPanel.appendChild(new FakeElement('p', { id: 'postAudioSourceHint' }));
const postTranscribeBtn = statusPanel.appendChild(new FakeElement('button', { id: 'postTranscribeBtn' }));
const resetBtn = statusPanel.appendChild(new FakeElement('button', { id: 'resetBtn', classes: ['hidden'] }));

const docListeners = {};
const documentObj = {
    documentElement: new FakeElement('html'),
    get activeElement() { return activeElement; },
    getElementById: (id) => byId.get(id) || null,
    querySelector: (sel) => root.querySelector(sel),
    querySelectorAll: (sel) => root.querySelectorAll(sel),
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    addEventListener: (t, fn) => { (docListeners[t] ||= []).push(fn); },
    dispatchEvent: (evt) => { (docListeners[evt.type] || []).forEach((fn) => fn(evt)); },
};

class FakeFormData {
    constructor() { this.entries = {}; }
    append(k, v) { this.entries[k] = v; }
}

const uploadCalls = [];
const postTranscribeCalls = [];
let statusResponses = {};

const fakeFetch = async (url, opts = {}) => {
    if (url === '/api/config') {
        return {
            ok: true,
            json: async () => ({
                version: '5.1.0',
                cpu_count: 8,
                default_workers: 4,
                max_workers: 7,
                transcription_default: true,
                default_asr_model: 'gigaam-v3',
                default_language: 'ru',
                asr_models: [
                    { id: 'gigaam-v3', label: 'GigaAM v3 FP32 — эталон для русского (RU, GPU/CPU, 892 МБ)', supports_cuda: false },
                    { id: 'breeze-asr-25', label: 'Breeze-ASR-25 — лучший для RU+EN IT-речи (3.1 ГБ)', supports_cuda: true },
                    { id: 'turbo', label: 'Whisper Turbo — универсальная (99 яз., 810 МБ)', supports_cuda: true },
                    { id: 'deepgram-nova-3', label: 'Deepgram Nova-3 — облачный API', supports_cuda: false, is_cloud: true },
                ],
                languages: [
                    { id: 'ru', label: 'Русский (RU)' },
                    { id: 'auto', label: 'Авто (мультиязычный)' },
                    { id: 'en', label: 'English (EN)' },
                ],
                cuda_available: true,
                default_compute_device: 'cpu',
                compute_devices: [{ id: 'cpu', label: 'CPU' }, { id: 'cuda', label: 'GPU (CUDA)' }],
            }),
        };
    }
    if (url === '/api/recordings') {
        return {
            ok: true,
            json: async () => ({
                recordings: [
                    {
                        task_id: 'old-rec-1',
                        original_filename: 'saved_interview.mp4',
                        video_ready: true,
                        video_size_bytes: 5 * 1024 * 1024,
                        transcript_ready: false,
                        has_original_audio: false,
                        can_transcribe: true,
                    },
                ],
            }),
        };
    }
    if (url === '/api/upload') {
        const payload = { ...opts.body.entries };
        uploadCalls.push(payload);
        const tid = `t-${uploadCalls.length}`;
        if (payload.mode === 'transcribe_only') {
            statusResponses[tid] = {
                status: 'completed',
                mode: 'transcribe_only',
                video_ready: false,
                transcript_ready: true,
                can_transcribe: true,
                has_original_audio: true,
                logs: ['Done!'],
            };
        } else {
            statusResponses[tid] = {
                status: 'completed',
                mode: 'anonymize',
                video_ready: true,
                transcript_ready: payload.transcribe === 'true',
                can_transcribe: true,
                has_original_audio: true,
                logs: ['Done!'],
            };
        }
        return { ok: true, json: async () => ({ task_id: tid, status: 'queued', worker_count: 4 }) };
    }
    if (url.startsWith('/api/transcribe/')) {
        const tid = url.split('/').pop();
        postTranscribeCalls.push({ tid, payload: { ...opts.body.entries } });
        if (tid === 'old-rec-1') {
            statusResponses[tid] = {
                status: 'processing',
                mode: 'anonymize',
                video_ready: true,
                transcript_ready: false,
                can_transcribe: true,
                has_original_audio: true,
                is_post_transcribe: true,
                logs: ['Working...'],
            };
        } else {
            statusResponses[tid] = {
                status: 'completed',
                mode: 'anonymize',
                video_ready: true,
                transcript_ready: true,
                can_transcribe: true,
                has_original_audio: true,
                is_post_transcribe: true,
                logs: ['Done!'],
            };
        }
        return { ok: true, json: async () => ({ task_id: tid, status: 'queued' }) };
    }
    if (url.startsWith('/api/cancel/')) {
        const tid = url.split('/').pop();
        return {
            ok: true,
            json: async () => ({
                task_id: tid,
                status: 'cancelled',
                cancelled: true,
                mode: 'anonymize',
                is_post_transcribe: true,
                video_ready: true,
                transcript_ready: false,
                has_original_audio: true,
                can_transcribe: true,
            }),
        };
    }
    if (url.startsWith('/api/status/')) {
        const tid = url.split('/').pop();
        return { ok: true, json: async () => statusResponses[tid] };
    }
    throw new Error('Unexpected URL: ' + url);
};

const ctx = vm.createContext({
    document: documentObj,
    window: { location: { href: '' }, matchMedia: () => ({ matches: false }) },
    localStorage,
    Event: FakeEvent,
    FormData: FakeFormData,
    AbortController: class { constructor() { this.signal = {}; } abort() {} },
    fetch: fakeFetch,
    setInterval: () => 1,
    clearInterval: () => {},
    alert: (m) => { throw new Error('Unexpected alert: ' + m); },
    console,
    Number, String, Boolean, Array, Map, Set, Date,
});
vm.runInContext(fs.readFileSync('static/script.js', 'utf8'), ctx);

(async () => {
    await new Promise((r) => setTimeout(r, 25));

    // 1. Default model is gigaam-v3 + ru, and recordings list rendered from /api/recordings
    assert.strictEqual(asrModelSelect.value, 'gigaam-v3');
    assert.strictEqual(languageSelect.value, 'ru');
    assert.strictEqual(recordingsSection.classList.contains('hidden'), false);
    assert.strictEqual(recordingsList.children.length, 1);

    // 2. Selecting GPU (CUDA) while gigaam-v3 is CPU-only (no ONNX CUDA) auto-switches to breeze-asr-25!
    const devMenuBtns = computeDeviceSelect.parentNode.children.find((c) => c.classList.contains('clay-select-menu')).children;
    assert.strictEqual(devMenuBtns[1].disabled, false);
    devMenuBtns[1].click();
    assert.strictEqual(computeDeviceSelect.value, 'cuda');
    assert.strictEqual(asrModelSelect.value, 'breeze-asr-25');

    // 3. Switch to 'Только транскрибация' mode -> accepts .wav, hides video download button!
    modeTranscribeOnlyBtn.click();
    assert.ok(dropZoneTitle.textContent.includes('аудио'));
    assert.ok(fileInput.accept.includes('.wav'));
    assert.strictEqual(transcribeSwitchWrap.classList.contains('hidden'), true);

    fileInput.files = [{ name: 'voice_memo.wav' }];
    fileInput.dispatchEvent(new ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 25));

    assert.strictEqual(uploadCalls.length, 1);
    assert.strictEqual(uploadCalls[0].mode, 'transcribe_only');
    assert.strictEqual(uploadCalls[0].language, 'ru');
    assert.strictEqual(uploadCalls[0].asr_model, 'breeze-asr-25');
    assert.strictEqual(downloadBtn.classList.contains('hidden'), true, 'Video download must stay hidden in transcribe_only');
    assert.strictEqual(downloadTranscriptBtn.classList.contains('hidden'), false);

    // 4. Return, switch to 'Анонимизация' with transcribe=false -> post-transcribe form opens immediately!
    resetBtn.click();
    modeAnonymizeBtn.click();
    transcribeToggle.checked = false;
    transcribeToggle.dispatchEvent(new ctx.Event('change'));

    fileInput.files = [{ name: 'interview.mp4' }];
    fileInput.dispatchEvent(new ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 25));

    assert.strictEqual(uploadCalls.length, 2);
    assert.strictEqual(uploadCalls[1].mode, 'anonymize');
    assert.strictEqual(uploadCalls[1].transcribe, 'false');
    assert.strictEqual(downloadBtn.classList.contains('hidden'), false);
    assert.strictEqual(downloadTranscriptBtn.classList.contains('hidden'), true);
    assert.strictEqual(postTranscribeSection.classList.contains('hidden'), false);
    assert.strictEqual(postTranscribeForm.classList.contains('hidden'), false);

    // 5. Click 'Создать транскрипт' -> calls POST /api/transcribe/t-2 and unlocks transcript download!
    postTranscribeBtn.click();
    await new Promise((r) => setTimeout(r, 25));

    assert.strictEqual(postTranscribeCalls.length, 1);
    assert.strictEqual(postTranscribeCalls[0].tid, 't-2');
    assert.strictEqual(downloadBtn.classList.contains('hidden'), false);
    assert.strictEqual(downloadTranscriptBtn.classList.contains('hidden'), false);

    // 6. Return to home, click 'Настроить' on saved recording, start post-transcribe and cancel immediately
    resetBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    const recCard = recordingsList.children[0];
    const actionsRow = recCard.children[2];
    const cfgChip = actionsRow.children.find((b) => b.textContent === 'Настроить');
    assert.ok(cfgChip, 'Expected Настроить button on recording card');
    cfgChip.click();
    assert.strictEqual(statusPanel.classList.contains('hidden'), false);
    assert.strictEqual(postTranscribeForm.classList.contains('hidden'), false);
    assert.strictEqual(downloadBtn.classList.contains('hidden'), false);

    postTranscribeBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    await cancelBtn.onclick();
    assert.strictEqual(downloadBtn.classList.contains('hidden'), false, 'Video download must stay visible after cancelling post-transcribe');
    assert.strictEqual(postTranscribeSection.classList.contains('hidden'), false, 'Post-transcribe section must stay visible after cancel');

    console.log('FRONTEND_V45_ALL_FEATURES_OK');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_test_js],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node v4.5 UI test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FRONTEND_V45_ALL_FEATURES_OK" in proc.stdout


def test_ffmpeg_metadata_handles_na_duration_and_invalid_sample_rate(monkeypatch):
    import ffmpeg as ffmpeg_pkg

    def _fake_probe(_path):
        return {
            "format": {"duration": "N/A"},
            "streams": [
                {"codec_type": "audio", "sample_rate": "N/A", "duration": "12.5"},
            ],
        }

    monkeypatch.setattr(ffmpeg_pkg, "probe", _fake_probe)
    width, height, fps, duration, has_audio, sample_rate = FFmpegUtils.get_video_metadata("stream.webm")
    assert width is None
    assert height is None
    assert has_audio is True
    assert abs(duration - 12.5) < 1e-6
    assert sample_rate == 44100


def test_dual_track_obs_mp4_automatic_split_and_mix_e2e(tmp_path, monkeypatch):
    from app import _audio_cache_path, _audio_cache_track2_path

    obs_mp4 = str(tmp_path / "obs_dual_track.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=1.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=44100:duration=1.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=44100:duration=1.6",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            obs_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    streams = FFmpegUtils.get_audio_streams(obs_mp4)
    assert len(streams) == 2

    dual_calls = []
    single_calls = []

    def _mock_dual(
        cls,
        track1_wav_path: str,
        track2_wav_path: str,
        output_txt_path: str,
        asr_model: str = "gigaam-v3",
        speaker_count: str = "auto",
        threads: int = 1,
        diar_threads=None,
        compute_device: str = "cpu",
        status_cb=None,
        cancel_token=None,
        language: str = "ru",
        custom_prompt: str = "",
        diarization_sensitivity: str = "normal",
        **kwargs,
    ):
        assert os.path.isfile(track1_wav_path)
        assert os.path.isfile(track2_wav_path)
        assert os.path.abspath(track1_wav_path) != os.path.abspath(track2_wav_path)
        with wave.open(track1_wav_path, "rb") as w1, wave.open(track2_wav_path, "rb") as w2:
            assert w1.getframerate() == 16000 and w1.getnchannels() == 1
            assert w2.getframerate() == 16000 and w2.getnchannels() == 1
        dual_calls.append(
            {
                "t1": track1_wav_path,
                "t2": track2_wav_path,
                "speakers": speaker_count,
                "sensitivity": diarization_sensitivity,
                "prompt": custom_prompt,
            }
        )
        txt = format_transcript_txt(
            [
                SpeakerTurn(0.1, 0.8, "Спикер 1", "Привет из микрофона."),
                SpeakerTurn(0.9, 1.5, "Спикер 2", "Привет из второй дорожки."),
            ],
            duration_sec=1.6,
            model_name=asr_model,
            num_speakers=2,
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return {"num_speakers": 2, "num_turns": 2, "elapsed": 0.01, "dual_track": True}

    def _mock_single(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = "gigaam-v3",
        speaker_count: str = "auto",
        threads: int = 1,
        diar_threads=None,
        compute_device: str = "cpu",
        status_cb=None,
        cancel_token=None,
        language: str = "ru",
        custom_prompt: str = "",
        diarization_sensitivity: str = "normal",
        **kwargs,
    ):
        assert os.path.isfile(wav_path)
        single_calls.append({"wav": wav_path, "prompt": custom_prompt, "sensitivity": diarization_sensitivity})
        txt = format_transcript_txt(
            [SpeakerTurn(0.1, 1.4, "Спикер 1", "Смикшированный звук.")],
            duration_sec=1.6,
            model_name=asr_model,
            num_speakers=1,
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.01}

    monkeypatch.setattr(Transcriber, "transcribe_dual_track_files", classmethod(_mock_dual))
    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_mock_single))

    client = TestClient(app)
    created_ids = []
    try:
        # 1. Auto mode on 2-track OBS MP4 -> automatic dual-track split + caches BOTH tracks + mixes audio in anon MP4
        with open(obs_mp4, "rb") as f:
            r1 = client.post(
                "/api/upload",
                files={"file": ("obs_interview.mp4", f, "video/mp4")},
                data={
                    "workers": "1",
                    "mode": "anonymize",
                    "transcribe": "true",
                    "asr_model": "gigaam-v3",
                    "speakers": "4",
                    "diarization_sensitivity": "high",
                    "audio_tracks": "auto",
                    "custom_prompt": "REST, gRPC, OAuth 2",
                },
            )
        assert r1.status_code == 200
        tid1 = r1.json()["task_id"]
        created_ids.append(tid1)

        st1 = client.get(f"/api/status/{tid1}").json()
        assert st1["status"] == "completed"
        assert st1["video_ready"] is True
        assert st1["transcript_ready"] is True
        assert len(dual_calls) == 1
        assert dual_calls[0]["speakers"] == "4"
        assert dual_calls[0]["sensitivity"] == "high"
        assert dual_calls[0]["prompt"] == "REST, gRPC, OAuth 2"
        assert os.path.isfile(_audio_cache_path(tid1))
        assert os.path.isfile(_audio_cache_track2_path(tid1))

        # Anonymized video must contain an audio stream (mixed from both OBS tracks)
        out_mp4 = os.path.join("output", f"{tid1}_anonymized.mp4")
        assert len(FFmpegUtils.get_audio_streams(out_mp4)) == 1

        # Post-recording re-transcription reuses BOTH cached original tracks!
        r_post = client.post(
            f"/api/transcribe/{tid1}",
            data={
                "workers": "1",
                "asr_model": "breeze-asr-25",
                "speakers": "4",
                "diarization_sensitivity": "high",
                "audio_tracks": "separate",
                "custom_prompt": "REST, gRPC",
            },
        )
        assert r_post.status_code == 200
        assert len(dual_calls) == 2
        assert os.path.abspath(dual_calls[1]["t1"]) == os.path.abspath(_audio_cache_path(tid1))
        assert os.path.abspath(dual_calls[1]["t2"]) == os.path.abspath(_audio_cache_track2_path(tid1))

        # 2. Explicit audio_tracks='mix' in transcribe_only mode -> mixes tracks and calls transcribe_audio_file
        with open(obs_mp4, "rb") as f:
            r2 = client.post(
                "/api/upload",
                files={"file": ("obs_mix.mp4", f, "video/mp4")},
                data={
                    "workers": "1",
                    "mode": "transcribe_only",
                    "asr_model": "gigaam-v3",
                    "audio_tracks": "mix",
                    "diarization_sensitivity": "low",
                    "custom_prompt": "Kafka, Redis",
                },
            )
        assert r2.status_code == 200
        tid2 = r2.json()["task_id"]
        created_ids.append(tid2)

        st2 = client.get(f"/api/status/{tid2}").json()
        assert st2["status"] == "completed"
        assert st2["transcript_ready"] is True
        assert len(dual_calls) == 2
        assert len(single_calls) == 1
        assert single_calls[0]["sensitivity"] == "low"
        assert single_calls[0]["prompt"] == "Kafka, Redis"
    finally:
        for tid in created_ids:
            tasks.pop(tid, None)
            for p in (
                os.path.join("output", f"{tid}_anonymized.mp4"),
                os.path.join("output", f"{tid}_transcript.txt"),
                _audio_cache_path(tid),
                _audio_cache_track2_path(tid),
            ):
                if os.path.exists(p):
                    os.remove(p)


def test_frontend_v46_glossary_presets_sensitivity_and_tracks():
    """
    Executes static/script.js in Node.js to verify v4.6.0 frontend features:
      1. Custom prompt starts empty (no unsolicited SQL prompt).
      2. Clicking 'Системный аналитик' preset chip populates #customPromptInput with the
         System Analyst glossary and shows the GigaAM Latin-terms hint in #transcribeHintText.
      3. Selecting speakers='4', diarization_sensitivity='high', audio_tracks='separate'
         submits all v4.6.0 fields to /api/upload and propagates them to /api/transcribe/{tid}.
      4. Clicking 'Очистить' clears the prompt back to ''.
    """
    node_test_js = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const storage = new Map();
const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
};

class ClassList {
    constructor(el, initial = []) { this.el = el; this.set = new Set(initial.filter(Boolean)); }
    add(...cls) { cls.forEach((c) => this.set.add(c)); }
    remove(...cls) { cls.forEach((c) => this.set.delete(c)); }
    contains(c) { return this.set.has(c); }
    toggle(c, force) {
        const next = force !== undefined ? Boolean(force) : !this.set.has(c);
        if (next) this.set.add(c); else this.set.delete(c);
        return next;
    }
}

class FakeEvent {
    constructor(type, opts = {}) {
        this.type = type;
        this.bubbles = Boolean(opts.bubbles);
        this.defaultPrevented = false;
        this.propagationStopped = false;
    }
    preventDefault() { this.defaultPrevented = true; }
    stopPropagation() { this.propagationStopped = true; }
}

const byId = new Map();
let activeElement = null;

class FakeElement {
    constructor(tagName, { id = '', classes = [], attrs = {}, dataset = {}, text = '' } = {}) {
        this.tagName = tagName.toUpperCase();
        this.id = id;
        if (id) byId.set(id, this);
        this.classList = new ClassList(this, classes);
        this.attrs = { ...attrs };
        this.dataset = { ...dataset };
        this.children = [];
        this.parentNode = null;
        this.listeners = {};
        this._textContent = text;
        this._innerHTML = '';
        this.disabled = false;
        this.checked = Boolean(attrs.checked);
        this.title = '';
        this.accept = attrs.accept || '';
        this.tabIndex = 0;
        this.scrollTop = 0;
        this.scrollHeight = 100;
        this.files = [];
        this._value = attrs.value !== undefined ? String(attrs.value) : '';
        this.selected = Boolean(attrs.selected);
        this.onclick = null;
    }
    get className() { return Array.from(this.classList.set).join(' '); }
    set className(v) { this.classList = new ClassList(this, String(v).split(/\s+/)); }
    get textContent() {
        if (this.children.length === 0) return this._textContent;
        return this._textContent + this.children.map((c) => c.textContent).join('');
    }
    set textContent(v) { this._textContent = String(v); this.children = []; }
    get innerHTML() { return this._innerHTML; }
    set innerHTML(v) {
        this._innerHTML = String(v);
        if (v === '') {
            this.children.forEach((c) => { c.parentNode = null; });
            this.children = [];
        }
    }
    get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
    get selectedIndex() {
        const opts = this.options;
        const idx = opts.findIndex((o) => o.value === this.value);
        return idx >= 0 ? idx : 0;
    }
    get value() {
        if (this.tagName === 'SELECT') {
            const opts = this.options;
            if (this._value && opts.some((o) => o.value === this._value)) return this._value;
            const sel = opts.find((o) => o.selected) || opts[0];
            return sel ? sel.value : '';
        }
        return this._value;
    }
    set value(v) { this._value = String(v); }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return this.attrs[k] ?? null; }
    appendChild(child) {
        if (child.parentNode) {
            const idx = child.parentNode.children.indexOf(child);
            if (idx >= 0) child.parentNode.children.splice(idx, 1);
        }
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    insertBefore(newNode, refNode) {
        if (newNode.parentNode) {
            const oldIdx = newNode.parentNode.children.indexOf(newNode);
            if (oldIdx >= 0) newNode.parentNode.children.splice(oldIdx, 1);
        }
        newNode.parentNode = this;
        const idx = this.children.indexOf(refNode);
        if (idx >= 0) this.children.splice(idx, 0, newNode);
        else this.children.push(newNode);
        return newNode;
    }
    addEventListener(type, fn) {
        if (!this.listeners[type]) this.listeners[type] = [];
        this.listeners[type].push(fn);
    }
    dispatchEvent(evt) {
        (this.listeners[evt.type] || []).forEach((fn) => fn.call(this, evt));
        if (evt.type === 'click' && typeof this.onclick === 'function') this.onclick.call(this, evt);
        if (evt.bubbles && !evt.propagationStopped) {
            if (this.parentNode) this.parentNode.dispatchEvent(evt);
            else documentObj.dispatchEvent(evt);
        }
        return !evt.defaultPrevented;
    }
    click() { this.dispatchEvent(new FakeEvent('click', { bubbles: true })); }
    focus() { activeElement = this; }
    matchesSelector(sel) {
        if (sel === '.sun-icon') return this.classList.contains('sun-icon');
        if (sel === '.moon-icon') return this.classList.contains('moon-icon');
        if (sel === '.prompt-chip') return this.classList.contains('prompt-chip');
        const lm = sel.match(/^label\[for="(.+)"\]$/);
        if (lm) return this.tagName === 'LABEL' && this.attrs.for === lm[1];
        if (sel === '.clay-select-option:not(:disabled)') return this.classList.contains('clay-select-option') && !this.disabled;
        if (sel === '.clay-select-option.is-selected:not(:disabled)') {
            return this.classList.contains('clay-select-option') && this.classList.contains('is-selected') && !this.disabled;
        }
        return false;
    }
    querySelectorAll(sel) {
        const out = [];
        const walk = (n) => { n.children.forEach((ch) => { if (ch.matchesSelector(sel)) out.push(ch); walk(ch); }); };
        walk(this);
        return out;
    }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

const root = new FakeElement('div', { id: 'root' });
const makeSelect = (id, opts) => {
    const container = new FakeElement('div');
    const sel = new FakeElement('select', { id, classes: ['worker-select'] });
    opts.forEach(([val, text, selected]) => sel.appendChild(new FakeElement('option', { attrs: { value: val, selected }, text })));
    container.appendChild(sel);
    root.appendChild(container);
    return sel;
};

const themeBtn = root.appendChild(new FakeElement('button', { id: 'themeToggle' }));
themeBtn.appendChild(new FakeElement('svg', { classes: ['sun-icon', 'hidden'] }));
themeBtn.appendChild(new FakeElement('svg', { classes: ['moon-icon', 'hidden'] }));

const uploadPanel = root.appendChild(new FakeElement('div', { id: 'uploadPanel' }));
uploadPanel.appendChild(new FakeElement('button', { id: 'modeAnonymizeBtn', classes: ['mode-tab', 'is-active'] }));
uploadPanel.appendChild(new FakeElement('button', { id: 'modeTranscribeOnlyBtn', classes: ['mode-tab'] }));
uploadPanel.appendChild(new FakeElement('div', { id: 'dropZone' }));
uploadPanel.appendChild(new FakeElement('h2', { id: 'dropZoneTitle' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'dropZoneHint' }));
const fileInput = uploadPanel.appendChild(new FakeElement('input', { id: 'fileInput', attrs: { accept: '.mp4' } }));
uploadPanel.appendChild(new FakeElement('span', { id: 'workerValue', text: '4' }));
makeSelect('workerSelect', [['1', '1'], ['4', '4', true], ['custom', 'Другое']]);
uploadPanel.appendChild(new FakeElement('div', { id: 'customWorkerWrap', classes: ['hidden'] }));
uploadPanel.appendChild(new FakeElement('input', { id: 'customWorkerInput' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'workerLimitText' }));
uploadPanel.appendChild(new FakeElement('input', { id: 'transcribeToggle', attrs: { checked: true } }));
uploadPanel.appendChild(new FakeElement('label', { id: 'transcribeSwitchWrap' }));
uploadPanel.appendChild(new FakeElement('span', { id: 'transcribeStateBadge', text: 'Вкл' }));
uploadPanel.appendChild(new FakeElement('div', { id: 'transcribeOptionsWrap' }));
const transcribeHintText = uploadPanel.appendChild(new FakeElement('p', { id: 'transcribeHintText' }));

const modelOpts = [
    ['gigaam-v3', 'GigaAM v3', true],
    ['breeze-asr-25', 'Breeze-ASR-25'],
    ['turbo', 'Whisper Turbo'],
];
const langOpts = [['ru', 'Русский (RU)', true], ['auto', 'Авто'], ['en', 'English (EN)']];
const devOpts = [['cpu', 'CPU', true], ['cuda', 'GPU (CUDA)']];
const spkOpts = [['auto', 'Авто', true], ['2', '2'], ['3', '3'], ['4', '4'], ['5', '5'], ['6', '6'], ['4+', '4+']];
const sensOpts = [['normal', 'Стандартное', true], ['high', 'Высокое (похожие голоса)'], ['low', 'Мягкое']];
const trackOpts = [['auto', 'Авто (2 дорожки → Спикер 1 + собеседники)', true], ['separate', 'Раздельно'], ['mix', 'Смешать'], ['first', 'Только 1-я']];

makeSelect('asrModelSelect', modelOpts);
makeSelect('languageSelect', langOpts);
makeSelect('computeDeviceSelect', devOpts);
const speakerCountSelect = makeSelect('speakerCountSelect', spkOpts);
const diarizationSensitivitySelect = makeSelect('diarizationSensitivitySelect', sensOpts);
const audioTracksSelect = makeSelect('audioTracksSelect', trackOpts);
const customPromptInput = uploadPanel.appendChild(new FakeElement('textarea', { id: 'customPromptInput' }));
const promptPresets = uploadPanel.appendChild(new FakeElement('div', { id: 'promptPresets' }));
const saChip = promptPresets.appendChild(new FakeElement('button', { classes: ['prompt-chip'], dataset: { preset: 'system_analyst' }, text: 'Системный аналитик' }));
const clearChip = promptPresets.appendChild(new FakeElement('button', { classes: ['prompt-chip'], dataset: { preset: 'clear' }, text: 'Очистить' }));

uploadPanel.appendChild(new FakeElement('div', { id: 'recordingsSection', classes: ['hidden'] }));
uploadPanel.appendChild(new FakeElement('span', { id: 'recordingsCount' }));
uploadPanel.appendChild(new FakeElement('div', { id: 'recordingsList' }));

const statusPanel = root.appendChild(new FakeElement('div', { id: 'statusPanel', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('div', { id: 'spinner' }));
statusPanel.appendChild(new FakeElement('svg', { id: 'successIcon', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('svg', { id: 'cancelIcon', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('h2', { id: 'statusText' }));
statusPanel.appendChild(new FakeElement('p', { id: 'statusSubText' }));
statusPanel.appendChild(new FakeElement('div', { id: 'debugLog', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'cancelBtn', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'downloadBtn', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'downloadTranscriptBtn', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('div', { id: 'postTranscribeSection', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'togglePostTranscribeBtn', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('div', { id: 'postTranscribeForm', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('span', { id: 'postTranscribeTitle' }));
makeSelect('postAsrModelSelect', modelOpts);
makeSelect('postLanguageSelect', langOpts);
makeSelect('postComputeDeviceSelect', devOpts);
makeSelect('postSpeakerCountSelect', spkOpts);
makeSelect('postDiarizationSensitivitySelect', sensOpts);
makeSelect('postAudioTracksSelect', trackOpts);
const postCustomPromptInput = statusPanel.appendChild(new FakeElement('textarea', { id: 'postCustomPromptInput' }));
const postPromptPresets = statusPanel.appendChild(new FakeElement('div', { id: 'postPromptPresets' }));
postPromptPresets.appendChild(new FakeElement('button', { classes: ['prompt-chip'], dataset: { preset: 'system_analyst' }, text: 'Системный аналитик' }));
statusPanel.appendChild(new FakeElement('p', { id: 'postAudioSourceHint' }));
const postTranscribeBtn = statusPanel.appendChild(new FakeElement('button', { id: 'postTranscribeBtn' }));
statusPanel.appendChild(new FakeElement('button', { id: 'resetBtn', classes: ['hidden'] }));

const docListeners = {};
const documentObj = {
    documentElement: new FakeElement('html'),
    get activeElement() { return activeElement; },
    getElementById: (id) => byId.get(id) || null,
    querySelector: (sel) => root.querySelector(sel),
    querySelectorAll: (sel) => root.querySelectorAll(sel),
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    addEventListener: (t, fn) => { (docListeners[t] ||= []).push(fn); },
    dispatchEvent: (evt) => { (docListeners[evt.type] || []).forEach((fn) => fn(evt)); },
};

class FakeFormData {
    constructor() { this.entries = {}; }
    append(k, v) { this.entries[k] = v; }
}

const uploadCalls = [];
const postCalls = [];
const fakeFetch = async (url, opts = {}) => {
    if (url === '/api/config') {
        return {
            ok: true,
            json: async () => ({
                version: '4.6.0',
                cpu_count: 8,
                default_workers: 4,
                max_workers: 7,
                transcription_default: true,
                default_asr_model: 'gigaam-v3',
                default_language: 'ru',
                default_diarization_sensitivity: 'normal',
                default_audio_track_mode: 'auto',
                asr_models: [
                    { id: 'gigaam-v3', label: 'GigaAM v3', supports_cuda: true },
                    { id: 'breeze-asr-25', label: 'Breeze-ASR-25', supports_cuda: true },
                    { id: 'turbo', label: 'Whisper Turbo', supports_cuda: true },
                ],
                languages: [{ id: 'ru', label: 'Русский (RU)' }, { id: 'auto', label: 'Авто' }],
                diarization_sensitivities: [
                    { id: 'normal', label: 'Стандартное' },
                    { id: 'high', label: 'Высокое (похожие голоса)' },
                    { id: 'low', label: 'Мягкое' },
                ],
                audio_track_modes: [
                    { id: 'auto', label: 'Авто' },
                    { id: 'separate', label: 'Раздельно' },
                    { id: 'mix', label: 'Смешать' },
                    { id: 'first', label: 'Только 1-я' },
                ],
                prompt_presets: [
                    {
                        id: 'system_analyst',
                        label: 'Системный аналитик',
                        prompt: 'Собеседование системного аналитика: REST, gRPC, HTTP 2.0, SOAP, JWT, Bearer, OAuth 2',
                    },
                ],
                cuda_available: false,
                default_compute_device: 'cpu',
                compute_devices: [{ id: 'cpu', label: 'CPU' }, { id: 'cuda', label: 'GPU (CUDA)' }],
            }),
        };
    }
    if (url === '/api/recordings') return { ok: true, json: async () => ({ recordings: [] }) };
    if (url === '/api/upload') {
        uploadCalls.push({ ...opts.body.entries });
        return { ok: true, json: async () => ({ task_id: 'v46-task', status: 'queued', worker_count: 4 }) };
    }
    if (url === '/api/status/v46-task') {
        return {
            ok: true,
            json: async () => ({
                status: 'completed',
                mode: 'anonymize',
                video_ready: true,
                transcript_ready: true,
                can_transcribe: true,
                has_original_audio: true,
                asr_model: 'gigaam-v3',
                language: 'ru',
                speakers: '4',
                diarization_sensitivity: 'high',
                audio_tracks: 'separate',
                custom_prompt: uploadCalls[0].custom_prompt,
                logs: ['Done!'],
            }),
        };
    }
    if (url === '/api/transcribe/v46-task') {
        postCalls.push({ ...opts.body.entries });
        return { ok: true, json: async () => ({ task_id: 'v46-task', status: 'queued' }) };
    }
    throw new Error('Unexpected URL: ' + url);
};

const ctx = vm.createContext({
    document: documentObj,
    window: { location: { href: '' }, matchMedia: () => ({ matches: false }) },
    localStorage,
    Event: FakeEvent,
    FormData: FakeFormData,
    AbortController: class { constructor() { this.signal = {}; } abort() {} },
    fetch: fakeFetch,
    setInterval: () => 1,
    clearInterval: () => {},
    alert: (m) => { throw new Error('Unexpected alert: ' + m); },
    console,
    Number, String, Boolean, Array, Map, Set, Date,
});
vm.runInContext(fs.readFileSync('static/script.js', 'utf8'), ctx);

(async () => {
    await new Promise((r) => setTimeout(r, 25));

    // 1. Default prompt is strictly empty (never auto-injected!)
    assert.strictEqual(customPromptInput.value, '');

    // 2. Click 'Системный аналитик' preset chip -> fills glossary & shows GigaAM Latin hint
    saChip.click();
    assert.ok(customPromptInput.value.includes('gRPC'));
    assert.ok(customPromptInput.value.includes('OAuth 2'));
    assert.ok(transcribeHintText.textContent.includes('латинские термины'));

    // 3. Pick 4 speakers, high sensitivity, separate tracks -> upload sends all fields
    speakerCountSelect.value = '4';
    speakerCountSelect.dispatchEvent(new ctx.Event('change'));
    diarizationSensitivitySelect.value = 'high';
    diarizationSensitivitySelect.dispatchEvent(new ctx.Event('change'));
    audioTracksSelect.value = 'separate';
    audioTracksSelect.dispatchEvent(new ctx.Event('change'));

    fileInput.files = [{ name: 'obs_4_women.mp4' }];
    fileInput.dispatchEvent(new ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 25));

    assert.strictEqual(uploadCalls.length, 1);
    assert.strictEqual(uploadCalls[0].speakers, '4');
    assert.strictEqual(uploadCalls[0].diarization_sensitivity, 'high');
    assert.strictEqual(uploadCalls[0].audio_tracks, 'separate');
    assert.ok(uploadCalls[0].custom_prompt.includes('gRPC'));

    // 4. Post-transcribe form inherited v4.6.0 values and submits them on /api/transcribe/{tid}
    assert.ok(postCustomPromptInput.value.includes('gRPC'));
    postTranscribeBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(postCalls.length, 1);
    assert.strictEqual(postCalls[0].speakers, '4');
    assert.strictEqual(postCalls[0].diarization_sensitivity, 'high');
    assert.strictEqual(postCalls[0].audio_tracks, 'separate');
    assert.ok(postCalls[0].custom_prompt.includes('OAuth 2'));

    // 5. Clicking 'Очистить' resets customPromptInput back to ''
    clearChip.click();
    assert.strictEqual(customPromptInput.value, '');

    console.log('FRONTEND_V46_GLOSSARY_SENSITIVITY_TRACKS_OK');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_test_js],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node v4.6 UI test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FRONTEND_V46_GLOSSARY_SENSITIVITY_TRACKS_OK" in proc.stdout


def test_orchestrator_parallel_gpu_mode_for_onnx_asr_gigaam_v3(tmp_path, monkeypatch):
    import numpy as np
    import src.transcriber as tr_mod
    from app import _audio_cache_path
    from src.transcriber import TimedUnit

    in_mp4 = str(tmp_path / "gigaam_gpu_input.mp4")
    out_mp4 = str(tmp_path / "gigaam_gpu_output.mp4")
    out_txt = str(tmp_path / "gigaam_gpu_transcript.txt")
    _generate_test_mp4(in_mp4, duration_sec=1.6, with_audio=True)

    phase_a_calls = []
    phase_b_calls = []

    def _mock_units_only(cls, wav_path, asr_model="gigaam-v3", compute_device="cuda", denoise=False, **kwargs):
        phase_a_calls.append({"asr_model": asr_model, "compute_device": compute_device, "denoise": denoise})
        return {
            "samples": np.ones(16000, dtype=np.float32) * 0.1,
            "sample_rate": 16000,
            "duration_sec": 1.6,
            "units": [TimedUnit(0.1, 1.0, " GigaAM GPU OK")],
            "profile_id": asr_model,
            "custom_prompt": kwargs.get("custom_prompt", ""),
            "elapsed_asr": 0.01,
        }

    def _mock_diar_write(
        cls,
        asr_state,
        output_txt_path,
        llm_polish=False,
        llm_base_url="",
        llm_model="",
        **kwargs,
    ):
        phase_b_calls.append(
            {
                "profile_id": asr_state["profile_id"],
                "llm_polish": llm_polish,
                "llm_base_url": llm_base_url,
                "llm_model": llm_model,
            }
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("[00:00:00 - 00:00:01] Спикер 1: GigaAM GPU OK\n")
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.02, "llm_applied": llm_polish}

    # Even when CTranslate2 CUDA is False, ONNX CUDA True MUST activate parallel GPU mode for gigaam-v3!
    monkeypatch.setattr(tr_mod, "probe_cuda_available", lambda: False)
    monkeypatch.setattr(tr_mod, "probe_onnx_cuda_available", lambda: True)
    monkeypatch.setattr(Transcriber, "transcribe_units_only", classmethod(_mock_units_only))
    monkeypatch.setattr(Transcriber, "diarize_and_write_transcript", classmethod(_mock_diar_write))

    iom = IOManager()
    job_dir = iom.create_job_dir()
    try:
        res = Orchestrator.process_video_job(
            input_path=in_mp4,
            output_path=out_mp4,
            job_dir=job_dir,
            worker_count=1,
            transcribe=True,
            transcript_path=out_txt,
            asr_model="gigaam-v3",
            compute_device="cuda",
            denoise=True,
            llm_polish=True,
            llm_base_url="http://localhost:11434/v1",
            llm_model="qwen3:8b",
        )
        assert res["video_ready"] is True
        assert res["transcript_ready"] is True
        assert phase_a_calls == [{"asr_model": "gigaam-v3", "compute_device": "cuda", "denoise": True}]
        assert phase_b_calls == [
            {
                "profile_id": "gigaam-v3",
                "llm_polish": True,
                "llm_base_url": "http://localhost:11434/v1",
                "llm_model": "qwen3:8b",
            }
        ]
    finally:
        iom.cleanup_job_dir(job_dir)

    # Also verify API transcribe_only forwards denoise, llm_polish, & cloud llm_api_key (never leaked!) AND preserves staged *_raw.txt backup!
    api_calls = []
    secret_api_key = "sk-deepseek-secret-never-leak-777"

    def _mock_tr_with_raw_backup(cls, wav_path, output_txt_path, **kwargs):
        api_calls.append(
            {
                "asr_model": kwargs.get("asr_model"),
                "denoise": kwargs.get("denoise"),
                "llm_polish": kwargs.get("llm_polish"),
                "llm_base_url": kwargs.get("llm_base_url"),
                "llm_model": kwargs.get("llm_model"),
                "llm_api_key": kwargs.get("llm_api_key"),
            }
        )
        raw_p = os.path.splitext(output_txt_path)[0] + "_raw.txt"
        with open(raw_p, "w", encoding="utf-8") as rf:
            rf.write("RAW_PRE_LLM_BACKUP\n")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("POLISHED_FINAL_TRANSCRIPT\n")
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.01, "llm_applied": True, "raw_transcript_path": raw_p}

    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_mock_tr_with_raw_backup))

    client = TestClient(app)
    tid = None
    try:
        with open(in_mp4, "rb") as f:
            resp = client.post(
                "/api/upload",
                files={"file": ("v50_test.mp4", f, "video/mp4")},
                data={
                    "mode": "transcribe_only",
                    "workers": "1",
                    "asr_model": "gigaam-v3-ctc",
                    "denoise": "true",
                    "llm_polish": "true",
                    "llm_base_url": "https://api.deepseek.com/v1",
                    "llm_model": "deepseek-flash",
                    "llm_api_key": secret_api_key,
                },
            )
        assert resp.status_code == 200
        assert secret_api_key not in resp.text
        body = resp.json()
        tid = body["task_id"]
        assert body["asr_model"] == "gigaam-v3-ctc"
        assert body["denoise"] is True
        assert body["llm_polish"] is True
        assert body["llm_base_url"] == "https://api.deepseek.com/v1"
        assert body["llm_model"] == "deepseek-flash"
        assert "llm_api_key" not in body

        st_resp = client.get(f"/api/status/{tid}")
        assert secret_api_key not in st_resp.text
        st = st_resp.json()
        assert st["status"] == "completed"
        assert "llm_api_key" not in tasks[tid]
        assert secret_api_key not in str(tasks[tid])
        assert api_calls == [
            {
                "asr_model": "gigaam-v3-ctc",
                "denoise": True,
                "llm_polish": True,
                "llm_base_url": "https://api.deepseek.com/v1",
                "llm_model": "deepseek-flash",
                "llm_api_key": secret_api_key,
            }
        ]
        raw_out = os.path.join("output", f"{tid}_transcript_raw.txt")
        final_out = os.path.join("output", f"{tid}_transcript.txt")
        assert os.path.isfile(final_out)
        assert os.path.isfile(raw_out)
        assert open(raw_out, encoding="utf-8").read().strip() == "RAW_PRE_LLM_BACKUP"
        assert open(final_out, encoding="utf-8").read().strip() == "POLISHED_FINAL_TRANSCRIPT"
    finally:
        if tid:
            tasks.pop(tid, None)
            for p in (
                os.path.join("output", f"{tid}_transcript.txt"),
                os.path.join("output", f"{tid}_transcript_raw.txt"),
                _audio_cache_path(tid),
            ):
                if os.path.exists(p):
                    os.remove(p)


def test_frontend_v50_gpu_gigaam_denoise_and_llm_polish_controls():
    """
    Executes static/script.js in Node.js to verify v5.0+ frontend features:
      1. When onnx_cuda_available=true, selecting GPU (CUDA) keeps gigaam-v3 (FP32) and gigaam-v3-ctc selected.
      2. Selecting cloud deepgram-nova-3 and switching to GPU (CUDA) switches to a CUDA-capable local model.
      3. When onnx_cuda_available=false and ct2_cuda_available=true, GPU (CUDA) switches gigaam-v3 -> breeze-asr-25.
      4. denoiseToggle & llmPolishToggle default OFF; enabling llmPolishToggle reveals #llmConfigWrap,
         clicking DeepSeek API / Qwen Cloud preset chips fills URL + model & flips #llmPrivacyHint to .is-cloud,
         API key stays ONLY in sessionStorage (never localStorage), and both /api/upload and /api/transcribe/{tid}
         submit denoise, llm_polish, llm_base_url, llm_model, and llm_api_key.
    """
    node_test_js = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function createHarness({ onnxCudaAvailable, ct2CudaAvailable }) {
    const storage = new Map();
    const localStorage = {
        getItem: (k) => (storage.has(k) ? storage.get(k) : null),
        setItem: (k, v) => storage.set(k, String(v)),
    };
    const sessStorage = new Map();
    const sessionStorage = {
        getItem: (k) => (sessStorage.has(k) ? sessStorage.get(k) : null),
        setItem: (k, v) => sessStorage.set(k, String(v)),
        removeItem: (k) => sessStorage.delete(k),
    };

    class ClassList {
        constructor(el, initial = []) { this.el = el; this.set = new Set(initial.filter(Boolean)); }
        add(...cls) { cls.forEach((c) => this.set.add(c)); }
        remove(...cls) { cls.forEach((c) => this.set.delete(c)); }
        contains(c) { return this.set.has(c); }
        toggle(c, force) {
            const next = force !== undefined ? Boolean(force) : !this.set.has(c);
            if (next) this.set.add(c); else this.set.delete(c);
            return next;
        }
    }

    class FakeEvent {
        constructor(type, opts = {}) {
            this.type = type;
            this.bubbles = Boolean(opts.bubbles);
            this.defaultPrevented = false;
            this.propagationStopped = false;
        }
        preventDefault() { this.defaultPrevented = true; }
        stopPropagation() { this.propagationStopped = true; }
    }

    const byId = new Map();
    let activeElement = null;

    class FakeElement {
        constructor(tagName, { id = '', classes = [], attrs = {}, dataset = {}, text = '' } = {}) {
            this.tagName = tagName.toUpperCase();
            this.id = id;
            if (id) byId.set(id, this);
            this.classList = new ClassList(this, classes);
            this.attrs = { ...attrs };
            this.dataset = { ...dataset };
            this.children = [];
            this.parentNode = null;
            this.listeners = {};
            this._textContent = text;
            this._innerHTML = '';
            this.disabled = false;
            this.checked = Boolean(attrs.checked);
            this.title = '';
            this.accept = attrs.accept || '';
            this.tabIndex = 0;
            this.scrollTop = 0;
            this.scrollHeight = 100;
            this.files = [];
            this._value = attrs.value !== undefined ? String(attrs.value) : '';
            this.selected = Boolean(attrs.selected);
            this.onclick = null;
        }
        get className() { return Array.from(this.classList.set).join(' '); }
        set className(v) { this.classList = new ClassList(this, String(v).split(/\s+/)); }
        get textContent() {
            if (this.children.length === 0) return this._textContent;
            return this._textContent + this.children.map((c) => c.textContent).join('');
        }
        set textContent(v) { this._textContent = String(v); this.children = []; }
        get innerHTML() { return this._innerHTML; }
        set innerHTML(v) {
            this._innerHTML = String(v);
            if (v === '') {
                this.children.forEach((c) => { c.parentNode = null; });
                this.children = [];
            }
        }
        get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
        get selectedIndex() {
            const opts = this.options;
            const idx = opts.findIndex((o) => o.value === this.value);
            return idx >= 0 ? idx : 0;
        }
        get value() {
            if (this.tagName === 'SELECT') {
                const opts = this.options;
                if (this._value && opts.some((o) => o.value === this._value)) return this._value;
                const sel = opts.find((o) => o.selected) || opts[0];
                return sel ? sel.value : '';
            }
            return this._value;
        }
        set value(v) { this._value = String(v); }
        setAttribute(k, v) { this.attrs[k] = String(v); }
        getAttribute(k) { return this.attrs[k] ?? null; }
        appendChild(child) {
            if (child.parentNode) {
                const idx = child.parentNode.children.indexOf(child);
                if (idx >= 0) child.parentNode.children.splice(idx, 1);
            }
            child.parentNode = this;
            this.children.push(child);
            return child;
        }
        insertBefore(newNode, refNode) {
            if (newNode.parentNode) {
                const oldIdx = newNode.parentNode.children.indexOf(newNode);
                if (oldIdx >= 0) newNode.parentNode.children.splice(oldIdx, 1);
            }
            newNode.parentNode = this;
            const idx = this.children.indexOf(refNode);
            if (idx >= 0) this.children.splice(idx, 0, newNode);
            else this.children.push(newNode);
            return newNode;
        }
        addEventListener(type, fn) {
            if (!this.listeners[type]) this.listeners[type] = [];
            this.listeners[type].push(fn);
        }
        dispatchEvent(evt) {
            (this.listeners[evt.type] || []).forEach((fn) => fn.call(this, evt));
            if (evt.type === 'click' && typeof this.onclick === 'function') this.onclick.call(this, evt);
            if (evt.bubbles && !evt.propagationStopped) {
                if (this.parentNode) this.parentNode.dispatchEvent(evt);
                else documentObj.dispatchEvent(evt);
            }
            return !evt.defaultPrevented;
        }
        click() { this.dispatchEvent(new FakeEvent('click', { bubbles: true })); }
        focus() { activeElement = this; }
        matchesSelector(sel) {
            if (sel === '.sun-icon') return this.classList.contains('sun-icon');
            if (sel === '.moon-icon') return this.classList.contains('moon-icon');
            if (sel === '.prompt-chip') return this.classList.contains('prompt-chip');
            if (sel === '[data-llm-preset]') {
                return Boolean((this.dataset && this.dataset.llmPreset) || this.attrs['data-llm-preset']);
            }
            const lm = sel.match(/^label\[for="(.+)"\]$/);
            if (lm) return this.tagName === 'LABEL' && this.attrs.for === lm[1];
            if (sel === '.clay-select-option:not(:disabled)') return this.classList.contains('clay-select-option') && !this.disabled;
            if (sel === '.clay-select-option.is-selected:not(:disabled)') {
                return this.classList.contains('clay-select-option') && this.classList.contains('is-selected') && !this.disabled;
            }
            return false;
        }
        querySelectorAll(sel) {
            const out = [];
            const walk = (n) => { n.children.forEach((ch) => { if (ch.matchesSelector(sel)) out.push(ch); walk(ch); }); };
            walk(this);
            return out;
        }
        querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
    }

    const root = new FakeElement('div', { id: 'root' });
    const makeSelect = (id, opts) => {
        const container = new FakeElement('div');
        const sel = new FakeElement('select', { id, classes: ['worker-select'] });
        opts.forEach(([val, text, selected]) => sel.appendChild(new FakeElement('option', { attrs: { value: val, selected }, text })));
        container.appendChild(sel);
        root.appendChild(container);
        return sel;
    };
    const makeLlmPresets = (containerId) => {
        const wrap = new FakeElement('div', { id: containerId, classes: ['prompt-presets'] });
        const btns = {};
        ['ollama_local', 'lmstudio_local', 'deepseek_cloud', 'qwen_cloud'].forEach((pid) => {
            const b = new FakeElement('button', {
                classes: ['prompt-chip'],
                attrs: { 'data-llm-preset': pid },
                dataset: { llmPreset: pid },
                text: pid,
            });
            wrap.appendChild(b);
            btns[pid] = b;
        });
        return { wrap, btns };
    };

    const themeBtn = root.appendChild(new FakeElement('button', { id: 'themeToggle' }));
    themeBtn.appendChild(new FakeElement('svg', { classes: ['sun-icon', 'hidden'] }));
    themeBtn.appendChild(new FakeElement('svg', { classes: ['moon-icon', 'hidden'] }));

    const uploadPanel = root.appendChild(new FakeElement('div', { id: 'uploadPanel' }));
    uploadPanel.appendChild(new FakeElement('button', { id: 'modeAnonymizeBtn', classes: ['mode-tab', 'is-active'] }));
    uploadPanel.appendChild(new FakeElement('button', { id: 'modeTranscribeOnlyBtn', classes: ['mode-tab'] }));
    uploadPanel.appendChild(new FakeElement('div', { id: 'dropZone' }));
    uploadPanel.appendChild(new FakeElement('h2', { id: 'dropZoneTitle' }));
    uploadPanel.appendChild(new FakeElement('p', { id: 'dropZoneHint' }));
    const fileInput = uploadPanel.appendChild(new FakeElement('input', { id: 'fileInput', attrs: { accept: '.mp4' } }));
    uploadPanel.appendChild(new FakeElement('span', { id: 'workerValue', text: '4' }));
    makeSelect('workerSelect', [['1', '1'], ['4', '4', true], ['custom', 'Другое']]);
    uploadPanel.appendChild(new FakeElement('div', { id: 'customWorkerWrap', classes: ['hidden'] }));
    uploadPanel.appendChild(new FakeElement('input', { id: 'customWorkerInput' }));
    uploadPanel.appendChild(new FakeElement('p', { id: 'workerLimitText' }));
    uploadPanel.appendChild(new FakeElement('input', { id: 'transcribeToggle', attrs: { checked: true } }));
    uploadPanel.appendChild(new FakeElement('label', { id: 'transcribeSwitchWrap' }));
    uploadPanel.appendChild(new FakeElement('span', { id: 'transcribeStateBadge', text: 'Вкл' }));
    uploadPanel.appendChild(new FakeElement('div', { id: 'transcribeOptionsWrap' }));
    const transcribeHintText = uploadPanel.appendChild(new FakeElement('p', { id: 'transcribeHintText' }));

    const modelOpts = [
        ['gigaam-v3', 'GigaAM v3 FP32', true],
        ['gigaam-v3-ctc', 'GigaAM v3 CTC FP32'],
        ['breeze-asr-25', 'Breeze-ASR-25'],
        ['turbo', 'Whisper Turbo'],
        ['deepgram-nova-3', 'Deepgram Nova-3'],
    ];
    const langOpts = [['ru', 'Русский (RU)', true], ['auto', 'Авто'], ['en', 'English (EN)']];
    const devOpts = [['cpu', 'CPU', true], ['cuda', 'GPU (CUDA)']];
    const spkOpts = [['auto', 'Авто', true], ['2', '2']];
    const sensOpts = [['normal', 'Обычная', true], ['high', 'Высокая']];
    const trackOpts = [['auto', 'Авто', true], ['separate', 'Раздельно']];

    const asrModelSelect = makeSelect('asrModelSelect', modelOpts);
    makeSelect('languageSelect', langOpts);
    const computeDeviceSelect = makeSelect('computeDeviceSelect', devOpts);
    makeSelect('speakerCountSelect', spkOpts);
    makeSelect('diarizationSensitivitySelect', sensOpts);
    makeSelect('audioTracksSelect', trackOpts);
    uploadPanel.appendChild(new FakeElement('textarea', { id: 'customPromptInput' }));
    uploadPanel.appendChild(new FakeElement('div', { id: 'promptPresets' }));
    const denoiseToggle = uploadPanel.appendChild(new FakeElement('input', { id: 'denoiseToggle' }));
    const llmPolishToggle = uploadPanel.appendChild(new FakeElement('input', { id: 'llmPolishToggle' }));
    const llmConfigWrap = uploadPanel.appendChild(new FakeElement('div', { id: 'llmConfigWrap', classes: ['hidden'] }));
    const mainLlmPresets = makeLlmPresets('llmPresets');
    llmConfigWrap.appendChild(mainLlmPresets.wrap);
    const llmBaseUrlInput = llmConfigWrap.appendChild(new FakeElement('input', { id: 'llmBaseUrlInput' }));
    const llmModelInput = llmConfigWrap.appendChild(new FakeElement('input', { id: 'llmModelInput' }));
    const llmApiKeyInput = llmConfigWrap.appendChild(new FakeElement('input', { id: 'llmApiKeyInput' }));
    const llmPrivacyHint = llmConfigWrap.appendChild(new FakeElement('p', { id: 'llmPrivacyHint' }));

    uploadPanel.appendChild(new FakeElement('div', { id: 'recordingsSection', classes: ['hidden'] }));
    uploadPanel.appendChild(new FakeElement('span', { id: 'recordingsCount' }));
    uploadPanel.appendChild(new FakeElement('div', { id: 'recordingsList' }));

    const statusPanel = root.appendChild(new FakeElement('div', { id: 'statusPanel', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('div', { id: 'spinner' }));
    statusPanel.appendChild(new FakeElement('svg', { id: 'successIcon', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('svg', { id: 'cancelIcon', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('h2', { id: 'statusText' }));
    statusPanel.appendChild(new FakeElement('p', { id: 'statusSubText' }));
    statusPanel.appendChild(new FakeElement('div', { id: 'debugLog', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'cancelBtn', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'downloadBtn', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'downloadTranscriptBtn', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('div', { id: 'postTranscribeSection', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('button', { id: 'togglePostTranscribeBtn', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('div', { id: 'postTranscribeForm', classes: ['hidden'] }));
    statusPanel.appendChild(new FakeElement('span', { id: 'postTranscribeTitle' }));
    makeSelect('postAsrModelSelect', modelOpts);
    makeSelect('postLanguageSelect', langOpts);
    makeSelect('postComputeDeviceSelect', devOpts);
    makeSelect('postSpeakerCountSelect', spkOpts);
    makeSelect('postDiarizationSensitivitySelect', sensOpts);
    makeSelect('postAudioTracksSelect', trackOpts);
    statusPanel.appendChild(new FakeElement('textarea', { id: 'postCustomPromptInput' }));
    statusPanel.appendChild(new FakeElement('div', { id: 'postPromptPresets' }));
    const postDenoiseToggle = statusPanel.appendChild(new FakeElement('input', { id: 'postDenoiseToggle' }));
    const postLlmPolishToggle = statusPanel.appendChild(new FakeElement('input', { id: 'postLlmPolishToggle' }));
    const postLlmConfigWrap = statusPanel.appendChild(new FakeElement('div', { id: 'postLlmConfigWrap', classes: ['hidden'] }));
    const postLlmPresets = makeLlmPresets('postLlmPresets');
    postLlmConfigWrap.appendChild(postLlmPresets.wrap);
    const postLlmBaseUrlInput = postLlmConfigWrap.appendChild(new FakeElement('input', { id: 'postLlmBaseUrlInput' }));
    const postLlmModelInput = postLlmConfigWrap.appendChild(new FakeElement('input', { id: 'postLlmModelInput' }));
    const postLlmApiKeyInput = postLlmConfigWrap.appendChild(new FakeElement('input', { id: 'postLlmApiKeyInput' }));
    const postLlmPrivacyHint = postLlmConfigWrap.appendChild(new FakeElement('p', { id: 'postLlmPrivacyHint' }));
    statusPanel.appendChild(new FakeElement('p', { id: 'postAudioSourceHint' }));
    const postTranscribeBtn = statusPanel.appendChild(new FakeElement('button', { id: 'postTranscribeBtn' }));
    statusPanel.appendChild(new FakeElement('button', { id: 'resetBtn', classes: ['hidden'] }));

    const docListeners = {};
    const documentObj = {
        documentElement: new FakeElement('html'),
        get activeElement() { return activeElement; },
        getElementById: (id) => byId.get(id) || null,
        querySelector: (sel) => root.querySelector(sel),
        querySelectorAll: (sel) => root.querySelectorAll(sel),
        createElement: (tag) => new FakeElement(tag),
        createElementNS: (_ns, tag) => new FakeElement(tag),
        addEventListener: (t, fn) => { (docListeners[t] ||= []).push(fn); },
        dispatchEvent: (evt) => { (docListeners[evt.type] || []).forEach((fn) => fn(evt)); },
    };

    class FakeFormData {
        constructor() { this.entries = {}; }
        append(k, v) { this.entries[k] = v; }
    }

    const uploadCalls = [];
    const postCalls = [];
    const fakeFetch = async (url, opts = {}) => {
        if (url === '/api/config') {
            return {
                ok: true,
                json: async () => ({
                    version: '5.1.0',
                    cpu_count: 8,
                    default_workers: 4,
                    max_workers: 7,
                    transcription_default: true,
                    default_asr_model: 'gigaam-v3',
                    default_language: 'ru',
                    asr_models: [
                        { id: 'gigaam-v3', label: 'GigaAM v3 FP32', engine: 'onnx_asr', supports_cuda: true },
                        { id: 'gigaam-v3-ctc', label: 'GigaAM v3 CTC FP32', engine: 'onnx_asr', supports_cuda: true },
                        { id: 'breeze-asr-25', label: 'Breeze-ASR-25', engine: 'faster_whisper', supports_cuda: true },
                        { id: 'turbo', label: 'Whisper Turbo', engine: 'faster_whisper', supports_cuda: true },
                        { id: 'deepgram-nova-3', label: 'Deepgram Nova-3', engine: 'deepgram_cloud', supports_cuda: false, is_cloud: true },
                    ],
                    cuda_available: Boolean(ct2CudaAvailable || onnxCudaAvailable),
                    ct2_cuda_available: ct2CudaAvailable,
                    onnx_cuda_available: onnxCudaAvailable,
                    default_compute_device: 'cpu',
                    compute_devices: [{ id: 'cpu', label: 'CPU' }, { id: 'cuda', label: 'GPU (CUDA)' }],
                    denoise_default: false,
                    llm_polish_default: false,
                    llm_defaults: { base_url: 'http://localhost:11434/v1', model: 'qwen3:8b' },
                    llm_presets: [
                        { id: 'ollama_local', base_url: 'http://localhost:11434/v1', model: 'qwen3:8b', is_cloud: false },
                        { id: 'lmstudio_local', base_url: 'http://localhost:1234/v1', model: 'qwen3-8b', is_cloud: false },
                        { id: 'deepseek_cloud', base_url: 'https://api.deepseek.com/v1', model: 'deepseek-flash', is_cloud: true },
                        { id: 'qwen_cloud', base_url: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus', is_cloud: true },
                    ],
                }),
            };
        }
        if (url === '/api/recordings') return { ok: true, json: async () => ({ recordings: [] }) };
        if (url === '/api/upload') {
            uploadCalls.push({ ...opts.body.entries });
            return { ok: true, json: async () => ({ task_id: 'v50-task', status: 'queued', worker_count: 4 }) };
        }
        if (url === '/api/status/v50-task') {
            return {
                ok: true,
                json: async () => ({
                    status: 'completed',
                    mode: 'anonymize',
                    video_ready: true,
                    transcript_ready: true,
                    can_transcribe: true,
                    has_original_audio: true,
                    asr_model: uploadCalls[0].asr_model,
                    compute_device: uploadCalls[0].compute_device,
                    denoise: uploadCalls[0].denoise === 'true',
                    llm_polish: uploadCalls[0].llm_polish === 'true',
                    llm_base_url: uploadCalls[0].llm_base_url,
                    llm_model: uploadCalls[0].llm_model,
                    logs: ['Done!'],
                }),
            };
        }
        if (url === '/api/transcribe/v50-task') {
            postCalls.push({ ...opts.body.entries });
            return { ok: true, json: async () => ({ task_id: 'v50-task', status: 'queued' }) };
        }
        throw new Error('Unexpected URL: ' + url);
    };

    const ctx = vm.createContext({
        document: documentObj,
        window: { location: { href: '' }, matchMedia: () => ({ matches: false }) },
        localStorage,
        sessionStorage,
        Event: FakeEvent,
        FormData: FakeFormData,
        AbortController: class { constructor() { this.signal = {}; } abort() {} },
        fetch: fakeFetch,
        setInterval: () => 1,
        clearInterval: () => {},
        alert: (m) => { throw new Error('Unexpected alert: ' + m); },
        console,
        Number, String, Boolean, Array, Map, Set, Date,
    });
    vm.runInContext(fs.readFileSync('static/script.js', 'utf8'), ctx);
    return {
        ctx, storage, sessStorage, asrModelSelect, computeDeviceSelect, fileInput, transcribeHintText,
        denoiseToggle, llmPolishToggle, llmConfigWrap, mainLlmPresets, llmBaseUrlInput, llmModelInput, llmApiKeyInput, llmPrivacyHint,
        postDenoiseToggle, postLlmPolishToggle, postLlmConfigWrap, postLlmPresets, postLlmBaseUrlInput, postLlmModelInput, postLlmApiKeyInput, postLlmPrivacyHint,
        postTranscribeBtn, uploadCalls, postCalls,
    };
}

(async () => {
    // Scenario A: Full GPU build (both ONNX CUDA + CTranslate2 CUDA available)
    const a = createHarness({ onnxCudaAvailable: true, ct2CudaAvailable: true });
    await new Promise((r) => setTimeout(r, 25));

    // Defaults OFF and privacy hint starts in local mode
    assert.strictEqual(a.denoiseToggle.checked, false);
    assert.strictEqual(a.llmPolishToggle.checked, false);
    assert.strictEqual(a.llmConfigWrap.classList.contains('hidden'), true);
    assert.strictEqual(a.llmPrivacyHint.classList.contains('is-cloud'), false);

    // Switching to GPU (CUDA) keeps gigaam-v3 (FP32) AND gigaam-v3-ctc selected!
    a.computeDeviceSelect.value = 'cuda';
    a.computeDeviceSelect.dispatchEvent(new a.ctx.Event('change'));
    assert.strictEqual(a.asrModelSelect.value, 'gigaam-v3');
    assert.strictEqual(a.computeDeviceSelect.value, 'cuda');

    a.asrModelSelect.value = 'gigaam-v3-ctc';
    a.asrModelSelect.dispatchEvent(new a.ctx.Event('change'));
    assert.strictEqual(a.asrModelSelect.value, 'gigaam-v3-ctc');
    assert.strictEqual(a.computeDeviceSelect.value, 'cuda');

    // Selecting cloud deepgram-nova-3 -> switches to CPU + shows cloud hint; GPU switches to turbo
    a.asrModelSelect.value = 'deepgram-nova-3';
    a.asrModelSelect.dispatchEvent(new a.ctx.Event('change'));
    assert.strictEqual(a.computeDeviceSelect.value, 'cpu');
    assert.ok(a.transcribeHintText.textContent.includes('Deepgram Nova-3'));
    a.computeDeviceSelect.value = 'cuda';
    a.computeDeviceSelect.dispatchEvent(new a.ctx.Event('change'));
    assert.strictEqual(a.asrModelSelect.value, 'turbo');
    assert.strictEqual(a.computeDeviceSelect.value, 'cuda');

    // Switch back to gigaam-v3 on GPU CUDA for upload check
    a.asrModelSelect.value = 'gigaam-v3';
    a.asrModelSelect.dispatchEvent(new a.ctx.Event('change'));
    assert.strictEqual(a.asrModelSelect.value, 'gigaam-v3');
    assert.strictEqual(a.computeDeviceSelect.value, 'cuda');

    // Enable GTCRN denoise + Cloud DeepSeek LLM polish via preset chip + API key
    a.denoiseToggle.checked = true;
    a.denoiseToggle.dispatchEvent(new a.ctx.Event('change'));
    a.llmPolishToggle.checked = true;
    a.llmPolishToggle.dispatchEvent(new a.ctx.Event('change'));
    assert.strictEqual(a.llmConfigWrap.classList.contains('hidden'), false);
    a.mainLlmPresets.btns.deepseek_cloud.click();
    assert.strictEqual(a.llmBaseUrlInput.value, 'https://api.deepseek.com/v1');
    assert.strictEqual(a.llmModelInput.value, 'deepseek-flash');
    assert.strictEqual(a.llmPrivacyHint.classList.contains('is-cloud'), true);
    assert.ok(a.llmPrivacyHint.textContent.includes('облачный API'));

    a.llmApiKeyInput.value = 'sk-deepseek-ui-secret';
    a.llmApiKeyInput.dispatchEvent(new a.ctx.Event('input'));
    assert.strictEqual(a.sessStorage.get('selectedLlmApiKey'), 'sk-deepseek-ui-secret');
    assert.strictEqual(a.storage.has('selectedLlmApiKey'), false, 'API key must NEVER be saved in localStorage');

    a.fileInput.files = [{ name: 'interview.mp4' }];
    a.fileInput.dispatchEvent(new a.ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 25));

    assert.strictEqual(a.uploadCalls.length, 1);
    assert.strictEqual(a.uploadCalls[0].asr_model, 'gigaam-v3');
    assert.strictEqual(a.uploadCalls[0].compute_device, 'cuda');
    assert.strictEqual(a.uploadCalls[0].denoise, 'true');
    assert.strictEqual(a.uploadCalls[0].llm_polish, 'true');
    assert.strictEqual(a.uploadCalls[0].llm_base_url, 'https://api.deepseek.com/v1');
    assert.strictEqual(a.uploadCalls[0].llm_model, 'deepseek-flash');
    assert.strictEqual(a.uploadCalls[0].llm_api_key, 'sk-deepseek-ui-secret');

    // Post-transcribe form inherited controls + API key; switching to Qwen Cloud preset updates URL/model & submits
    assert.strictEqual(a.postDenoiseToggle.checked, true);
    assert.strictEqual(a.postLlmPolishToggle.checked, true);
    assert.strictEqual(a.postLlmConfigWrap.classList.contains('hidden'), false);
    assert.strictEqual(a.postLlmApiKeyInput.value, 'sk-deepseek-ui-secret');
    assert.strictEqual(a.postLlmPrivacyHint.classList.contains('is-cloud'), true);
    a.postLlmPresets.btns.qwen_cloud.click();
    assert.strictEqual(a.postLlmBaseUrlInput.value, 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1');
    assert.strictEqual(a.postLlmModelInput.value, 'qwen-plus');
    assert.strictEqual(a.postLlmPrivacyHint.classList.contains('is-cloud'), true);
    a.postTranscribeBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(a.postCalls.length, 1);
    assert.strictEqual(a.postCalls[0].denoise, 'true');
    assert.strictEqual(a.postCalls[0].llm_polish, 'true');
    assert.strictEqual(a.postCalls[0].llm_base_url, 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1');
    assert.strictEqual(a.postCalls[0].llm_model, 'qwen-plus');
    assert.strictEqual(a.postCalls[0].llm_api_key, 'sk-deepseek-ui-secret');
    assert.strictEqual(a.storage.has('selectedLlmApiKey'), false);

    // Scenario B: CTranslate2 CUDA available, ONNX CUDA NOT available -> GPU switches gigaam-v3 -> breeze-asr-25
    const b = createHarness({ onnxCudaAvailable: false, ct2CudaAvailable: true });
    await new Promise((r) => setTimeout(r, 25));
    b.computeDeviceSelect.value = 'cuda';
    b.computeDeviceSelect.dispatchEvent(new b.ctx.Event('change'));
    assert.strictEqual(b.asrModelSelect.value, 'breeze-asr-25');
    assert.strictEqual(b.computeDeviceSelect.value, 'cuda');

    console.log('FRONTEND_V50_GPU_GIGAAM_DENOISE_LLM_OK');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_test_js],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node v5.0 UI test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FRONTEND_V50_GPU_GIGAAM_DENOISE_LLM_OK" in proc.stdout


def test_video_processor_18_frame_adaptive_bisection_window_drag_and_dropout_bridging():
    import numpy as np
    from src.config import (
        ADAPTIVE_MIN_DETECT_STEP,
        DETECT_EVERY_N_FRAMES,
        YUNET_MIN_CONFIDENCE,
        YUNET_TARGET_HEIGHT,
    )
    from src.video_processor import VideoProcessor

    # 1. Verify tightened config defaults for 100% face anonymity
    assert DETECT_EVERY_N_FRAMES == 18
    assert ADAPTIVE_MIN_DETECT_STEP == 2
    assert YUNET_MIN_CONFIDENCE <= 0.25
    assert YUNET_TARGET_HEIGHT >= 576

    vp = VideoProcessor()
    w, h = 1280, 720
    vp._init_detector(w, h)
    assert vp.ml_height == YUNET_TARGET_HEIGHT

    # 2. Globally optimal matching tracks wide 30% screen browser drags without big/small cross-stealing
    big_s = (100, 200, 140, 180)
    small_s = (650, 120, 40, 52)
    big_e = (460, 220, 142, 182)      # moved +360 px (~28% width)
    small_e = (1010, 140, 41, 51)     # moved +360 px (~28% width)
    matched, unm_s, unm_e = vp._match_faces_detailed([small_s, big_s], [big_e, small_e], w, h)
    assert len(matched) == 2
    assert not unm_s and not unm_e
    assert (big_s, big_e) in matched
    assert (small_s, small_e) in matched

    # 3. Calm 18-frame interval must NOT trigger unnecessary bisection
    frame_size = (w * h * 3) // 2
    calm_frame = np.full(frame_size, 100, dtype=np.uint8)
    calm_buf = [calm_frame.copy() for _ in range(DETECT_EVERY_N_FRAMES)]
    calm_calls = []
    vp._detect_faces = lambda fr: (calm_calls.append(1) or [(300, 200, 80, 100)])
    prev_thumb = vp._extract_motion_thumb(calm_frame, w, h)
    kp_calm, end_calm, _, sticky_calm = vp._resolve_batch_keypoints(
        calm_buf, [(300, 200, 80, 100)], prev_thumb, w, h
    )
    assert sorted(kp_calm.keys()) == [0, DETECT_EVERY_N_FRAMES - 1]
    assert len(calm_calls) == 1  # only last frame detected when continuing calm footage
    assert end_calm == [(300, 200, 80, 100)]
    assert sticky_calm == []

    # 4. Fast browser-window drag triggers recursive bisection down to <= ADAPTIVE_MIN_DETECT_STEP
    #    AND bridges isolated 1-keyframe YuNet drop-outs mid-drag
    drag_buf = []
    for idx in range(DETECT_EVERY_N_FRAMES):
        fr = np.full(frame_size, 40, dtype=np.uint8)
        y_plane = fr[: w * h].reshape((h, w))
        # Moving bright browser window rectangle shifts horizontally across 400 px
        wx = 60 + idx * 24
        y_plane[80:560, wx : wx + 520] = 210
        drag_buf.append(fr)

    # Simulate YuNet detecting the face at moving x, EXCEPT dropping out on frame 8 (mid-drag flicker)
    frame_id_map = {id(fr): idx for idx, fr in enumerate(drag_buf)}
    def _mock_drag_detect(fr):
        idx = frame_id_map[id(fr)]
        if idx == 8:
            return []  # momentary motion-blur drop-out
        return [(200 + idx * 24, 180, 50, 64)]

    vp._detect_faces = _mock_drag_detect
    kp_drag, _, _, _ = vp._resolve_batch_keypoints(drag_buf, None, None, w, h)
    keys = sorted(kp_drag.keys())
    # Must subdivide aggressively across the 18-frame drag (max gap <= ADAPTIVE_MIN_DETECT_STEP)
    assert len(keys) >= 7
    assert max(b - a for a, b in zip(keys, keys[1:])) <= ADAPTIVE_MIN_DETECT_STEP
    # Frame 8 drop-out must have been bridged from neighboring keyframes
    assert 8 in kp_drag and len(kp_drag[8]) == 1
    assert abs(kp_drag[8][0][0] - (200 + 8 * 24)) <= 10

    # 5. Mid-batch Alt-Tab popup (faces absent at frame 0 & 17, visible only at frames 6..11) is caught
    pop_buf = []
    for idx in range(DETECT_EVERY_N_FRAMES):
        fr = np.full(frame_size, 30, dtype=np.uint8)
        if 6 <= idx <= 11:
            y_plane = fr[: w * h].reshape((h, w))
            y_plane[100:600, 150:900] = 220
        pop_buf.append(fr)
    pop_map = {id(fr): idx for idx, fr in enumerate(pop_buf)}
    vp._detect_faces = lambda fr: [(400, 220, 60, 80)] if 6 <= pop_map[id(fr)] <= 11 else []
    kp_pop, _, _, _ = vp._resolve_batch_keypoints(pop_buf, None, None, w, h)
    assert any(kp_pop[k] for k in kp_pop if 6 <= k <= 11)

    # 6. 1-cycle static-region anti-flicker persistence holds static face on momentary profile turn
    vp._detect_faces = lambda _fr: []
    _, _, _, sticky_static = vp._resolve_batch_keypoints(
        calm_buf, [(300, 200, 80, 100)], prev_thumb, w, h
    )
    assert sticky_static == [(300, 200, 80, 100)]

    # 7. _blur_and_write_buffer writes exact frame count without duplicates or drops
    written_frames = []
    class _DummyPipe:
        class _Stdin:
            def write(self, mv):
                written_frames.append(len(mv))
        stdin = _Stdin()

    cancelled = vp._blur_and_write_buffer(drag_buf, kp_drag, [], w, h, _DummyPipe())
    assert cancelled is False
    assert len(written_frames) == DETECT_EVERY_N_FRAMES
    assert all(nbytes == frame_size for nbytes in written_frames)


def test_api_deepgram_nova3_upload_and_post_transcribe_keep_asr_api_key_secret(tmp_path, monkeypatch):
    """
    Verifies that both /api/upload and /api/transcribe/{tid} accept asr_model=deepgram-nova-3
    and asr_api_key, forward asr_api_key to Orchestrator, and NEVER leak or store the key
    in HTTP responses, tasks[tid], or logs.
    """
    from src.orchestrator import Orchestrator
    from app import _audio_cache_path

    in_mp4 = str(tmp_path / "dg_api.mp4")
    with open(in_mp4, "wb") as mf:
        mf.write(b"\x00\x00\x00\x18ftypmp42")

    secret_dg_key = "dg-api-super-secret-4242"
    seen_calls = []

    def _fake_transcribe_only(
        input_path,
        transcript_path,
        job_dir,
        status_cb=None,
        asr_model="gigaam-v3",
        asr_api_key="",
        cached_audio_path=None,
        **kwargs,
    ):
        seen_calls.append({"asr_model": asr_model, "asr_api_key": asr_api_key})
        os.makedirs(os.path.dirname(os.path.abspath(transcript_path)), exist_ok=True)
        with open(transcript_path, "w", encoding="utf-8") as tf:
            tf.write("# Mock Deepgram Nova-3 transcript\n[00:00:00 - 00:00:01] Спикер 1: Привет!\n")
        if cached_audio_path:
            os.makedirs(os.path.dirname(os.path.abspath(cached_audio_path)), exist_ok=True)
            with open(cached_audio_path, "wb") as cf:
                cf.write(b"RIFFmockwav")
        return {
            "video_ready": False,
            "transcript_ready": True,
            "transcript_path": transcript_path,
            "audio_cached": bool(cached_audio_path),
            "dual_track_cached": False,
        }

    monkeypatch.setattr(Orchestrator, "process_transcription_only_job", staticmethod(_fake_transcribe_only))

    client = TestClient(app)
    tid = None
    try:
        with open(in_mp4, "rb") as f:
            up_resp = client.post(
                "/api/upload",
                files={"file": ("dg_api.mp4", f, "video/mp4")},
                data={
                    "mode": "transcribe_only",
                    "workers": "1",
                    "asr_model": "deepgram-nova-3",
                    "language": "ru",
                    "asr_api_key": secret_dg_key,
                },
            )
        assert up_resp.status_code == 200
        assert secret_dg_key not in up_resp.text
        body = up_resp.json()
        tid = body["task_id"]
        assert body["asr_model"] == "deepgram-nova-3"
        assert "asr_api_key" not in body

        st_resp = client.get(f"/api/status/{tid}")
        assert st_resp.status_code == 200
        assert secret_dg_key not in st_resp.text
        assert "asr_api_key" not in tasks[tid]
        assert secret_dg_key not in str(tasks[tid])

        # Re-transcribe via /api/transcribe/{tid} with a new Deepgram key
        secret_dg_key_2 = "dg-post-secret-9999"
        post_resp = client.post(
            f"/api/transcribe/{tid}",
            data={
                "workers": "1",
                "asr_model": "deepgram-nova-3",
                "language": "auto",
                "asr_api_key": secret_dg_key_2,
            },
        )
        assert post_resp.status_code == 200
        assert secret_dg_key_2 not in post_resp.text
        assert "asr_api_key" not in tasks[tid]
        assert secret_dg_key_2 not in str(tasks[tid])

        assert seen_calls == [
            {"asr_model": "deepgram-nova-3", "asr_api_key": secret_dg_key},
            {"asr_model": "deepgram-nova-3", "asr_api_key": secret_dg_key_2},
        ]
    finally:
        if tid:
            tasks.pop(tid, None)
            for p in (
                os.path.join("output", f"{tid}_transcript.txt"),
                _audio_cache_path(tid),
            ):
                if os.path.exists(p):
                    os.remove(p)


def test_frontend_v51_deepseek_flash_deepgram_nova3_and_live_version_footer():
    """
    Executes real static/script.js in Node.js to verify v5.1.0 frontend behavior:
      1. /api/config is fetched with { cache: 'no-store' } and updates #appVersionFooter dynamically.
      2. Default LLM model is 'deepseek-flash' (https://api.deepseek.com/v1) and stale
         localStorage 'deepseek-chat' is auto-migrated to 'deepseek-flash'.
      3. #asrApiKeyWrap is hidden for local models, shown when 'deepgram-nova-3' is selected,
         stores selectedAsrApiKey ONLY in sessionStorage (never localStorage), sends asr_api_key
         for deepgram-nova-3 uploads & post-transcribe, and omits asr_api_key for local models.
    """
    node_test_js = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

class FakeClassList {
    constructor(init = []) { this.set = new Set(init); }
    add(...c) { c.forEach((x) => this.set.add(x)); }
    remove(...c) { c.forEach((x) => this.set.delete(x)); }
    toggle(c, force) {
        if (force === undefined) {
            if (this.set.has(c)) { this.set.delete(c); return false; }
            this.set.add(c); return true;
        }
        if (force) this.set.add(c); else this.set.delete(c);
        return Boolean(force);
    }
    contains(c) { return this.set.has(c); }
}

class FakeEvent {
    constructor(type, opts = {}) { this.type = type; this.bubbles = Boolean(opts.bubbles); }
    preventDefault() {}
    stopPropagation() {}
}

class FakeElement {
    constructor(tag = 'div', id = '', classes = []) {
        this.tagName = tag.toUpperCase();
        this.id = id;
        this.className = classes.join(' ');
        this.classList = new FakeClassList(classes);
        this.children = [];
        this.parentNode = null;
        this.listeners = {};
        this.attributes = {};
        this.dataset = {};
        this._textContent = '';
        this._innerHTML = '';
        this.value = '';
        this.checked = false;
        this.disabled = false;
        this.title = '';
        this.files = [];
    }
    set textContent(v) { this._textContent = String(v); }
    get textContent() { return this._textContent; }
    set innerHTML(v) { this._innerHTML = String(v); if (v === '') this.children = []; }
    get innerHTML() { return this._innerHTML; }
    appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
    insertBefore(node, ref) {
        node.parentNode = this;
        const idx = this.children.indexOf(ref);
        if (idx === -1) this.children.push(node); else this.children.splice(idx, 0, node);
        return node;
    }
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
    dispatchEvent(evt) { (this.listeners[evt.type] || []).forEach((fn) => fn.call(this, evt)); }
    click() {
        if (this.disabled) return;
        if (typeof this.onclick === 'function') this.onclick(new FakeEvent('click'));
        this.dispatchEvent(new FakeEvent('click'));
    }
    setAttribute(k, v) { this.attributes[k] = String(v); }
    getAttribute(k) { return this.attributes[k] || null; }
    focus() {}
    querySelectorAll() { return []; }
    querySelector() { return null; }
}

class FakeSelect extends FakeElement {
    constructor(id, options) {
        super('select', id);
        this.parentNode = new FakeElement('div');
        this.parentNode.appendChild(this);
        options.forEach((o) => {
            const opt = new FakeElement('option');
            opt.value = o.value;
            opt.textContent = o.label;
            this.appendChild(opt);
        });
        this.value = options[0].value;
    }
    get options() { return this.children; }
    get selectedIndex() { return Math.max(0, this.children.findIndex((o) => o.value === this.value)); }
}

function runScenario({ initialLocalStorage = {} } = {}) {
    const storage = new Map(Object.entries(initialLocalStorage));
    const localStorage = {
        getItem: (k) => (storage.has(k) ? storage.get(k) : null),
        setItem: (k, v) => storage.set(k, String(v)),
        removeItem: (k) => storage.delete(k),
    };
    const sessStorage = new Map();
    const sessionStorage = {
        getItem: (k) => (sessStorage.has(k) ? sessStorage.get(k) : null),
        setItem: (k, v) => sessStorage.set(k, String(v)),
        removeItem: (k) => sessStorage.delete(k),
    };

    const byId = new Map();
    const reg = (el) => { if (el.id) byId.set(el.id, el); return el; };

    [
        'dropZone', 'fileInput', 'dropZoneTitle', 'dropZoneHint', 'themeToggle',
        'uploadPanel', 'statusPanel', 'statusText', 'statusSubText', 'spinner',
        'successIcon', 'cancelIcon', 'cancelBtn', 'downloadBtn', 'downloadTranscriptBtn',
        'resetBtn', 'debugLog', 'modeAnonymizeBtn', 'modeTranscribeOnlyBtn',
        'workerValue', 'customWorkerWrap', 'customWorkerInput', 'workerLimitText',
        'transcribeSwitchWrap', 'transcribeStateBadge', 'transcribeOptionsWrap',
        'customPromptInput', 'promptPresets', 'transcribeHintText',
        'recordingsSection', 'recordingsCount', 'recordingsList',
        'postTranscribeSection', 'togglePostTranscribeBtn', 'postTranscribeForm',
        'postTranscribeTitle', 'postCustomPromptInput', 'postPromptPresets',
        'postAudioSourceHint', 'postTranscribeBtn',
    ].forEach((id) => reg(new FakeElement('div', id)));

    const appVersionFooter = reg(new FakeElement('footer', 'appVersionFooter'));
    appVersionFooter.textContent = 'v5.0.0 (Stale Cached)';

    const asrApiKeyWrap = reg(new FakeElement('div', 'asrApiKeyWrap', ['llm-config-wrap', 'hidden']));
    const asrApiKeyInput = reg(new FakeElement('input', 'asrApiKeyInput'));
    const postAsrApiKeyWrap = reg(new FakeElement('div', 'postAsrApiKeyWrap', ['llm-config-wrap', 'hidden']));
    const postAsrApiKeyInput = reg(new FakeElement('input', 'postAsrApiKeyInput'));

    const transcribeToggle = reg(new FakeElement('input', 'transcribeToggle'));
    transcribeToggle.checked = true;
    const denoiseToggle = reg(new FakeElement('input', 'denoiseToggle'));
    const llmPolishToggle = reg(new FakeElement('input', 'llmPolishToggle'));
    const llmConfigWrap = reg(new FakeElement('div', 'llmConfigWrap', ['hidden']));
    const llmBaseUrlInput = reg(new FakeElement('input', 'llmBaseUrlInput'));
    const llmModelInput = reg(new FakeElement('input', 'llmModelInput'));
    const llmApiKeyInput = reg(new FakeElement('input', 'llmApiKeyInput'));
    const llmPrivacyHint = reg(new FakeElement('p', 'llmPrivacyHint'));
    const llmPresets = reg(new FakeElement('div', 'llmPresets'));

    const postDenoiseToggle = reg(new FakeElement('input', 'postDenoiseToggle'));
    const postLlmPolishToggle = reg(new FakeElement('input', 'postLlmPolishToggle'));
    const postLlmConfigWrap = reg(new FakeElement('div', 'postLlmConfigWrap', ['hidden']));
    const postLlmBaseUrlInput = reg(new FakeElement('input', 'postLlmBaseUrlInput'));
    const postLlmModelInput = reg(new FakeElement('input', 'postLlmModelInput'));
    const postLlmApiKeyInput = reg(new FakeElement('input', 'postLlmApiKeyInput'));
    const postLlmPrivacyHint = reg(new FakeElement('p', 'postLlmPrivacyHint'));
    const postLlmPresets = reg(new FakeElement('div', 'postLlmPresets'));

    const modelOpts = [
        { value: 'gigaam-v3', label: 'GigaAM v3 FP32' },
        { value: 'deepgram-nova-3', label: 'Deepgram Nova-3' },
    ];
    reg(new FakeSelect('workerSelect', [{ value: '1', label: '1' }, { value: '4', label: '4' }, { value: 'custom', label: 'Другое' }]));
    const asrModelSelect = reg(new FakeSelect('asrModelSelect', modelOpts));
    const postAsrModelSelect = reg(new FakeSelect('postAsrModelSelect', modelOpts));
    reg(new FakeSelect('languageSelect', [{ value: 'ru', label: 'RU' }, { value: 'auto', label: 'AUTO' }, { value: 'en', label: 'EN' }]));
    reg(new FakeSelect('postLanguageSelect', [{ value: 'ru', label: 'RU' }, { value: 'auto', label: 'AUTO' }, { value: 'en', label: 'EN' }]));
    const computeDeviceSelect = reg(new FakeSelect('computeDeviceSelect', [{ value: 'cpu', label: 'CPU' }, { value: 'cuda', label: 'GPU (CUDA)' }]));
    reg(new FakeSelect('postComputeDeviceSelect', [{ value: 'cpu', label: 'CPU' }, { value: 'cuda', label: 'GPU (CUDA)' }]));
    reg(new FakeSelect('speakerCountSelect', [{ value: 'auto', label: 'Авто' }, { value: '2', label: '2' }]));
    reg(new FakeSelect('postSpeakerCountSelect', [{ value: 'auto', label: 'Авто' }, { value: '2', label: '2' }]));
    reg(new FakeSelect('diarizationSensitivitySelect', [{ value: 'normal', label: 'Обычная' }]));
    reg(new FakeSelect('postDiarizationSensitivitySelect', [{ value: 'normal', label: 'Обычная' }]));
    reg(new FakeSelect('audioTracksSelect', [{ value: 'auto', label: 'Авто' }]));
    reg(new FakeSelect('postAudioTracksSelect', [{ value: 'auto', label: 'Авто' }]));

    const sunIcon = new FakeElement('svg');
    const moonIcon = new FakeElement('svg');
    const documentObj = {
        documentElement: new FakeElement('html'),
        getElementById: (id) => byId.get(id) || null,
        querySelector: (sel) => (sel === '.sun-icon' ? sunIcon : sel === '.moon-icon' ? moonIcon : null),
        createElement: (tag) => new FakeElement(tag),
        createElementNS: (_ns, tag) => new FakeElement(tag),
        addEventListener: () => {},
        activeElement: null,
    };

    class FakeFormData {
        constructor() { this.entries = {}; }
        append(k, v) { this.entries[k] = v; }
    }

    const fetchCalls = [];
    const uploadCalls = [];
    const postCalls = [];
    const fakeFetch = async (url, opts = {}) => {
        fetchCalls.push({ url, opts });
        if (url === '/api/config') {
            return {
                ok: true,
                json: async () => ({
                    version: '5.1.0',
                    build_id: '5.1.0-abc1234-fedcba',
                    cpu_count: 8,
                    default_workers: 4,
                    max_workers: 7,
                    transcription_default: true,
                    default_asr_model: 'gigaam-v3',
                    default_language: 'ru',
                    asr_models: [
                        { id: 'gigaam-v3', label: 'GigaAM v3 FP32', engine: 'onnx_asr', supports_cuda: true, is_cloud: false, requires_api_key: false },
                        { id: 'deepgram-nova-3', label: 'Deepgram Nova-3 — облачный API', engine: 'deepgram_cloud', supports_cuda: false, is_cloud: true, requires_api_key: true, has_native_diarization: true },
                    ],
                    cuda_available: true,
                    ct2_cuda_available: true,
                    onnx_cuda_available: true,
                    default_compute_device: 'cpu',
                    compute_devices: [{ id: 'cpu', label: 'CPU' }, { id: 'cuda', label: 'GPU (CUDA)' }],
                    denoise_default: false,
                    llm_polish_default: false,
                    llm_defaults: { base_url: 'https://api.deepseek.com/v1', model: 'deepseek-flash' },
                    llm_presets: [
                        { id: 'deepseek_cloud', base_url: 'https://api.deepseek.com/v1', model: 'deepseek-flash', is_cloud: true },
                        { id: 'qwen_cloud', base_url: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus', is_cloud: true },
                        { id: 'ollama_local', base_url: 'http://localhost:11434/v1', model: 'qwen3:8b', is_cloud: false },
                    ],
                }),
            };
        }
        if (url === '/api/recordings') return { ok: true, json: async () => ({ recordings: [] }) };
        if (url === '/api/upload') {
            uploadCalls.push({ ...opts.body.entries });
            return { ok: true, json: async () => ({ task_id: 'v51-task', status: 'queued', worker_count: 4 }) };
        }
        if (url === '/api/status/v51-task') {
            return {
                ok: true,
                json: async () => ({
                    status: 'completed',
                    mode: 'anonymize',
                    video_ready: true,
                    transcript_ready: true,
                    can_transcribe: true,
                    has_original_audio: true,
                    asr_model: uploadCalls[uploadCalls.length - 1].asr_model,
                    compute_device: uploadCalls[uploadCalls.length - 1].compute_device,
                    logs: ['Done!'],
                }),
            };
        }
        if (url === '/api/transcribe/v51-task') {
            postCalls.push({ ...opts.body.entries });
            return { ok: true, json: async () => ({ task_id: 'v51-task', status: 'queued' }) };
        }
        throw new Error('Unexpected URL: ' + url);
    };

    const ctx = vm.createContext({
        document: documentObj,
        window: { location: { href: '' }, matchMedia: () => ({ matches: false }) },
        localStorage,
        sessionStorage,
        Event: FakeEvent,
        FormData: FakeFormData,
        AbortController: class { constructor() { this.signal = {}; } abort() {} },
        fetch: fakeFetch,
        setInterval: () => 1,
        clearInterval: () => {},
        alert: (m) => { throw new Error('Unexpected alert: ' + m); },
        console,
        Number, String, Boolean, Array, Map, Set, Date,
    });
    vm.runInContext(fs.readFileSync('static/script.js', 'utf8'), ctx);
    return {
        ctx, storage, sessStorage, fetchCalls, uploadCalls, postCalls,
        appVersionFooter, asrModelSelect, asrApiKeyWrap, asrApiKeyInput,
        postAsrModelSelect, postAsrApiKeyWrap, postAsrApiKeyInput,
        computeDeviceSelect, llmBaseUrlInput, llmModelInput, llmPrivacyHint,
        fileInput: byId.get('fileInput'), postTranscribeBtn: byId.get('postTranscribeBtn'),
    };
}

(async () => {
    // 1. Fresh load: deepseek-flash default, live footer version from /api/config with cache: no-store
    const h = runScenario();
    await new Promise((r) => setTimeout(r, 25));

    const cfgCall = h.fetchCalls.find((c) => c.url === '/api/config');
    assert.ok(cfgCall, '/api/config must be fetched');
    assert.strictEqual(cfgCall.opts.cache, 'no-store');
    assert.strictEqual(h.appVersionFooter.textContent, 'v5.1.0 (Stable Edition • abc1234)');

    assert.strictEqual(h.llmBaseUrlInput.value, 'https://api.deepseek.com/v1');
    assert.strictEqual(h.llmModelInput.value, 'deepseek-flash');
    assert.strictEqual(h.llmPrivacyHint.classList.contains('is-cloud'), true);

    // 2. Local model keeps #asrApiKeyWrap hidden
    assert.strictEqual(h.asrModelSelect.value, 'gigaam-v3');
    assert.strictEqual(h.asrApiKeyWrap.classList.contains('hidden'), true);

    // Switch to GPU first, then pick deepgram-nova-3 -> auto-switches to CPU & shows API key field
    h.computeDeviceSelect.value = 'cuda';
    h.computeDeviceSelect.dispatchEvent(new h.ctx.Event('change'));
    assert.strictEqual(h.computeDeviceSelect.value, 'cuda');

    h.asrModelSelect.value = 'deepgram-nova-3';
    h.asrModelSelect.dispatchEvent(new h.ctx.Event('change'));
    assert.strictEqual(h.computeDeviceSelect.value, 'cpu');
    assert.strictEqual(h.asrApiKeyWrap.classList.contains('hidden'), false);

    // Entering Deepgram key stores ONLY in sessionStorage, NEVER localStorage
    h.asrApiKeyInput.value = 'dg-session-only-secret';
    h.asrApiKeyInput.dispatchEvent(new h.ctx.Event('input'));
    assert.strictEqual(h.sessStorage.get('selectedAsrApiKey'), 'dg-session-only-secret');
    assert.strictEqual(h.storage.has('selectedAsrApiKey'), false);

    // Upload sends asr_api_key for deepgram-nova-3
    h.fileInput.files = [{ name: 'interview.mp4' }];
    h.fileInput.dispatchEvent(new h.ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(h.uploadCalls.length, 1);
    assert.strictEqual(h.uploadCalls[0].asr_model, 'deepgram-nova-3');
    assert.strictEqual(h.uploadCalls[0].asr_api_key, 'dg-session-only-secret');

    // Post-transcribe form inherits deepgram-nova-3 & shows #postAsrApiKeyWrap
    assert.strictEqual(h.postAsrApiKeyWrap.classList.contains('hidden'), false);
    assert.strictEqual(h.postAsrApiKeyInput.value, 'dg-session-only-secret');
    h.postTranscribeBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(h.postCalls.length, 1);
    assert.strictEqual(h.postCalls[0].asr_model, 'deepgram-nova-3');
    assert.strictEqual(h.postCalls[0].asr_api_key, 'dg-session-only-secret');

    // Switching back to local gigaam-v3 hides API key input and omits asr_api_key on post-transcribe
    h.postAsrModelSelect.value = 'gigaam-v3';
    h.postAsrModelSelect.dispatchEvent(new h.ctx.Event('change'));
    assert.strictEqual(h.postAsrApiKeyWrap.classList.contains('hidden'), true);
    h.postTranscribeBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(h.postCalls.length, 2);
    assert.strictEqual(h.postCalls[1].asr_model, 'gigaam-v3');
    assert.strictEqual(h.postCalls[1].asr_api_key, undefined);
    assert.strictEqual(h.storage.has('selectedAsrApiKey'), false);

    // 3. Stale localStorage 'deepseek-chat' and removed ASR model 'canary-1b-v2' are auto-migrated
    const mig = runScenario({
        initialLocalStorage: {
            selectedLlmBaseUrl: 'https://api.deepseek.com/v1',
            selectedLlmModel: 'deepseek-chat',
            selectedAsrModel: 'canary-1b-v2',
        },
    });
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(mig.llmModelInput.value, 'deepseek-flash');
    assert.strictEqual(mig.storage.get('selectedLlmModel'), 'deepseek-flash');
    assert.strictEqual(mig.asrModelSelect.value, 'gigaam-v3');
    assert.strictEqual(mig.storage.get('selectedAsrModel'), 'gigaam-v3');

    console.log('FRONTEND_V51_ALL_OK');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_test_js],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node v5.1 frontend test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FRONTEND_V51_ALL_OK" in proc.stdout


def test_api_raw_transcript_backup_download_and_status():
    import uuid
    from app import _raw_transcript_path

    client = TestClient(app)
    tid = str(uuid.uuid4())
    txt_path = os.path.join("output", f"{tid}_transcript.txt")
    raw_path = _raw_transcript_path(tid)
    os.makedirs("output", exist_ok=True)

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("[00:00:01 - 00:00:04] Спикер 1: Работаем по gRPC.\n")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("[00:00:01 - 00:00:04] Спикер 1: Работаем поджирописью.\n")

        # 1. Disk fallback status reports raw_transcript_ready=True
        st_resp = client.get(f"/api/status/{tid}")
        assert st_resp.status_code == 200
        st_data = st_resp.json()
        assert st_data["transcript_ready"] is True
        assert st_data["raw_transcript_ready"] is True

        # 2. /api/recordings reports raw_transcript_ready=True
        rec_resp = client.get("/api/recordings")
        assert rec_resp.status_code == 200
        rec = next((r for r in rec_resp.json()["recordings"] if r["task_id"] == tid), None)
        assert rec is not None and rec["raw_transcript_ready"] is True

        # 3. Default download returns polished transcript; ?raw=true returns raw ASR backup
        dl_main = client.get(f"/api/download-transcript/{tid}")
        assert dl_main.status_code == 200
        assert "по gRPC" in dl_main.text

        dl_raw = client.get(f"/api/download-transcript/{tid}?raw=true")
        assert dl_raw.status_code == 200
        assert "поджирописью" in dl_raw.text
        assert f"transcript_{tid}_raw.txt" in dl_raw.headers.get("content-disposition", "")

        # 4. Once _raw.txt is removed, raw_transcript_ready becomes False and ?raw=true returns 404
        os.remove(raw_path)
        assert client.get(f"/api/status/{tid}").json()["raw_transcript_ready"] is False
        assert client.get(f"/api/download-transcript/{tid}?raw=true").status_code == 404
    finally:
        for p in (txt_path, raw_path):
            if os.path.exists(p):
                os.remove(p)


def test_frontend_stage_file_configure_and_start_button():
    """
    Executes static/script.js in Node.js to verify the 2-step staged upload flow:
      1. Selecting/dropping a recording stages it WITHOUT immediately calling /api/upload,
         reveals #selectedFileBar with filename/size, and enables #startProcessingBtn ('Начать').
      2. Changing mode/model/speakers/sensitivity/LLM AFTER picking the file submits the updated
         settings only when the user clicks 'Начать'.
      3. Dropping an audio file (.mp3) while in 'Анонимизация' mode auto-switches to 'Только транскрибация'
         without starting; switching back to 'Анонимизация' disables 'Начать' with a warning until fixed;
         clicking 'Убрать' or 'Загрузить ещё' clears the staged file.
    """
    node_test_js = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

class FakeEvent {
    constructor(type) { this.type = type; }
    preventDefault() {}
    stopPropagation() {}
}

const byId = new Map();
class FakeElement {
    constructor(tag, opts = {}) {
        this.tagName = tag.toUpperCase();
        this.id = opts.id || '';
        this.children = [];
        this.parentNode = null;
        this.listeners = {};
        this.attributes = {};
        this.dataset = opts.dataset ? { ...opts.dataset } : {};
        this.style = {};
        this.disabled = Boolean(opts.disabled);
        this.checked = Boolean(opts.checked);
        this.value = opts.value || '';
        this.textContent = opts.text || '';
        this.innerHTML = '';
        this.title = '';
        this.files = [];
        this.onclick = null;
        this._classes = new Set(opts.classes || []);
        this.classList = {
            add: (...c) => c.forEach((x) => this._classes.add(x)),
            remove: (...c) => c.forEach((x) => this._classes.delete(x)),
            toggle: (c, force) => {
                if (force === undefined) {
                    if (this._classes.has(c)) this._classes.delete(c);
                    else this._classes.add(c);
                } else if (force) {
                    this._classes.add(c);
                } else {
                    this._classes.delete(c);
                }
            },
            contains: (c) => this._classes.has(c),
        };
        if (this.id) byId.set(this.id, this);
    }
    get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
    appendChild(child) {
        child.parentNode = this;
        this.children.push(child);
        if (this.tagName === 'SELECT' && child.tagName === 'OPTION') {
            if (child.selected || this.value === '') this.value = child.value;
        }
        return child;
    }
    insertBefore(child) { return this.appendChild(child); }
    setAttribute(k, v) { this.attributes[k] = String(v); }
    getAttribute(k) { return this.attributes[k] ?? null; }
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
    dispatchEvent(evt) { (this.listeners[evt.type] || []).forEach((fn) => fn.call(this, evt)); }
    click() {
        if (this.disabled) return;
        if (typeof this.onclick === 'function') this.onclick();
        this.dispatchEvent(new FakeEvent('click'));
    }
    focus() {}
    querySelectorAll() { return []; }
    querySelector() { return null; }
}

const root = new FakeElement('div', { id: 'root' });
const makeSelect = (id, opts) => {
    const sel = new FakeElement('select', { id });
    opts.forEach(([val, text, selected]) => {
        const opt = new FakeElement('option', { value: val, text });
        opt.selected = Boolean(selected);
        sel.appendChild(opt);
    });
    root.appendChild(sel);
    return sel;
};

const themeBtn = root.appendChild(new FakeElement('button', { id: 'themeToggle' }));
const sunIcon = themeBtn.appendChild(new FakeElement('svg', { classes: ['sun-icon', 'hidden'] }));
const moonIcon = themeBtn.appendChild(new FakeElement('svg', { classes: ['moon-icon', 'hidden'] }));

const uploadPanel = root.appendChild(new FakeElement('div', { id: 'uploadPanel' }));
const modeAnonymizeBtn = uploadPanel.appendChild(new FakeElement('button', { id: 'modeAnonymizeBtn', classes: ['is-active'] }));
const modeTranscribeOnlyBtn = uploadPanel.appendChild(new FakeElement('button', { id: 'modeTranscribeOnlyBtn' }));
const dropZone = uploadPanel.appendChild(new FakeElement('div', { id: 'dropZone' }));
uploadPanel.appendChild(new FakeElement('h2', { id: 'dropZoneTitle' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'dropZoneHint' }));
const fileInput = uploadPanel.appendChild(new FakeElement('input', { id: 'fileInput' }));
const selectedFileBar = uploadPanel.appendChild(new FakeElement('div', { id: 'selectedFileBar', classes: ['hidden'] }));
const selectedFileName = selectedFileBar.appendChild(new FakeElement('span', { id: 'selectedFileName' }));
const selectedFileMeta = selectedFileBar.appendChild(new FakeElement('span', { id: 'selectedFileMeta' }));
const replaceSelectedFileBtn = selectedFileBar.appendChild(new FakeElement('button', { id: 'replaceSelectedFileBtn' }));
const clearSelectedFileBtn = selectedFileBar.appendChild(new FakeElement('button', { id: 'clearSelectedFileBtn' }));

uploadPanel.appendChild(new FakeElement('span', { id: 'workerValue', text: '4' }));
makeSelect('workerSelect', [['1', '1'], ['4', '4', true], ['custom', 'Другое']]);
uploadPanel.appendChild(new FakeElement('div', { id: 'customWorkerWrap', classes: ['hidden'] }));
uploadPanel.appendChild(new FakeElement('input', { id: 'customWorkerInput' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'workerLimitText' }));
uploadPanel.appendChild(new FakeElement('input', { id: 'transcribeToggle', checked: true }));
uploadPanel.appendChild(new FakeElement('span', { id: 'transcribeStateBadge', text: 'Вкл' }));
uploadPanel.appendChild(new FakeElement('div', { id: 'transcribeOptionsWrap' }));
const asrModelSelect = makeSelect('asrModelSelect', [
    ['gigaam-v3', 'GigaAM v3 FP32', true],
    ['breeze-asr-25', 'Breeze-ASR-25'],
    ['turbo', 'Whisper Turbo'],
]);
makeSelect('languageSelect', [['ru', 'Русский (RU)', true], ['auto', 'Авто']]);
makeSelect('computeDeviceSelect', [['cpu', 'CPU', true], ['cuda', 'GPU (CUDA)']]);
const speakerCountSelect = makeSelect('speakerCountSelect', [['auto', 'Авто', true], ['4', '4 спикера']]);
const diarizationSensitivitySelect = makeSelect('diarizationSensitivitySelect', [['normal', 'Обычная', true], ['high', 'Высокая']]);
makeSelect('audioTracksSelect', [['auto', 'Авто', true]]);
uploadPanel.appendChild(new FakeElement('textarea', { id: 'customPromptInput' }));
const llmPolishToggle = uploadPanel.appendChild(new FakeElement('input', { id: 'llmPolishToggle', checked: false }));
uploadPanel.appendChild(new FakeElement('div', { id: 'llmConfigWrap', classes: ['hidden'] }));
uploadPanel.appendChild(new FakeElement('input', { id: 'llmBaseUrlInput', value: 'https://api.deepseek.com/v1' }));
uploadPanel.appendChild(new FakeElement('input', { id: 'llmModelInput', value: 'deepseek-flash' }));
uploadPanel.appendChild(new FakeElement('input', { id: 'llmApiKeyInput' }));
uploadPanel.appendChild(new FakeElement('p', { id: 'llmPrivacyHint' }));
const startProcessingBtn = uploadPanel.appendChild(new FakeElement('button', { id: 'startProcessingBtn', disabled: true, text: 'Начать' }));
const startProcessingHint = uploadPanel.appendChild(new FakeElement('p', { id: 'startProcessingHint' }));

const statusPanel = root.appendChild(new FakeElement('div', { id: 'statusPanel', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('div', { id: 'spinner' }));
statusPanel.appendChild(new FakeElement('svg', { id: 'successIcon', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('svg', { id: 'cancelIcon', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('h2', { id: 'statusText' }));
statusPanel.appendChild(new FakeElement('p', { id: 'statusSubText' }));
statusPanel.appendChild(new FakeElement('div', { id: 'debugLog', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'cancelBtn', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'downloadBtn', classes: ['hidden'] }));
statusPanel.appendChild(new FakeElement('button', { id: 'downloadTranscriptBtn', classes: ['hidden'] }));
const downloadRawTranscriptBtn = statusPanel.appendChild(new FakeElement('button', { id: 'downloadRawTranscriptBtn', classes: ['hidden'] }));
const resetBtn = statusPanel.appendChild(new FakeElement('button', { id: 'resetBtn', classes: ['hidden'] }));

const storage = new Map();
const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
    removeItem: (k) => storage.delete(k),
};
const sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

const documentObj = {
    documentElement: new FakeElement('html'),
    getElementById: (id) => byId.get(id) || null,
    querySelector: (sel) => (sel === '.sun-icon' ? sunIcon : sel === '.moon-icon' ? moonIcon : null),
    querySelectorAll: () => [],
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    addEventListener: () => {},
};

class FakeFormData {
    constructor() { this.entries = {}; }
    append(k, v) { this.entries[k] = v; }
}

const uploadCalls = [];
const alerts = [];
const fakeFetch = async (url, opts = {}) => {
    if (url === '/api/config') {
        return {
            ok: true,
            json: async () => ({
                version: '5.1.0',
                build_id: '5.1.0-stage',
                cpu_count: 8,
                default_workers: 4,
                max_workers: 7,
                transcription_default: true,
                default_asr_model: 'gigaam-v3',
                asr_models: [
                    { id: 'gigaam-v3', label: 'GigaAM v3 FP32', supports_cuda: true, engine: 'onnx_asr' },
                    { id: 'breeze-asr-25', label: 'Breeze-ASR-25', supports_cuda: true, engine: 'faster_whisper' },
                    { id: 'turbo', label: 'Whisper Turbo', supports_cuda: true, engine: 'faster_whisper' },
                ],
                cuda_available: true,
                ct2_cuda_available: true,
                onnx_cuda_available: true,
                default_compute_device: 'cpu',
                compute_devices: [{ id: 'cpu', label: 'CPU' }, { id: 'cuda', label: 'GPU (CUDA)' }],
            }),
        };
    }
    if (url === '/api/recordings') return { ok: true, json: async () => ({ recordings: [] }) };
    if (url === '/api/upload') {
        uploadCalls.push({ ...opts.body.entries });
        return { ok: true, json: async () => ({ task_id: `stage-${uploadCalls.length}`, status: 'queued' }) };
    }
    if (url.startsWith('/api/status/')) {
        return {
            ok: true,
            json: async () => ({
                status: 'completed',
                mode: uploadCalls[uploadCalls.length - 1].mode,
                video_ready: false,
                transcript_ready: true,
                raw_transcript_ready: true,
                logs: ['Done!'],
            }),
        };
    }
    throw new Error('Unexpected URL: ' + url);
};

const ctx = vm.createContext({
    document: documentObj,
    window: { location: { href: '' }, matchMedia: () => ({ matches: false }) },
    localStorage,
    sessionStorage,
    Event: FakeEvent,
    FormData: FakeFormData,
    AbortController: class { constructor() { this.signal = {}; } abort() {} },
    fetch: fakeFetch,
    setInterval: () => 1,
    clearInterval: () => {},
    alert: (m) => alerts.push(m),
    console,
    Number, String, Boolean, Array, Map, Set, Date,
});

vm.runInContext(fs.readFileSync('static/script.js', 'utf8'), ctx);

(async () => {
    await new Promise((r) => setTimeout(r, 25));

    // 1. Initial state: Start button disabled, file bar hidden
    assert.strictEqual(startProcessingBtn.disabled, true);
    assert.strictEqual(selectedFileBar.classList.contains('hidden'), true);
    startProcessingBtn.click();
    assert.strictEqual(uploadCalls.length, 0);

    // 2. Select interview.mp4 -> staged ONLY, NO immediate upload!
    fileInput.files = [{ name: 'interview.mp4', size: 50 * 1024 * 1024 }];
    fileInput.dispatchEvent(new FakeEvent('change'));
    await new Promise((r) => setTimeout(r, 15));
    assert.strictEqual(uploadCalls.length, 0);
    assert.strictEqual(selectedFileBar.classList.contains('hidden'), false);
    assert.strictEqual(dropZone.classList.contains('has-file'), true);
    assert.strictEqual(selectedFileName.textContent, 'interview.mp4');
    assert.ok(selectedFileMeta.textContent.includes('50.0 МБ'));
    assert.strictEqual(startProcessingBtn.disabled, false);

    // 3. Configure settings AFTER picking the file
    modeTranscribeOnlyBtn.click();
    asrModelSelect.value = 'breeze-asr-25';
    asrModelSelect.dispatchEvent(new FakeEvent('change'));
    speakerCountSelect.value = '4';
    speakerCountSelect.dispatchEvent(new FakeEvent('change'));
    diarizationSensitivitySelect.value = 'high';
    diarizationSensitivitySelect.dispatchEvent(new FakeEvent('change'));
    llmPolishToggle.checked = true;
    llmPolishToggle.dispatchEvent(new FakeEvent('change'));
    assert.strictEqual(uploadCalls.length, 0);

    // 4. Click 'Начать' -> uploads staged file with all newly configured settings!
    startProcessingBtn.click();
    await new Promise((r) => setTimeout(r, 25));
    assert.strictEqual(uploadCalls.length, 1);
    assert.strictEqual(uploadCalls[0].file.name, 'interview.mp4');
    assert.strictEqual(uploadCalls[0].mode, 'transcribe_only');
    assert.strictEqual(uploadCalls[0].asr_model, 'breeze-asr-25');
    assert.strictEqual(uploadCalls[0].speakers, '4');
    assert.strictEqual(uploadCalls[0].diarization_sensitivity, 'high');
    assert.strictEqual(uploadCalls[0].llm_polish, 'true');

    // Also verify raw ASR download button becomes visible when raw_transcript_ready=true
    assert.strictEqual(downloadRawTranscriptBtn.classList.contains('hidden'), false);

    // 5. Reset clears staged file and re-disables Start button
    resetBtn.click();
    assert.strictEqual(selectedFileBar.classList.contains('hidden'), true);
    assert.strictEqual(startProcessingBtn.disabled, true);
    assert.strictEqual(downloadRawTranscriptBtn.classList.contains('hidden'), true);

    // 6. Drop .mp3 while on 'Анонимизация' -> auto-switches to 'Только транскрибация' & stages without upload
    modeAnonymizeBtn.click();
    fileInput.files = [{ name: 'voice_note.mp3', size: 3 * 1024 * 1024 }];
    fileInput.dispatchEvent(new FakeEvent('change'));
    assert.strictEqual(uploadCalls.length, 1);
    assert.strictEqual(modeTranscribeOnlyBtn.classList.contains('is-active'), true);
    assert.strictEqual(selectedFileName.textContent, 'voice_note.mp3');
    assert.strictEqual(startProcessingBtn.disabled, false);

    // Switching back to 'Анонимизация' with .mp3 staged disables 'Начать' with warning
    modeAnonymizeBtn.click();
    assert.strictEqual(startProcessingBtn.disabled, true);
    assert.strictEqual(startProcessingHint.classList.contains('is-warning'), true);
    assert.ok(startProcessingHint.textContent.includes('.mp4'));

    // Switching back to 'Только транскрибация' re-enables 'Начать'
    modeTranscribeOnlyBtn.click();
    assert.strictEqual(startProcessingBtn.disabled, false);
    assert.strictEqual(startProcessingHint.classList.contains('is-warning'), false);

    // Clicking 'Убрать' clears the staged file
    clearSelectedFileBtn.click();
    assert.strictEqual(selectedFileBar.classList.contains('hidden'), true);
    assert.strictEqual(startProcessingBtn.disabled, true);
    assert.strictEqual(alerts.length, 0);

    console.log('FRONTEND_STAGE_AND_START_OK');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_test_js],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node staged-upload test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FRONTEND_STAGE_AND_START_OK" in proc.stdout


