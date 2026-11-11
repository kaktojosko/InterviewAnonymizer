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
    assert args_default.asr_model == "turbo"
    assert args_default.speakers == "auto"
    assert args_default.device == "cpu"
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
            "parakeet",
            "--speakers",
            "2",
            "--device",
            "cuda",
            "-w",
            "1",
        ]
    )
    assert args_custom.no_transcribe is True
    assert args_custom.asr_model == "parakeet"
    assert args_custom.speakers == "2"
    assert args_custom.device == "cuda"
    assert args_custom.transcript_output == "dialog.txt"
    assert args_custom.workers == 1


def test_api_root_and_config():
    client = TestClient(app)
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "transcribeToggle" in root_resp.text
    assert "computeDeviceSelect" in root_resp.text
    assert "downloadTranscriptBtn" in root_resp.text
    assert "cancelBtn" in root_resp.text
    assert "Прервать" in root_resp.text

    cfg_resp = client.get("/api/config")
    assert cfg_resp.status_code == 200
    data = cfg_resp.json()
    assert data["version"] == "4.4.0"

    assert data["transcription_default"] is True
    assert data["default_asr_model"] == "turbo"
    assert data["default_compute_device"] == "cpu"
    assert isinstance(data["cuda_available"], bool)
    assert [d["id"] for d in data["compute_devices"]] == ["cpu", "cuda"]
    model_ids = [m["id"] for m in data["asr_models"]]
    assert model_ids == ["turbo", "large-v3", "parakeet", "small"]
    assert data["speaker_options"] == ["auto", "2", "3", "4+"]


def test_api_upload_validation_errors():
    client = TestClient(app)

    r_ext = client.post(
        "/api/upload",
        files={"file": ("clip.avi", b"dummy", "video/x-msvideo")},
    )
    assert r_ext.status_code == 400

    r_model = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"asr_model": "nonexistent"},
    )
    assert r_model.status_code == 400

    r_spk = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", b"dummy", "video/mp4")},
        data={"speakers": "99"},
    )
    assert r_spk.status_code == 400

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
            asr_model="small",
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
        assert cancel_resp.json() == {
            "task_id": task_id,
            "status": "cancelled",
            "cancelled": True,
        }

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

        dl_vid2 = client.get(f"/api/download/{tid2}")
        assert dl_vid2.status_code == 200

        dl_txt2 = client.get(f"/api/download-transcript/{tid2}")
        assert dl_txt2.status_code == 404
    finally:
        for tid in created_task_ids:
            tasks.pop(tid, None)
            for suffix in ("_anonymized.mp4", "_transcript.txt"):
                out_file = os.path.join("output", f"{tid}{suffix}")
                if os.path.exists(out_file):
                    os.remove(out_file)


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
      1. After processing a CPU job (including with CPU-only 'parakeet') and clicking
         'Загрузить ещё' (#resetBtn), 'GPU (CUDA)' in #computeDeviceSelect is enabled,
         clickable, auto-switches ASR model from 'parakeet' to 'turbo', and submits
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
        ['turbo', 'Whisper Turbo — точная (99 яз., 810 МБ)', true],
        ['large-v3', 'Whisper Large v3 — макс. точность (99 яз., 3.1 ГБ)'],
        ['parakeet', 'Parakeet v3 — быстрая (25 яз., 600 МБ)'],
        ['small', 'Whisper Small — лёгкая (99 яз., 245 МБ)'],
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
                    version: '4.4.0',
                    cpu_count: 8,
                    default_workers: 4,
                    max_workers: 7,
                    transcription_default: true,
                    default_asr_model: 'turbo',
                    asr_models: [
                        { id: 'turbo', label: 'Whisper Turbo — точная (99 яз., 810 МБ)' },
                        { id: 'large-v3', label: 'Whisper Large v3 — макс. точность (99 яз., 3.1 ГБ)' },
                        { id: 'parakeet', label: 'Parakeet v3 — быстрая (25 яз., 600 МБ)' },
                        { id: 'small', label: 'Whisper Small — лёгкая (99 яз., 245 МБ)' },
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

    // Step 1: User selects CPU-only 'Parakeet v3 — быстрая' for fast CPU job
    const parakeetBtn = getMenuButtons(asrWrapper).find((b) => b.textContent.includes('Parakeet'));
    assert.ok(parakeetBtn, 'Parakeet option must exist in ASR dropdown');
    parakeetBtn.click();
    assert.strictEqual(s1.asrGroup.sel.value, 'parakeet');
    assert.strictEqual(s1.devGroup.sel.value, 'cpu');

    // Verify GPU (CUDA) button in menu is STILL enabled (not locked out!) with 'Whisper' hint
    const gpuBtnBefore = getMenuButtons(devWrapper)[1];
    assert.strictEqual(gpuBtnBefore.disabled, false, 'GPU option must NOT be disabled when Parakeet is selected');
    assert.ok(gpuBtnBefore.textContent.includes('Whisper'), 'GPU option should show Whisper auto-switch hint');

    // Step 2: Upload & complete first video job on CPU with Parakeet
    const fileInput = s1.byId.get('fileInput');
    fileInput.files = [{ name: 'interview1.mp4' }];
    fileInput.dispatchEvent(new s1.ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 30));

    assert.strictEqual(s1.uploadedPayloads.length, 1);
    assert.strictEqual(s1.uploadedPayloads[0].asr_model, 'parakeet');
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
    assert.strictEqual(s1.asrGroup.sel.value, 'turbo', 'asrModelSelect must auto-switch from parakeet to turbo');
    assert.strictEqual(s1.localStorage.getItem('selectedComputeDevice'), 'cuda');
    assert.strictEqual(s1.localStorage.getItem('selectedAsrModel'), 'turbo');

    // Step 5b: User can also switch to 'Whisper Large v3' while keeping GPU (CUDA) active
    const largeV3Btn = getMenuButtons(asrWrapper).find((b) => b.textContent.includes('Large v3'));
    assert.ok(largeV3Btn, 'Whisper Large v3 option must exist in ASR dropdown');
    largeV3Btn.click();
    assert.strictEqual(s1.asrGroup.sel.value, 'large-v3');
    assert.strictEqual(s1.devGroup.sel.value, 'cuda', 'GPU must remain active when Whisper Large v3 is selected');

    // Step 6: Upload second video -> verify compute_device='cuda' & asr_model='large-v3' are sent!
    fileInput.files = [{ name: 'interview2.mp4' }];
    fileInput.dispatchEvent(new s1.ctx.Event('change'));
    await new Promise((r) => setTimeout(r, 30));

    assert.strictEqual(s1.uploadedPayloads.length, 2);
    assert.strictEqual(s1.uploadedPayloads[1].compute_device, 'cuda');
    assert.strictEqual(s1.uploadedPayloads[1].asr_model, 'large-v3');

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

