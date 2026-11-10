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

    cfg_resp = client.get("/api/config")
    assert cfg_resp.status_code == 200
    data = cfg_resp.json()
    assert data["version"] == "4.1.0"
    assert data["transcription_default"] is True
    assert data["default_asr_model"] == "turbo"
    assert data["default_compute_device"] == "cpu"
    assert isinstance(data["cuda_available"], bool)
    assert [d["id"] for d in data["compute_devices"]] == ["cpu", "cuda"]
    model_ids = [m["id"] for m in data["asr_models"]]
    assert model_ids == ["turbo", "parakeet", "small"]
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
    import src.transcriber as tr_mod

    in_mp4 = str(tmp_path / "gpu_input.mp4")
    out_mp4 = str(tmp_path / "gpu_output.mp4")
    out_txt = str(tmp_path / "gpu_transcript.txt")
    _generate_test_mp4(in_mp4, duration_sec=1.2, with_audio=True)

    recorded_calls = []

    def _mock_gpu_transcribe(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = "turbo",
        speaker_count: str = "auto",
        threads: int = 1,
        diar_threads=None,
        compute_device: str = "cpu",
        status_cb=None,
    ):
        recorded_calls.append({"compute_device": compute_device, "diar_threads": diar_threads})
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("[00:00:00 - 00:00:01] Спикер 1: GPU OK\n")
        return {"num_speakers": 1, "num_turns": 1, "elapsed": 0.01}

    monkeypatch.setattr(tr_mod, "probe_cuda_available", lambda: True)
    monkeypatch.setattr(Transcriber, "transcribe_audio_file", classmethod(_mock_gpu_transcribe))

    logs = []
    iom = IOManager()
    job_dir = iom.create_job_dir()
    try:
        res = Orchestrator.process_video_job(
            input_path=in_mp4,
            output_path=out_mp4,
            job_dir=job_dir,
            status_cb=logs.append,
            worker_count=4,
            transcribe=True,
            transcript_path=out_txt,
            asr_model="turbo",
            speaker_count="auto",
            compute_device="cuda",
        )
        assert res["video_ready"] is True
        assert res["transcript_ready"] is True
        assert recorded_calls == [{"compute_device": "cuda", "diar_threads": 2}]
        assert any("Параллельная транскрибация на GPU" in m for m in logs)
    finally:
        iom.cleanup_job_dir(job_dir)


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
