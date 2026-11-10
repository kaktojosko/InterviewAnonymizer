import os
import wave

import numpy as np
import pytest

import src.model_manager as mm_mod
from src.config import ASR_MODEL_PROFILES, DIARIZATION_MODELS
from src.model_manager import ModelManager
from src.transcriber import (
    DiarizationSegment,
    SpeakerTurn,
    TimedUnit,
    _ModelCache,
    align_units_to_speakers,
    build_speaker_asr_windows,
    format_timestamp,
    format_transcript_txt,
    group_bpe_tokens_to_units,
    load_wav_16k_mono,
    smooth_diarization_segments,
    split_audio_ranges_at_silence,
)


def _write_pcm16_wav(path: str, samples_i16: np.ndarray, sample_rate: int = 16000, channels: int = 1):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.astype(np.int16).tobytes())


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00:00"
    assert format_timestamp(-3.2) == "00:00:00"
    assert format_timestamp(9.6) == "00:00:10"
    assert format_timestamp(3661.2) == "01:01:01"


def test_load_wav_16k_mono_stereo_downmix_and_gain_normalization(tmp_path):
    wav_path = str(tmp_path / "quiet_stereo.wav")
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Quiet signal with peak ~0.08 in [-1, 1]
    left = (0.08 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (0.04 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo = np.column_stack([left, right]).reshape(-1)
    _write_pcm16_wav(wav_path, stereo, sample_rate=sr, channels=2)

    samples, out_sr = load_wav_16k_mono(wav_path, normalize_gain=True)
    assert out_sr == sr
    assert samples.dtype == np.float32
    assert samples.shape == (sr,)
    peak = float(np.max(np.abs(samples)))
    # Mean of 0.08 and 0.04 is 0.06; gain capped at 10x -> ~0.60
    assert 0.55 <= peak <= 0.91

    raw_samples, _ = load_wav_16k_mono(wav_path, normalize_gain=False)
    assert float(np.max(np.abs(raw_samples))) < 0.07


def test_load_wav_16k_mono_errors_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wav_16k_mono(str(tmp_path / "missing.wav"))

    empty_wav = str(tmp_path / "empty.wav")
    _write_pcm16_wav(empty_wav, np.zeros(0, dtype=np.int16))
    arr, sr = load_wav_16k_mono(empty_wav)
    assert arr.size == 0
    assert sr == 16000

    wav_8bit = str(tmp_path / "eight_bit.wav")
    with wave.open(wav_8bit, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(16000)
        wf.writeframes(b"\x80" * 100)
    with pytest.raises(ValueError, match="sampwidth=2"):
        load_wav_16k_mono(wav_8bit)


def test_smooth_diarization_segments():
    raw = [
        DiarizationSegment(0.0, 2.0, speaker=0),
        # Short pause <= 0.8s same speaker -> merged
        DiarizationSegment(2.4, 4.0, speaker=0),
        # Micro-glitch < 0.2s from speaker 1 sandwiched between speaker 0 -> absorbed
        DiarizationSegment(4.1, 4.22, speaker=1),
        DiarizationSegment(4.3, 6.5, speaker=0),
        # Real speaker 1 turn
        DiarizationSegment(6.8, 10.2, speaker=1),
    ]
    smoothed = smooth_diarization_segments(raw)
    assert len(smoothed) == 2
    assert smoothed[0].speaker == 0
    assert smoothed[0].start == pytest.approx(0.0)
    assert smoothed[0].end == pytest.approx(6.5)
    assert smoothed[1].speaker == 1
    assert smoothed[1].start == pytest.approx(6.8)
    assert smoothed[1].end == pytest.approx(10.2)


def test_split_audio_ranges_at_silence():
    sr = 16000
    # 40 seconds of synthetic speech with clear 0.5s silence valleys at 15s and 31s
    samples = np.random.RandomState(42).normal(0, 0.2, sr * 40).astype(np.float32)
    samples[int(14.8 * sr) : int(15.4 * sr)] = 0.0
    samples[int(30.8 * sr) : int(31.4 * sr)] = 0.0

    ranges = split_audio_ranges_at_silence(samples, sr, diar_segments=[], max_chunk_sec=18.0)
    assert len(ranges) >= 3
    for s, e in ranges:
        assert 0.0 < (e - s) <= 18.01
    # First split should land inside the 14.8..15.4 silence valley
    assert 14.75 <= ranges[0][1] <= 15.45


def test_build_speaker_asr_windows_never_merges_across_speakers():
    diar = [
        DiarizationSegment(0.1, 4.0, speaker=0),
        DiarizationSegment(4.5, 10.5, speaker=0),  # same speaker <= 1.2s gap -> merged
        DiarizationSegment(10.7, 15.8, speaker=1),  # different speaker only 0.2s away -> kept separate!
    ]
    windows = build_speaker_asr_windows(diar, total_dur=16.0)
    assert len(windows) == 2
    assert windows[0][0] == pytest.approx(0.0)
    # Midpoint between 10.5 and 10.7 is 10.6 -> zero overlap
    assert windows[0][1] == pytest.approx(10.6)
    assert windows[1][0] == pytest.approx(10.6)
    assert windows[1][1] == pytest.approx(16.0)


def test_group_bpe_tokens_to_units_code_switching():
    tokens = ["▁Привет", ",", "▁это", "▁deploy", "ment", "▁pipeline", "."]
    timestamps = [0.2, 0.55, 0.8, 1.1, 1.3, 1.6, 2.0]
    units = group_bpe_tokens_to_units(tokens, timestamps, offset_sec=10.0, chunk_end_sec=15.0)
    assert len(units) == 4
    assert [u.text.strip() for u in units] == ["Привет,", "это", "deployment", "pipeline."]
    assert units[0].start == pytest.approx(10.2)
    assert units[2].text.strip() == "deployment"


def test_align_units_to_speakers_chronological_and_pause_breaks():
    # Raw speaker IDs 7 and 3 -> first heard (7) becomes Спикер 1, second (3) becomes Спикер 2
    diar = [
        DiarizationSegment(0.5, 4.2, speaker=7),
        DiarizationSegment(4.6, 9.0, speaker=3),
        DiarizationSegment(9.5, 16.0, speaker=7),
    ]
    units = [
        TimedUnit(0.6, 1.2, " Привет,"),
        TimedUnit(1.3, 2.5, " расскажи"),
        TimedUnit(2.6, 4.3, " про production."),  # slightly spills 0.1s past seg end
        TimedUnit(4.7, 6.2, " Мы"),
        TimedUnit(6.3, 8.8, " используем Docker."),
        TimedUnit(9.6, 10.8, " Отлично."),
        # Long pause > 2.5s within same speaker 7 -> splits into a new paragraph turn
        TimedUnit(14.0, 15.5, " Идём дальше."),
    ]
    turns, spk_count = align_units_to_speakers(units, diar, pause_break_sec=2.5)
    assert spk_count == 2
    assert len(turns) == 4
    assert turns[0].speaker_label == "Спикер 1"
    assert turns[0].text == "Привет, расскажи про production."
    assert turns[1].speaker_label == "Спикер 2"
    assert turns[1].text == "Мы используем Docker."
    assert turns[2].speaker_label == "Спикер 1"
    assert turns[2].text == "Отлично."
    assert turns[3].speaker_label == "Спикер 1"
    assert turns[3].text == "Идём дальше."


def test_align_units_zero_diarization_fallback():
    units = [
        TimedUnit(0.2, 0.9, " Короткая"),
        TimedUnit(1.0, 1.8, " фраза."),
    ]
    turns, spk_count = align_units_to_speakers(units, diar_segments=[])
    assert spk_count == 1
    assert len(turns) == 1
    assert turns[0].speaker_label == "Спикер 1"
    assert turns[0].text == "Короткая фраза."


def test_format_transcript_txt():
    turns = [
        SpeakerTurn(1.2, 8.9, "Спикер 1", "Привет! Как прошёл code review?"),
        SpeakerTurn(9.6, 15.1, "Спикер 2", "Всё отлично, PR уже в main."),
    ]
    txt = format_transcript_txt(
        turns=turns,
        duration_sec=125.0,
        model_name="Whisper Large v3 Turbo (INT8)",
        num_speakers=2,
    )
    assert "# Транскрибация интервью (Interview Anonymizer v4.0.0)" in txt
    assert "Длительность: 00:02:05 | Модель: Whisper Large v3 Turbo (INT8) | Обнаружено спикеров: 2" in txt
    assert "[00:00:01 - 00:00:09] Спикер 1: Привет! Как прошёл code review?" in txt
    assert "[00:00:10 - 00:00:15] Спикер 2: Всё отлично, PR уже в main." in txt

    empty_txt = format_transcript_txt([], duration_sec=10.0, model_name="Whisper Small (INT8)", num_speakers=0)
    assert "[Речь в аудиодорожке не обнаружена]" in empty_txt


def test_model_cache_evicts_previous_engine():
    created = []

    def make_engine(name):
        obj = {"name": name}
        created.append(name)
        return obj

    e1 = _ModelCache.get_or_create(("test", "a"), lambda: make_engine("a"))
    e1_again = _ModelCache.get_or_create(("test", "a"), lambda: make_engine("a2"))
    assert e1 is e1_again
    assert created == ["a"]

    e2 = _ModelCache.get_or_create(("test", "b"), lambda: make_engine("b"))
    assert e2["name"] == "b"
    assert _ModelCache.active_key == ("test", "b")


def test_model_manager_prefers_valid_preloaded_dir_without_network(tmp_path, monkeypatch):
    preloaded = tmp_path / "preloaded"
    models_dir = tmp_path / "models"
    preloaded.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(mm_mod, "PRELOADED_MODELS_DIR", str(preloaded))
    monkeypatch.setattr(mm_mod, "MODELS_DIR", str(models_dir))

    # Populate valid mock files for diarization + 'small' profile with lowered byte thresholds
    seg_dir = preloaded / "diarization" / "segmentation"
    emb_dir = preloaded / "diarization" / "embedding"
    asr_dir = preloaded / "asr" / "small"
    seg_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)
    asr_dir.mkdir(parents=True)

    seg_file = seg_dir / DIARIZATION_MODELS["segmentation"]["filename"]
    emb_file = emb_dir / DIARIZATION_MODELS["embedding"]["filename"]
    seg_file.write_bytes(b"x" * 128)
    emb_file.write_bytes(b"y" * 256)

    for fname in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        (asr_dir / fname).write_bytes(b"z" * 128)

    monkeypatch.setitem(
        DIARIZATION_MODELS,
        "segmentation",
        {**DIARIZATION_MODELS["segmentation"], "min_bytes": 64},
    )
    monkeypatch.setitem(
        DIARIZATION_MODELS,
        "embedding",
        {**DIARIZATION_MODELS["embedding"], "min_bytes": 64},
    )
    monkeypatch.setitem(
        ASR_MODEL_PROFILES,
        "small",
        {**ASR_MODEL_PROFILES["small"], "min_total_bytes": 256},
    )

    def _forbid_network(*args, **kwargs):
        raise AssertionError("Network download should not be called when preloaded models exist")

    monkeypatch.setattr(mm_mod, "hf_hub_download", _forbid_network)
    monkeypatch.setattr(mm_mod, "snapshot_download", _forbid_network)

    got_seg, got_emb = ModelManager.ensure_diarization_models()
    assert got_seg == str(seg_file)
    assert got_emb == str(emb_file)

    got_asr = ModelManager.ensure_asr_model("small")
    assert got_asr == str(asr_dir)
