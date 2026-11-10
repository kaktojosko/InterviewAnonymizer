import os
import wave

import numpy as np
import pytest

import src.model_manager as mm_mod
import src.transcriber as tr_mod
from src.config import ASR_MODEL_PROFILES, DIARIZATION_MODELS
from src.model_manager import ModelManager
from src.transcriber import (
    DiarizationSegment,
    SpeakerTurn,
    TimedUnit,
    _ModelCache,
    align_units_to_speakers,
    build_speaker_asr_windows,
    clean_hallucinated_text,
    cluster_anchor_embeddings,
    consolidate_minor_speakers,
    format_timestamp,
    format_transcript_txt,
    group_bpe_tokens_to_units,
    is_narrowband_tone,
    is_repetitive_noise_token,
    load_wav_16k_mono,
    resolve_compute_device,
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


def test_consolidate_minor_speakers_absorbs_transient_clusters():
    # Total speech > 20s: Speaker 0 (30s) and Speaker 1 (25s) are real participants;
    # Speaker 99 (0.6s cough) is a transient ghost cluster (<1.5% and <10s) and gets absorbed.
    segs = [
        DiarizationSegment(0.0, 30.0, speaker=0),
        DiarizationSegment(30.2, 30.8, speaker=99),
        DiarizationSegment(31.0, 56.0, speaker=1),
    ]
    cleaned = consolidate_minor_speakers(segs)
    speakers = {s.speaker for s in cleaned}
    assert speakers == {0, 1}
    assert cleaned[0].end == pytest.approx(30.8)


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


def test_build_speaker_asr_windows_splits_substantive_turns_and_packs_backchannels():
    diar = [
        DiarizationSegment(0.1, 4.0, speaker=0),
        DiarizationSegment(4.5, 10.5, speaker=0),  # same speaker <= 1.2s gap -> merged (10.4s >= 6s)
        DiarizationSegment(10.7, 15.8, speaker=1),  # substantive turn (5.1s >= 2.5s) -> kept separate!
    ]
    windows = build_speaker_asr_windows(diar, total_dur=16.0)
    assert len(windows) == 2
    assert windows[0][0] == pytest.approx(0.0)
    assert windows[0][1] == pytest.approx(10.6)
    assert windows[1][0] == pytest.approx(10.6)
    assert windows[1][1] == pytest.approx(16.0)

    # Short <2.5s backchannels ("Угу", "Да") are packed into one window instead of 3x 30s passes
    rapid = [
        DiarizationSegment(1.0, 5.0, speaker=0),
        DiarizationSegment(5.3, 6.2, speaker=1),  # 0.9s backchannel
        DiarizationSegment(6.5, 11.0, speaker=0),
    ]
    packed = build_speaker_asr_windows(rapid, total_dur=12.0)
    assert len(packed) == 1
    assert packed[0] == pytest.approx((0.8, 11.2))


def test_resolve_compute_device_cpu_cuda_and_fallback(monkeypatch):
    # 1. Default CPU mode
    assert resolve_compute_device("cpu") == ("cpu", "int8")

    # 2. CUDA requested when unavailable -> emits clear Russian fallback log and uses CPU int8
    monkeypatch.setattr(tr_mod, "probe_cuda_available", lambda: False)
    logs = []
    dev, ctype = resolve_compute_device("cuda", status_cb=logs.append)
    assert (dev, ctype) == ("cpu", "int8")
    assert any("CUDA недоступна" in m for m in logs)

    # 3. CUDA requested when available -> selects int8_float16
    monkeypatch.setattr(tr_mod, "probe_cuda_available", lambda: True)
    monkeypatch.setattr(
        tr_mod.ctranslate2,
        "get_supported_compute_types",
        lambda device: {"float32", "float16", "int8_float16"},
    )
    assert resolve_compute_device("cuda") == ("cuda", "int8_float16")


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
    assert "# Транскрибация интервью (Interview Anonymizer v4.1.0)" in txt
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


def test_is_narrowband_tone_rejects_ringback_beeps_and_keeps_broadband_speech():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)

    # 1. Standard 425 Hz call-waiting / Zoom / telephony ringback tone -> True
    beep_425 = 0.5 * np.sin(2 * np.pi * 425.0 * t).astype(np.float32)
    assert is_narrowband_tone(beep_425, sample_rate=sr) is True

    # 2. Dual-tone 440 + 480 Hz US ringback tone -> True
    dual_beep = (0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 480.0 * t)).astype(
        np.float32
    )
    assert is_narrowband_tone(dual_beep, sample_rate=sr) is True

    # 3. Rich multi-formant broadband speech-like signal -> False
    rng = np.random.RandomState(7)
    speech_like = rng.normal(0.0, 0.08, sr).astype(np.float32)
    for f0 in (180, 360, 540, 720, 1100, 1650, 2400, 3100):
        speech_like += (0.06 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    assert is_narrowband_tone(speech_like, sample_rate=sr) is False


def test_clean_hallucinated_text_strips_ringback_stutters_and_preserves_real_speech():
    # Exact dial-tone repetition hallucinations observed in real interview logs
    assert clean_hallucinated_text("뚜루뚜뚜뚜뚜뚜뚜뚜뚜뚜뚜") == ""
    assert clean_hallucinated_text("Doototototototototototototototototo...") == ""
    assert clean_hallucinated_text("Дудудудудудудудудудудудуду...") == ""
    assert is_repetitive_noise_token("뚜루뚜뚜뚜뚜뚜뚜뚜뚜뚜뚜") is True
    assert is_repetitive_noise_token("preprocessing") is False

    # Stutter prefix followed by real speech in the same segment
    mixed = (
        "Ту -ру -ту -ту -ту -ту -ту -ту -ту -ту -ту -ту -ту, "
        "Тэфи, ты серьезно? Ты решил сейчас это сделать?"
    )
    assert clean_hallucinated_text(mixed) == "Тэфи, ты серьезно? Ты решил сейчас это сделать?"

    # Natural speech & short interjections must remain completely intact
    assert (
        clean_hallucinated_text("Да, да, да, Дмитрий, добрый день!")
        == "Да, да, да, Дмитрий, добрый день!"
    )
    assert clean_hallucinated_text("Mm -hmm.") == "Mm -hmm."
    assert clean_hallucinated_text("Я же не на то шел.") == "Я же не на то шел."
    assert (
        clean_hallucinated_text("Нет, в целом все понятненько, все хорошо. Okay.")
        == "Нет, в целом все понятненько, все хорошо. Okay."
    )


def test_align_units_drops_noise_turn_without_consuming_speaker_one():
    diar = [
        DiarizationSegment(0.5, 4.0, speaker=99),  # leaked noise segment
        DiarizationSegment(5.0, 9.5, speaker=7),
        DiarizationSegment(10.0, 14.0, speaker=3),
    ]
    units = [
        TimedUnit(0.6, 3.8, " Дудудудудудудудудудуду"),
        TimedUnit(5.2, 9.2, " Добрый день, коллеги!"),
        TimedUnit(10.2, 13.5, " Привет, начинаем интервью."),
    ]
    turns, spk_count = align_units_to_speakers(units, diar)
    assert spk_count == 2
    assert len(turns) == 2
    assert turns[0].speaker_label == "Спикер 1"
    assert turns[0].text == "Добрый день, коллеги!"
    assert turns[1].speaker_label == "Спикер 2"
    assert turns[1].text == "Привет, начинаем интервью."


def test_cluster_anchor_embeddings_prevents_speaker_explosion_on_3_person_interview():
    rng = np.random.RandomState(123)
    dim = 192

    # 3 orthogonal base voiceprints (Дмитрий, Виктор, Вероника)
    qr, _ = np.linalg.qr(rng.randn(dim, 3))
    true_centers = [qr[:, i].astype(np.float32) for i in range(3)]

    segments: list[DiarizationSegment] = []
    embeddings: list[np.ndarray] = []
    expected_speakers: list[int] = []
    cur_t = 0.0

    # Simulate 45 rapid + long turns including noisy <1s interjections ('Mm-hmm', 'Добрый день')
    for turn_idx in range(45):
        true_spk = turn_idx % 3
        is_short = (turn_idx % 4 == 1)
        dur = 0.85 if is_short else 4.2
        noise_scale = 0.095 if is_short else 0.042  # short <1s remarks have much noisier embeddings
        vec = true_centers[true_spk] + noise_scale * rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        segments.append(DiarizationSegment(cur_t, cur_t + dur, speaker=turn_idx))
        embeddings.append(vec)
        expected_speakers.append(true_spk)
        cur_t += dur + 1.2  # >0.8s gap so turns don't trivially time-merge

    # Also inject a 2.2s overlapped-speech turn (50% spk 0 + 50% spk 1)
    ov_vec = 0.5 * true_centers[0] + 0.5 * true_centers[1] + 0.03 * rng.randn(dim).astype(np.float32)
    ov_vec /= np.linalg.norm(ov_vec)
    segments.append(DiarizationSegment(cur_t, cur_t + 2.2, speaker=999))
    embeddings.append(ov_vec)

    clustered = cluster_anchor_embeddings(segments, embeddings, speaker_count="auto")
    unique_spks = {s.speaker for s in clustered}
    assert len(unique_spks) == 3

    # Verify all 45 non-overlap turns mapped consistently to their 3 true speakers
    label_by_true: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    for idx in range(45):
        label_by_true[expected_speakers[idx]].add(clustered[idx].speaker)
    for true_spk in (0, 1, 2):
        assert len(label_by_true[true_spk]) == 1

    # Single-speaker monologue in auto mode must stay 1 speaker
    mono_segs = [DiarizationSegment(i * 5.0, i * 5.0 + 3.5, speaker=i) for i in range(10)]
    mono_embs = [
        (true_centers[0] + 0.04 * rng.randn(dim).astype(np.float32)) for _ in range(10)
    ]
    mono_clustered = cluster_anchor_embeddings(mono_segs, mono_embs, speaker_count="auto")
    assert len({s.speaker for s in mono_clustered}) == 1

