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
    Transcriber,
    _ModelCache,
    align_units_to_speakers,
    apply_custom_glossary_terms,
    build_longform_asr_chunks,
    build_speaker_asr_windows,
    build_vad_asr_windows,
    clean_hallucinated_text,
    cluster_anchor_embeddings,
    consolidate_minor_speakers,
    format_timestamp,
    format_transcript_txt,
    group_bpe_tokens_to_units,
    is_narrowband_tone,
    is_repetitive_noise_token,
    is_whisper_prompt_echo,
    load_wav_16k_mono,
    normalize_chunk_level,
    resolve_compute_device,
    smooth_diarization_segments,
    split_audio_ranges_at_silence,
    subdivide_long_asr_windows,
    window_has_multiple_speakers,
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

    # 3. CUDA on Tensor Core GPUs (incl. RTX 50-series Blackwell) -> prefers native float16
    monkeypatch.setattr(tr_mod, "probe_cuda_available", lambda: True)
    monkeypatch.setattr(
        tr_mod.ctranslate2,
        "get_supported_compute_types",
        lambda device: {"float32", "float16", "int8_float16", "int8"},
    )
    assert resolve_compute_device("cuda") == ("cuda", "float16")

    # 4. CUDA on older Pascal GPUs without FP16 Tensor Cores -> selects int8_float32
    monkeypatch.setattr(
        tr_mod.ctranslate2,
        "get_supported_compute_types",
        lambda device: {"float32", "int8_float32", "int8"},
    )
    assert resolve_compute_device("cuda") == ("cuda", "int8_float32")


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
    assert "# Транскрибация интервью (Interview Anonymizer v4.6.0)" in txt

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


def test_build_vad_asr_windows_filters_ringback_and_packs_speech(monkeypatch):
    sr = 16000
    total_sec = 20.0
    samples = np.zeros(int(sr * total_sec), dtype=np.float32)

    # 1..2s: pure 425 Hz dial-tone beep -> must be rejected
    t1 = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    samples[1 * sr : 2 * sr] = 0.5 * np.sin(2 * np.pi * 425.0 * t1)

    # 3..10s (7s) and 11..15s (4s): broadband speech-like spans separated by 1.0s (>=0.6s) -> 2 windows
    rng = np.random.RandomState(99)
    for s_sec, e_sec in ((3, 10), (11, 15)):
        n = (e_sec - s_sec) * sr
        tt = np.linspace(0, float(e_sec - s_sec), n, endpoint=False, dtype=np.float32)
        chunk = rng.normal(0.0, 0.08, n).astype(np.float32)
        for f0 in (200, 420, 880, 1550, 2600):
            chunk += (0.06 * np.sin(2 * np.pi * f0 * tt)).astype(np.float32)
        samples[s_sec * sr : e_sec * sr] = chunk

    monkeypatch.setattr(
        tr_mod,
        "get_speech_timestamps",
        lambda audio, vad_options=None, sampling_rate=16000, **kw: [
            {"start": 1 * sr, "end": 2 * sr},
            {"start": 3 * sr, "end": 10 * sr},
            {"start": 11 * sr, "end": 15 * sr},
        ],
    )

    windows = build_vad_asr_windows(samples, sample_rate=sr)
    assert len(windows) == 2
    assert windows[0] == pytest.approx((2.85, 10.15))
    assert windows[1] == pytest.approx((10.85, 15.15))


def test_run_faster_whisper_retries_cuda_float32_before_cpu_fallback(monkeypatch):
    monkeypatch.setattr(tr_mod, "resolve_compute_device", lambda dev, status_cb=None: ("cuda", "float16"))

    attempted = []

    def _mock_decode_fp32_succeeds(**kwargs):
        attempted.append((kwargs["device"], kwargs["compute_type"]))
        if kwargs["device"] == "cuda" and kwargs["compute_type"] == "float16":
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")
        return [TimedUnit(0.0, 1.0, " GPU-FP32")]

    monkeypatch.setattr(Transcriber, "_decode_whisper_windows", staticmethod(_mock_decode_fp32_succeeds))
    logs = []
    out = Transcriber._run_faster_whisper(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        model_dir="/tmp/mock",
        profile_id="turbo",
        compute_device="cuda",
        status_cb=logs.append,
    )
    assert [u.text for u in out] == [" GPU-FP32"]
    assert attempted == [("cuda", "float16"), ("cuda", "float32")]
    assert any("float32" in m for m in logs)

    # If CUDA float32 ALSO fails, it safely falls back to CPU int8
    attempted.clear()

    def _mock_decode_all_cuda_fail(**kwargs):
        attempted.append((kwargs["device"], kwargs["compute_type"]))
        if kwargs["device"] == "cuda":
            raise RuntimeError("CUDA driver error")
        return [TimedUnit(0.0, 1.0, " CPU-INT8")]

    monkeypatch.setattr(Transcriber, "_decode_whisper_windows", staticmethod(_mock_decode_all_cuda_fail))
    out_cpu = Transcriber._run_faster_whisper(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        model_dir="/tmp/mock",
        profile_id="turbo",
        compute_device="cuda",
    )
    assert [u.text for u in out_cpu] == [" CPU-INT8"]
    assert attempted == [("cuda", "float16"), ("cuda", "float32"), ("cpu", "int8")]


def test_decoupled_gpu_transcribe_units_and_diarize(tmp_path, monkeypatch):
    wav_path = str(tmp_path / "sample.wav")
    out_txt = str(tmp_path / "out.txt")
    sr = 16000
    tone = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, sr * 2, endpoint=False)) * 32767).astype(
        np.int16
    )
    _write_pcm16_wav(wav_path, tone, sample_rate=sr)

    monkeypatch.setattr(ModelManager, "ensure_asr_model", classmethod(lambda cls, pid, status_cb=None: "/tmp/asr"))
    monkeypatch.setattr(
        ModelManager,
        "ensure_diarization_models",
        classmethod(lambda cls, status_cb=None: ("/tmp/seg.onnx", "/tmp/emb.onnx")),
    )
    monkeypatch.setattr(tr_mod, "resolve_compute_device", lambda dev, status_cb=None: ("cuda", "float16"))
    monkeypatch.setattr(
        Transcriber,
        "_run_faster_whisper",
        classmethod(
            lambda cls, **kw: [
                TimedUnit(0.1, 0.9, " Привет!"),
                TimedUnit(1.1, 1.9, " Hello!"),
            ]
        ),
    )
    monkeypatch.setattr(
        Transcriber,
        "_run_diarization",
        classmethod(
            lambda cls, **kw: [
                DiarizationSegment(0.0, 1.0, speaker=0),
                DiarizationSegment(1.05, 2.0, speaker=1),
            ]
        ),
    )

    state = Transcriber.transcribe_units_only(wav_path, asr_model="turbo", compute_device="cuda")
    assert len(state["units"]) == 2

    res = Transcriber.diarize_and_write_transcript(
        asr_state=state,
        output_txt_path=out_txt,
        speaker_count="auto",
        threads=4,
    )
    assert res["num_speakers"] == 2
    assert res["num_turns"] == 2
    txt = open(out_txt, encoding="utf-8").read()
    assert "Спикер 1: Привет!" in txt
    assert "Спикер 2: Hello!" in txt


def test_cancellation_token_terminates_subprocess_and_transcriber(tmp_path):
    import subprocess
    import sys
    import time
    from src.cancellation import CancellationToken, TaskCancelledError

    token = CancellationToken()
    cb_called = []
    token.register_cancel_callback(lambda: cb_called.append(True))

    proc1 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    token.register_proc(proc1)

    t0 = time.time()
    assert token.cancel() is True
    assert token.cancel() is False  # idempotent
    assert token.is_cancelled() is True
    assert cb_called == [True]

    proc1.wait(timeout=2.0)
    assert (time.time() - t0) < 2.0
    assert proc1.poll() is not None

    # Registering a new subprocess onto an already-cancelled token immediately kills it & raises
    proc2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(TaskCancelledError):
        token.register_proc(proc2)
    proc2.wait(timeout=2.0)
    assert proc2.poll() is not None

    # Transcriber immediately raises TaskCancelledError when cancel_token is set
    wav_path = str(tmp_path / "cancel.wav")
    _write_pcm16_wav(wav_path, np.zeros(16000, dtype=np.int16))
    with pytest.raises(TaskCancelledError):
        Transcriber.transcribe_audio_file(
            wav_path=wav_path,
            output_txt_path=str(tmp_path / "cancel.txt"),
            cancel_token=token,
        )





def test_cpu_dense_window_packing_and_monologue_subdivision():
    diar = [
        DiarizationSegment(0.5, 9.0, speaker=0),
        DiarizationSegment(10.0, 18.0, speaker=1),
        DiarizationSegment(19.0, 26.0, speaker=0),
    ]
    # Default mode splits substantive speaker turns -> 3 windows
    split_wins = build_speaker_asr_windows(diar, total_dur=27.0, split_speaker_turns=True)
    assert len(split_wins) == 3

    # Dense CPU mode packs all 3 turns into a single <=28.5s window (3x fewer Encoder passes!)
    dense_wins = build_speaker_asr_windows(
        diar, total_dur=27.0, max_same_speaker_sec=28.5, merge_gap_sec=3.5, split_speaker_turns=False
    )
    assert len(dense_wins) == 1
    assert dense_wins[0] == pytest.approx((0.3, 26.2))
    assert window_has_multiple_speakers(dense_wins[0][0], dense_wins[0][1], diar) is True

    # Single-speaker monologue window -> window_has_multiple_speakers is False (skips DTW word_timestamps)
    mono_diar = [DiarizationSegment(1.0, 62.0, speaker=0)]
    sr = 16000
    samples = np.random.RandomState(11).normal(0.0, 0.15, int(65.0 * sr)).astype(np.float32)
    # Insert quiet breath pauses at 26.5s and 53.0s
    samples[int(26.3 * sr) : int(26.8 * sr)] = 0.0
    samples[int(52.8 * sr) : int(53.3 * sr)] = 0.0

    raw_mono_wins = build_speaker_asr_windows(
        mono_diar, total_dur=65.0, max_same_speaker_sec=28.5, split_speaker_turns=False
    )
    sub_wins = subdivide_long_asr_windows(raw_mono_wins, samples, sample_rate=sr, max_window_sec=29.0)
    assert len(sub_wins) == 3
    for ws, we in sub_wins:
        assert (we - ws) <= 29.01
        assert window_has_multiple_speakers(ws, we, mono_diar) is False


def test_decode_whisper_windows_cpu_optimizations_and_parallel_workers(monkeypatch):
    import time
    from types import SimpleNamespace

    created_init_kwargs = {}
    transcribe_calls = []

    class _FakeWhisperModel:
        def __init__(self, model_dir, **kwargs):
            created_init_kwargs.update(kwargs)

        def transcribe(self, audio_slice, **kwargs):
            # Sleep slightly longer on the 9.4s first window so worker 2 finishes first,
            # proving deterministic chronological reassembly of results!
            dur_s = len(audio_slice) / 16000.0
            transcribe_calls.append(dict(kwargs))
            if dur_s > 9.35:
                time.sleep(0.04)
                seg = SimpleNamespace(
                    start=0.1,
                    end=min(2.0, dur_s),
                    text=" Монолог первого спикера.",
                    words=[
                        SimpleNamespace(start=0.1, end=1.9, word=" Монолог первого спикера."),
                    ],
                    no_speech_prob=0.05,
                    avg_logprob=-0.2,
                )
            else:
                seg = SimpleNamespace(
                    start=0.1,
                    end=min(3.0, dur_s),
                    text=" Вопрос? Ответ!",
                    words=[
                        SimpleNamespace(start=0.1, end=1.2, word=" Вопрос?"),
                        SimpleNamespace(start=1.6, end=2.8, word=" Ответ!"),
                    ],
                    no_speech_prob=0.05,
                    avg_logprob=-0.2,
                )
            return [seg], SimpleNamespace(language="ru")

    monkeypatch.setattr(tr_mod, "WhisperModel", _FakeWhisperModel)
    _ModelCache.active_key = None
    _ModelCache.active_engine = None

    sr = 16000
    samples = np.ones(sr * 40, dtype=np.float32) * 0.05
    # Window 1 (0.3..9.7s = 9.4s): speaker 0
    # Gap of 6.5s (>3.0s merge_gap_sec) forces second window:
    # Window 2 (15.8..25.0s = 9.2s): speakers 0 and 1
    diar = [
        DiarizationSegment(0.5, 9.5, speaker=0),
        DiarizationSegment(16.0, 20.0, speaker=0),
        DiarizationSegment(20.5, 24.8, speaker=1),
    ]

    units = Transcriber._decode_whisper_windows(
        samples=samples,
        sample_rate=sr,
        model_dir="/tmp/fake_whisper_cpu",
        profile_id="turbo",
        device="cpu",
        compute_type="int8",
        diar_segments=diar,
        threads=8,
        status_cb=None,
    )

    assert created_init_kwargs["num_workers"] == 2
    assert created_init_kwargs["cpu_threads"] == 4
    assert len(transcribe_calls) == 2
    for call_kw in transcribe_calls:
        assert call_kw["language"] == "ru"
        assert call_kw["multilingual"] is False
        assert call_kw["word_timestamps"] is True
        assert call_kw["beam_size"] == 2
        assert call_kw["temperature"] == (0.0, 0.2, 0.4)
        assert call_kw["compression_ratio_threshold"] == pytest.approx(2.4)
        assert call_kw["log_prob_threshold"] == pytest.approx(-1.0)
        assert call_kw["repetition_penalty"] == pytest.approx(1.08)
        assert call_kw["hallucination_silence_threshold"] == pytest.approx(1.5)
        assert "-" not in call_kw["prepend_punctuations"]
        # Hardcoded SQL/ClickHouse prompt must never be injected!
        assert call_kw["initial_prompt"] is None

    # Chronological order preserved despite window 1 sleeping longer than window 2!
    assert [u.text.strip() for u in units] == ["Монолог первого спикера.", "Вопрос?", "Ответ!"]
    assert units[0].start < units[1].start < units[2].start


def test_run_sherpa_parakeet_batched_decode_streams(monkeypatch):
    from types import SimpleNamespace

    batch_sizes = []

    class _FakeStream:
        def __init__(self):
            self.result = None

        def accept_waveform(self, sr, wav):
            self.dur = len(wav) / float(sr)

    class _FakeParakeetRecognizer:
        def create_stream(self):
            return _FakeStream()

        def decode_streams(self, streams):
            batch_sizes.append(len(streams))
            for idx, s in enumerate(streams):
                s.result = SimpleNamespace(
                    tokens=["▁Тест", f"▁{idx}"],
                    timestamps=[0.1, 0.5],
                    text=f"Тест {idx}",
                )

        def decode_stream(self, stream):
            batch_sizes.append(1)
            stream.result = SimpleNamespace(
                tokens=["▁Один"],
                timestamps=[0.1],
                text="Один",
            )

    _ModelCache.active_key = ("sherpa_parakeet", "/tmp/fake_parakeet", 4)
    _ModelCache.active_engine = _FakeParakeetRecognizer()

    sr = 16000
    samples = np.ones(sr * 30, dtype=np.float32) * 0.05
    # 5 separate speech islands (>0.4s apart, >18s spans prevented by explicit gaps)
    diar = [DiarizationSegment(i * 5.0 + 0.5, i * 5.0 + 3.0, speaker=i % 2) for i in range(5)]

    units = Transcriber._run_sherpa_parakeet(
        samples=samples,
        sample_rate=sr,
        model_dir="/tmp/fake_parakeet",
        diar_segments=diar,
        threads=4,
    )
    # 5 chunks -> batch of 4 via decode_streams + remaining 1 via decode_stream
    assert batch_sizes == [4, 1]
    assert len(units) == 9
    assert units[0].start < units[-1].start


def test_compound_hyphen_orthography_and_spaced_dash_preservation():
    from src.transcriber import _join_unit_texts

    # Exact Whisper hyphen artifacts from Russian + IT interviews
    assert _join_unit_texts([" что", " -то", " подобное", " по -моему", " где -то"]) == (
        "что-то подобное по-моему где-то"
    )
    assert _join_unit_texts([" ключи", " -кандидаты,", " а", " что", " такое", " ключ", "-кандидат?"]) == (
        "ключи-кандидаты, а что такое ключ-кандидат?"
    )
    assert _join_unit_texts([" 28 -й", " кадр", " кто- то", " по- разному"]) == (
        "28-й кадр кто-то по-разному"
    )
    # Legitimate spaced dashes and negative numbers must remain untouched
    assert _join_unit_texts([" Это базовые - есть ещё расширенные, шаг -2."]) == (
        "Это базовые - есть ещё расширенные, шаг -2."
    )


def test_clause_alignment_prevents_mid_sentence_split_on_overlapping_turn():
    """
    Reproduces real interview bug [00:13:53 - 00:13:58]:
      Спикер 1 finishes 'Есть естественные ключи, есть суррогатные ключи.' at 55.65s,
      while Спикер 2 starts overlapping at 55.68s with 'А что такое ключ-кандидат? Это прям интересно.'
      and coarse diarization only switches from speaker 0 to speaker 1 at 56.05s.
    'А что' MUST NOT be glued onto Спикер 1's turn!
    """
    diar = [
        DiarizationSegment(53.00, 56.05, speaker=0),
        DiarizationSegment(56.05, 58.60, speaker=1),
    ]
    units = [
        TimedUnit(53.10, 53.45, " Есть"),
        TimedUnit(53.46, 54.10, " естественные"),
        TimedUnit(54.12, 54.50, " ключи,"),
        TimedUnit(54.55, 54.82, " есть"),
        TimedUnit(54.84, 55.30, " суррогатные"),
        TimedUnit(55.31, 55.65, " ключи."),
        # Speaker 2 jumps in while Speaker 1's mic decays (before diar boundary 56.05s)
        TimedUnit(55.68, 55.82, " А"),
        TimedUnit(55.83, 56.02, " что"),
        TimedUnit(56.04, 56.38, " такое"),
        TimedUnit(56.40, 57.22, " ключ -кандидат?"),
        TimedUnit(57.32, 57.52, " Это"),
        TimedUnit(57.53, 57.80, " прям"),
        TimedUnit(57.82, 58.40, " интересно."),
    ]
    turns, spk_count = align_units_to_speakers(units, diar)
    assert spk_count == 2
    assert len(turns) == 2
    assert turns[0].speaker_label == "Спикер 1"
    assert turns[0].text == "Есть естественные ключи, есть суррогатные ключи."
    assert turns[1].speaker_label == "Спикер 2"
    assert turns[1].text == "А что такое ключ-кандидат? Это прям интересно."


def test_unpunctuated_overlap_keeps_trailing_verb_and_splits_at_discourse_starter():
    """
    Reproduces real interview bug [00:13:22 - 00:13:36]:
      Even if ASR omits punctuation across a simultaneous transition ('...сложно
      джойном отнести хорошо так ладно раз мы залезли...'), trailing verb 'отнести'
      stays with Спикер 1 and Спикер 2's turn opens at discourse marker 'хорошо'.
    """
    diar = [
        DiarizationSegment(22.0, 24.95, speaker=0),
        DiarizationSegment(24.95, 31.0, speaker=1),
    ]
    words_with_spans = [
        (22.10, 22.35, " так"),
        (22.36, 22.55, " не"),
        (22.56, 22.95, " скажу"),
        (23.00, 23.18, " но"),
        (23.20, 23.45, " ещё"),
        (23.46, 23.90, " юнионы"),
        (23.92, 24.18, " есть"),
        (24.22, 24.38, " но"),
        (24.40, 24.60, " это"),
        (24.62, 24.92, " сложно"),
        (24.93, 25.12, " джойном"),
        (25.13, 25.42, " отнести"),
        # Speaker 2 takes the floor with 'хорошо так ладно...'
        (25.48, 25.88, " хорошо"),
        (25.90, 26.15, " так"),
        (26.18, 26.55, " ладно"),
        (26.58, 26.85, " раз"),
        (26.86, 27.10, " мы"),
        (27.12, 27.65, " залезли"),
        (27.66, 27.80, " в"),
        (27.82, 28.30, " базу"),
        (28.32, 28.90, " данных"),
    ]
    units = [TimedUnit(s, e, w) for s, e, w in words_with_spans]
    turns, spk_count = align_units_to_speakers(units, diar)
    assert spk_count == 2
    assert len(turns) == 2
    assert turns[0].speaker_label == "Спикер 1"
    assert turns[0].text.endswith("сложно джойном отнести")
    assert turns[1].speaker_label == "Спикер 2"
    assert turns[1].text.startswith("хорошо так ладно раз мы залезли")


def test_rapid_ping_pong_qa_voiceprint_verification_and_unbiased_clustering():
    """
    Reproduces real interview bugs [00:12:18 - 00:12:21] and [00:12:39 - 00:12:51]:
      1. Short <1s replies ('Ну да.', 'Все.') 0.1s after a question are no longer
         swallowed into the questioner by prev_spk bias in cluster_anchor_embeddings.
      2. Even when coarse pyannote diarization fuses a fast 5-turn Q&A exchange into
         a single speaker segment, CAM++ clause-level voiceprint verification cleanly
         separates every question and answer while keeping rhetorical self-questions intact.
    """
    dim = 64
    rng = np.random.RandomState(2026)
    qr, _ = np.linalg.qr(rng.randn(dim, 2))
    v_spk1 = qr[:, 0].astype(np.float32)  # Candidate (Спикер 1)
    v_spk2 = qr[:, 1].astype(np.float32)  # Interviewer (Спикер 2)

    # Part A: cluster_anchor_embeddings on a close-margin short reply ('Ну да.', 0.6s)
    # where v_reply is slightly closer to v_spk1 by 0.025 (previously +0.04 prev_spk bias flipped it!)
    close_reply = (0.52 * v_spk1 + 0.48 * v_spk2)
    close_reply /= np.linalg.norm(close_reply)
    segs = [
        DiarizationSegment(0.0, 4.0, speaker=0),    # Anchor Spk 1
        DiarizationSegment(5.0, 9.0, speaker=1),    # Anchor Spk 2 question
        DiarizationSegment(9.15, 9.75, speaker=2),  # 0.6s rapid reply 0.15s after Spk 2!
    ]
    embs = [v_spk1, v_spk2, close_reply]
    clustered, cents = cluster_anchor_embeddings(segs, embs, speaker_count="auto", return_centroids=True)
    assert len(clustered) == 3
    assert clustered[0].speaker != clustered[1].speaker
    assert clustered[2].speaker == clustered[0].speaker  # Short reply stays with Candidate!
    assert set(cents.keys()) == {clustered[0].speaker, clustered[1].speaker}

    # Part B: Full 5-turn rapid Q&A fused into one diarization segment + rhetorical self-question
    sr = 16000
    samples = (0.05 * np.sin(2 * np.pi * 220.0 * np.linspace(0, 30.0, sr * 30, endpoint=False))).astype(
        np.float32
    )
    # Speaker 0 speaks first at 1..3s so Speaker 0 = 'Спикер 1', Speaker 1 = 'Спикер 2'.
    # Then 4..16s is fused by coarse diarization into a single Speaker 1 block,
    # followed by 18..24s Speaker 0 rhetorical self-question + self-answer.
    fused_diar = [
        DiarizationSegment(1.0, 3.0, speaker=0),
        DiarizationSegment(4.0, 16.0, speaker=1),
        DiarizationSegment(18.0, 24.0, speaker=0),
    ]
    qa_units = [
        TimedUnit(1.1, 2.8, " Да, у нас все с курсором."),
        TimedUnit(4.1, 6.0, " Так, чего я хотел спросить."),
        TimedUnit(6.1, 8.8, " Так, SQL ты знаешь, да, насколько понимаю?"),
        TimedUnit(8.95, 10.45, " Ну да, есть немного."),
        TimedUnit(10.55, 11.25, " Немного?"),
        TimedUnit(11.35, 13.30, " Ну, джойны, подзапросы."),
        TimedUnit(13.45, 15.60, " Какие джойны знаешь?"),
        # Rhetorical self-question by Speaker 0 (both clauses have v_spk1 voiceprint!)
        TimedUnit(18.2, 20.5, " Какие ещё джойны у нас есть?"),
        TimedUnit(20.7, 23.2, " Так сразу не скажу."),
    ]

    def _mock_clause_embed(_wav: np.ndarray, _sr: int, t_s: float, t_e: float):
        mid = 0.5 * (t_s + t_e)
        # Candidate answers at ~9.7s and ~12.3s, plus Candidate monologue at 1..3s and 18..24s
        if (8.9 <= mid <= 10.5) or (11.3 <= mid <= 13.35) or mid < 3.5 or mid > 17.5:
            return v_spk1
        return v_spk2

    turns, spk_count = align_units_to_speakers(
        qa_units,
        fused_diar,
        samples=samples,
        sample_rate=sr,
        speaker_centroids={0: v_spk1, 1: v_spk2},
        clause_embed_fn=_mock_clause_embed,
    )
    assert spk_count == 2
    assert [(t.speaker_label, t.text) for t in turns] == [
        ("Спикер 1", "Да, у нас все с курсором."),
        ("Спикер 2", "Так, чего я хотел спросить. Так, SQL ты знаешь, да, насколько понимаю?"),
        ("Спикер 1", "Ну да, есть немного."),
        ("Спикер 2", "Немного?"),
        ("Спикер 1", "Ну, джойны, подзапросы."),
        ("Спикер 2", "Какие джойны знаешь?"),
        ("Спикер 1", "Какие ещё джойны у нас есть? Так сразу не скажу."),
    ]


def test_sequential_whisper_rolling_context_and_large_v3_single_worker(monkeypatch):
    from types import SimpleNamespace

    init_kwargs = {}
    prompts_seen = []
    calls_seen = []

    class _FakeLargeV3Model:
        def __init__(self, model_dir, **kwargs):
            init_kwargs.update(kwargs)

        def detect_language(self, audio=None, **kwargs):
            return "ru", 0.96, [("ru", 0.96)]

        def transcribe(self, audio_slice, **kwargs):
            prompts_seen.append(kwargs.get("initial_prompt"))
            calls_seen.append(dict(kwargs))
            idx = len(prompts_seen)
            txt = (
                " Мы обсуждали организацию проекта и сроки."
                if idx == 1
                else " А ещё важно согласовать бюджет на квартал."
            )
            seg = SimpleNamespace(
                start=0.1,
                end=2.0,
                text=txt,
                words=[SimpleNamespace(start=0.1, end=2.0, word=txt)],
                no_speech_prob=0.02,
                avg_logprob=-0.15,
            )
            return [seg], SimpleNamespace(language="ru")

    monkeypatch.setattr(tr_mod, "WhisperModel", _FakeLargeV3Model)
    _ModelCache.active_key = None
    _ModelCache.active_engine = None

    sr = 16000
    samples = np.ones(sr * 65, dtype=np.float32) * 0.04
    # Two adjacent ~26s windows separated by <1s gap, followed by a window after a 6s silence
    diar = [
        DiarizationSegment(0.5, 26.5, speaker=0),
        DiarizationSegment(27.0, 52.0, speaker=1),
        DiarizationSegment(59.0, 63.0, speaker=0),
    ]

    Transcriber._decode_whisper_windows(
        samples=samples,
        sample_rate=sr,
        model_dir="/tmp/fake_large_v3",
        profile_id="large-v3",
        device="cpu",
        compute_type="int8",
        diar_segments=diar,
        threads=12,
        status_cb=None,
        language="ru",
    )

    # large-v3 caps num_workers=1 on CPU to preserve RAM bandwidth & enable rolling context
    assert init_kwargs["num_workers"] == 1
    assert init_kwargs["cpu_threads"] == 12
    assert len(prompts_seen) == 3
    # Window 1 gets NO hardcoded SQL/IT prompt; Window 2 (<2.5s gap) gets Window 1 clean tail;
    # Window 3 (>2.5s silence gap) resets to None!
    assert prompts_seen[0] is None
    assert "Мы обсуждали организацию проекта и сроки." in (prompts_seen[1] or "")
    assert prompts_seen[2] is None
    # Explicit Russian language locks multilingual=False and uses stronger CPU beam_size=3
    assert all(c["language"] == "ru" and c["multilingual"] is False for c in calls_seen)
    assert all(c["beam_size"] == 3 for c in calls_seen)


def test_gigaam_v3_transducer_engine_feature_dim_64_and_bpe_tokens(monkeypatch):
    from types import SimpleNamespace

    from_transducer_kwargs = {}

    class _FakeStream:
        def __init__(self):
            self.result = None

        def accept_waveform(self, sr, wav):
            self.sr = sr
            self.wav = wav

    class _FakeGigaAMRecognizer:
        def create_stream(self):
            return _FakeStream()

        def decode_stream(self, stream):
            # Mimics real GigaAM v3 e2e RNN-T BPE output with space-prefixed Cyrillic tokens + punctuation
            stream.result = SimpleNamespace(
                tokens=[" При", "вет", ",", " ", "как", " де", "ла", "?"],
                timestamps=[0.10, 0.25, 0.40, 0.50, 0.60, 0.80, 0.95, 1.10],
                text="Привет, как дела?",
            )

        def decode_streams(self, streams):
            for s in streams:
                self.decode_stream(s)

    def _mock_from_transducer(**kwargs):
        from_transducer_kwargs.update(kwargs)
        return _FakeGigaAMRecognizer()

    monkeypatch.setattr(tr_mod.sherpa_onnx.OfflineRecognizer, "from_transducer", staticmethod(_mock_from_transducer))
    _ModelCache.active_key = None
    _ModelCache.active_engine = None

    sr = 16000
    samples = np.ones(sr * 5, dtype=np.float32) * 0.05
    diar = [DiarizationSegment(0.2, 2.5, speaker=0)]

    units = Transcriber._run_sherpa_transducer(
        samples=samples,
        sample_rate=sr,
        model_dir="/tmp/fake_gigaam_v3",
        diar_segments=diar,
        profile_id="gigaam-v3",
        threads=4,
    )

    assert from_transducer_kwargs["feature_dim"] == 64
    assert from_transducer_kwargs["model_type"] == "nemo_transducer"
    assert from_transducer_kwargs["encoder"].endswith("encoder.int8.onnx")
    assert from_transducer_kwargs["decoder"].endswith("decoder.onnx")
    assert from_transducer_kwargs["joiner"].endswith("joiner.onnx")
    assert from_transducer_kwargs["tokens"].endswith("tokens.txt")

    turns, spk_count = align_units_to_speakers(units, diar)
    assert spk_count == 1
    assert len(turns) == 1
    assert turns[0].text == "Привет, как дела?"


def test_whisper_language_resolution_and_russian_phantom_filter():
    from src.transcriber import _resolve_whisper_language, is_whisper_phantom_hallucination

    sr = 16000
    samples = np.ones(sr * 30, dtype=np.float32) * 0.05
    wins = [(0.0, 5.0, 0, 5 * sr), (6.0, 11.0, 6 * sr, 11 * sr), (12.0, 17.0, 12 * sr, 17 * sr)]

    class _UnanimousRuModel:
        def detect_language(self, audio=None):
            return "ru", 0.94, [("ru", 0.94)]

    class _MixedLangModel:
        def __init__(self):
            self.n = 0

        def detect_language(self, audio=None):
            self.n += 1
            return ("ru", 0.90, []) if self.n % 2 == 1 else ("en", 0.88, [])

    # 1. Explicit 'ru' always locks language='ru', multilingual=False
    assert _resolve_whisper_language(_MixedLangModel(), wins, samples, "ru", "large-v3") == ("ru", False)
    # 2. 'large-v3-ru' defaults to locked Russian even if 'auto' is passed
    assert _resolve_whisper_language(_MixedLangModel(), wins, samples, "auto", "large-v3-ru") == ("ru", False)
    # 3. 'auto' with unanimous high-confidence probes locks language to avoid per-window flips
    assert _resolve_whisper_language(_UnanimousRuModel(), wins, samples, "auto", "large-v3") == ("ru", False)
    # 4. 'auto' with genuinely mixed languages enables multilingual=True
    assert _resolve_whisper_language(_MixedLangModel(), wins, samples, "auto", "large-v3") == ("ru", True)

    # Phantom Russian subtitle hallucinations are dropped while normal speech stays intact
    assert is_whisper_phantom_hallucination("Продолжение следует...", no_speech_prob=0.4, avg_logprob=-0.6) is True
    assert is_whisper_phantom_hallucination("Редактор субтитров А.Семкин", no_speech_prob=0.05, avg_logprob=-0.2) is True
    assert is_whisper_phantom_hallucination("Субтитры сделал DimaTorzok") is True
    assert is_whisper_phantom_hallucination("Спасибо за просмотр!", no_speech_prob=0.35, avg_logprob=-0.5) is True
    assert is_whisper_phantom_hallucination("Здравствуйте, расскажите о вашем проекте.", no_speech_prob=0.05, avg_logprob=-0.2) is False


def test_load_wav_16k_mono_click_resistant_normalization(tmp_path):
    wav_path = str(tmp_path / "quiet_with_click.wav")
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.04 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    # Inject an isolated full-scale mic click that previously disabled peak normalization!
    sig[100] = 32000
    sig[101] = -32000
    _write_pcm16_wav(wav_path, sig, sample_rate=sr, channels=1)

    samples, _ = load_wav_16k_mono(wav_path, normalize_gain=True)
    # Quiet speech body (excluding click) must still be boosted ~10x (0.04 -> ~0.40)
    speech_p95 = float(np.percentile(np.abs(samples[500:]), 95))
    assert speech_p95 >= 0.35


def test_build_longform_asr_chunks_enforces_20s_cap_with_overlap_and_dedup(monkeypatch):
    from types import SimpleNamespace

    sr = 16000
    rng = np.random.RandomState(42)

    # 1. Per-chunk quiet speech leveling boosts quiet remote speakers while keeping loud audio intact
    quiet = rng.normal(0.0, 0.015, sr).astype(np.float32)
    leveled = normalize_chunk_level(quiet)
    assert float(np.percentile(np.abs(leveled), 99.0)) >= 0.30
    loud = rng.normal(0.0, 0.25, sr).astype(np.float32)
    assert np.allclose(normalize_chunk_level(loud), loud)

    # 2. 75-second breathless monologue (no pauses): MUST split <=20.0s with 1.5s overlap
    #    and exact non-overlapping [keep_start, keep_end] ownership cores covering [0, 75]
    samples_75 = rng.normal(0.0, 0.12, sr * 75).astype(np.float32)
    diar_75 = [DiarizationSegment(0.0, 75.0, speaker=0)]
    monkeypatch.setattr(
        tr_mod,
        "get_speech_timestamps",
        lambda audio, vad_options=None, sampling_rate=16000, **kw: [{"start": 0, "end": len(audio)}],
    )
    chunks = build_longform_asr_chunks(
        samples=samples_75,
        sample_rate=sr,
        diar_segments=diar_75,
        max_chunk_sec=20.0,
        overlap_sec=1.5,
        use_silero_vad=True,
    )
    assert len(chunks) >= 4
    assert chunks[0][2] == pytest.approx(0.0, abs=1e-3)
    assert chunks[-1][3] == pytest.approx(75.0, abs=1e-3)
    for idx, (c_s, c_e, k_s, k_e) in enumerate(chunks):
        assert (c_e - c_s) <= 20.01
        assert c_s <= k_s < k_e <= c_e
        if idx > 0:
            prev_s, prev_e, prev_ks, prev_ke = chunks[idx - 1]
            # Sliding overlap ~1.5s and contiguous non-overlapping ownership cores
            assert prev_e - c_s == pytest.approx(1.5, abs=0.05)
            assert k_s == pytest.approx(prev_ke, abs=1e-4)

    # 3. Genuine silent breath pause at 16.5s -> splits cleanly right at the pause valley
    samples_pause = rng.normal(0.0, 0.12, sr * 32).astype(np.float32)
    samples_pause[int(16.2 * sr) : int(16.8 * sr)] = 0.0
    pause_chunks = build_longform_asr_chunks(
        samples=samples_pause,
        sample_rate=sr,
        diar_segments=[DiarizationSegment(0.0, 32.0, speaker=0)],
        max_chunk_sec=20.0,
        overlap_sec=1.5,
        use_silero_vad=False,
    )
    assert len(pause_chunks) == 2
    assert 16.0 <= pause_chunks[0][1] <= 17.0
    assert pause_chunks[1][0] == pytest.approx(pause_chunks[0][1], abs=1e-4)

    # 4. Silero VAD rescues quiet speech missed completely by diarization!
    samples_rescue = rng.normal(0.0, 0.10, sr * 25).astype(np.float32)
    monkeypatch.setattr(
        tr_mod,
        "get_speech_timestamps",
        lambda audio, vad_options=None, sampling_rate=16000, **kw: [
            {"start": 1 * sr, "end": 4 * sr},
            {"start": 12 * sr, "end": 18 * sr},
        ],
    )
    rescued_chunks = build_longform_asr_chunks(
        samples=samples_rescue,
        sample_rate=sr,
        diar_segments=[DiarizationSegment(1.0, 4.0, speaker=0)],
        max_chunk_sec=20.0,
        overlap_sec=1.5,
        use_silero_vad=True,
    )
    assert any(c_s <= 12.1 and c_e >= 17.9 for c_s, c_e, _, _ in rescued_chunks)

    # 5. _run_sherpa_transducer deduplicates overlap boundary words, recovers from batch exception,
    #    and bisects an energetic chunk that initially returned empty output!
    samples_35 = rng.normal(0.0, 0.12, sr * 35).astype(np.float32)
    monkeypatch.setattr(
        tr_mod,
        "get_speech_timestamps",
        lambda audio, vad_options=None, sampling_rate=16000, **kw: [{"start": 0, "end": len(audio)}],
    )

    decode_calls = []

    class _FakeStream:
        def __init__(self):
            self.dur = 0.0
            self.result = SimpleNamespace(tokens=[], timestamps=[], text="")

        def accept_waveform(self, s_rate, wav):
            self.dur = len(wav) / float(s_rate)

    class _FlakyTransducer:
        def create_stream(self):
            return _FakeStream()

        def decode_streams(self, streams):
            # Force fallback path to per-stream decode_stream
            raise RuntimeError("simulated batch decode failure")

        def decode_stream(self, stream):
            decode_calls.append(round(stream.dur, 2))
            if len(decode_calls) == 1:
                # First chunk (0..20s, keep_end=19.25s): emit a token inside core AND an overlap tail at 19.6s
                stream.result = SimpleNamespace(
                    tokens=["▁Первый", "▁стык"],
                    timestamps=[5.0, 19.6],
                    text="Первый стык",
                )
            elif len(decode_calls) == 2:
                # Second chunk (18.5..35s, dur=16.5s): return empty first to trigger >10s bisection retry!
                stream.result = SimpleNamespace(tokens=[], timestamps=[], text="")
            elif len(decode_calls) == 3:
                # Left half of bisected chunk 2 (18.5..26.75s): emit word before keep_start (18.7s -> dropped!)
                # and the real boundary word at 19.6s (rel=1.1s -> kept once!)
                stream.result = SimpleNamespace(
                    tokens=["▁дубль", "▁стык", "▁второй"],
                    timestamps=[0.2, 1.1, 4.0],
                    text="дубль стык второй",
                )
            else:
                # Right half of bisected chunk 2 (26.75..35.0s)
                stream.result = SimpleNamespace(
                    tokens=["▁финал"],
                    timestamps=[3.0],
                    text="финал",
                )

    _ModelCache.active_key = ("sherpa_transducer", "gigaam-v3", "/tmp/fake_longform_gigaam", 2)
    _ModelCache.active_engine = _FlakyTransducer()

    units = Transcriber._run_sherpa_transducer(
        samples=samples_35,
        sample_rate=sr,
        model_dir="/tmp/fake_longform_gigaam",
        diar_segments=[DiarizationSegment(0.0, 35.0, speaker=0)],
        profile_id="gigaam-v3",
        threads=2,
    )
    texts = [u.text.strip() for u in units]
    assert texts == ["Первый", "стык", "второй", "финал"]
    # Verify bisection retry happened (2 initial chunks + 2 bisected sub-chunks = 4 single calls)
    assert len(decode_calls) == 4


def test_custom_glossary_initial_prompt_echo_guard_and_phonetic_normalizer(monkeypatch):
    from types import SimpleNamespace
    from src.config import ASR_PROMPT_PRESETS

    sa_prompt = ASR_PROMPT_PRESETS["system_analyst"]["prompt"]

    # 1. Empty custom_prompt MUST be a 100% no-op (never silently rewrites anything)
    raw_text = "Мы обсуждали джи эр пи си, gpc, оауф 2, джвт, постгрес, сваггер, камунду, сиквенс и докс эз код."
    assert apply_custom_glossary_terms(raw_text, "") == raw_text
    assert apply_custom_glossary_terms(raw_text, "   ") == raw_text

    # 2. With System Analyst preset, requested IT terms are normalized cleanly
    norm_sa = apply_custom_glossary_terms(raw_text, sa_prompt)
    for expected_term in ("gRPC", "OAuth 2", "JWT", "PostgreSQL", "Swagger", "Camunda", "Sequence", "Docs as Code"):
        assert expected_term in norm_sa
    assert "gpc" not in norm_sa.lower()

    # 3. Selective activation: terms NOT in custom_prompt remain untouched!
    norm_partial = apply_custom_glossary_terms(
        "Используем джи эр пи си, постгрес и камунду.", "REST, gRPC"
    )
    assert "gRPC" in norm_partial
    assert "постгрес" in norm_partial
    assert "камунду" in norm_partial

    # 4. Whisper prompt-echo guard blocks verbatim echoes while preserving real technical answers
    assert is_whisper_prompt_echo(
        "Собеседование системного аналитика: REST, gRPC, HTTP 2.0, SOAP, JWT",
        sa_prompt,
    ) is True
    assert is_whisper_prompt_echo(
        "REST, gRPC, HTTP 2.0, SOAP, JWT, Bearer, OAuth 2",
        sa_prompt,
        no_speech_prob=0.45,
    ) is True
    assert is_whisper_prompt_echo(
        "Мы перевели наш сервис с REST на gRPC и подключили OAuth 2.",
        sa_prompt,
        no_speech_prob=0.05,
        avg_logprob=-0.2,
    ) is False
    assert is_whisper_prompt_echo("Обычная фраза.", "") is False

    # 5. _decode_whisper_windows passes custom_prompt into initial_prompt and filters echoed segments
    seen_prompts = []

    class _FakePromptWhisper:
        def __init__(self, model_dir, **kwargs):
            pass

        def transcribe(self, audio_slice, **kwargs):
            seen_prompts.append(kwargs.get("initial_prompt"))
            echo_seg = SimpleNamespace(
                start=0.0,
                end=1.5,
                text=" Собеседование системного аналитика: REST, gRPC, HTTP 2.0, SOAP",
                words=[
                    SimpleNamespace(
                        start=0.0,
                        end=1.5,
                        word=" Собеседование системного аналитика: REST, gRPC, HTTP 2.0, SOAP",
                    )
                ],
                no_speech_prob=0.4,
                avg_logprob=-0.6,
            )
            real_seg = SimpleNamespace(
                start=1.8,
                end=4.0,
                text=" Используем gRPC и OAuth 2.",
                words=[
                    SimpleNamespace(start=1.8, end=2.5, word=" Используем"),
                    SimpleNamespace(start=2.6, end=3.1, word=" gRPC"),
                    SimpleNamespace(start=3.2, end=3.4, word=" и"),
                    SimpleNamespace(start=3.5, end=4.0, word=" OAuth 2."),
                ],
                no_speech_prob=0.03,
                avg_logprob=-0.15,
            )
            return [echo_seg, real_seg], SimpleNamespace(language="ru")

    monkeypatch.setattr(tr_mod, "WhisperModel", _FakePromptWhisper)
    monkeypatch.setattr(tr_mod, "build_vad_asr_windows", lambda *a, **kw: [])
    _ModelCache.active_key = None
    _ModelCache.active_engine = None

    sr = 16000
    samples = np.ones(sr * 6, dtype=np.float32) * 0.05
    diar = [DiarizationSegment(0.2, 5.0, speaker=0)]

    units = Transcriber._decode_whisper_windows(
        samples=samples,
        sample_rate=sr,
        model_dir="/tmp/fake_whisper_prompt",
        profile_id="large-v3-ru",
        device="cpu",
        compute_type="int8",
        diar_segments=diar,
        threads=4,
        status_cb=None,
        language="ru",
        custom_prompt=sa_prompt,
    )
    assert seen_prompts == [sa_prompt]
    assert [u.text.strip() for u in units] == ["Используем", "gRPC", "и", "OAuth 2."]


def test_cluster_anchor_embeddings_separates_4_similar_female_voices():
    """
    Reproduces the real 4-female-participant interview failure where pairwise
    inter-speaker CAM++ cosine similarity sits around ~0.59 (same gender + same
    conference codec). UPGMA average-linkage with 'high' sensitivity (auto) or
    exact speaker_count='4' MUST preserve all 4 speakers instead of collapsing to 2.
    """
    rng = np.random.RandomState(202604)
    dim = 192
    qr, _ = np.linalg.qr(rng.randn(dim, 5))
    common_female_axis = qr[:, 0].astype(np.float32)

    # Inter-speaker cosine similarity = 0.59 across every pair of the 4 speakers!
    shared_w = float(np.sqrt(0.59))
    indiv_w = float(np.sqrt(0.41))
    true_centers = []
    for i in range(4):
        v = shared_w * common_female_axis + indiv_w * qr[:, i + 1].astype(np.float32)
        v /= np.linalg.norm(v)
        true_centers.append(v)

    segments: list[DiarizationSegment] = []
    embeddings: list[np.ndarray] = []
    expected_speakers: list[int] = []
    cur_t = 0.0
    for turn_idx in range(48):
        true_spk = turn_idx % 4
        is_short = (turn_idx % 4 == 3) and (turn_idx >= 12)
        dur = 0.9 if is_short else 3.6
        noise = 0.045 if is_short else 0.025
        emb = true_centers[true_spk] + noise * rng.randn(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        segments.append(DiarizationSegment(cur_t, cur_t + dur, speaker=turn_idx))
        embeddings.append(emb)
        expected_speakers.append(true_spk)
        cur_t += dur + 1.2

    # Case A: Auto speaker discovery with sensitivity='high' (recommended for similar voices)
    clustered_high = cluster_anchor_embeddings(
        segments, embeddings, speaker_count="auto", sensitivity="high"
    )
    assert len({s.speaker for s in clustered_high}) == 4
    by_true_high: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    for idx in range(48):
        by_true_high[expected_speakers[idx]].add(clustered_high[idx].speaker)
    for spk in range(4):
        assert len(by_true_high[spk]) == 1

    # Case B: Exact speaker_count='4' even with default sensitivity='normal'
    clustered_exact4 = cluster_anchor_embeddings(
        segments, embeddings, speaker_count="4", sensitivity="normal"
    )
    assert len({s.speaker for s in clustered_exact4}) == 4
    by_true_4: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    for idx in range(48):
        by_true_4[expected_speakers[idx]].add(clustered_exact4[idx].speaker)
    for spk in range(4):
        assert len(by_true_4[spk]) == 1


def test_transcribe_dual_track_files_separates_speaker_1_deterministically(tmp_path, monkeypatch):
    sr = 16000
    t1_wav = str(tmp_path / "track1_candidate.wav")
    t2_wav = str(tmp_path / "track2_interviewers.wav")
    out_txt = str(tmp_path / "dual_transcript.txt")

    t = np.linspace(0, 10.0, sr * 10, endpoint=False)
    tone = (0.15 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    _write_pcm16_wav(t1_wav, tone, sample_rate=sr)
    _write_pcm16_wav(t2_wav, tone, sample_rate=sr)

    monkeypatch.setattr(ModelManager, "ensure_asr_model", classmethod(lambda cls, pid, status_cb=None: "/tmp/asr"))
    monkeypatch.setattr(
        ModelManager,
        "ensure_diarization_models",
        classmethod(lambda cls, status_cb=None: ("/tmp/seg.onnx", "/tmp/emb.onnx")),
    )

    diar_calls = []

    def _mock_diar(cls, **kw):
        diar_calls.append(
            {"speaker_count": kw.get("speaker_count"), "sensitivity": kw.get("sensitivity")}
        )
        segs = [
            DiarizationSegment(0.2, 2.5, speaker=0),
            DiarizationSegment(5.0, 7.0, speaker=1),
            DiarizationSegment(7.5, 9.5, speaker=2),
        ]
        return (segs, {}, None) if kw.get("return_voiceprints") else segs

    asr_call_idx = {"n": 0}

    def _mock_asr_engine(cls, **kw):
        asr_call_idx["n"] += 1
        if asr_call_idx["n"] == 1:
            # Track 1 (Candidate Mic): must have diar_segments=None!
            assert kw["diar_segments"] is None
            return [TimedUnit(3.0, 4.8, " Мы используем джи эр пи си и постгрес.")]
        # Track 2 (3 Interviewers)
        assert kw["diar_segments"] is not None
        return [
            TimedUnit(0.3, 2.2, " Добрый день, расскажите про интеграции."),
            TimedUnit(5.1, 6.8, " А как насчёт оауф 2?"),
            TimedUnit(7.6, 9.2, " Спасибо, всё понятно."),
        ]

    monkeypatch.setattr(Transcriber, "_run_diarization", classmethod(_mock_diar))
    monkeypatch.setattr(Transcriber, "_run_asr_engine", classmethod(_mock_asr_engine))

    res = Transcriber.transcribe_dual_track_files(
        track1_wav_path=t1_wav,
        track2_wav_path=t2_wav,
        output_txt_path=out_txt,
        asr_model="gigaam-v3",
        speaker_count="4",
        threads=2,
        language="ru",
        custom_prompt="gRPC, OAuth 2, PostgreSQL",
        diarization_sensitivity="high",
    )

    # Diarization ran ONLY ONCE (for Track 2), allocating speaker_count = 4 - 1 = '3'!
    assert diar_calls == [{"speaker_count": "3", "sensitivity": "high"}]
    assert res["dual_track"] is True
    assert res["num_speakers"] == 4
    assert res["num_turns"] == 4

    txt = open(out_txt, encoding="utf-8").read()
    assert "Дорожка 1 = Спикер 1" in txt
    assert "Обнаружено спикеров: 4" in txt
    lines = [ln for ln in txt.splitlines() if ln.startswith("[00:")]
    assert len(lines) == 4
    assert "Спикер 2: Добрый день, расскажите про интеграции." in lines[0]
    assert "Спикер 1: Мы используем gRPC и PostgreSQL." in lines[1]
    assert "Спикер 3: А как насчёт OAuth 2?" in lines[2]
    assert "Спикер 4: Спасибо, всё понятно." in lines[3]



