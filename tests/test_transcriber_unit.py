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
    assert "# Транскрибация интервью (Interview Anonymizer v5.1.0)" in txt

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

    # Populate valid mock files for diarization + 'turbo' profile with lowered byte thresholds
    seg_dir = preloaded / "diarization" / "segmentation"
    emb_dir = preloaded / "diarization" / "embedding"
    asr_dir = preloaded / "asr" / "turbo"
    seg_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)
    asr_dir.mkdir(parents=True)

    seg_file = seg_dir / DIARIZATION_MODELS["segmentation"]["filename"]
    emb_file = emb_dir / DIARIZATION_MODELS["embedding"]["filename"]
    seg_file.write_bytes(b"x" * 128)
    emb_file.write_bytes(b"y" * 256)

    for fname in ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.txt"):
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
        "turbo",
        {**ASR_MODEL_PROFILES["turbo"], "min_total_bytes": 256},
    )

    def _forbid_network(*args, **kwargs):
        raise AssertionError("Network download should not be called when preloaded models exist")

    monkeypatch.setattr(mm_mod, "hf_hub_download", _forbid_network)
    monkeypatch.setattr(mm_mod, "snapshot_download", _forbid_network)

    got_seg, got_emb = ModelManager.ensure_diarization_models()
    assert got_seg == str(seg_file)
    assert got_emb == str(emb_file)

    got_asr = ModelManager.ensure_asr_model("turbo")
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

    # Long real English & Russian monologues (>160-240 chars) naturally have unique/total <= 0.16
    # because alphabets are finite (26/33 letters); they must NEVER be discarded.
    long_en = (
        "Good morning! For our backend architecture, we use PostgreSQL for storage, "
        "Kafka for event streaming, and Redis for caching. Authentication relies on "
        "OAuth with JSON Web Tokens."
    )
    assert clean_hallucinated_text(long_en) == long_en

    long_ru = (
        "Добрый день! В архитектуре нашего высоконагруженного сервиса мы используем "
        "микросервисы, реляционную базу данных PostgreSQL, брокер сообщений Kafka и "
        "распределённый кэш Redis. Авторизация пользователей работает через протокол "
        "OAuth и токены JWT, а все контракты описаны в OpenAPI."
    )
    assert clean_hallucinated_text(long_ru) == long_ru
    assert clean_hallucinated_text("ду ду ду ду ду ду ду ду ду") == ""



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


def test_sequential_whisper_rolling_context_and_breeze_single_worker(monkeypatch):
    from types import SimpleNamespace

    init_kwargs = {}
    prompts_seen = []
    calls_seen = []

    class _FakeBreezeModel:
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

    monkeypatch.setattr(tr_mod, "WhisperModel", _FakeBreezeModel)
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
        model_dir="/tmp/fake_breeze",
        profile_id="breeze-asr-25",
        device="cpu",
        compute_type="int8",
        diar_segments=diar,
        threads=12,
        status_cb=None,
        language="ru",
    )

    # breeze-asr-25 caps num_workers=1 on CPU to preserve RAM bandwidth & enable rolling context
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
    assert _resolve_whisper_language(_MixedLangModel(), wins, samples, "ru", "turbo") == ("ru", False)
    # 2. 'breeze-asr-25' defaults to locked Russian even if 'auto' is passed
    assert _resolve_whisper_language(_MixedLangModel(), wins, samples, "auto", "breeze-asr-25") == ("ru", False)
    # 3. 'auto' with unanimous high-confidence probes locks language to avoid per-window flips
    assert _resolve_whisper_language(_UnanimousRuModel(), wins, samples, "auto", "turbo") == ("ru", False)
    # 4. 'auto' with genuinely mixed languages enables multilingual=True
    assert _resolve_whisper_language(_MixedLangModel(), wins, samples, "auto", "turbo") == ("ru", True)

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
    assert 16.0 <= pause_chunks[0][3] <= 17.0
    assert pause_chunks[1][2] == pytest.approx(pause_chunks[0][3], abs=1e-4)
    assert pause_chunks[0][1] - pause_chunks[1][0] == pytest.approx(1.0, abs=0.05)

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
        profile_id="breeze-asr-25",
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


def test_run_onnx_asr_gpu_batching_overlap_dedup_and_cpu_fallback(monkeypatch):
    from types import SimpleNamespace

    sr = 16000
    rng = np.random.RandomState(500)
    samples_35 = rng.normal(0.0, 0.12, sr * 35).astype(np.float32)

    monkeypatch.setattr(
        tr_mod,
        "get_speech_timestamps",
        lambda audio, vad_options=None, sampling_rate=16000, **kw: [{"start": 0, "end": len(audio)}],
    )

    loaded_calls = []
    batch_sizes_seen = []

    class _FakeOnnxAsrModel:
        def __init__(self, fail_once_on_cuda: bool = False):
            self.fail_once_on_cuda = fail_once_on_cuda
            self.calls = 0

        def with_timestamps(self):
            return self

        def recognize(self, waveforms, sample_rate=16000):
            if self.fail_once_on_cuda:
                raise RuntimeError("CUDAExecutionProvider out of memory")
            if isinstance(waveforms, list):
                batch_sizes_seen.append(len(waveforms))
                results = []
                for idx, wav in enumerate(waveforms):
                    dur = len(wav) / float(sample_rate)
                    if idx == 0:
                        # Chunk 0 (0..20s, keep_end=19.25s): emit interior token + overlap tail at 19.6s
                        results.append(
                            SimpleNamespace(
                                text="Привет стык",
                                tokens=["▁Привет", "▁стык"],
                                timestamps=[5.0, 19.6],
                                logprobs=[-0.1, -0.1],
                            )
                        )
                    else:
                        # Chunk 1 (18.5..35.0s, dur=16.5s): return empty to trigger bisection retry!
                        results.append(SimpleNamespace(text="", tokens=[], timestamps=[], logprobs=[]))
                return results
            # Single-chunk bisection calls
            self.calls += 1
            if self.calls == 1:
                # Left half (18.5..26.75s): token before keep_start (18.7s -> dropped) + boundary word at 19.6s (rel=1.1s)
                return SimpleNamespace(
                    text="дубль стык второй",
                    tokens=["▁дубль", "▁стык", "▁второй"],
                    timestamps=[0.2, 1.1, 4.0],
                    logprobs=[-0.1, -0.1, -0.1],
                )
            return SimpleNamespace(
                text="финал",
                tokens=["▁финал"],
                timestamps=[2.5],
                logprobs=[-0.1],
            )

    def _fake_load_model(model_name, path=None, quantization=None, sess_options=None, providers=None):
        loaded_calls.append(
            {
                "model_name": model_name,
                "path": path,
                "quantization": quantization,
                "providers": list(providers or []),
            }
        )
        is_cuda = "CUDAExecutionProvider" in (providers or [])
        return _FakeOnnxAsrModel(fail_once_on_cuda=is_cuda)

    fake_onnx_asr_mod = SimpleNamespace(load_model=_fake_load_model)
    monkeypatch.setattr(tr_mod, "onnx_asr", fake_onnx_asr_mod)
    monkeypatch.setattr(
        tr_mod,
        "resolve_onnx_provider",
        lambda req, status_cb=None: (
            ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])
            if req == "cuda"
            else ("cpu", ["CPUExecutionProvider"])
        ),
    )
    _ModelCache.active_key = None
    _ModelCache.active_engine = None

    logs = []
    # Request CUDA -> simulated CUDA error -> MUST transparently fall back to CPU and complete!
    units = Transcriber._run_onnx_asr(
        samples=samples_35,
        sample_rate=sr,
        model_dir="/tmp/fake_gigaam_fp32",
        profile_id="gigaam-v3",
        diar_segments=[DiarizationSegment(0.0, 35.0, speaker=0)],
        threads=4,
        compute_device="cuda",
        status_cb=logs.append,
    )

    assert len(loaded_calls) == 2
    assert loaded_calls[0]["model_name"] == "gigaam-v3-e2e-rnnt"
    assert loaded_calls[0]["quantization"] is None
    assert loaded_calls[0]["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert loaded_calls[1]["providers"] == ["CPUExecutionProvider"]
    assert any("переключение" in m and "на CPU" in m for m in logs)
    assert batch_sizes_seen == [2]
    assert [u.text.strip() for u in units] == ["Привет", "стык", "второй", "финал"]


def test_denoise_speech_gtcrn_preserves_sample_length_and_falls_back_safely(monkeypatch):
    from types import SimpleNamespace

    sr = 16000
    # 32.3 seconds -> 3 GTCRN blocks (15s + 15s + 2.3s)
    orig = np.linspace(-0.3, 0.3, int(sr * 32.3), dtype=np.float32)

    monkeypatch.setattr(
        ModelManager,
        "ensure_denoiser_model",
        classmethod(lambda cls, status_cb=None: "/tmp/fake_gtcrn_simple.onnx"),
    )

    block_lengths = []

    class _FakeDenoiser:
        def __init__(self, cfg):
            self.cfg = cfg

        def __call__(self, samples_list, sample_rate):
            block_lengths.append(len(samples_list))
            # Slightly attenuate signal to prove denoiser ran and slightly vary length to verify exact trimming/padding
            arr = np.asarray(samples_list, dtype=np.float32) * 0.5
            return SimpleNamespace(samples=arr.tolist(), sample_rate=sample_rate)

    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeechDenoiser", _FakeDenoiser)
    tr_mod._ModelCache.denoiser_key = None
    tr_mod._ModelCache.denoiser_engine = None

    logs = []
    out = Transcriber._denoise_speech(orig, sample_rate=sr, threads=2, status_cb=logs.append)
    assert out.shape == orig.shape
    assert out.dtype == np.float32
    assert len(block_lengths) == 3
    assert np.allclose(out, orig * 0.5, atol=1e-5)
    assert any("GTCRN" in m for m in logs)

    # Error resilience: if denoiser fails at runtime, MUST return original samples untouched!
    class _BrokenDenoiser:
        def __init__(self, cfg):
            pass

        def __call__(self, samples_list, sample_rate):
            raise RuntimeError("simulated GTCRN failure")

    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeechDenoiser", _BrokenDenoiser)
    tr_mod._ModelCache.denoiser_key = None
    tr_mod._ModelCache.denoiser_engine = None

    err_logs = []
    fallback = Transcriber._denoise_speech(orig[:sr], sample_rate=sr, threads=2, status_cb=err_logs.append)
    assert np.array_equal(fallback, orig[:sr])
    assert any("исходный звук" in m for m in err_logs)


def test_llm_postprocessor_restores_terms_preserves_timestamps_and_guards_corruption(tmp_path, monkeypatch):
    import json
    import httpx
    import src.llm_postprocessor as llm_mod
    from src.llm_postprocessor import polish_turns_with_local_llm

    turns = [
        SpeakerTurn(1.0, 5.5, "Спикер 1", "Мы перевели сервис с реста на GPC и настроили ауф 2."),
        SpeakerTurn(6.0, 11.2, "Спикер 2", "А постгрес и камунда у вас в кубере крутятся?"),
    ]

    # Case 1: Valid Qwen3-style response with <think> block and ```json fence -> terms restored, timestamps intact
    class _FakeResp:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def _make_fake_client(reply_contents: list[str]):
        idx = {"i": 0}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, headers=None, json=None):
                content = reply_contents[min(idx["i"], len(reply_contents) - 1)]
                idx["i"] += 1
                return _FakeResp({"choices": [{"message": {"content": content}}]})

        return _FakeClient

    qwen_content = (
        "<think>Исправляю IT-термины по словарю: GPC -> gRPC, ауф 2 -> OAuth 2.</think>\n"
        "```json\n"
        + json.dumps(
            {
                "turns": [
                    {"id": 0, "text": "Мы перевели сервис с REST на gRPC и настроили OAuth 2."},
                    {"id": 1, "text": "А PostgreSQL и Camunda у вас в Kubernetes крутятся?"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    monkeypatch.setattr(llm_mod.httpx, "Client", _make_fake_client([qwen_content]))

    out_txt = str(tmp_path / "interview_transcript.txt")
    fin = Transcriber._finalize_and_write_transcript(
        turns=turns,
        duration_sec=12.0,
        model_name="GigaAM v3 E2E RNN-T (FP32)",
        num_speakers=2,
        output_txt_path=out_txt,
        custom_prompt="REST, gRPC, OAuth 2, PostgreSQL, Camunda, Kubernetes",
        llm_polish=True,
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen3:8b",
    )
    assert fin["llm_applied"] is True
    assert fin["raw_transcript_path"] is not None
    assert os.path.isfile(fin["raw_transcript_path"])
    # Timestamps and speaker labels are 100% preserved
    assert fin["turns"][0].start == 1.0 and fin["turns"][0].end == 5.5 and fin["turns"][0].speaker_label == "Спикер 1"
    assert fin["turns"][1].start == 6.0 and fin["turns"][1].end == 11.2 and fin["turns"][1].speaker_label == "Спикер 2"
    assert "gRPC" in fin["turns"][0].text and "OAuth 2" in fin["turns"][0].text
    assert "PostgreSQL" in fin["turns"][1].text and "Kubernetes" in fin["turns"][1].text

    final_txt = open(out_txt, encoding="utf-8").read()
    raw_txt = open(fin["raw_transcript_path"], encoding="utf-8").read()
    assert "LLM-корректура (qwen3:8b" in final_txt
    assert "gRPC" in final_txt
    assert "LLM-корректура" not in raw_txt

    # Case 2: Corrupted LLM output (meta-leak + wrong length) -> guardrail rejects batch & preserves raw ASR!
    corrupted = json.dumps(
        {
            "turns": [
                {"id": 0, "text": "Вот исправленный текст: Мы перевели сервис."},
                {"id": 1, "text": "Ок."},
            ]
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(llm_mod.httpx, "Client", _make_fake_client([corrupted]))
    bad_logs = []
    guarded_turns, stats_bad = polish_turns_with_local_llm(
        turns=turns, base_url="http://localhost:11434/v1", model="qwen3:8b", status_cb=bad_logs.append
    )
    assert stats_bad["applied"] is False
    assert stats_bad["batches_ok"] == 0
    assert stats_bad["batches_total"] == 1
    assert any("не прошёл валидацию" in m for m in bad_logs)
    assert [t.text for t in guarded_turns] == [t.text for t in turns]

    # Case 3: Local Ollama offline (ConnectError) -> safe immediate fallback without crashing!
    class _OfflineClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(llm_mod.httpx, "Client", _OfflineClient)
    offline_logs = []
    safe_turns, stats_off = polish_turns_with_local_llm(
        turns=turns,
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        status_cb=offline_logs.append,
    )
    assert stats_off["applied"] is False
    assert stats_off["batches_ok"] == 0
    assert [t.text for t in safe_turns] == [t.text for t in turns]
    assert any("ошибка на батче" in m and "Connection refused" in m for m in offline_logs)


def test_llm_postprocessor_cloud_deepseek_qwen_auth_header_and_401_abort(monkeypatch):
    import json
    import httpx
    import src.llm_postprocessor as llm_mod
    from src.llm_postprocessor import polish_turns_with_local_llm

    secret_key = "sk-cloud-secret-do-not-log-999"
    turns = [
        SpeakerTurn(0.5, 4.0, "Спикер 1", "Мы развернули кубер и переехали на GPC."),
        SpeakerTurn(4.5, 9.0, "Спикер 2", "А постгрес с кафкой тоже подняли?"),
    ]
    seen_requests = []

    class _FakeOkResp:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class _CloudClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            seen_requests.append({"url": url, "headers": dict(headers or {}), "model": (json or {}).get("model")})
            return _FakeOkResp(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"turns": ['
                                    '{"id": 0, "text": "Мы развернули Kubernetes и переехали на gRPC."}, '
                                    '{"id": 1, "text": "А PostgreSQL с Kafka тоже подняли?"}'
                                    ']}'
                                )
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _CloudClient())

    # 1. DeepSeek Cloud API (deepseek-flash default model)
    ds_logs = []
    out_ds, stats_ds = polish_turns_with_local_llm(
        turns=turns,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        api_key=secret_key,
        custom_prompt="Kubernetes, gRPC, PostgreSQL, Kafka",
        status_cb=ds_logs.append,
    )
    assert stats_ds["applied"] is True
    assert stats_ds["is_cloud"] is True
    assert seen_requests[-1]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen_requests[-1]["headers"]["Authorization"] == f"Bearer {secret_key}"
    assert seen_requests[-1]["model"] == "deepseek-flash"
    assert "gRPC" in out_ds[0].text and "PostgreSQL" in out_ds[1].text
    assert any("облачный API" in m for m in ds_logs)
    assert all(secret_key not in m for m in ds_logs)

    # 2. Qwen Cloud (DashScope OpenAI-compatible)
    qw_logs = []
    out_qw, stats_qw = polish_turns_with_local_llm(
        turns=turns,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        api_key=secret_key,
        status_cb=qw_logs.append,
    )
    assert stats_qw["applied"] is True
    assert stats_qw["is_cloud"] is True
    assert seen_requests[-1]["url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert seen_requests[-1]["headers"]["Authorization"] == f"Bearer {secret_key}"
    assert seen_requests[-1]["model"] == "qwen-plus"
    assert all(secret_key not in m for m in qw_logs)

    # 3. HTTP 401 on multi-batch cloud call aborts immediately without hammering remaining batches or leaking key
    many_turns = [
        SpeakerTurn(float(i * 3), float(i * 3 + 2), "Спикер 1", f"Реплика номер {i} про GPC.")
        for i in range(6)
    ]
    auth_attempts = {"count": 0}

    class _AuthFailClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            auth_attempts["count"] += 1
            req = httpx.Request("POST", url)
            resp = httpx.Response(401, request=req)
            raise httpx.HTTPStatusError("401 Unauthorized", request=req, response=resp)

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _AuthFailClient())
    auth_logs = []
    safe_turns, stats_401 = polish_turns_with_local_llm(
        turns=many_turns,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        api_key=secret_key,
        batch_size=2,
        status_cb=auth_logs.append,
    )
    assert auth_attempts["count"] == 1  # Fast-aborted after batch 1 instead of 3 failed calls
    assert stats_401["applied"] is False
    assert stats_401["is_cloud"] is True
    assert [t.text for t in safe_turns] == [t.text for t in many_turns]
    assert any("401" in m and "API-ключ" in m for m in auth_logs)
    assert all(secret_key not in m for m in auth_logs)


def test_per_model_tuned_decoding_params_hotwords_ctc_batch_and_llm_json_fallback(monkeypatch):
    from types import SimpleNamespace
    import httpx
    import src.llm_postprocessor as llm_mod
    from src.config import ASR_MODEL_PROFILES
    from src.llm_postprocessor import polish_turns_with_local_llm

    # 1. Verify Breeze-ASR-25 disables hotwords (avoids hallucination loops on fine-tuned Whisper)
    #    while Whisper Turbo passes capped hotwords (<=220 chars) when custom_prompt is provided
    seen_calls = []

    class _FakeWhisperCapture:
        def __init__(self, model_dir, **kwargs):
            pass

        def transcribe(self, audio_slice, **kwargs):
            seen_calls.append(dict(kwargs))
            seg = SimpleNamespace(
                start=0.1,
                end=1.5,
                text=" Используем gRPC и OAuth 2.",
                words=[SimpleNamespace(start=0.1, end=1.4, word=" Используем gRPC и OAuth 2.")],
                no_speech_prob=0.05,
                avg_logprob=-0.2,
                compression_ratio=1.2,
            )
            return [seg], SimpleNamespace(language="ru")

    monkeypatch.setattr(tr_mod, "WhisperModel", _FakeWhisperCapture)
    _ModelCache.active_key = None
    _ModelCache.active_engine = None

    sr = 16000
    wav_4s = np.ones(sr * 4, dtype=np.float32) * 0.05
    diar_1 = [DiarizationSegment(0.2, 3.5, speaker=0)]

    Transcriber._decode_whisper_windows(
        samples=wav_4s,
        sample_rate=sr,
        model_dir="/tmp/fake_breeze",
        profile_id="breeze-asr-25",
        device="cpu",
        compute_type="int8",
        diar_segments=diar_1,
        threads=4,
        status_cb=None,
        language="ru",
        custom_prompt="Собеседование системного аналитика: REST, gRPC, OAuth 2, JWT",
    )
    breeze_kw = seen_calls[-1]
    assert breeze_kw["beam_size"] == 3
    assert breeze_kw["best_of"] == 3
    assert breeze_kw["patience"] == pytest.approx(1.2)
    assert breeze_kw["repetition_penalty"] == pytest.approx(1.05)
    assert breeze_kw["no_repeat_ngram_size"] == 4
    assert breeze_kw["no_speech_threshold"] == pytest.approx(0.58)
    assert breeze_kw["compression_ratio_threshold"] == pytest.approx(2.3)
    assert breeze_kw["log_prob_threshold"] == pytest.approx(-0.95)
    # Breeze-ASR-25 has use_hotwords=False to avoid Whisper hotwords looping
    assert "hotwords" not in breeze_kw

    long_prompt = "Собеседование: " + ", ".join(f"ServiceTerm{i:02d}" for i in range(40))
    Transcriber._decode_whisper_windows(
        samples=wav_4s,
        sample_rate=sr,
        model_dir="/tmp/fake_turbo",
        profile_id="turbo",
        device="cpu",
        compute_type="int8",
        diar_segments=diar_1,
        threads=4,
        status_cb=None,
        language="ru",
        custom_prompt=long_prompt,
    )
    turbo_kw = seen_calls[-1]
    assert turbo_kw["beam_size"] == 2
    assert turbo_kw["patience"] == pytest.approx(1.0)
    assert turbo_kw["repetition_penalty"] == pytest.approx(1.08)
    assert "hotwords" in turbo_kw
    assert len(turbo_kw["hotwords"]) <= 220
    assert "ServiceTerm00" in turbo_kw["hotwords"]

    Transcriber._decode_whisper_windows(
        samples=wav_4s,
        sample_rate=sr,
        model_dir="/tmp/fake_turbo",
        profile_id="turbo",
        device="cpu",
        compute_type="int8",
        diar_segments=diar_1,
        threads=4,
        status_cb=None,
        language="ru",
        custom_prompt="",
    )
    assert "hotwords" not in seen_calls[-1]

    # 2. Verify clean 5-model core and ONNX profile tuning invariants
    assert list(ASR_MODEL_PROFILES.keys()) == [
        "gigaam-v3",
        "gigaam-v3-ctc",
        "breeze-asr-25",
        "turbo",
        "deepgram-nova-3",
    ]
    assert ASR_MODEL_PROFILES["gigaam-v3-ctc"]["batch_size_cuda"] == 16
    assert ASR_MODEL_PROFILES["gigaam-v3-ctc"]["batch_size_cpu"] == 6
    assert ASR_MODEL_PROFILES["gigaam-v3"]["batch_size_cuda"] == 8
    assert ASR_MODEL_PROFILES["breeze-asr-25"]["decode_params"]["use_hotwords"] is False
    assert ASR_MODEL_PROFILES["turbo"]["decode_params"]["use_hotwords"] is True

    # 3. Verify GTCRN denoiser is cached across back-to-back calls
    init_count = {"n": 0}

    class _CountingDenoiser:
        def __init__(self, cfg):
            init_count["n"] += 1

        def __call__(self, blk, sr_in):
            return SimpleNamespace(samples=blk, sample_rate=sr_in)

    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeechDenoiser", _CountingDenoiser)
    monkeypatch.setattr(
        ModelManager,
        "ensure_denoiser_model",
        classmethod(lambda cls, status_cb=None: "/tmp/fake_gtcrn_cached.onnx"),
    )
    _ModelCache.denoiser_key = None
    _ModelCache.denoiser_engine = None
    Transcriber._denoise_speech(wav_4s, sample_rate=sr, threads=2)
    Transcriber._denoise_speech(wav_4s, sample_rate=sr, threads=2)
    assert init_count["n"] == 1

    # 5. Verify LLM postprocessor sends temperature=0.05 + response_format json_object & retries on HTTP 400
    llm_bodies = []

    class _FallbackJsonClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            llm_bodies.append(dict(json or {}))
            if "response_format" in (json or {}):
                req = httpx.Request("POST", url)
                raise httpx.HTTPStatusError("400 unsupported response_format", request=req, response=httpx.Response(400, request=req))
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "choices": [{"message": {"content": '{"turns": [{"id": 0, "text": "Мы используем gRPC."}]}'}}]
                },
            )

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _FallbackJsonClient())
    out_turns, stats = polish_turns_with_local_llm(
        turns=[SpeakerTurn(0.0, 2.0, "Спикер 1", "Мы используем GPC.")],
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
    )
    assert stats["applied"] is True
    assert out_turns[0].text == "Мы используем gRPC."
    assert len(llm_bodies) == 2
    assert llm_bodies[0]["temperature"] == pytest.approx(0.05)
    assert llm_bodies[0]["top_p"] == pytest.approx(0.9)
    assert llm_bodies[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in llm_bodies[1]


def test_v501_zoom_4_female_svd_split_punctuated_boundary_overlap_and_glossary(monkeypatch):
    from src.config import ASR_PROMPT_PRESETS
    from src.transcriber import apply_custom_glossary_terms, build_longform_asr_chunks

    # 1. Realistic 4-female Zoom interview: 1 local mic (Alina) + 3 remote Zoom women sharing
    #    a heavy channel+gender axis (raw inter-speaker cosine = 0.70, unbalanced 24/12/10/8).
    #    MUST separate into ALL 4 speakers even in DEFAULT speaker_count='auto', sensitivity='normal'!
    rng = np.random.RandomState(501)
    dim = 192
    qr, _ = np.linalg.qr(rng.randn(dim, 6))
    local_mic_vec = qr[:, 0].astype(np.float32)
    zoom_axis = qr[:, 1].astype(np.float32)
    sh, ind = float(np.sqrt(0.70)), float(np.sqrt(0.30))
    zoom_women = [
        (sh * zoom_axis + ind * qr[:, 2 + k].astype(np.float32)) for k in range(3)
    ]
    zoom_women = [v / np.linalg.norm(v) for v in zoom_women]
    centers = [local_mic_vec] + zoom_women
    counts = [24, 12, 10, 8]

    segs: list[DiarizationSegment] = []
    embs: list[np.ndarray] = []
    truth: list[int] = []
    t_cur = 0.0
    max_turns = max(counts)
    for round_i in range(max_turns):
        for spk_i, cnt in enumerate(counts):
            if round_i >= cnt:
                continue
            dur = 3.5
            v = centers[spk_i] + 0.032 * rng.randn(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            segs.append(DiarizationSegment(t_cur, t_cur + dur, speaker=len(segs)))
            embs.append(v)
            truth.append(spk_i)
            t_cur += dur + 1.0

    clustered_default = cluster_anchor_embeddings(
        segs, embs, speaker_count="auto", sensitivity="normal"
    )
    assert len({s.speaker for s in clustered_default}) == 4
    by_truth: dict[int, set[int]] = {i: set() for i in range(4)}
    for idx, t_spk in enumerate(truth):
        by_truth[t_spk].add(clustered_default[idx].speaker)
    for spk_i in range(4):
        assert len(by_truth[spk_i]) == 1

    # Single speaker with mild volume/mood drift (cos=0.88) MUST NOT falsely split!
    drift_b = np.sqrt(0.88) * local_mic_vec + np.sqrt(0.12) * qr[:, 5].astype(np.float32)
    drift_segs = [DiarizationSegment(i * 4.5, i * 4.5 + 3.5, speaker=i) for i in range(36)]
    drift_embs = []
    for i in range(36):
        base = local_mic_vec if i < 18 else drift_b
        v = base + 0.04 * rng.randn(dim).astype(np.float32)
        drift_embs.append(v / np.linalg.norm(v))
    clustered_drift = cluster_anchor_embeddings(
        drift_segs, drift_embs, speaker_count="auto", sensitivity="normal"
    )
    assert len({s.speaker for s in clustered_drift}) == 1

    # 2. Punctuated glued 2-speaker GigaAM clause MUST split across speaker boundary
    diar_glued = [
        DiarizationSegment(0.0, 2.2, speaker=1),   # Interviewer finishes without period
        DiarizationSegment(2.2, 4.4, speaker=0),   # Alina starts answering and ends with '.'
    ]
    glued_units = [
        TimedUnit(0.0, 0.45, " Твоей"),
        TimedUnit(0.45, 0.95, " готовности"),
        TimedUnit(0.95, 1.35, " в дальнейшем"),
        TimedUnit(1.35, 1.75, " сильно"),
        TimedUnit(1.75, 2.18, " погрузиться,"),
        TimedUnit(2.22, 2.60, " да,"),
        TimedUnit(2.60, 3.15, " конечно,"),
        TimedUnit(3.15, 3.65, " я"),
        TimedUnit(3.65, 4.35, " готова."),
    ]
    turns_glued, n_spks_glued = align_units_to_speakers(glued_units, diar_glued)
    assert n_spks_glued == 2
    assert len(turns_glued) == 2
    assert "погрузиться" in turns_glued[0].text
    assert "готова" in turns_glued[1].text

    # 3. build_longform_asr_chunks supplies overlap context even at true pause splits (no cold cut!)
    sr = 16000
    speech_sig = rng.normal(0.0, 0.14, sr * 34).astype(np.float32)
    # Create a genuine 300ms silence valley at 16.0s..16.3s inside a 34s speech island
    speech_sig[int(16.0 * sr) : int(16.3 * sr)] = 0.0
    monkeypatch.setattr(
        "src.transcriber.get_speech_timestamps",
        lambda wav, vad_options, sampling_rate: [{"start": 0, "end": len(wav)}],
    )
    chunks = build_longform_asr_chunks(
        speech_sig, sample_rate=sr, max_chunk_sec=20.0, overlap_sec=1.6, use_silero_vad=True
    )
    assert len(chunks) == 2
    (c1_s, c1_e, k1_s, k1_e), (c2_s, c2_e, k2_s, k2_e) = chunks
    assert (c1_e - c1_s) <= 20.0 + 1e-6
    assert (c2_e - c2_s) <= 20.0 + 1e-6
    # Must overlap acoustically across the pause seam while keeping ownership non-overlapping!
    assert c1_e > k1_e + 0.30
    assert c2_s < k2_s - 0.30
    assert k1_e == pytest.approx(k2_s, abs=1e-4)

    # 4. Expanded opt-in System Analyst phonetic normalizer fixes real GigaAM v3 FP32 distortions
    #    while preserving legitimate homonyms like 'GPT-токены' untouched.
    sa_prompt = ASR_PROMPT_PRESETS["system_analyst"]["prompt"]
    raw_gigaam = (
        "Работаем на C+ и от ЛП, интеграция ЦФА-система поджирописью, BPen в Комунде, "
        "передаём Geson, джвт-токены, бирутокены и ауф два, рисуем в План Юля и дру Ио, "
        "задачи в Джерри, UI-гид, user Kase и SpecDrive and development."
    )
    fixed = apply_custom_glossary_terms(raw_gigaam, sa_prompt)
    for expected_term in (
        "C++",
        "OTLP",
        "SFA-система",
        "по gRPC",
        "BPMN",
        "Camunda",
        "JSON",
        "JWT-токены",
        "Bearer-токены",
        "OAuth 2",
        "PlantUML",
        "draw.io",
        "в Jira",
        "UI-kit",
        "use cases",
        "Spec-Driven Development",
    ):
        assert expected_term in fixed, f"Missing '{expected_term}' in: {fixed}"
    # Homonym protection: real 'GPT-токены' MUST NOT be silently corrupted to 'JWT-токены'
    assert apply_custom_glossary_terms("Считаем GPT-токены для LLM.", sa_prompt) == "Считаем GPT-токены для LLM."
    # Strict opt-in invariant: empty custom_prompt leaves text 100% untouched
    assert apply_custom_glossary_terms(raw_gigaam, "") == raw_gigaam


def test_deepgram_nova3_client_keyterms_diarization_400_fallback_and_skip_local_campp(tmp_path, monkeypatch):
    import httpx
    import src.deepgram_asr as dg_mod
    from src.deepgram_asr import extract_deepgram_keyterms, transcribe_with_deepgram_nova3
    from src.model_manager import ModelManager

    # 1. Keyterm extraction strips leading descriptive prefix, skips 1-2 char pure-alpha terms (Go/UI/x),
    #    preserves non-alpha short tokens like C4 / C++, and deduplicates cleanly
    terms = extract_deepgram_keyterms(
        "Собеседование системного аналитика: REST, gRPC, HTTP/1.1, gRPC, OAuth 2, x, Go, UI, C4, C++, Camunda"
    )
    assert terms == ["REST", "gRPC", "HTTP/1.1", "OAuth 2", "C4", "C++", "Camunda"]
    assert extract_deepgram_keyterms("") == []

    # 2. ModelManager skips local download for cloud profile
    assert ModelManager.ensure_asr_model("deepgram-nova-3") == ""

    # Prepare a tiny real 16 kHz mono WAV file
    wav_path = str(tmp_path / "dg_sample.wav")
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    samples = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    Transcriber._write_wav_16k_mono(wav_path, samples, sample_rate=sr)

    # 3. Missing API key fails fast with helpful Russian error
    with pytest.raises(RuntimeError, match="Deepgram API-ключ"):
        transcribe_with_deepgram_nova3(wav_path=wav_path, api_key="")

    secret_dg_key = "dg-secret-token-987654"
    recorded_calls = []

    class _FakeResp:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                req = httpx.Request("POST", "https://api.deepgram.com/v1/listen")
                resp = httpx.Response(self.status_code, request=req)
                raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=req, response=resp)

        def json(self):
            return self._payload

    dg_payload = {
        "metadata": {"duration": 2.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Привет! Мы используем gRPC и Camunda.",
                            "words": [
                                {"word": "привет", "punctuated_word": "Привет!", "start": 0.1, "end": 0.5, "speaker": 0},
                                {"word": "мы", "punctuated_word": "Мы", "start": 0.8, "end": 1.0, "speaker": 1},
                                {"word": "используем", "punctuated_word": "используем", "start": 1.0, "end": 1.4, "speaker": 1},
                                {"word": "grpc", "punctuated_word": "gRPC", "start": 1.4, "end": 1.65, "speaker": 1},
                                {"word": "и", "punctuated_word": "и", "start": 1.65, "end": 1.75, "speaker": 1},
                                {"word": "camunda", "punctuated_word": "Camunda.", "start": 1.75, "end": 1.98, "speaker": 1},
                            ],
                        }
                    ]
                }
            ]
        },
    }

    # 4. Success with keyterms + native diarization
    class _SuccessClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, params=None, headers=None, content=None):
            recorded_calls.append({"url": url, "params": list(params or []), "headers": dict(headers or {})})
            return _FakeResp(200, dg_payload)

    monkeypatch.setattr(dg_mod, "_create_deepgram_httpx_client", lambda timeout_sec: _SuccessClient())
    logs = []
    units, diar_segs = transcribe_with_deepgram_nova3(
        wav_path=wav_path,
        api_key=secret_dg_key,
        language="ru",
        custom_prompt="Собеседование: gRPC, Camunda",
        diarize=True,
        status_cb=logs.append,
    )
    assert len(units) == 6
    assert len(diar_segs) == 2
    assert {s.speaker for s in diar_segs} == {0, 1}
    assert recorded_calls[-1]["url"] == "https://api.deepgram.com/v1/listen"
    assert recorded_calls[-1]["headers"]["Authorization"] == f"Token {secret_dg_key}"
    assert ("model", "nova-3") in recorded_calls[-1]["params"]
    assert ("diarize", "true") in recorded_calls[-1]["params"]
    assert ("keyterm", "gRPC") in recorded_calls[-1]["params"]
    assert ("keyterm", "Camunda") in recorded_calls[-1]["params"]
    assert all(secret_dg_key not in m for m in logs)

    # 5a. language="auto" (multi) skips keyterm upfront in a single request (no wasted double upload!)
    multi_calls = []

    class _MultiCaptureClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, params=None, headers=None, content=None):
            multi_calls.append(list(params or []))
            return _FakeResp(200, dg_payload)

    monkeypatch.setattr(dg_mod, "_create_deepgram_httpx_client", lambda timeout_sec: _MultiCaptureClient())
    units_m, diar_m = transcribe_with_deepgram_nova3(
        wav_path=wav_path,
        api_key=secret_dg_key,
        language="auto",
        custom_prompt="gRPC, Camunda",
        diarize=True,
    )
    assert len(multi_calls) == 1
    assert ("language", "multi") in multi_calls[0]
    assert not any(k == "keyterm" for k, _ in multi_calls[0])
    assert len(units_m) == 6 and len(diar_m) == 2

    # 5b. HTTP 400 with keyterms on ru/en transparently retries without keyterm and succeeds
    retry_calls = []

    class _Retry400Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, params=None, headers=None, content=None):
            p_list = list(params or [])
            retry_calls.append(p_list)
            if any(k == "keyterm" for k, _ in p_list):
                return _FakeResp(400, {"err_msg": "keyterm not supported"})
            return _FakeResp(200, dg_payload)

    monkeypatch.setattr(dg_mod, "_create_deepgram_httpx_client", lambda timeout_sec: _Retry400Client())
    units_r, diar_r = transcribe_with_deepgram_nova3(
        wav_path=wav_path,
        api_key=secret_dg_key,
        language="ru",
        custom_prompt="gRPC, Camunda",
        diarize=True,
    )
    assert len(retry_calls) == 2
    assert ("language", "ru") in retry_calls[0]
    assert any(k == "keyterm" for k, _ in retry_calls[0])
    assert not any(k == "keyterm" for k, _ in retry_calls[1])
    assert len(units_r) == 6 and len(diar_r) == 2

    # 6. HTTP 401 raises Russian RuntimeError and NEVER leaks secret API key
    class _Fail401Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, params=None, headers=None, content=None):
            return _FakeResp(401, {"err_msg": "invalid"})

    monkeypatch.setattr(dg_mod, "_create_deepgram_httpx_client", lambda timeout_sec: _Fail401Client())
    with pytest.raises(RuntimeError) as exc_info:
        transcribe_with_deepgram_nova3(wav_path=wav_path, api_key=secret_dg_key)
    assert "401" in str(exc_info.value)
    assert secret_dg_key not in str(exc_info.value)

    # 7. Transcriber.transcribe_audio_file skips local CAM++ diarization when Deepgram returns speaker segments
    monkeypatch.setattr(dg_mod, "_create_deepgram_httpx_client", lambda timeout_sec: _SuccessClient())

    def _forbidden_local_diar(*args, **kwargs):
        raise AssertionError("Local CAM++ diarization must NOT run when Deepgram provides native speakers")

    monkeypatch.setattr(ModelManager, "ensure_diarization_models", staticmethod(_forbidden_local_diar))
    monkeypatch.setattr(Transcriber, "_run_diarization", classmethod(lambda cls, *a, **kw: _forbidden_local_diar()))

    out_txt = str(tmp_path / "dg_out.txt")
    res = Transcriber.transcribe_audio_file(
        wav_path=wav_path,
        output_txt_path=out_txt,
        asr_model="deepgram-nova-3",
        speaker_count="auto",
        language="ru",
        custom_prompt="gRPC, Camunda",
        asr_api_key=secret_dg_key,
    )
    assert res["num_speakers"] == 2
    assert res["num_turns"] == 2
    txt = open(out_txt, encoding="utf-8").read()
    assert "Deepgram Nova-3 (Cloud API)" in txt
    assert "Спикер 1: Привет!" in txt
    assert "Спикер 2: Мы используем gRPC и Camunda." in txt
    assert secret_dg_key not in txt


def test_llm_postprocessor_deepseek_non_thinking_blank_retry_truncation_and_per_turn_guard(monkeypatch):
    import json as json_mod
    import src.llm_postprocessor as llm_mod
    from src.llm_postprocessor import _build_turn_batches, polish_turns_with_local_llm

    class _Resp:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    # 1. DeepSeek endpoint sends thinking={"type": "disabled"}, generous max_tokens, omits top_p,
    #    and recovers automatically from DeepSeek's documented JSON-mode blank-spaces response.
    turns = [
        SpeakerTurn(0.0, 4.0, "Спикер 1", "Мы работаем по GPC и рисуем диаграммы в Комунде."),
        SpeakerTurn(4.5, 9.0, "Спикер 2", "А для авторизации используете ауф два и джвт?"),
    ]
    seen_bodies = []

    class _BlankThenOkClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            seen_bodies.append(dict(json or {}))
            if len(seen_bodies) == 1:
                # DeepSeek JSON-mode quirk: spaces-only content on attempt 1
                return _Resp({"choices": [{"finish_reason": "stop", "message": {"content": "     "}}]})
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"turns": ['
                                    '{"id": 0, "text": "Мы работаем по gRPC и рисуем диаграммы в Camunda."}, '
                                    '{"id": 1, "text": "А для авторизации используете OAuth 2 и JWT?"}'
                                    ']}'
                                )
                            },
                        }
                    ]
                }
            )

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _BlankThenOkClient())
    ds_logs = []
    out_turns, stats = polish_turns_with_local_llm(
        turns=turns,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        api_key="sk-ds-test",
        status_cb=ds_logs.append,
    )
    assert stats["applied"] is True
    assert stats["batches_ok"] == 1
    assert len(seen_bodies) == 2
    assert seen_bodies[0]["thinking"] == {"type": "disabled"}
    assert "top_p" not in seen_bodies[0]
    assert seen_bodies[0]["max_tokens"] >= 2048
    assert seen_bodies[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen_bodies[1]
    assert seen_bodies[1]["thinking"] == {"type": "disabled"}
    assert "gRPC" in out_turns[0].text and "Camunda" in out_turns[0].text
    assert "OAuth 2" in out_turns[1].text and "JWT" in out_turns[1].text
    assert any("non-thinking" in m for m in ds_logs)

    # 2. Truncated finish_reason='length' retries once with 2x max_tokens and succeeds
    trunc_bodies = []

    class _TruncThenOkClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            trunc_bodies.append(dict(json or {}))
            if len(trunc_bodies) == 1:
                return _Resp(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": '{"turns": [{"id": 0, "text": "Мы работаем по gR'},
                            }
                        ]
                    }
                )
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json_mod.dumps(
                                    {
                                        "turns": [
                                            {"id": 0, "text": "Мы работаем по gRPC и рисуем диаграммы в Camunda."},
                                            {"id": 1, "text": "А для авторизации используете OAuth 2 и JWT?"},
                                        ]
                                    },
                                    ensure_ascii=False,
                                )
                            },
                        }
                    ]
                }
            )

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _TruncThenOkClient())
    out_trunc, stats_trunc = polish_turns_with_local_llm(
        turns=turns,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        api_key="sk-ds-test",
    )
    assert stats_trunc["applied"] is True
    assert len(trunc_bodies) == 2
    assert trunc_bodies[1]["max_tokens"] == trunc_bodies[0]["max_tokens"] * 2
    assert "Camunda" in out_trunc[0].text

    # 3. Per-turn guardrail: 1 corrupted turn keeps original ASR while valid turn in same batch IS applied
    class _PartialBadClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"turns": ['
                                    '{"id": 0, "text": "Мы работаем по gRPC и рисуем диаграммы в Camunda."}, '
                                    '{"id": 1, "text": "Вот исправленный текст: Окей."}'
                                    ']}'
                                )
                            },
                        }
                    ]
                }
            )

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _PartialBadClient())
    partial_logs = []
    out_partial, stats_partial = polish_turns_with_local_llm(
        turns=turns,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        api_key="sk-ds-test",
        status_cb=partial_logs.append,
    )
    assert stats_partial["applied"] is True
    assert stats_partial["batches_ok"] == 1
    assert "gRPC" in out_partial[0].text
    assert out_partial[1].text == turns[1].text  # Corrupted turn safely preserved as original ASR
    assert any("оставлено без правок по guard: 1/2" in m for m in partial_logs)

    # 4. Character-aware batching splits oversized monologues cleanly
    long_turns = [SpeakerTurn(float(i * 10), float(i * 10 + 9), "Спикер 1", "А" * 1500) for i in range(5)]
    char_batches = _build_turn_batches(long_turns, batch_size=12, max_batch_chars=3600)
    assert char_batches == [(0, 2), (2, 4), (4, 5)]


def test_v510_glossary_homonym_word_boundary_echo_and_llm_bypass(tmp_path, monkeypatch):
    from src.config import ASR_PROMPT_PRESETS
    from src.llm_postprocessor import _build_polish_messages
    from src.transcriber import apply_custom_glossary_terms, is_whisper_prompt_echo

    sa_prompt = ASR_PROMPT_PRESETS["system_analyst"]["prompt"]

    # 1. Dangerous homonyms MUST remain untouched even when full SA preset is active
    homonym_text = (
        "Мы считаем GPT-токены для модели, в еде полезные жиры и редис, "
        "мяч ушёл, о аут, а за экзамен поставили оценку C+."
    )
    assert apply_custom_glossary_terms(homonym_text, sa_prompt) == homonym_text

    # 2. Substring triggers MUST use word boundaries ('digital' != 'Git', 'restore' != 'REST')
    unrelated_prompt = "digital marketing, restore backup, hospitality, restful sleep"
    assert apply_custom_glossary_terms("Мы обсуждали гит и рест.", unrelated_prompt) == (
        "Мы обсуждали гит и рест."
    )
    # Explicit 'Git, REST' DOES activate those rules
    assert apply_custom_glossary_terms("Мы обсуждали гит и рест.", "Git, REST") == (
        "Мы обсуждали Git и REST."
    )

    # 3. Confident real tech-stack enumeration MUST NOT be falsely deleted as Whisper prompt echo
    #    (5 terms in candidate's own order, high confidence — previously deleted by old >=5 rule!)
    assert (
        is_whisper_prompt_echo(
            "PostgreSQL, Redis, Kafka, gRPC, REST.",
            sa_prompt,
            no_speech_prob=0.02,
            avg_logprob=-0.15,
        )
        is False
    )
    # True low-confidence/silence prompt echo AND verbatim header echo ARE still caught
    assert (
        is_whisper_prompt_echo(
            "PostgreSQL, Redis, Kafka, gRPC, REST.",
            sa_prompt,
            no_speech_prob=0.45,
            avg_logprob=-0.75,
        )
        is True
    )
    assert (
        is_whisper_prompt_echo(
            "Собеседование системного аналитика: REST, gRPC, HTTP 2.0.",
            sa_prompt,
            no_speech_prob=0.02,
            avg_logprob=-0.15,
        )
        is True
    )

    # 4. _finalize_and_write_transcript passes RAW ASR text to LLM (no blind pre-regex)
    #    and skips regex when LLM applies, while applying regex when LLM is off or fails.
    seen_llm_inputs = []

    def _fake_llm_ok(turns, **kwargs):
        seen_llm_inputs.append([t.text for t in turns])
        out = [
            SpeakerTurn(t.start, t.end, t.speaker_label, t.text.replace("поджирописью", "по gRPC"))
            for t in turns
        ]
        return out, {"applied": True, "changed_turns": len(out)}

    def _fake_llm_fail(turns, **kwargs):
        seen_llm_inputs.append([t.text for t in turns])
        return list(turns), {"applied": False, "changed_turns": 0}

    raw_turns = [
        SpeakerTurn(0.0, 3.0, "Спикер 1", "Работаем поджирописью, передаём Geson."),
    ]
    monkeypatch.setattr(tr_mod, "polish_turns_with_local_llm", _fake_llm_ok)
    out_llm_ok = str(tmp_path / "llm_ok.txt")
    fin_ok = Transcriber._finalize_and_write_transcript(
        turns=raw_turns,
        duration_sec=3.0,
        model_name="GigaAM v3",
        num_speakers=1,
        output_txt_path=out_llm_ok,
        custom_prompt=sa_prompt,
        llm_polish=True,
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-flash",
    )
    assert fin_ok["llm_applied"] is True
    # LLM received RAW unmutated ASR text ('поджирописью' and 'Geson'), NOT regex-rewritten 'JSON'
    assert seen_llm_inputs[-1] == ["Работаем поджирописью, передаём Geson."]
    # Regex did NOT overwrite LLM output after LLM succeeded ('Geson' stays as LLM left it)
    assert fin_ok["turns"][0].text == "Работаем по gRPC, передаём Geson."

    # When LLM fails (applied=False), regex fallback DOES run
    monkeypatch.setattr(tr_mod, "polish_turns_with_local_llm", _fake_llm_fail)
    out_llm_fail = str(tmp_path / "llm_fail.txt")
    fin_fail = Transcriber._finalize_and_write_transcript(
        turns=raw_turns,
        duration_sec=3.0,
        model_name="GigaAM v3",
        num_speakers=1,
        output_txt_path=out_llm_fail,
        custom_prompt=sa_prompt,
        llm_polish=True,
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-flash",
    )
    assert fin_fail["llm_applied"] is False
    assert "по gRPC" in fin_fail["turns"][0].text and "JSON" in fin_fail["turns"][0].text

    # When LLM is disabled, regex runs directly
    out_no_llm = str(tmp_path / "no_llm.txt")
    fin_no_llm = Transcriber._finalize_and_write_transcript(
        turns=raw_turns,
        duration_sec=3.0,
        model_name="GigaAM v3",
        num_speakers=1,
        output_txt_path=out_no_llm,
        custom_prompt=sa_prompt,
        llm_polish=False,
    )
    assert fin_no_llm["llm_applied"] is False
    assert "по gRPC" in fin_no_llm["turns"][0].text and "JSON" in fin_no_llm["turns"][0].text

    # 5. _build_polish_messages system prompt is domain-neutral when custom_prompt=""
    msgs = _build_polish_messages(
        batch_turns=[(0, "Спикер 1", "Привет, обсудим план.")],
        prev_context=[],
        custom_prompt="",
    )
    sys_prompt = msgs[0]["content"]
    for leaked_sa_example in ("Комунде", "План Юля", "в Джерри", "CRM-систему", "HR-специалиста"):
        assert leaked_sa_example not in sys_prompt


def test_v510_diarization_50min_4speaker_anti_explosion_and_direct_onnx_pyannote(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.transcriber import _DiarizationCache, decode_pyannote_powerset_windows

    # Part A: Anti-explosion on a 50-minute (~2900s) noisy 4-speaker Zoom recording
    # (340 clean turns with noisy CAM++ intra-speaker cosine ~0.55-0.62 + 15 short crosstalk outliers)
    # MUST converge to exactly 4 speakers (NOT 19!) under BOTH sensitivity='high' and 'normal'.
    rng = np.random.RandomState(5102026)
    dim = 192
    qr, _ = np.linalg.qr(rng.randn(dim, 8))
    local_cand = qr[:, 0].astype(np.float32)
    zoom_axis = qr[:, 1].astype(np.float32)
    # 3 remote speakers share a moderate codec axis (inter-speaker cos ~0.48)
    sh, ind = float(np.sqrt(0.48)), float(np.sqrt(0.52))
    centers = [local_cand] + [
        (sh * zoom_axis + ind * qr[:, 2 + k].astype(np.float32)) for k in range(3)
    ]
    centers = [c / np.linalg.norm(c) for c in centers]

    segs: list[DiarizationSegment] = []
    embs: list[np.ndarray] = []
    clean_indices: list[tuple[int, int]] = []
    t_cur = 0.0
    # 325 real speaker turns distributed across 4 speakers (160/65/55/45) + 15 noisy 2.5-3.5s crosstalk blips
    speaker_pool = [0] * 160 + [1] * 65 + [2] * 55 + [3] * 45
    rng.shuffle(speaker_pool)

    for idx, spk_i in enumerate(speaker_pool):
        dur = float(rng.uniform(2.8, 7.5))
        # High realistic intra-speaker variance (~0.56 mean intra-speaker cosine on Zoom)
        noise = rng.randn(dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
        v = 0.78 * centers[spk_i] + 0.62 * noise
        v /= np.linalg.norm(v)
        clean_indices.append((len(segs), spk_i))
        segs.append(DiarizationSegment(t_cur, t_cur + dur, speaker=len(segs)))
        embs.append(v)
        t_cur += dur + float(rng.uniform(0.6, 2.0))

        # Inject 15 crosstalk/laughter outlier segments (2.2-3.5s) spread throughout the call
        if idx % 22 == 10 and (len(segs) - len(clean_indices)) < 15:
            odur = float(rng.uniform(2.2, 3.4))
            other_spk = (spk_i + 1) % 4
            mix_noise = rng.randn(dim).astype(np.float32)
            mix_noise /= np.linalg.norm(mix_noise)
            ov = 0.45 * centers[spk_i] + 0.45 * centers[other_spk] + 0.77 * mix_noise
            ov /= np.linalg.norm(ov)
            segs.append(DiarizationSegment(t_cur, t_cur + odur, speaker=len(segs)))
            embs.append(ov)
            t_cur += odur + 0.5

    assert len(segs) == 340

    for sens in ("high", "normal"):
        clustered = cluster_anchor_embeddings(segs, embs, speaker_count="auto", sensitivity=sens)
        found_speakers = {s.speaker for s in clustered}
        assert len(found_speakers) == 4, (
            f"Expected exactly 4 speakers for sensitivity={sens}, got {len(found_speakers)}"
        )
        # Verify >=95% purity on clean turns (majority mapping from true speaker -> predicted label)
        correct = 0
        used_pred: set[int] = set()
        for true_spk in range(4):
            preds = [clustered[i].speaker for i, ts in clean_indices if ts == true_spk]
            best_pred = max(set(preds), key=preds.count)
            used_pred.add(best_pred)
            correct += preds.count(best_pred)
        assert len(used_pred) == 4
        assert correct / float(len(clean_indices)) >= 0.95

    # Part B1: Direct pyannote 7-class powerset decoder + overlap permutation stitching
    # 589 frames per 10s window, hop_frames=296 (~5s), rf_size=991, rf_shift=270
    n_frames = 589
    hop_frames = 296
    hop_samples = hop_frames * 270

    # Window 0: local speaker 0 speaks frames 50..500; local speaker 1 overlaps on frames 200..260 (class 4 = s0+s1)
    #           plus a tiny 4-frame (<0.18s) blip on local speaker 2 at frames 20..24 (must be dropped)
    logits_w0 = np.full((n_frames, 7), -5.0, dtype=np.float32)
    logits_w0[:, 0] = 5.0  # silence default
    logits_w0[20:24, 0] = -5.0
    logits_w0[20:24, 3] = 5.0  # 4 frames = ~0.07s blip -> dropped by min_duration_on=0.18
    logits_w0[50:200, 0] = -5.0
    logits_w0[50:200, 1] = 5.0  # s0 only
    logits_w0[200:260, 0] = -5.0
    logits_w0[200:260, 4] = 5.0  # s0 + s1 simultaneous overlap!
    logits_w0[260:500, 0] = -5.0
    logits_w0[260:500, 1] = 5.0  # s0 continues

    # Window 1 (shifted by hop_frames=296): pyannote swaps local slots!
    # Former s0 is now in local slot 1 (class 2) across overlap frames 0..204 AND continues to frame 520!
    logits_w1 = np.full((n_frames, 7), -5.0, dtype=np.float32)
    logits_w1[:, 0] = 5.0
    logits_w1[0:520, 0] = -5.0
    logits_w1[0:520, 2] = 5.0  # swapped local slot 1 -> MUST stitch to Window 0's s0 track!

    decoded = decode_pyannote_powerset_windows(
        window_outputs=[(0, logits_w0), (hop_samples, logits_w1)],
        hop_frames=hop_frames,
        sample_rate=16000,
        rf_size=991,
        rf_shift=270,
        total_dur_s=15.0,
        min_duration_on=0.18,
        min_duration_off=0.18,
        max_turn_sec=6.0,
    )
    # Exactly 2 stitched tracks: s0 (long stitched speech subdivided into <=6s parts) + s1 (overlap segment)
    # The tiny 4-frame blip on s2 MUST be dropped.
    track_ids = sorted({s.speaker for s in decoded})
    assert len(track_ids) == 2
    assert all((s.end - s.start) <= 6.01 for s in decoded)
    # Verify stitched s0 spans continuously from ~0.85s through ~13.7s
    main_track = max(track_ids, key=lambda tid: sum(s.end - s.start for s in decoded if s.speaker == tid))
    main_segs = [s for s in decoded if s.speaker == main_track]
    assert main_segs[0].start < 1.2 and main_segs[-1].end > 13.0
    assert len(main_segs) >= 3  # >12s continuous speech subdivided cleanly into <=6s turns

    # Part B2: Transcriber._run_diarization uses direct ONNX pyannote (NOT sherpa OfflineSpeakerDiarization)
    #          and falls back cleanly to sherpa OfflineSpeakerDiarization if ONNX raises.
    fake_seg_onnx = tmp_path / "pyannote_seg.onnx"
    fake_seg_onnx.write_bytes(b"fake_onnx_bytes")
    fake_emb_onnx = tmp_path / "campp_emb.onnx"
    fake_emb_onnx.write_bytes(b"fake_campp_bytes")

    sr = 16000
    wav_8s = rng.normal(0.0, 0.12, sr * 8).astype(np.float32)

    class _FakeSegSession:
        def __init__(self):
            self.calls = 0

        def get_modelmeta(self):
            return SimpleNamespace(
                custom_metadata_map={
                    "window_size": "160000",
                    "receptive_field_size": "991",
                    "receptive_field_shift": "270",
                }
            )

        def get_inputs(self):
            return [SimpleNamespace(name="x", shape=[1, 1, 160000])]

        def get_outputs(self):
            return [SimpleNamespace(name="y")]

        def run(self, out_names, feed):
            self.calls += 1
            out = np.full((1, 589, 7), -5.0, dtype=np.float32)
            out[0, :, 0] = 5.0
            out[0, 30:220, 0] = -5.0
            out[0, 30:220, 1] = 5.0  # Spk A: ~0.5s..3.7s
            out[0, 260:450, 0] = -5.0
            out[0, 260:450, 2] = 5.0  # Spk B: ~4.4s..7.6s
            return [out]

    v_a = qr[:, 0].astype(np.float32)
    v_b = qr[:, 1].astype(np.float32)
    embed_call_count = {"n": 0}

    class _FakeEmbStream:
        def __init__(self):
            self.wav = None

        def accept_waveform(self, s_rate, wav):
            self.wav = wav

        def input_finished(self):
            pass

    class _FakeEmbExtractor:
        def __init__(self, cfg):
            pass

        def create_stream(self):
            return _FakeEmbStream()

        def is_ready(self, stream):
            return True

        def compute(self, stream):
            embed_call_count["n"] += 1
            return v_a if embed_call_count["n"] % 2 == 1 else v_b

    fake_sess_inst = _FakeSegSession()
    monkeypatch.setattr(tr_mod, "resolve_onnx_provider", lambda req, status_cb=None: ("cpu", ["CPUExecutionProvider"]))
    monkeypatch.setattr(tr_mod.ort, "InferenceSession", lambda *a, **kw: fake_sess_inst)
    monkeypatch.setattr(tr_mod.sherpa_onnx, "SpeakerEmbeddingExtractor", _FakeEmbExtractor)
    monkeypatch.setattr(
        tr_mod.sherpa_onnx,
        "SpeakerEmbeddingExtractorConfig",
        lambda **kw: SimpleNamespace(**kw),
    )

    def _forbid_sherpa_sd(*args, **kwargs):
        raise AssertionError("sherpa_onnx.OfflineSpeakerDiarization MUST NOT run when direct ONNX pyannote succeeds")

    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeakerDiarization", _forbid_sherpa_sd)
    _DiarizationCache.active_key = None
    _DiarizationCache.active_pair = None

    direct_segs = Transcriber._run_diarization(
        samples=wav_8s,
        seg_model_path=str(fake_seg_onnx),
        emb_model_path=str(fake_emb_onnx),
        speaker_count="auto",
        threads=4,
        compute_device="cpu",
    )
    assert fake_sess_inst.calls == 1
    assert len(direct_segs) == 2
    assert len({s.speaker for s in direct_segs}) == 2
    assert embed_call_count["n"] == 2  # Exactly 1 CAM++ pass (zero redundant Pass-1 calls!)

    # Part B3: Fallback to sherpa_onnx.OfflineSpeakerDiarization when direct ONNX session raises
    fallback_used = {"called": False, "window_shift_ratio": None}

    def _boom_onnx(*args, **kwargs):
        raise RuntimeError("simulated ONNX load error")

    class _FakeSherpaSD:
        def __init__(self, cfg):
            fallback_used["called"] = True

        def process(self, wav, callback=None):
            return SimpleNamespace(
                sort_by_start_time=lambda: [
                    SimpleNamespace(start=0.5, end=3.5, speaker=0),
                    SimpleNamespace(start=4.2, end=7.4, speaker=1),
                ]
            )

    def _capture_pyannote_cfg(**kw):
        fallback_used["window_shift_ratio"] = kw.get("window_shift_ratio")
        return SimpleNamespace(**kw)

    monkeypatch.setattr(tr_mod.ort, "InferenceSession", _boom_onnx)
    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeakerDiarization", _FakeSherpaSD)
    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeakerDiarizationConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeakerSegmentationModelConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(tr_mod.sherpa_onnx, "OfflineSpeakerSegmentationPyannoteModelConfig", _capture_pyannote_cfg)
    monkeypatch.setattr(tr_mod.sherpa_onnx, "FastClusteringConfig", lambda **kw: SimpleNamespace(**kw))
    _DiarizationCache.active_key = None
    _DiarizationCache.active_pair = None

    fb_segs = Transcriber._run_diarization(
        samples=wav_8s,
        seg_model_path=str(fake_seg_onnx),
        emb_model_path=str(fake_emb_onnx),
        speaker_count="auto",
        threads=4,
        compute_device="cpu",
    )
    assert fallback_used["called"] is True
    assert fallback_used["window_shift_ratio"] == pytest.approx(0.25)
    assert len(fb_segs) == 2


def test_v511_overlap_pack_batched_pyannote_cloud_llm_concurrency_retry_and_deepgram_resilience(tmp_path, monkeypatch):
    import json as json_mod
    import threading
    import time as time_mod
    from types import SimpleNamespace
    import httpx
    import src.deepgram_asr as dg_mod
    import src.llm_postprocessor as llm_mod
    from src.config import ASR_PROMPT_PRESETS
    from src.deepgram_asr import transcribe_with_deepgram_nova3
    from src.llm_postprocessor import polish_turns_with_local_llm
    from src.transcriber import _DiarizationCache, build_longform_asr_chunks

    sr = 16000
    rng = np.random.RandomState(5112026)

    # 1. Chunk packing MUST merge nearby spans (<=0.55s) into one continuous island BEFORE max_len splitting,
    #    never hard-cutting at 19.6s with 0s overlap when two adjacent spans exceed 20s!
    speech_34 = rng.normal(0.0, 0.14, sr * 34).astype(np.float32)
    diar_adj = [
        DiarizationSegment(0.0, 19.6, speaker=0),
        DiarizationSegment(19.8, 33.8, speaker=0),
    ]
    chunks = build_longform_asr_chunks(
        speech_34,
        sample_rate=sr,
        diar_segments=diar_adj,
        max_chunk_sec=20.0,
        overlap_sec=1.6,
        use_silero_vad=False,
    )
    assert len(chunks) == 2
    (c1_s, c1_e, k1_s, k1_e), (c2_s, c2_e, k2_s, k2_e) = chunks
    assert (c1_e - c1_s) <= 20.01 and (c2_e - c2_s) <= 20.01
    # Full sliding overlap (~1.6s), NOT a 0s hard cut!
    assert (c1_e - c2_s) >= 1.4
    assert k1_e == pytest.approx(k2_s, abs=1e-4)

    # 2. Batched dynamic-dim ONNX pyannote + CUDA->CPU direct ONNX retry + overlapping <1.2s CAM++ bounds
    fake_seg = tmp_path / "pyannote_dyn.onnx"
    fake_seg.write_bytes(b"fake_dyn_onnx")
    fake_emb = tmp_path / "campp_dyn.onnx"
    fake_emb.write_bytes(b"fake_campp")

    wav_20 = rng.normal(0.0, 0.12, sr * 19).astype(np.float32)
    seen_batch_dims = []
    seen_providers = []

    class _DynamicBatchSegSession:
        def __init__(self, fail_cuda: bool):
            self.fail_cuda = fail_cuda

        def get_modelmeta(self):
            return SimpleNamespace(
                custom_metadata_map={
                    "window_size": "160000",
                    "receptive_field_size": "991",
                    "receptive_field_shift": "270",
                }
            )

        def get_inputs(self):
            return [SimpleNamespace(name="x", shape=["batch_size", 1, 160000])]

        def get_outputs(self):
            return [SimpleNamespace(name="y")]

        def run(self, out_names, feed):
            arr = feed["x"]
            bs = int(arr.shape[0])
            seen_batch_dims.append(bs)
            if self.fail_cuda:
                raise RuntimeError("CUDA OOM on pyannote")
            out = np.full((bs, 589, 7), -5.0, dtype=np.float32)
            out[:, :, 0] = 5.0
            # Long speaker 0 turn (frames 30..350) with a short 0.6s overlapping interjection
            # by speaker 1 (powerset class 4 = s0+s1) on frames 120..156 (~2.0s..2.6s),
            # plus a regular ~2.36s speaker 1 anchor turn on frames 380..520 (16.4s..18.8s in win 2)!
            out[:, 30:120, 0] = -5.0
            out[:, 30:120, 1] = 5.0
            out[:, 120:156, 0] = -5.0
            out[:, 120:156, 4] = 5.0
            out[:, 156:350, 0] = -5.0
            out[:, 156:350, 1] = 5.0
            out[:, 380:520, 0] = -5.0
            out[:, 380:520, 2] = 5.0
            return [out]

    qr, _ = np.linalg.qr(rng.randn(192, 3))
    v0, v1 = qr[:, 0].astype(np.float32), qr[:, 1].astype(np.float32)
    extracted_durs = []

    class _RecEmbStream:
        def __init__(self):
            self.dur = 0.0

        def accept_waveform(self, s_rate, wav):
            self.dur = len(wav) / float(s_rate)
            extracted_durs.append(self.dur)

        def input_finished(self):
            pass

    class _RecEmbExtractor:
        def __init__(self, cfg):
            pass

        def create_stream(self):
            return _RecEmbStream()

        def is_ready(self, stream):
            return True

        def compute(self, stream):
            # Speaker 1 owns both the 0.6s overlapping interjection and the ~2.36s anchor turn;
            # Speaker 0 owns the >3.2s long turns.
            return v1 if stream.dur < 2.8 else v0

    def _fake_ort_sess(path, sess_options=None, providers=None):
        seen_providers.append(list(providers or []))
        is_cuda = "CUDAExecutionProvider" in (providers or [])
        return _DynamicBatchSegSession(fail_cuda=is_cuda)

    monkeypatch.setattr(
        tr_mod,
        "resolve_onnx_provider",
        lambda req, status_cb=None: ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    monkeypatch.setattr(tr_mod.ort, "InferenceSession", _fake_ort_sess)
    monkeypatch.setattr(tr_mod.sherpa_onnx, "SpeakerEmbeddingExtractor", _RecEmbExtractor)
    monkeypatch.setattr(tr_mod.sherpa_onnx, "SpeakerEmbeddingExtractorConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        tr_mod.sherpa_onnx,
        "OfflineSpeakerDiarization",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("sherpa SD must NOT run when CPU direct ONNX succeeds")),
    )
    _DiarizationCache.active_key = None
    _DiarizationCache.active_pair = None

    diar_out = Transcriber._run_diarization(
        samples=wav_20,
        seg_model_path=str(fake_seg),
        emb_model_path=str(fake_emb),
        speaker_count="auto",
        threads=4,
        compute_device="cuda",
    )
    # Tried CUDA first, retried direct 1-pass ONNX on CPU cleanly (never touched sherpa SD!)
    assert seen_providers == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    # Batched ONNX execution was actually used (3 windows on 20s audio -> batch_size=3)
    assert any(b > 1 for b in seen_batch_dims)
    # Overlapping <1.2s interjections were NOT inverted to empty slices — both speakers extracted & clustered!
    assert any(0.3 <= d <= 1.1 for d in extracted_durs)
    assert len({s.speaker for s in diar_out}) == 2

    # 3. Cloud LLM bounded concurrency + 429 retry + step-wise 400 fallback + partial-batch regex rescue
    monkeypatch.setattr(llm_mod, "_sleep_with_cancel", lambda *a, **kw: None)
    turns_8 = [
        SpeakerTurn(float(i * 4), float(i * 4 + 3), "Спикер 1", f"Реплика {i}: работаем поджирописью и передаём Geson.")
        for i in range(8)
    ]

    lock = threading.Lock()
    active_cnt = {"cur": 0, "peak": 0}
    seen_bodies = []
    batch0_400_fired = {"done": False}
    rate_limit_fired = {"done": False}

    class _ConcurrentCloudClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            body = dict(json or {})
            payload = json_mod.loads(body["messages"][1]["content"])
            batch_ids = [item["id"] for item in payload["turns"]]
            with lock:
                seen_bodies.append((batch_ids, dict(body)))
                # Batch 0 rejects response_format with HTTP 400 -> must strip ONLY response_format, keeping thinking=disabled!
                if batch_ids[0] == 0 and "response_format" in body and not batch0_400_fired["done"]:
                    batch0_400_fired["done"] = True
                    req = httpx.Request("POST", url)
                    raise httpx.HTTPStatusError("400", request=req, response=httpx.Response(400, request=req))
                # Batch 1 hits transient HTTP 429 once -> must retry and succeed!
                if batch_ids[0] == 2 and not rate_limit_fired["done"]:
                    rate_limit_fired["done"] = True
                    req = httpx.Request("POST", url)
                    raise httpx.HTTPStatusError(
                        "429", request=req, response=httpx.Response(429, headers={"Retry-After": "1"}, request=req)
                    )
                active_cnt["cur"] += 1
                active_cnt["peak"] = max(active_cnt["peak"], active_cnt["cur"])

            try:
                if batch_ids[0] >= 2:
                    time_mod.sleep(0.05)
                # Simulate batch 3 (ids 6,7) returning invalid JSON so partial-batch regex rescue is tested!
                if batch_ids[0] == 6:
                    content = "not valid json"
                else:
                    content = json_mod.dumps(
                        {
                            "turns": [
                                {"id": tid, "text": f"Реплика {tid}: работаем по gRPC (LLM) и передаём Geson."}
                                for tid in batch_ids
                            ]
                        },
                        ensure_ascii=False,
                    )
                return SimpleNamespace(
                    raise_for_status=lambda: None,
                    json=lambda c=content: {"choices": [{"finish_reason": "stop", "message": {"content": c}}]},
                )
            finally:
                with lock:
                    active_cnt["cur"] -= 1

    monkeypatch.setattr(llm_mod, "_create_httpx_client", lambda endpoint, timeout_sec: _ConcurrentCloudClient())

    sa_prompt = ASR_PROMPT_PRESETS["system_analyst"]["prompt"]
    out_txt = str(tmp_path / "partial_llm.txt")
    fin_partial = Transcriber._finalize_and_write_transcript(
        turns=turns_8,
        duration_sec=32.0,
        model_name="GigaAM v3",
        num_speakers=1,
        output_txt_path=out_txt,
        custom_prompt=sa_prompt,
        llm_polish=True,
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-flash",
        llm_api_key="sk-test-conc",
    )
    # 1) Probed batch 0 stripped ONLY response_format while keeping thinking={"type": "disabled"}!
    b0_second = [b for ids, b in seen_bodies if ids[0] == 0][1]
    assert "response_format" not in b0_second
    assert b0_second["thinking"] == {"type": "disabled"}
    assert fin_partial["llm_applied"] is True
    assert fin_partial["raw_transcript_path"] is not None
    assert os.path.isfile(fin_partial["raw_transcript_path"])

    # 2) Explicit multi-batch concurrency + partial failed_indices check with batch_size=2 (4 batches)
    active_cnt["cur"] = 0
    active_cnt["peak"] = 0
    batch0_400_fired["done"] = False
    rate_limit_fired["done"] = False
    seen_bodies.clear()
    polished_4b, stats_4b = polish_turns_with_local_llm(
        turns=turns_8,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        api_key="sk-test-conc",
        custom_prompt=sa_prompt,
        batch_size=2,
    )
    assert active_cnt["peak"] >= 2
    assert stats_4b["applied"] is True
    assert stats_4b["batches_ok"] == 3
    assert stats_4b["batches_total"] == 4
    assert stats_4b["failed_indices"] == [6, 7]
    assert "по gRPC (LLM)" in polished_4b[0].text
    assert "по gRPC (LLM)" in polished_4b[2].text  # Batch 1 recovered after 429 retry!
    assert polished_4b[6].text == turns_8[6].text  # Batch 3 raw preserved by LLM layer

    # Verify _finalize_and_write_transcript rescues ONLY failed_indices [6, 7] with regex glossary
    monkeypatch.setattr(
        tr_mod,
        "polish_turns_with_local_llm",
        lambda turns, **kw: (polished_4b, stats_4b),
    )
    fin_rescue = Transcriber._finalize_and_write_transcript(
        turns=turns_8,
        duration_sec=32.0,
        model_name="GigaAM v3",
        num_speakers=1,
        output_txt_path=out_txt,
        custom_prompt=sa_prompt,
        llm_polish=True,
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-flash",
    )
    # Turn 0 (LLM succeeded): regex did NOT touch 'Geson'
    assert "Geson" in fin_rescue["turns"][0].text and "(LLM)" in fin_rescue["turns"][0].text
    # Turn 6 (LLM batch failed): regex rescued 'поджирописью' -> 'по gRPC' and 'Geson' -> 'JSON'!
    assert "по gRPC" in fin_rescue["turns"][6].text and "JSON" in fin_rescue["turns"][6].text

    # Re-running with llm_polish=False MUST remove stale _raw.txt backup
    raw_backup_file = fin_rescue["raw_transcript_path"]
    assert raw_backup_file and os.path.isfile(raw_backup_file)
    Transcriber._finalize_and_write_transcript(
        turns=turns_8,
        duration_sec=32.0,
        model_name="GigaAM v3",
        num_speakers=1,
        output_txt_path=out_txt,
        custom_prompt=sa_prompt,
        llm_polish=False,
    )
    assert not os.path.exists(raw_backup_file)

    # 4. Deepgram Nova-3 transient HTTP 503 retry recovers cleanly
    monkeypatch.setattr(dg_mod, "_sleep_with_cancel", lambda *a, **kw: None)
    dg_wav = str(tmp_path / "dg_503.wav")
    Transcriber._write_wav_16k_mono(dg_wav, rng.normal(0.0, 0.1, sr * 2).astype(np.float32), sample_rate=sr)
    dg_attempts = {"n": 0}

    class _Transient503Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, params=None, headers=None, content=None):
            dg_attempts["n"] += 1
            if dg_attempts["n"] == 1:
                req = httpx.Request("POST", url)
                return SimpleNamespace(
                    raise_for_status=lambda: (_ for _ in ()).throw(
                        httpx.HTTPStatusError("503", request=req, response=httpx.Response(503, request=req))
                    )
                )
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "results": {
                        "channels": [
                            {
                                "alternatives": [
                                    {
                                        "words": [
                                            {"word": "ok", "punctuated_word": "Ок.", "start": 0.2, "end": 0.6, "speaker": 0}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                },
            )

    monkeypatch.setattr(dg_mod, "_create_deepgram_httpx_client", lambda timeout_sec: _Transient503Client())
    dg_units, dg_diar = transcribe_with_deepgram_nova3(wav_path=dg_wav, api_key="dg-test-key", language="ru")
    assert dg_attempts["n"] == 2
    assert len(dg_units) == 1 and dg_units[0].text.strip() == "Ок."
    assert len(dg_diar) == 1


