import bisect
from dataclasses import dataclass
import gc
import math
import os
import threading
import time
from typing import Callable, Optional
import wave

import ctranslate2
import numpy as np
import sherpa_onnx
from faster_whisper import WhisperModel

from src.config import (
    ASR_MODEL_PROFILES,
    DEFAULT_ASR_MODEL,
    DIARIZATION_CLUSTER_THRESHOLD,
    DIARIZATION_MERGE_GAP_SEC,
    DIARIZATION_MIN_SEGMENT_SEC,
    DIARIZATION_MULTI_SPEAKER_THRESHOLD,
    TRANSCRIPT_PAUSE_BREAK_SEC,
)
from src.model_manager import ModelManager


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: int


@dataclass
class TimedUnit:
    start: float
    end: float
    text: str


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker_label: str
    text: str


def format_timestamp(seconds: float) -> str:
    total_sec = max(0, int(round(seconds)))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def load_wav_16k_mono(wav_path: str, normalize_gain: bool = True) -> tuple[np.ndarray, int]:
    """
    Loads a 16 kHz PCM s16le WAV file into a 1-D float32 numpy array in [-1, 1].
    Applies zero-phase peak gain normalization on quiet recordings to avoid
    timestamp drift while improving VAD, diarization, and ASR sensitivity.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw_bytes = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV (sampwidth=2), got {sampwidth}")

    if nframes == 0 or len(raw_bytes) == 0:
        return np.zeros(0, dtype=np.float32), sample_rate

    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    if normalize_gain and pcm.size > 0:
        peak = float(np.max(np.abs(pcm)))
        if 0.005 < peak < 0.5:
            gain = min(10.0, 0.9 / peak)
            pcm = np.clip(pcm * gain, -1.0, 1.0)

    return np.ascontiguousarray(pcm, dtype=np.float32), sample_rate


def smooth_diarization_segments(
    segments: list[DiarizationSegment],
    merge_gap_sec: float = DIARIZATION_MERGE_GAP_SEC,
    min_segment_sec: float = DIARIZATION_MIN_SEGMENT_SEC,
) -> list[DiarizationSegment]:
    """
    Merges adjacent same-speaker segments across short pauses and absorbs
    isolated micro-glitches sandwiched between the same speaker.
    """
    valid = [
        DiarizationSegment(float(s.start), float(s.end), int(s.speaker))
        for s in segments
        if s.end > s.start
    ]
    if not valid:
        return []

    valid.sort(key=lambda s: (s.start, s.end))

    def _merge_same_speaker(items: list[DiarizationSegment]) -> list[DiarizationSegment]:
        if not items:
            return []
        merged = [DiarizationSegment(items[0].start, items[0].end, items[0].speaker)]
        for cur in items[1:]:
            prev = merged[-1]
            if cur.speaker == prev.speaker and (cur.start - prev.end) <= merge_gap_sec:
                prev.end = max(prev.end, cur.end)
            else:
                merged.append(DiarizationSegment(cur.start, cur.end, cur.speaker))
        return merged

    pass1 = _merge_same_speaker(valid)

    # Absorb tiny interruptions surrounded by the same speaker, or drop isolated noise blips
    cleaned: list[DiarizationSegment] = []
    for idx, seg in enumerate(pass1):
        dur = seg.end - seg.start
        if dur < min_segment_sec and len(pass1) > 1:
            prev_seg = cleaned[-1] if cleaned else None
            next_seg = pass1[idx + 1] if idx + 1 < len(pass1) else None
            if (
                prev_seg is not None
                and next_seg is not None
                and prev_seg.speaker == next_seg.speaker
                and (seg.start - prev_seg.end) <= merge_gap_sec
                and (next_seg.start - seg.end) <= merge_gap_sec
            ):
                prev_seg.end = max(prev_seg.end, seg.end)
                continue
        cleaned.append(seg)

    return _merge_same_speaker(cleaned)


def split_audio_ranges_at_silence(
    samples: np.ndarray,
    sample_rate: int,
    diar_segments: list[DiarizationSegment],
    max_chunk_sec: float = 18.0,
) -> list[tuple[float, float]]:
    """
    Produces non-overlapping speech windows <= max_chunk_sec aligned to speaker
    boundaries or local RMS energy minima (silences/breaths). Essential for
    non-streaming transducer models like Parakeet-TDT-0.6B-v3.
    """
    total_dur = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    if total_dur <= 0.0:
        return []

    def _find_quiet_split(start_s: float, target_s: float, hard_end_s: float) -> float:
        search_lo = max(start_s + 4.0, target_s - 4.0)
        search_hi = min(hard_end_s - 1.0, target_s + 2.0)
        if search_hi <= search_lo:
            return min(target_s, hard_end_s)

        win_samples = max(1, int(0.05 * sample_rate))  # 50 ms frame
        lo_idx = max(0, int(search_lo * sample_rate))
        hi_idx = min(len(samples), int(search_hi * sample_rate))
        if hi_idx - lo_idx <= win_samples:
            return min(target_s, hard_end_s)

        best_idx = int(target_s * sample_rate)
        best_rms = float("inf")
        for pos in range(lo_idx, hi_idx - win_samples + 1, win_samples):
            frame = samples[pos : pos + win_samples]
            rms = float(np.dot(frame, frame))
            if rms < best_rms:
                best_rms = rms
                best_idx = pos + (win_samples // 2)
        return best_idx / float(sample_rate)

    def _subdivide_long_span(s_sec: float, e_sec: float) -> list[tuple[float, float]]:
        spans = []
        cur = s_sec
        while (e_sec - cur) > max_chunk_sec:
            split_pt = _find_quiet_split(cur, cur + max_chunk_sec - 2.0, e_sec)
            if split_pt <= cur + 0.5:
                split_pt = cur + max_chunk_sec
            spans.append((cur, min(split_pt, e_sec)))
            cur = split_pt
        if e_sec - cur > 0.05:
            spans.append((cur, e_sec))
        return spans

    if not diar_segments:
        return _subdivide_long_span(0.0, total_dur)

    # Pad and merge nearby speech islands up to max_chunk_sec
    padded: list[tuple[float, float]] = []
    for seg in diar_segments:
        s = max(0.0, seg.start - 0.15)
        e = min(total_dur, seg.end + 0.15)
        if not padded:
            padded.append((s, e))
        else:
            prev_s, prev_e = padded[-1]
            if s <= prev_e + 0.4 and (e - prev_s) <= max_chunk_sec:
                padded[-1] = (prev_s, max(prev_e, e))
            else:
                # Prevent slight overlap from padding
                s = max(s, prev_e)
                if e > s + 0.05:
                    padded.append((s, e))

    final_ranges: list[tuple[float, float]] = []
    for s, e in padded:
        final_ranges.extend(_subdivide_long_span(s, e))
    return final_ranges


def build_speaker_asr_windows(
    diar_segments: list[DiarizationSegment],
    total_dur: float,
    max_same_speaker_sec: float = 28.0,
    merge_gap_sec: float = 1.2,
    pad_sec: float = 0.2,
) -> list[tuple[float, float]]:
    """
    Groups consecutive same-speaker diarization segments into non-overlapping
    ASR windows without ever merging across speaker changes. Ensures Whisper
    performs fresh language detection per speaker turn when interlocutors speak
    different languages inside the same 30-second interval.
    """
    if not diar_segments or total_dur <= 0.0:
        return []

    ordered = sorted(
        (s for s in diar_segments if s.end > s.start),
        key=lambda s: (s.start, s.end),
    )
    if not ordered:
        return []

    grouped: list[list[float | int]] = [[ordered[0].start, ordered[0].end, ordered[0].speaker]]
    for seg in ordered[1:]:
        prev_start, prev_end, prev_spk = grouped[-1]
        if (
            seg.speaker == prev_spk
            and (seg.start - prev_end) <= merge_gap_sec
            and (seg.end - prev_start) <= max_same_speaker_sec
        ):
            grouped[-1][1] = max(prev_end, seg.end)
        else:
            grouped.append([seg.start, seg.end, seg.speaker])

    windows: list[tuple[float, float]] = []
    for i, (g_start, g_end, _) in enumerate(grouped):
        if i == 0:
            w_start = max(0.0, g_start - pad_sec)
        else:
            prev_end = grouped[i - 1][1]
            mid = 0.5 * (prev_end + g_start)
            cand_start = mid if (g_start - prev_end) < 2 * pad_sec else (g_start - pad_sec)
            w_start = max(windows[-1][1] if windows else 0.0, cand_start)

        if i == len(grouped) - 1:
            w_end = min(total_dur, g_end + pad_sec)
        else:
            next_start = grouped[i + 1][0]
            mid = 0.5 * (g_end + next_start)
            cand_end = mid if (next_start - g_end) < 2 * pad_sec else (g_end + pad_sec)
            w_end = min(total_dur, cand_end)

        if w_end - w_start >= 0.1:
            windows.append((w_start, w_end))

    return windows


def group_bpe_tokens_to_units(
    tokens: list[str],
    timestamps: list[float],
    offset_sec: float,
    chunk_end_sec: float,
) -> list[TimedUnit]:
    """
    Groups SentencePiece / BPE subword tokens and per-token timestamps into
    word-level TimedUnits on the global audio timeline.
    """
    units: list[TimedUnit] = []
    cur_text = ""
    cur_start = None
    cur_last_t = None

    for tok, rel_t in zip(tokens, timestamps):
        if not tok:
            continue
        abs_t = offset_sec + max(0.0, float(rel_t))
        starts_new_word = tok.startswith(" ") or tok.startswith("▁")
        clean_piece = tok.replace("▁", " ")

        if starts_new_word and cur_text.strip():
            word_end = min(chunk_end_sec, max(cur_start + 0.08, abs_t))
            units.append(TimedUnit(start=cur_start, end=word_end, text=cur_text))
            cur_text = clean_piece
            cur_start = abs_t
            cur_last_t = abs_t
        else:
            if cur_start is None:
                cur_start = abs_t
            cur_text += clean_piece
            cur_last_t = abs_t

    if cur_text.strip() and cur_start is not None:
        last_end = min(chunk_end_sec, (cur_last_t or cur_start) + 0.25)
        units.append(TimedUnit(start=cur_start, end=max(cur_start + 0.08, last_end), text=cur_text))

    return units


def _join_unit_texts(pieces: list[str]) -> str:
    if not pieces:
        return ""
    raw = "".join(
        p if (idx == 0 or p.startswith(" ")) else f" {p}"
        for idx, p in enumerate(pieces)
    )
    # Normalize multiple spaces and spaces before punctuation
    compact = " ".join(raw.split())
    for punct in (".", ",", "!", "?", ":", ";"):
        compact = compact.replace(f" {punct}", punct)
    return compact.strip()


def align_units_to_speakers(
    units: list[TimedUnit],
    diar_segments: list[DiarizationSegment],
    pause_break_sec: float = TRANSCRIPT_PAUSE_BREAK_SEC,
) -> tuple[list[SpeakerTurn], int]:
    """
    Assigns each TimedUnit to the speaker with maximum temporal overlap (or
    nearest speaker boundary within 1.5s if inside a micro-gap), normalizes
    speaker IDs chronologically (Спикер 1, Спикер 2, ...), and merges contiguous
    same-speaker units into readable dialogue turns.
    """
    non_empty_units = [u for u in units if u.text and u.text.strip()]
    if not non_empty_units:
        return [], 0

    if not diar_segments:
        # Fallback if speech was recognized on a very short clip with 0 diarized clusters
        merged_turns: list[SpeakerTurn] = []
        buf_texts: list[str] = []
        t_start = non_empty_units[0].start
        t_end = non_empty_units[0].end
        for u in non_empty_units:
            if buf_texts and (u.start - t_end) > pause_break_sec:
                merged_turns.append(
                    SpeakerTurn(t_start, t_end, "Спикер 1", _join_unit_texts(buf_texts))
                )
                buf_texts = [u.text]
                t_start = u.start
                t_end = u.end
            else:
                buf_texts.append(u.text)
                t_end = max(t_end, u.end)
        if buf_texts:
            merged_turns.append(
                SpeakerTurn(t_start, t_end, "Спикер 1", _join_unit_texts(buf_texts))
            )
        return merged_turns, 1

    seg_ends = [s.end for s in diar_segments]
    prev_spk: Optional[int] = None
    assigned: list[tuple[TimedUnit, int]] = []

    for u in non_empty_units:
        u_start = float(u.start)
        u_end = max(u_start + 0.02, float(u.end))

        # Binary search first segment whose end > u_start - 1.5
        idx = max(0, bisect.bisect_left(seg_ends, u_start - 1.5) - 1)
        overlap_by_spk: dict[int, float] = {}
        nearest_dist = float("inf")
        nearest_spk = diar_segments[0].speaker

        for j in range(idx, len(diar_segments)):
            seg = diar_segments[j]
            if seg.start > u_end + 1.5:
                break

            ov = min(u_end, seg.end) - max(u_start, seg.start)
            if ov > 0.0:
                overlap_by_spk[seg.speaker] = overlap_by_spk.get(seg.speaker, 0.0) + ov
            else:
                dist = min(abs(u_start - seg.end), abs(seg.start - u_end))
                if dist < nearest_dist or (
                    math.isclose(dist, nearest_dist, abs_tol=0.05) and seg.speaker == prev_spk
                ):
                    nearest_dist = dist
                    nearest_spk = seg.speaker

        if overlap_by_spk:
            chosen_spk = max(
                overlap_by_spk.items(),
                key=lambda kv: (kv[1], 1 if kv[0] == prev_spk else 0),
            )[0]
        elif nearest_dist <= 1.5:
            chosen_spk = nearest_spk
        elif prev_spk is not None:
            chosen_spk = prev_spk
        else:
            # Pick globally closest diarization segment
            closest_seg = min(
                diar_segments,
                key=lambda s: min(abs(u_start - s.end), abs(s.start - u_end)),
            )
            chosen_spk = closest_seg.speaker

        prev_spk = chosen_spk
        assigned.append((u, chosen_spk))

    # Chronological speaker mapping: first speaker heard -> Спикер 1, etc.
    speaker_map: dict[int, str] = {}
    for _, raw_spk in assigned:
        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"Спикер {len(speaker_map) + 1}"

    turns: list[SpeakerTurn] = []
    cur_spk_raw = assigned[0][1]
    cur_start = assigned[0][0].start
    cur_end = assigned[0][0].end
    cur_pieces: list[str] = [assigned[0][0].text]

    for u, raw_spk in assigned[1:]:
        gap = u.start - cur_end
        if raw_spk != cur_spk_raw or gap > pause_break_sec:
            text_out = _join_unit_texts(cur_pieces)
            if text_out:
                turns.append(
                    SpeakerTurn(
                        start=cur_start,
                        end=cur_end,
                        speaker_label=speaker_map[cur_spk_raw],
                        text=text_out,
                    )
                )
            cur_spk_raw = raw_spk
            cur_start = u.start
            cur_end = max(u.start, u.end)
            cur_pieces = [u.text]
        else:
            cur_pieces.append(u.text)
            cur_end = max(cur_end, u.end)

    last_text = _join_unit_texts(cur_pieces)
    if last_text:
        turns.append(
            SpeakerTurn(
                start=cur_start,
                end=cur_end,
                speaker_label=speaker_map[cur_spk_raw],
                text=last_text,
            )
        )

    return turns, len(speaker_map)


def format_transcript_txt(
    turns: list[SpeakerTurn],
    duration_sec: float,
    model_name: str,
    num_speakers: int,
    note: Optional[str] = None,
) -> str:
    header_lines = [
        "# Транскрибация интервью (Interview Anonymizer v4.0.0)",
        f"# Длительность: {format_timestamp(duration_sec)} | Модель: {model_name} | Обнаружено спикеров: {num_speakers}",
        "# " + ("=" * 72),
        "",
    ]
    if note:
        header_lines.append(note)
        header_lines.append("")

    if not turns:
        if not note:
            header_lines.append("[Речь в аудиодорожке не обнаружена]")
        return "\n".join(header_lines).rstrip() + "\n"

    body_lines = [
        f"[{format_timestamp(t.start)} - {format_timestamp(t.end)}] {t.speaker_label}: {t.text}"
        for t in turns
    ]
    return "\n".join(header_lines) + "\n\n".join(body_lines) + "\n"


class _ModelCache:
    """
    Holds at most one warm ASR engine in process memory to avoid redundant model
    reloads across sequential uploads while preventing multi-model RAM bloat.
    """

    lock = threading.Lock()
    active_key: Optional[tuple] = None
    active_engine = None

    @classmethod
    def get_or_create(cls, key: tuple, factory: Callable):
        with cls.lock:
            if cls.active_key == key and cls.active_engine is not None:
                return cls.active_engine
            if cls.active_engine is not None:
                cls.active_engine = None
                cls.active_key = None
                gc.collect()
            cls.active_engine = factory()
            cls.active_key = key
            return cls.active_engine


def _detect_ct2_device() -> tuple[str, str]:
    forced_device = os.environ.get("ASR_DEVICE", "").strip().lower()
    if forced_device in ("cpu", "cuda"):
        device = forced_device
    else:
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    compute_type = "int8_float16" if device == "cuda" else "int8"
    return device, compute_type


class Transcriber:
    """
    100% local speaker-diarized multilingual speech transcription pipeline.
    """

    @staticmethod
    def _run_diarization(
        samples: np.ndarray,
        seg_model_path: str,
        emb_model_path: str,
        speaker_count: str = "auto",
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[DiarizationSegment]:
        if samples.size < 1600:  # < 0.1s
            return []

        diar_threads = max(1, min(int(threads), 8))
        mode = (speaker_count or "auto").strip().lower()

        if mode in ("2", "3"):
            num_clusters = int(mode)
            threshold = DIARIZATION_CLUSTER_THRESHOLD
        elif mode == "4+":
            num_clusters = -1
            threshold = DIARIZATION_MULTI_SPEAKER_THRESHOLD
        else:
            num_clusters = -1
            threshold = DIARIZATION_CLUSTER_THRESHOLD

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=seg_model_path
                ),
                num_threads=diar_threads,
                provider="cpu",
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=emb_model_path,
                num_threads=diar_threads,
                provider="cpu",
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters,
                threshold=threshold,
            ),
            min_duration_on=0.25,
            min_duration_off=0.45,
        )

        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        last_pct = -15
        last_log_t = 0.0

        def _progress_cb(done_chunks: int, total_chunks: int) -> int:
            nonlocal last_pct, last_log_t
            if status_cb and total_chunks > 0:
                pct = int((done_chunks * 100) / total_chunks)
                now = time.time()
                if pct >= last_pct + 25 and (now - last_log_t) >= 2.0:
                    last_pct = pct
                    last_log_t = now
                    status_cb(f"Диаризация спикеров: {pct}%...")
            return 0

        result = sd.process(samples, callback=_progress_cb)
        raw_objs = result.sort_by_start_time()
        raw_segments = [
            DiarizationSegment(start=float(s.start), end=float(s.end), speaker=int(s.speaker))
            for s in raw_objs
        ]
        return smooth_diarization_segments(raw_segments)

    @staticmethod
    def _run_faster_whisper(
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        profile_id: str,
        diar_segments: Optional[list[DiarizationSegment]] = None,
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[TimedUnit]:
        device, compute_type = _detect_ct2_device()
        cpu_threads = max(1, int(threads)) if device == "cpu" else 2
        cache_key = ("faster_whisper", profile_id, model_dir, device, compute_type, cpu_threads)

        model: WhisperModel = _ModelCache.get_or_create(
            cache_key,
            lambda: WhisperModel(
                model_dir,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=1,
            ),
        )

        total_dur = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
        windows = build_speaker_asr_windows(diar_segments or [], total_dur)
        if not windows:
            windows = [(0.0, total_dur)]
            used_diar_windows = False
        else:
            used_diar_windows = True

        units: list[TimedUnit] = []
        last_pct = -15
        last_log_t = time.time()

        for w_start, w_end in windows:
            s_idx = max(0, int(w_start * sample_rate))
            e_idx = min(len(samples), int(w_end * sample_rate))
            if e_idx - s_idx < 800:
                continue

            win_dur = w_end - w_start
            use_vad = (not used_diar_windows) or (win_dur > 30.0)
            transcribe_kwargs = dict(
                language=None,
                task="transcribe",
                multilingual=True,
                word_timestamps=True,
                vad_filter=use_vad,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                beam_size=5,
            )
            if use_vad:
                transcribe_kwargs["vad_parameters"] = dict(
                    min_silence_duration_ms=400, speech_pad_ms=250
                )
                transcribe_kwargs["hallucination_silence_threshold"] = 2.0

            segments_iter, _ = model.transcribe(samples[s_idx:e_idx], **transcribe_kwargs)

            for seg in segments_iter:
                abs_seg_end = min(w_end, w_start + float(seg.end))
                if seg.words:
                    for w in seg.words:
                        if w.word and w.word.strip():
                            units.append(
                                TimedUnit(
                                    start=w_start + float(w.start),
                                    end=min(w_end, w_start + float(w.end)),
                                    text=w.word,
                                )
                            )
                elif seg.text and seg.text.strip():
                    units.append(
                        TimedUnit(
                            start=w_start + float(seg.start),
                            end=abs_seg_end,
                            text=seg.text,
                        )
                    )

                if status_cb and total_dur > 0:
                    pct = min(99, int((abs_seg_end / total_dur) * 100))
                    now = time.time()
                    if pct >= last_pct + 15 and (now - last_log_t) >= 2.0:
                        last_pct = pct
                        last_log_t = now
                        status_cb(
                            f"Распознавание речи: {pct}% ({format_timestamp(abs_seg_end)} / {format_timestamp(total_dur)})..."
                        )

        return units

    @staticmethod
    def _run_sherpa_parakeet(
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        diar_segments: list[DiarizationSegment],
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[TimedUnit]:
        asr_threads = max(1, int(threads))
        cache_key = ("sherpa_parakeet", model_dir, asr_threads)

        recognizer = _ModelCache.get_or_create(
            cache_key,
            lambda: sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=os.path.join(model_dir, "encoder.int8.onnx"),
                decoder=os.path.join(model_dir, "decoder.int8.onnx"),
                joiner=os.path.join(model_dir, "joiner.int8.onnx"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=asr_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
                provider="cpu",
            ),
        )

        chunks = split_audio_ranges_at_silence(
            samples, sample_rate, diar_segments, max_chunk_sec=18.0
        )
        if not chunks:
            return []

        units: list[TimedUnit] = []
        total_chunks = len(chunks)
        last_pct = -20
        last_log_t = time.time()

        for idx, (c_start, c_end) in enumerate(chunks, start=1):
            s_idx = max(0, int(c_start * sample_rate))
            e_idx = min(len(samples), int(c_end * sample_rate))
            if e_idx - s_idx < 800:
                continue

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples[s_idx:e_idx])
            recognizer.decode_stream(stream)
            res = stream.result

            if res.tokens and res.timestamps and len(res.tokens) == len(res.timestamps):
                units.extend(
                    group_bpe_tokens_to_units(
                        list(res.tokens), list(res.timestamps), c_start, c_end
                    )
                )
            elif res.text and res.text.strip():
                units.append(TimedUnit(start=c_start, end=c_end, text=res.text.strip()))

            if status_cb and total_chunks > 0:
                pct = min(99, int((idx * 100) / total_chunks))
                now = time.time()
                if pct >= last_pct + 20 and (now - last_log_t) >= 2.0:
                    last_pct = pct
                    last_log_t = now
                    status_cb(f"Распознавание речи (Parakeet): {pct}% ({idx}/{total_chunks})...")

        return units

    @classmethod
    def transcribe_audio_file(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = DEFAULT_ASR_MODEL,
        speaker_count: str = "auto",
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Runs local speaker diarization + multilingual ASR on a 16 kHz mono WAV
        file and writes the formatted role-split transcript to output_txt_path.
        """
        def update(msg: str):
            if status_cb:
                status_cb(msg)
            else:
                print(msg)

        profile_id = (asr_model or DEFAULT_ASR_MODEL).strip().lower()
        if profile_id not in ASR_MODEL_PROFILES:
            raise ValueError(f"Unsupported ASR profile: {profile_id}")
        profile = ASR_MODEL_PROFILES[profile_id]

        t0 = time.time()
        samples, sample_rate = load_wav_16k_mono(wav_path, normalize_gain=True)
        duration_sec = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0

        os.makedirs(os.path.dirname(os.path.abspath(output_txt_path)), exist_ok=True)

        if samples.size < 1600:
            txt = format_transcript_txt(
                turns=[],
                duration_sec=duration_sec,
                model_name=profile["short_name"],
                num_speakers=0,
                note="[Речь в аудиодорожке не обнаружена]",
            )
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(txt)
            return {"num_speakers": 0, "num_turns": 0, "elapsed": time.time() - t0}

        # 1. Ensure local models are cached & verified
        seg_onnx, emb_onnx = ModelManager.ensure_diarization_models(status_cb=update)
        asr_dir = ModelManager.ensure_asr_model(profile_id, status_cb=update)

        # 2. Speaker Diarization
        update(f"Разделение говорящих по ролям (режим спикеров: {speaker_count})...")
        t_diar = time.time()
        diar_segments = cls._run_diarization(
            samples=samples,
            seg_model_path=seg_onnx,
            emb_model_path=emb_onnx,
            speaker_count=speaker_count,
            threads=threads,
            status_cb=update,
        )
        detected_raw_speakers = len({s.speaker for s in diar_segments})
        update(
            f"[Profile] Диаризация завершена за {time.time() - t_diar:.2f}s "
            f"(сегментов: {len(diar_segments)}, голосов: {detected_raw_speakers})"
        )

        # 3. Multilingual Speech Recognition
        update(f"Локальная транскрибация речи ({profile['short_name']})...")
        t_asr = time.time()
        if profile["engine"] == "faster_whisper":
            units = cls._run_faster_whisper(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                profile_id=profile_id,
                diar_segments=diar_segments,
                threads=threads,
                status_cb=update,
            )
        elif profile["engine"] == "sherpa_parakeet":
            units = cls._run_sherpa_parakeet(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                diar_segments=diar_segments,
                threads=threads,
                status_cb=update,
            )
        else:
            raise ValueError(f"Unsupported engine: {profile['engine']}")

        update(f"[Profile] ASR завершена за {time.time() - t_asr:.2f}s (единиц текста: {len(units)})")

        # 4. Align words/units to speaker roles and write UTF-8 .txt
        turns, final_speaker_count = align_units_to_speakers(units, diar_segments)
        transcript_text = format_transcript_txt(
            turns=turns,
            duration_sec=duration_sec,
            model_name=profile["short_name"],
            num_speakers=final_speaker_count,
        )

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        total_elapsed = time.time() - t0
        update(
            f"[Profile] Транскрипт сохранён ({len(turns)} реплик, спикеров: {final_speaker_count}) "
            f"за {total_elapsed:.2f}s"
        )
        return {
            "num_speakers": final_speaker_count,
            "num_turns": len(turns),
            "elapsed": total_elapsed,
        }
