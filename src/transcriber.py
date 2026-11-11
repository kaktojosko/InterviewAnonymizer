import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
import gc
import glob
import math
import os
import re
import site
import sys
import threading
import time
from typing import Callable, Optional
import wave

import ctranslate2
import numpy as np
import sherpa_onnx
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from src.cancellation import CancellationToken
from src.config import (
    ASR_MODEL_PROFILES,
    COMPUTE_DEVICE_OPTIONS,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    DIARIZATION_CENTROID_SIM_THRESHOLD,
    DIARIZATION_MERGE_GAP_SEC,
    DIARIZATION_MIN_SEGMENT_SEC,
    DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD,
    DIARIZATION_SEGMENTATION_THRESHOLD,
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


_TONE_WIN = 512
_TONE_HOP = 256
_TONE_HANN = np.hanning(_TONE_WIN).astype(np.float32)


def is_narrowband_tone(seg_samples: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Detects synthetic single- or dual-frequency call-waiting / ringback / busy
    tones (e.g. Zoom, Teams, Telemost, telephony 425 Hz / 440+480 Hz beeps) so
    they are excluded before speaker clustering and ASR.
    Vectorized over STFT frames via NumPy batched rfft for ~30x lower CPU overhead.
    """
    if sample_rate <= 0 or seg_samples.size < max(_TONE_WIN, int(0.18 * sample_rate)):
        return False

    freqs = np.fft.rfftfreq(_TONE_WIN, d=1.0 / float(sample_rate))
    band_mask = (freqs >= 150.0) & (freqs <= 3800.0)
    if int(np.count_nonzero(band_mask)) < 6:
        return False

    frames = np.lib.stride_tricks.sliding_window_view(seg_samples, _TONE_WIN)[::_TONE_HOP]
    if frames.shape[0] < 4:
        return False
    if frames.shape[0] > 512:
        frames = frames[:: max(1, frames.shape[0] // 256)]

    energy = np.mean(frames * frames, axis=1)
    active = frames[energy >= 1e-6]
    if active.shape[0] < 4:
        return False

    spec = (np.abs(np.fft.rfft(active * _TONE_HANN, axis=1)) ** 2)[:, band_mask]
    band_sums = np.sum(spec, axis=1)
    valid_mask = band_sums > 1e-10
    if int(np.count_nonzero(valid_mask)) < 4:
        return False

    spec = spec[valid_mask]
    band_sums = band_sums[valid_mask]
    top3 = np.sum(np.partition(spec, -3, axis=1)[:, -3:], axis=1)
    top3_ratios = top3 / band_sums
    geom_means = np.exp(np.mean(np.log(spec + 1e-12), axis=1))
    arith_means = np.mean(spec, axis=1) + 1e-12
    flatnesses = geom_means / arith_means

    return bool(np.median(top3_ratios) > 0.78 and np.median(flatnesses) < 0.02)



_WORD_CHARS_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё가-힣]")
_SYLLABLE_STUTTER_RE = re.compile(
    r"(?<!\w)(?:[A-Za-zА-Яа-яЁё가-힣]{1,3})(?:[\s,\-\.]+[A-Za-zА-Яа-яЁё가-힣]{1,3}){4,}(?!\w)"
)
_SYLLABLE_SPLIT_RE = re.compile(r"[\s,\-\.]+")


def is_repetitive_noise_token(token: str) -> bool:
    """
    Returns True for single-token Whisper tone/noise repetition loops such as
    '뚜루뚜뚜뚜뚜뚜뚜뚜뚜뚜뚜', 'Dootototototototo', or 'Дудудудудудуду'.
    """
    letters = "".join(_WORD_CHARS_RE.findall(token)).lower()
    if len(letters) >= 10 and len(set(letters)) <= 3:
        return True
    if len(letters) >= 12 and re.search(r"(.{1,3})\1{4,}", letters):
        return True
    return False


def clean_hallucinated_text(text: str) -> str:
    """
    Strips dial-tone / ringback repetition loops (e.g. 'Ту -ру -ту -ту -ту -ту...')
    while preserving real speech that may follow in the same segment and keeping
    natural expressions and punctuation ('Да, да, да!', 'Mm -hmm.') intact.
    """
    if not text or not text.strip():
        return ""

    t = text.strip()
    modified = False

    # 1. Remove 5+ syllable repetitive tone stutters (<=4 unique chars & <=3 distinct syllables)
    def _strip_stutter(match: re.Match) -> str:
        nonlocal modified
        chunk = match.group(0)
        parts = [p.lower() for p in _SYLLABLE_SPLIT_RE.split(chunk) if p]
        letters = "".join(parts)
        if len(parts) >= 5 and len(set(letters)) <= 4 and len(set(parts)) <= 3:
            modified = True
            return " "
        return chunk

    t = _SYLLABLE_STUTTER_RE.sub(_strip_stutter, t)

    # 2. Drop individual noise-loop tokens inside multi-word strings
    words = t.split()
    kept_words = [w for w in words if not is_repetitive_noise_token(w)]
    if len(kept_words) != len(words):
        modified = True
    if not kept_words:
        return ""

    if modified:
        t = " ".join(kept_words)
        t = re.sub(r"^[\s,\-\.:;!?]+", "", t)
        t = re.sub(r"[\s,\-:;]+$", "", t)
        if t in (".", "!", "?", "..."):
            return ""

    letters = "".join(_WORD_CHARS_RE.findall(t)).lower()
    if not letters:
        return ""
    if is_repetitive_noise_token(letters):
        return ""
    if len(letters) >= 16 and (len(set(letters)) / float(len(letters))) <= 0.16:
        return ""

    return t.strip()


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


def consolidate_minor_speakers(
    segments: list[DiarizationSegment],
    merge_gap_sec: float = DIARIZATION_MERGE_GAP_SEC,
) -> list[DiarizationSegment]:
    """
    Absorbs transient noise/laugh/interjection micro-clusters on long recordings
    (speakers accounting for <1% of speech and <6s total) into the temporally
    nearest primary speaker, preventing false speaker explosions in auto mode.
    """
    if len(segments) <= 2:
        return segments

    dur_by_spk: dict[int, float] = {}
    for s in segments:
        dur_by_spk[s.speaker] = dur_by_spk.get(s.speaker, 0.0) + max(0.0, s.end - s.start)

    total_speech = sum(dur_by_spk.values())
    if total_speech < 20.0 or len(dur_by_spk) <= 2:
        return segments

    min_major_dur = min(6.0, max(2.5, 0.01 * total_speech))
    major_spks = {spk for spk, dur in dur_by_spk.items() if dur >= min_major_dur}
    if not major_spks or len(major_spks) == len(dur_by_spk):
        return segments

    major_segments = [s for s in segments if s.speaker in major_spks]
    reassigned: list[DiarizationSegment] = []
    for s in segments:
        if s.speaker in major_spks:
            reassigned.append(s)
        else:
            nearest = min(
                major_segments,
                key=lambda m: min(abs(s.start - m.end), abs(m.start - s.end)),
            )
            reassigned.append(DiarizationSegment(s.start, s.end, nearest.speaker))

    return smooth_diarization_segments(reassigned, merge_gap_sec=merge_gap_sec)


def cluster_anchor_embeddings(
    segments: list[DiarizationSegment],
    embeddings: list[Optional[np.ndarray]],
    speaker_count: str = "auto",
    sim_threshold: float = DIARIZATION_CENTROID_SIM_THRESHOLD,
    multi_sim_threshold: float = DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD,
    merge_gap_sec: float = DIARIZATION_MERGE_GAP_SEC,
) -> list[DiarizationSegment]:
    """
    Two-stage Anchor-Centroid speaker clustering:
      1. Discovers true speaker voiceprint centroids exclusively from reliable
         anchor utterances (>=2.0s) via duration-weighted cosine centroid
         agglomeration (immune to Complete-Linkage outlier shattering on long
         45-90 min interviews).
      2. Assigns all segments — including <2.0s backchannels ('Mm-hmm',
         'Добрый день', 'Пожалуйста') — to the closest anchor speaker centroid
         with a light temporal continuity prior, followed by one centroid
         refinement pass.
    """
    if not segments:
        return []
    if len(segments) != len(embeddings):
        raise ValueError("segments and embeddings must have identical length")

    mode = (speaker_count or "auto").strip().lower()
    if mode.isdigit() and int(mode) >= 1:
        target_k: Optional[int] = int(mode)
        eff_sim_thr = float(sim_threshold)
    elif mode == "4+":
        target_k = None
        eff_sim_thr = float(multi_sim_threshold)
    else:
        target_k = None
        eff_sim_thr = float(sim_threshold)

    if target_k == 1:
        return smooth_diarization_segments(
            [DiarizationSegment(s.start, s.end, 0) for s in segments],
            merge_gap_sec=merge_gap_sec,
        )

    durations = [max(0.0, float(s.end - s.start)) for s in segments]
    norm_embs: list[Optional[np.ndarray]] = []
    valid_indices: list[int] = []
    for idx, emb in enumerate(embeddings):
        if emb is None or emb.size == 0:
            norm_embs.append(None)
            continue
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8 or not np.isfinite(norm):
            norm_embs.append(None)
            continue
        norm_embs.append(vec / norm)
        valid_indices.append(idx)

    if not valid_indices:
        return smooth_diarization_segments(
            [DiarizationSegment(s.start, s.end, 0) for s in segments],
            merge_gap_sec=merge_gap_sec,
        )

    min_needed = max(2, target_k or 2)
    anchor_idx = [i for i in valid_indices if durations[i] >= 2.0]
    if len(anchor_idx) < min_needed:
        anchor_idx = [i for i in valid_indices if durations[i] >= 1.2]
    if not anchor_idx:
        anchor_idx = list(valid_indices)

    def _compute_centroid(member_ids: list[int]) -> np.ndarray:
        acc = np.zeros_like(norm_embs[member_ids[0]])
        for m_idx in member_ids:
            emb_vec = norm_embs[m_idx]
            if emb_vec is not None:
                acc += min(max(durations[m_idx], 0.25), 8.0) * emb_vec
        nrm = float(np.linalg.norm(acc))
        return acc / max(nrm, 1e-12)

    clusters: list[list[int]] = [[i] for i in anchor_idx]

    while len(clusters) > 1:
        if target_k is not None and len(clusters) <= target_k:
            break

        cent_mat = np.stack([_compute_centroid(c) for c in clusters], axis=0)
        sim_mat = cent_mat @ cent_mat.T
        np.fill_diagonal(sim_mat, -2.0)

        flat_best = int(np.argmax(sim_mat))
        bi, bj = divmod(flat_best, sim_mat.shape[1])
        best_sim = float(sim_mat[bi, bj])

        if target_k is None:
            if mode == "4+" and len(clusters) <= 4 and best_sim < max(eff_sim_thr, 0.64):
                break
            if best_sim < eff_sim_thr:
                break

        if bi > bj:
            bi, bj = bj, bi
        clusters[bi].extend(clusters[bj])
        clusters.pop(bj)

    # Absorb isolated 1-segment overlap/laugh outlier anchors on longer recordings
    if target_k is None and len(clusters) > 1:
        cluster_durs = [sum(durations[i] for i in c) for c in clusters]
        total_anchor_dur = sum(cluster_durs)
        if total_anchor_dur >= 25.0:
            min_keep_dur = min(6.0, max(2.5, 0.012 * total_anchor_dur))
            major_ids = [
                ci
                for ci, c_dur in enumerate(cluster_durs)
                if c_dur >= min_keep_dur or len(clusters[ci]) >= 2
            ]
            if major_ids and len(major_ids) < len(clusters):
                major_cents = {ci: _compute_centroid(clusters[ci]) for ci in major_ids}
                to_drop: set[int] = set()
                for ci in range(len(clusters)):
                    if ci in major_cents:
                        continue
                    c_vec = _compute_centroid(clusters[ci])
                    best_m = max(major_ids, key=lambda m: float(np.dot(c_vec, major_cents[m])))
                    best_m_sim = float(np.dot(c_vec, major_cents[best_m]))
                    if best_m_sim >= 0.36 or cluster_durs[ci] < 3.0:
                        clusters[best_m].extend(clusters[ci])
                        to_drop.add(ci)
                if to_drop:
                    clusters = [c for ci, c in enumerate(clusters) if ci not in to_drop]

    def _assign_to_centroids(cent_mat: np.ndarray) -> list[int]:
        assigned_ids: list[int] = []
        prev_spk: Optional[int] = None
        for idx, emb_vec in enumerate(norm_embs):
            if emb_vec is None:
                spk = prev_spk if prev_spk is not None else 0
            else:
                sims = cent_mat @ emb_vec
                if (
                    prev_spk is not None
                    and prev_spk < len(sims)
                    and durations[idx] < 1.5
                    and (segments[idx].start - segments[idx - 1].end) <= 1.0
                ):
                    sims = sims.copy()
                    sims[prev_spk] += 0.04
                spk = int(np.argmax(sims))
            assigned_ids.append(spk)
            prev_spk = spk
        return assigned_ids

    cent_mat = np.stack([_compute_centroid(c) for c in clusters], axis=0)
    assigned = _assign_to_centroids(cent_mat)

    # Refinement pass: recompute centroids from >=1.5s assigned segments & re-check merges
    if len(clusters) > 1:
        refined_groups: list[list[int]] = []
        for k in range(len(clusters)):
            strong_members = [
                i
                for i, spk in enumerate(assigned)
                if spk == k and norm_embs[i] is not None and durations[i] >= 1.5
            ]
            if not strong_members:
                strong_members = [
                    i for i, spk in enumerate(assigned) if spk == k and norm_embs[i] is not None
                ]
            if strong_members:
                refined_groups.append(strong_members)

        if len(refined_groups) >= 2:
            while target_k is None and len(refined_groups) > 1:
                if mode == "4+" and len(refined_groups) <= 4:
                    break
                r_mat = np.stack([_compute_centroid(g) for g in refined_groups], axis=0)
                r_sim = r_mat @ r_mat.T
                np.fill_diagonal(r_sim, -2.0)
                flat_best = int(np.argmax(r_sim))
                bi, bj = divmod(flat_best, r_sim.shape[1])
                if float(r_sim[bi, bj]) < eff_sim_thr:
                    break
                if bi > bj:
                    bi, bj = bj, bi
                refined_groups[bi].extend(refined_groups[bj])
                refined_groups.pop(bj)

            cent_mat = np.stack([_compute_centroid(g) for g in refined_groups], axis=0)
            assigned = _assign_to_centroids(cent_mat)

    out_segments = [
        DiarizationSegment(start=s.start, end=s.end, speaker=assigned[i])
        for i, s in enumerate(segments)
    ]
    smoothed = smooth_diarization_segments(out_segments, merge_gap_sec=merge_gap_sec)
    if target_k is None:
        smoothed = consolidate_minor_speakers(smoothed, merge_gap_sec=merge_gap_sec)
    return smoothed



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
    max_same_speaker_sec: float = 27.0,
    merge_gap_sec: float = 1.8,
    pad_sec: float = 0.2,
    split_speaker_turns: bool = True,
) -> list[tuple[float, float]]:
    """
    Packs diarization segments into efficient <=28.5s Whisper windows.
    When split_speaker_turns is True (default / GPU mode), substantial turns
    across different speakers (>=6s followed by >=2.5s) are split; when False
    (CPU mode), adjacent speech turns are packed densely up to max_same_speaker_sec
    to eliminate 30-second Mel zero-padding waste while multi-speaker windows
    use word_timestamps for exact speaker turn attribution.
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
        gap = seg.start - prev_end
        combined_span = seg.end - prev_start
        cur_group_dur = prev_end - prev_start
        next_seg_dur = seg.end - seg.start

        same_speaker_ok = seg.speaker == prev_spk
        short_exchange_ok = (not split_speaker_turns) or (cur_group_dur < 6.0) or (next_seg_dur < 2.5)

        if (
            gap <= merge_gap_sec
            and combined_span <= max_same_speaker_sec
            and (same_speaker_ok or short_exchange_ok)
        ):
            grouped[-1][1] = max(prev_end, seg.end)
            if not same_speaker_ok and next_seg_dur > cur_group_dur:
                grouped[-1][2] = seg.speaker
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


def subdivide_long_asr_windows(
    windows: list[tuple[float, float]],
    samples: np.ndarray,
    sample_rate: int = 16000,
    max_window_sec: float = 29.0,
) -> list[tuple[float, float]]:
    """
    Splits any >max_window_sec monologue window at local 50ms RMS silence/breath
    valleys so every Whisper window fits inside a single <=30s Encoder pass
    without triggering redundant Silero VAD or hallucination re-seeks.
    """
    if not windows or sample_rate <= 0 or samples.size == 0:
        return windows

    win_samples = max(1, int(0.05 * sample_rate))
    out: list[tuple[float, float]] = []

    for w_start, w_end in windows:
        if (w_end - w_start) <= max_window_sec:
            out.append((w_start, w_end))
            continue

        cur = w_start
        while (w_end - cur) > max_window_sec:
            target_s = cur + max_window_sec - 2.0
            search_lo = max(cur + 6.0, target_s - 4.0)
            search_hi = min(w_end - 1.0, cur + max_window_sec - 0.2)
            lo_idx = max(0, int(search_lo * sample_rate))
            hi_idx = min(len(samples), int(search_hi * sample_rate))

            if hi_idx - lo_idx > win_samples:
                n_frames = (hi_idx - lo_idx) // win_samples
                frames = samples[lo_idx : lo_idx + n_frames * win_samples].reshape(n_frames, win_samples)
                energies = np.einsum("ij,ij->i", frames, frames)
                best_f = int(np.argmin(energies))
                split_pt = (lo_idx + best_f * win_samples + (win_samples // 2)) / float(sample_rate)
            else:
                split_pt = min(target_s, w_end)

            if split_pt <= cur + 0.5:
                split_pt = cur + max_window_sec
            out.append((cur, min(split_pt, w_end)))
            cur = split_pt

        if w_end - cur >= 0.1:
            out.append((cur, w_end))

    return out


def window_has_multiple_speakers(
    w_start: float,
    w_end: float,
    diar_segments: Optional[list[DiarizationSegment]],
    min_overlap_sec: float = 0.15,
) -> bool:
    """
    Returns True if >=2 distinct diarized speakers overlap [w_start, w_end].
    Single-speaker monologue windows skip cross-attention DTW word_timestamps
    alignment because phrase-level segments already belong to one speaker.
    """
    if not diar_segments:
        return True
    seen: set[int] = set()
    for seg in diar_segments:
        if seg.end <= w_start:
            continue
        if seg.start >= w_end:
            break
        if (min(w_end, seg.end) - max(w_start, seg.start)) >= min_overlap_sec:
            seen.add(seg.speaker)
            if len(seen) >= 2:
                return True
    return False



def build_vad_asr_windows(
    samples: np.ndarray,
    sample_rate: int = 16000,
    max_window_sec: float = 26.0,
    merge_gap_sec: float = 1.4,
    pad_sec: float = 0.15,
) -> list[tuple[float, float]]:
    """
    Slices raw audio into <=26s speech-only windows via Silero VAD (single CPU
    thread, ~1s for a 1-hour file) and filters narrowband dial-tone beeps.
    Used in parallel GPU mode so Whisper immediately saturates CUDA without
    waiting for CPU speaker diarization.
    """
    total_dur = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    if total_dur <= 0.0 or samples.size < 800:
        return []

    raw_chunks = get_speech_timestamps(
        samples,
        vad_options=VadOptions(
            threshold=0.45,
            min_speech_duration_ms=200,
            max_speech_duration_s=max_window_sec,
            min_silence_duration_ms=450,
            speech_pad_ms=150,
        ),
        sampling_rate=sample_rate,
    )
    if not raw_chunks:
        return []

    speech_spans: list[tuple[float, float]] = []
    for ch in raw_chunks:
        s_idx = max(0, int(ch["start"]))
        e_idx = min(len(samples), int(ch["end"]))
        if e_idx <= s_idx:
            continue
        if is_narrowband_tone(samples[s_idx:e_idx], sample_rate=sample_rate):
            continue
        speech_spans.append((s_idx / float(sample_rate), e_idx / float(sample_rate)))

    if not speech_spans:
        return []

    grouped: list[list[float]] = [[speech_spans[0][0], speech_spans[0][1]]]
    for s_sec, e_sec in speech_spans[1:]:
        prev_start, prev_end = grouped[-1]
        gap = s_sec - prev_end
        combined_span = e_sec - prev_start
        cur_dur = prev_end - prev_start
        next_dur = e_sec - s_sec
        turn_boundary = gap >= 0.6 and cur_dur >= 6.0 and next_dur >= 2.5

        if gap <= merge_gap_sec and combined_span <= max_window_sec and not turn_boundary:
            grouped[-1][1] = max(prev_end, e_sec)
        else:
            grouped.append([s_sec, e_sec])

    windows: list[tuple[float, float]] = []
    for i, (g_start, g_end) in enumerate(grouped):
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
    return clean_hallucinated_text(compact.strip())


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
                joined = _join_unit_texts(buf_texts)
                if joined:
                    merged_turns.append(SpeakerTurn(t_start, t_end, "Спикер 1", joined))
                buf_texts = [u.text]
                t_start = u.start
                t_end = u.end
            else:
                buf_texts.append(u.text)
                t_end = max(t_end, u.end)
        if buf_texts:
            joined = _join_unit_texts(buf_texts)
            if joined:
                merged_turns.append(SpeakerTurn(t_start, t_end, "Спикер 1", joined))
        return merged_turns, (1 if merged_turns else 0)

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

    raw_turns: list[tuple[float, float, int, str]] = []
    cur_spk_raw = assigned[0][1]
    cur_start = assigned[0][0].start
    cur_end = assigned[0][0].end
    cur_pieces: list[str] = [assigned[0][0].text]

    for u, raw_spk in assigned[1:]:
        gap = u.start - cur_end
        if raw_spk != cur_spk_raw or gap > pause_break_sec:
            text_out = _join_unit_texts(cur_pieces)
            if text_out:
                raw_turns.append((cur_start, cur_end, cur_spk_raw, text_out))
            cur_spk_raw = raw_spk
            cur_start = u.start
            cur_end = max(u.start, u.end)
            cur_pieces = [u.text]
        else:
            cur_pieces.append(u.text)
            cur_end = max(cur_end, u.end)

    last_text = _join_unit_texts(cur_pieces)
    if last_text:
        raw_turns.append((cur_start, cur_end, cur_spk_raw, last_text))

    if not raw_turns:
        return [], 0

    # Chronological speaker mapping over real speech turns: first speaker heard -> Спикер 1
    speaker_map: dict[int, str] = {}
    turns: list[SpeakerTurn] = []
    for t_start, t_end, raw_spk, text_out in raw_turns:
        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"Спикер {len(speaker_map) + 1}"
        turns.append(
            SpeakerTurn(
                start=t_start,
                end=t_end,
                speaker_label=speaker_map[raw_spk],
                text=text_out,
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
        "# Транскрибация интервью (Interview Anonymizer v4.3.0)",
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


class _DiarizationCache:
    """
    Keeps warm pyannote segmentation + CAM++ voiceprint ONNX sessions (~34 MB)
    so sequential jobs skip repeated disk loads and ONNX graph compilation.
    """

    lock = threading.Lock()
    active_key: Optional[tuple] = None
    active_pair = None

    @classmethod
    def get_or_create(cls, key: tuple, factory: Callable):
        with cls.lock:
            if cls.active_key == key and cls.active_pair is not None:
                return cls.active_pair
            if cls.active_pair is not None:
                cls.active_pair = None
                cls.active_key = None
                gc.collect()
            cls.active_pair = factory()
            cls.active_key = key
            return cls.active_pair



_nvidia_libs_loaded = False


def _preload_nvidia_pip_libs() -> None:
    """
    Ensures pip-installed NVIDIA cuBLAS shared libraries (nvidia-cublas-cu12)
    are visible to CTranslate2's dynamic dlopen() both in local virtualenvs and
    inside Docker containers without requiring manual LD_LIBRARY_PATH exports.
    """
    global _nvidia_libs_loaded
    if _nvidia_libs_loaded:
        return
    _nvidia_libs_loaded = True

    search_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        search_roots.append(user_site)
    # Also include active virtualenv site-packages relative to sys.prefix
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        search_roots.extend(glob.glob(os.path.join(sys.prefix, pattern)))

    for root in search_roots:
        for sub in ("cublas", "cuda_nvrtc"):
            lib_dir = os.path.join(root, "nvidia", sub, "lib")
            bin_dir = os.path.join(root, "nvidia", sub, "bin")
            for cand_dir in (lib_dir, bin_dir):
                if not os.path.isdir(cand_dir):
                    continue
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(cand_dir)
                    except OSError:
                        pass
                else:
                    # Load cublasLt before cublas with RTLD_GLOBAL so soname lookup succeeds
                    for pat in ("libcublasLt.so*", "libcublas.so*", "libnvrtc.so*"):
                        for so_path in sorted(glob.glob(os.path.join(cand_dir, pat))):
                            try:
                                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                            except OSError:
                                pass


def probe_cuda_available() -> bool:
    """
    Returns True if at least one NVIDIA CUDA GPU and a compatible cuBLAS 12
    runtime are usable by CTranslate2 in the current process.
    """
    _preload_nvidia_pip_libs()
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        supported = ctranslate2.get_supported_compute_types("cuda")
        if not supported:
            return False
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libcublas.so.12")
        return True
    except Exception:
        return False


def resolve_compute_device(
    requested_device: Optional[str] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str, str]:
    """
    Resolves user-selected compute device ('cpu' or 'cuda') into a concrete
    (device, compute_type) pair for CTranslate2, falling back safely to CPU if
    CUDA is unavailable.
    """
    req = (requested_device or os.environ.get("ASR_DEVICE") or DEFAULT_COMPUTE_DEVICE).strip().lower()
    if req not in COMPUTE_DEVICE_OPTIONS and req != "auto":
        req = DEFAULT_COMPUTE_DEVICE

    if req == "cpu":
        return "cpu", "int8"

    if probe_cuda_available():
        try:
            supported = ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            supported = set()
        # Prefer native FP16 on NVIDIA Tensor Core GPUs (including RTX 50-series Blackwell sm_120,
        # where CTranslate2's int8/int8_float16 GEMM triggers CUBLAS_STATUS_NOT_SUPPORTED)
        for preferred in ("float16", "int8_float32", "float32", "int8_float16", "int8"):
            if preferred in supported:
                return "cuda", preferred
        return "cuda", "float16"

    if req == "cuda" and status_cb:
        status_cb("Видеокарта CUDA недоступна в текущем окружении — переключение на CPU...")
    return "cpu", "int8"


class Transcriber:
    """
    100% local speaker-diarized multilingual speech transcription pipeline.
    """

    @staticmethod
    def _extract_segment_embedding(
        extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
        samples: np.ndarray,
        start_sec: float,
        end_sec: float,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        dur = end_sec - start_sec
        pad = 0.06 if dur < 1.2 else 0.0
        s_idx = max(0, int((start_sec - pad) * sample_rate))
        e_idx = min(len(samples), int((end_sec + pad) * sample_rate))
        seg_wav = samples[s_idx:e_idx]
        if seg_wav.size < int(0.18 * sample_rate):
            return None

        def _embed_slice(wav_slice: np.ndarray) -> Optional[np.ndarray]:
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate, wav_slice)
            stream.input_finished()
            if not extractor.is_ready(stream):
                return None
            raw_vec = np.asarray(extractor.compute(stream), dtype=np.float32).reshape(-1)
            nrm = float(np.linalg.norm(raw_vec))
            if nrm <= 1e-8 or not np.isfinite(nrm):
                return None
            return raw_vec / nrm

        # CAM++ voiceprint accuracy saturates around ~3.2s; cap long segments to 2x 3.2s slices
        if seg_wav.size <= int(4.5 * sample_rate):
            return _embed_slice(seg_wav)

        max_win = int(3.2 * sample_rate)
        starts = np.linspace(0, seg_wav.size - max_win, num=2, dtype=int)
        sub_vecs = []
        for st in starts:
            v = _embed_slice(seg_wav[st : st + max_win])
            if v is not None:
                sub_vecs.append(v)
        if not sub_vecs:
            return None
        mean_vec = np.mean(sub_vecs, axis=0)
        nrm = float(np.linalg.norm(mean_vec))
        return mean_vec / max(nrm, 1e-12)

    @classmethod
    def _run_diarization(
        cls,
        samples: np.ndarray,
        seg_model_path: str,
        emb_model_path: str,
        speaker_count: str = "auto",
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[DiarizationSegment]:
        if cancel_token:
            cancel_token.check()
        if samples.size < 1600:  # < 0.1s
            return []

        diar_threads = max(1, min(int(threads), 8))
        emb_workers = min(4, max(1, diar_threads // 2))
        emb_intra_threads = max(1, diar_threads // emb_workers)
        mode = (speaker_count or "auto").strip().lower()

        # Pass 1: High-recall speaker-turn boundary detection via pyannote-segmentation-3.0.
        # Note: sherpa-onnx's built-in FastClustering uses Complete Linkage, which shatters
        # real speakers into 50+ clusters on long interviews; therefore we use a tight
        # local threshold here ONLY to prevent cross-speaker turn merges, then extract clean
        # full-turn CAM++ voiceprints and cluster via Anchor-Centroid clustering.
        diar_key = (
            seg_model_path,
            emb_model_path,
            diar_threads,
            emb_intra_threads,
            DIARIZATION_SEGMENTATION_THRESHOLD,
        )

        def _create_diar_engines():
            cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
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
                    num_clusters=-1,
                    threshold=DIARIZATION_SEGMENTATION_THRESHOLD,
                ),
                min_duration_on=0.30,
                min_duration_off=0.45,
            )
            sd_inst = sherpa_onnx.OfflineSpeakerDiarization(cfg)
            ext_inst = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=emb_model_path,
                    num_threads=emb_intra_threads,
                    provider="cpu",
                )
            )
            return sd_inst, ext_inst

        sd, extractor = _DiarizationCache.get_or_create(diar_key, _create_diar_engines)
        last_pct = 0
        last_log_t = 0.0

        def _progress_cb(done_chunks: int, total_chunks: int) -> int:
            nonlocal last_pct, last_log_t
            if cancel_token and cancel_token.is_cancelled():
                return 1  # Non-zero return immediately aborts sherpa-onnx C++ diarization
            if status_cb and total_chunks > 0:
                pct = int((done_chunks * 85) / total_chunks)
                now = time.time()
                if pct >= last_pct + 10 and (now - last_log_t) >= 1.5:
                    last_pct = pct
                    last_log_t = now
                    status_cb(f"Диаризация спикеров: {pct}%...")
            return 0

        result = sd.process(samples, callback=_progress_cb)
        if cancel_token:
            cancel_token.check()
        raw_objs = result.sort_by_start_time()

        # Filter out synthetic call-waiting / dial-tone ringback beeps before clustering
        speech_segments: list[DiarizationSegment] = []
        for s in raw_objs:
            st = float(s.start)
            en = float(s.end)
            if en <= st:
                continue
            s_idx = max(0, int(st * 16000))
            e_idx = min(len(samples), int(en * 16000))
            if is_narrowband_tone(samples[s_idx:e_idx], sample_rate=16000):
                continue
            speech_segments.append(DiarizationSegment(start=st, end=en, speaker=int(s.speaker)))

        if not speech_segments:
            return []

        # Merge micro-pauses (<=0.45s) within the same local turn up to 10s for cleaner embeddings
        pre_merged: list[DiarizationSegment] = [
            DiarizationSegment(
                speech_segments[0].start, speech_segments[0].end, speech_segments[0].speaker
            )
        ]
        for cur in speech_segments[1:]:
            prev = pre_merged[-1]
            if (
                cur.speaker == prev.speaker
                and (cur.start - prev.end) <= 0.45
                and (cur.end - prev.start) <= 10.0
            ):
                prev.end = max(prev.end, cur.end)
            else:
                pre_merged.append(DiarizationSegment(cur.start, cur.end, cur.speaker))

        if status_cb and len(pre_merged) > 30:
            status_cb("Диаризация спикеров: 90% (сопоставление голосовых отпечатков)...")

        if emb_workers <= 1 or len(pre_merged) < 8:
            embeddings: list[Optional[np.ndarray]] = []
            for seg in pre_merged:
                if cancel_token:
                    cancel_token.check()
                embeddings.append(
                    cls._extract_segment_embedding(
                        extractor, samples, seg.start, seg.end, sample_rate=16000
                    )
                )
        else:
            embeddings = [None] * len(pre_merged)

            def _embed_idx(idx_seg: tuple[int, DiarizationSegment]) -> tuple[int, Optional[np.ndarray]]:
                i, seg = idx_seg
                if cancel_token:
                    cancel_token.check()
                vec = cls._extract_segment_embedding(
                    extractor, samples, seg.start, seg.end, sample_rate=16000
                )
                return i, vec

            emb_pool = ThreadPoolExecutor(max_workers=emb_workers)
            try:
                futs = [emb_pool.submit(_embed_idx, (i, seg)) for i, seg in enumerate(pre_merged)]
                for fut in as_completed(futs):
                    if cancel_token:
                        cancel_token.check()
                    idx_out, vec_out = fut.result()
                    embeddings[idx_out] = vec_out
            except BaseException:
                emb_pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                emb_pool.shutdown(wait=True)

        return cluster_anchor_embeddings(
            segments=pre_merged,
            embeddings=embeddings,
            speaker_count=mode,
        )

    @staticmethod
    def _decode_whisper_windows(
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        profile_id: str,
        device: str,
        compute_type: str,
        diar_segments: Optional[list[DiarizationSegment]],
        threads: int,
        status_cb: Optional[Callable[[str], None]],
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[TimedUnit]:
        if cancel_token:
            cancel_token.check()

        if device == "cpu":
            total_threads = max(1, int(threads))
            max_workers_cap = 2 if profile_id == "turbo" else 3
            if total_threads >= 8:
                num_workers = min(max_workers_cap, total_threads // 4)
                cpu_threads = max(1, total_threads // num_workers)
            else:
                num_workers = 1
                cpu_threads = total_threads
        else:
            num_workers = 1
            cpu_threads = 1

        cache_key = (
            "faster_whisper",
            profile_id,
            model_dir,
            device,
            compute_type,
            cpu_threads,
            num_workers,
        )

        model: WhisperModel = _ModelCache.get_or_create(
            cache_key,
            lambda: WhisperModel(
                model_dir,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            ),
        )

        if cancel_token:
            cancel_token.check()

        total_dur = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
        if total_dur <= 0.0:
            return []

        ordered_diar: Optional[list[DiarizationSegment]] = None
        if diar_segments:
            ordered_diar = sorted(
                (s for s in diar_segments if s.end > s.start),
                key=lambda s: (s.start, s.end),
            )
            windows = build_speaker_asr_windows(
                ordered_diar,
                total_dur,
                max_same_speaker_sec=28.5 if device == "cpu" else 27.0,
                merge_gap_sec=3.5 if device == "cpu" else 1.8,
                split_speaker_turns=(device != "cpu"),
            )
            if windows:
                windows = subdivide_long_asr_windows(
                    windows, samples, sample_rate=sample_rate, max_window_sec=29.0
                )
            used_diar_windows = bool(windows)
        else:
            windows = build_vad_asr_windows(samples, sample_rate=sample_rate)
            used_diar_windows = False

        if not windows:
            if total_dur <= 5.0 and not is_narrowband_tone(samples, sample_rate=sample_rate):
                windows = [(0.0, total_dur)]
            else:
                return []

        valid_windows: list[tuple[float, float, int, int]] = []
        for w_start, w_end in windows:
            s_idx = max(0, int(w_start * sample_rate))
            e_idx = min(len(samples), int(w_end * sample_rate))
            if e_idx - s_idx >= 800:
                valid_windows.append((w_start, w_end, s_idx, e_idx))

        if not valid_windows:
            return []

        beam_size = 4 if device == "cuda" else 1
        temperatures = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if device == "cuda" else (0.0, 0.4)
        comp_ratio_thr = 2.2 if device == "cuda" else 2.4
        dev_tag = "GPU CUDA" if device == "cuda" else "CPU"

        last_pct = 0
        last_log_t = 0.0
        prog_lock = threading.Lock()

        def _maybe_report_progress(progress_sec: float):
            nonlocal last_pct, last_log_t
            if not status_cb or total_dur <= 0:
                return
            pct = min(99, int((progress_sec / total_dur) * 100))
            now = time.time()
            with prog_lock:
                if pct >= last_pct + 10 and (now - last_log_t) >= 1.5:
                    last_pct = pct
                    last_log_t = now
                    msg = (
                        f"Распознавание речи ({dev_tag}): {pct}% "
                        f"({format_timestamp(progress_sec)} / {format_timestamp(total_dur)})..."
                    )
                else:
                    msg = None
            if msg:
                status_cb(msg)

        def _decode_single_window(win_item: tuple[float, float, int, int]) -> list[TimedUnit]:
            w_start, w_end, s_idx, e_idx = win_item
            if cancel_token:
                cancel_token.check()

            win_dur = w_end - w_start
            use_vad = win_dur > 30.0
            need_word_ts = (
                True
                if (device == "cuda" or not used_diar_windows)
                else window_has_multiple_speakers(w_start, w_end, ordered_diar)
            )
            transcribe_kwargs = dict(
                # Non-None default language + multilingual=True skips faster-whisper's
                # discarded pre-encode detect_language() pass (saving a full 30s Encoder
                # pass per window!) while generate_segments still runs model.detect_language()
                # on encoder_output for every window across all 99 languages.
                language="ru",
                task="transcribe",
                multilingual=True,
                word_timestamps=need_word_ts,
                vad_filter=use_vad,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=comp_ratio_thr,
                log_prob_threshold=-1.0,
                beam_size=beam_size,
                temperature=temperatures,
            )
            if use_vad:
                transcribe_kwargs["vad_parameters"] = dict(
                    min_silence_duration_ms=400, speech_pad_ms=250
                )
                transcribe_kwargs["hallucination_silence_threshold"] = 2.0

            segments_iter, _ = model.transcribe(samples[s_idx:e_idx], **transcribe_kwargs)

            win_units: list[TimedUnit] = []
            for seg in segments_iter:
                if cancel_token:
                    cancel_token.check()

                abs_seg_start = w_start + float(seg.start)
                abs_seg_end = min(w_end, w_start + float(seg.end))

                if getattr(seg, "no_speech_prob", 0.0) > 0.75 and getattr(seg, "avg_logprob", 0.0) < -0.9:
                    continue

                if not used_diar_windows:
                    seg_s = max(0, int(abs_seg_start * sample_rate))
                    seg_e = min(len(samples), int(abs_seg_end * sample_rate))
                    if is_narrowband_tone(samples[seg_s:seg_e], sample_rate=sample_rate):
                        continue

                cleaned_seg_text = clean_hallucinated_text(seg.text or "")
                if not cleaned_seg_text:
                    continue

                if seg.words:
                    for w in seg.words:
                        if w.word and w.word.strip() and not is_repetitive_noise_token(w.word):
                            win_units.append(
                                TimedUnit(
                                    start=w_start + float(w.start),
                                    end=min(w_end, w_start + float(w.end)),
                                    text=w.word,
                                )
                            )
                else:
                    win_units.append(
                        TimedUnit(
                            start=abs_seg_start,
                            end=abs_seg_end,
                            text=cleaned_seg_text,
                        )
                    )

                if num_workers <= 1:
                    _maybe_report_progress(abs_seg_end)

            return win_units

        if num_workers <= 1 or len(valid_windows) <= 1:
            units: list[TimedUnit] = []
            for win_item in valid_windows:
                units.extend(_decode_single_window(win_item))
            return units

        # Parallel multi-worker CTranslate2 CPU execution preserving deterministic window order
        results_by_idx: list[Optional[list[TimedUnit]]] = [None] * len(valid_windows)
        total_speech_span = sum(max(0.01, w_e - w_s) for w_s, w_e, _, _ in valid_windows)
        done_speech_span = 0.0

        asr_pool = ThreadPoolExecutor(max_workers=num_workers)
        try:
            future_to_idx = {
                asr_pool.submit(_decode_single_window, win_item): idx
                for idx, win_item in enumerate(valid_windows)
            }
            for fut in as_completed(future_to_idx):
                if cancel_token:
                    cancel_token.check()
                w_idx = future_to_idx[fut]
                results_by_idx[w_idx] = fut.result()
                w_s, w_e, _, _ = valid_windows[w_idx]
                done_speech_span += max(0.01, w_e - w_s)
                est_pos = total_dur * min(1.0, done_speech_span / max(0.01, total_speech_span))
                _maybe_report_progress(est_pos)
        except BaseException:
            asr_pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            asr_pool.shutdown(wait=True)

        units = []
        for win_units in results_by_idx:
            if win_units:
                units.extend(win_units)
        return units

    @classmethod
    def _run_faster_whisper(
        cls,
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        profile_id: str,
        diar_segments: Optional[list[DiarizationSegment]] = None,
        threads: int = 4,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[TimedUnit]:
        if cancel_token:
            cancel_token.check()
        device, compute_type = resolve_compute_device(compute_device, status_cb=status_cb)
        try:
            return cls._decode_whisper_windows(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=model_dir,
                profile_id=profile_id,
                device=device,
                compute_type=compute_type,
                diar_segments=diar_segments,
                threads=threads,
                status_cb=status_cb,
                cancel_token=cancel_token,
            )
        except RuntimeError as err:
            if device == "cuda":
                if compute_type != "float32":
                    if status_cb:
                        status_cb(
                            f"Предупреждение CUDA {compute_type} ({err}) — повтор на GPU CUDA (float32)..."
                        )
                    try:
                        return cls._decode_whisper_windows(
                            samples=samples,
                            sample_rate=sample_rate,
                            model_dir=model_dir,
                            profile_id=profile_id,
                            device="cuda",
                            compute_type="float32",
                            diar_segments=diar_segments,
                            threads=threads,
                            status_cb=status_cb,
                            cancel_token=cancel_token,
                        )
                    except RuntimeError as err_fp32:
                        err = err_fp32
                if status_cb:
                    status_cb(f"Ошибка CUDA ({err}) — автоматический переход на CPU...")
                return cls._decode_whisper_windows(
                    samples=samples,
                    sample_rate=sample_rate,
                    model_dir=model_dir,
                    profile_id=profile_id,
                    device="cpu",
                    compute_type="int8",
                    diar_segments=diar_segments,
                    threads=threads,
                    status_cb=status_cb,
                    cancel_token=cancel_token,
                )
            raise

    @staticmethod
    def _run_sherpa_parakeet(
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        diar_segments: list[DiarizationSegment],
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[TimedUnit]:
        if cancel_token:
            cancel_token.check()

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

        valid_chunks: list[tuple[float, float, int, int]] = []
        for c_start, c_end in chunks:
            s_idx = max(0, int(c_start * sample_rate))
            e_idx = min(len(samples), int(c_end * sample_rate))
            if e_idx - s_idx >= 800:
                valid_chunks.append((c_start, c_end, s_idx, e_idx))

        if not valid_chunks:
            return []

        units: list[TimedUnit] = []
        total_chunks = len(valid_chunks)
        batch_size = 4
        last_pct = -20
        last_log_t = time.time()

        for b_start in range(0, total_chunks, batch_size):
            if cancel_token:
                cancel_token.check()

            batch = valid_chunks[b_start : b_start + batch_size]
            streams = []
            for _, _, s_idx, e_idx in batch:
                stream = recognizer.create_stream()
                stream.accept_waveform(sample_rate, samples[s_idx:e_idx])
                streams.append(stream)

            if hasattr(recognizer, "decode_streams") and len(streams) > 1:
                recognizer.decode_streams(streams)
            else:
                for stream in streams:
                    recognizer.decode_stream(stream)

            if cancel_token:
                cancel_token.check()

            for (c_start, c_end, _, _), stream in zip(batch, streams):
                res = stream.result
                if res.tokens and res.timestamps and len(res.tokens) == len(res.timestamps):
                    units.extend(
                        group_bpe_tokens_to_units(
                            list(res.tokens), list(res.timestamps), c_start, c_end
                        )
                    )
                elif res.text and res.text.strip():
                    units.append(TimedUnit(start=c_start, end=c_end, text=res.text.strip()))

            done_idx = b_start + len(batch)
            if status_cb and total_chunks > 0:
                pct = min(99, int((done_idx * 100) / total_chunks))
                now = time.time()
                if pct >= last_pct + 20 and (now - last_log_t) >= 2.0:
                    last_pct = pct
                    last_log_t = now
                    status_cb(f"Распознавание речи (Parakeet): {pct}% ({done_idx}/{total_chunks})...")

        return units


    @classmethod
    def transcribe_units_only(
        cls,
        wav_path: str,
        asr_model: str = DEFAULT_ASR_MODEL,
        compute_device: str = "cuda",
        threads: int = 1,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> dict:
        """
        Phase A of hardware-decoupled GPU mode: runs Silero VAD + Whisper on
        CUDA immediately during CPU video blurring, without competing with CPU
        video workers by deferring CPU speaker diarization until Phase B.
        """
        def update(msg: str):
            if status_cb:
                status_cb(msg)
            else:
                print(msg)

        if cancel_token:
            cancel_token.check()

        profile_id = (asr_model or DEFAULT_ASR_MODEL).strip().lower()
        if profile_id not in ASR_MODEL_PROFILES:
            raise ValueError(f"Unsupported ASR profile: {profile_id}")
        profile = ASR_MODEL_PROFILES[profile_id]

        t0 = time.time()
        samples, sample_rate = load_wav_16k_mono(wav_path, normalize_gain=True)
        duration_sec = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0

        if samples.size < 1600:
            return {
                "samples": samples,
                "sample_rate": sample_rate,
                "duration_sec": duration_sec,
                "units": [],
                "profile_id": profile_id,
                "elapsed_asr": time.time() - t0,
            }

        asr_dir = ModelManager.ensure_asr_model(profile_id, status_cb=update)
        if cancel_token:
            cancel_token.check()

        req_dev = (compute_device or "cuda").strip().lower()
        eff_dev, eff_ctype = resolve_compute_device(req_dev)
        dev_label = f"GPU CUDA ({eff_ctype})" if eff_dev == "cuda" else "CPU"
        update(f"Локальная транскрибация речи ({profile['short_name']}, {dev_label})...")

        t_asr = time.time()
        units = cls._run_faster_whisper(
            samples=samples,
            sample_rate=sample_rate,
            model_dir=asr_dir,
            profile_id=profile_id,
            diar_segments=None,
            threads=threads,
            compute_device=req_dev,
            status_cb=update,
            cancel_token=cancel_token,
        )
        elapsed_asr = time.time() - t_asr
        update(f"[Profile] GPU ASR завершена за {elapsed_asr:.2f}s (единиц текста: {len(units)})")

        return {
            "samples": samples,
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "units": units,
            "profile_id": profile_id,
            "elapsed_asr": elapsed_asr,
        }

    @classmethod
    def diarize_and_write_transcript(
        cls,
        asr_state: dict,
        output_txt_path: str,
        speaker_count: str = "auto",
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> dict:
        """
        Phase B of hardware-decoupled GPU mode: runs CPU speaker diarization
        using uncontested CPU cores after video blurring completes, then aligns
        the pre-computed GPU Whisper words to speaker turns and writes .txt.
        """
        def update(msg: str):
            if status_cb:
                status_cb(msg)
            else:
                print(msg)

        if cancel_token:
            cancel_token.check()

        t0 = time.time()
        samples: np.ndarray = asr_state["samples"]
        duration_sec = float(asr_state.get("duration_sec", 0.0))
        units: list[TimedUnit] = list(asr_state.get("units", []))
        profile_id = str(asr_state.get("profile_id", DEFAULT_ASR_MODEL))
        elapsed_asr = float(asr_state.get("elapsed_asr", 0.0))
        profile = ASR_MODEL_PROFILES.get(profile_id, ASR_MODEL_PROFILES[DEFAULT_ASR_MODEL])

        os.makedirs(os.path.dirname(os.path.abspath(output_txt_path)), exist_ok=True)

        if samples.size < 1600 or not units:
            txt = format_transcript_txt(
                turns=[],
                duration_sec=duration_sec,
                model_name=profile["short_name"],
                num_speakers=0,
            )
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(txt)
            total_elapsed = elapsed_asr + (time.time() - t0)
            update(f"[Profile] Транскрипт сохранён (0 реплик, спикеров: 0) за {total_elapsed:.2f}s")
            return {"num_speakers": 0, "num_turns": 0, "elapsed": total_elapsed}

        seg_onnx, emb_onnx = ModelManager.ensure_diarization_models(status_cb=update)
        if cancel_token:
            cancel_token.check()

        eff_threads = max(1, min(int(threads), 8))
        update(
            f"Разделение говорящих по ролям (режим спикеров: {speaker_count}, потоков CPU: {eff_threads})..."
        )
        t_diar = time.time()
        diar_segments = cls._run_diarization(
            samples=samples,
            seg_model_path=seg_onnx,
            emb_model_path=emb_onnx,
            speaker_count=speaker_count,
            threads=eff_threads,
            status_cb=update,
            cancel_token=cancel_token,
        )
        detected_raw_speakers = len({s.speaker for s in diar_segments})
        update(
            f"[Profile] Диаризация завершена за {time.time() - t_diar:.2f}s "
            f"(сегментов: {len(diar_segments)}, голосов: {detected_raw_speakers})"
        )

        if cancel_token:
            cancel_token.check()

        turns, final_speaker_count = align_units_to_speakers(units, diar_segments)
        transcript_text = format_transcript_txt(
            turns=turns,
            duration_sec=duration_sec,
            model_name=profile["short_name"],
            num_speakers=final_speaker_count,
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        total_elapsed = elapsed_asr + (time.time() - t0)
        update(
            f"[Profile] Транскрипт сохранён ({len(turns)} реплик, спикеров: {final_speaker_count}) "
            f"за {total_elapsed:.2f}s"
        )
        return {
            "num_speakers": final_speaker_count,
            "num_turns": len(turns),
            "elapsed": total_elapsed,
        }

    @classmethod
    def transcribe_audio_file(
        cls,
        wav_path: str,
        output_txt_path: str,
        asr_model: str = DEFAULT_ASR_MODEL,
        speaker_count: str = "auto",
        threads: int = 4,
        diar_threads: Optional[int] = None,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
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

        if cancel_token:
            cancel_token.check()

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
        if cancel_token:
            cancel_token.check()
        asr_dir = ModelManager.ensure_asr_model(profile_id, status_cb=update)
        if cancel_token:
            cancel_token.check()

        # 2. Speaker Diarization
        eff_diar_threads = diar_threads if diar_threads is not None else threads
        update(f"Разделение говорящих по ролям (режим спикеров: {speaker_count})...")
        t_diar = time.time()
        diar_segments = cls._run_diarization(
            samples=samples,
            seg_model_path=seg_onnx,
            emb_model_path=emb_onnx,
            speaker_count=speaker_count,
            threads=eff_diar_threads,
            status_cb=update,
            cancel_token=cancel_token,
        )
        detected_raw_speakers = len({s.speaker for s in diar_segments})
        update(
            f"[Profile] Диаризация завершена за {time.time() - t_diar:.2f}s "
            f"(сегментов: {len(diar_segments)}, голосов: {detected_raw_speakers})"
        )

        # 3. Multilingual Speech Recognition
        req_dev = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()
        t_asr = time.time()
        if profile["engine"] == "faster_whisper":
            eff_dev, _ = resolve_compute_device(req_dev)
            dev_label = "GPU CUDA" if eff_dev == "cuda" else "CPU"
            update(f"Локальная транскрибация речи ({profile['short_name']}, {dev_label})...")
            units = cls._run_faster_whisper(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                profile_id=profile_id,
                diar_segments=diar_segments,
                threads=threads,
                compute_device=req_dev,
                status_cb=update,
                cancel_token=cancel_token,
            )
        elif profile["engine"] == "sherpa_parakeet":
            if req_dev == "cuda":
                update(
                    "Модель Parakeet оптимизирована под CPU (INT8); для GPU рекомендуется Whisper Turbo."
                )
            update(f"Локальная транскрибация речи ({profile['short_name']}, CPU)...")
            units = cls._run_sherpa_parakeet(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                diar_segments=diar_segments,
                threads=threads,
                status_cb=update,
                cancel_token=cancel_token,
            )
        else:
            raise ValueError(f"Unsupported engine: {profile['engine']}")

        update(f"[Profile] ASR завершена за {time.time() - t_asr:.2f}s (единиц текста: {len(units)})")

        if cancel_token:
            cancel_token.check()

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

