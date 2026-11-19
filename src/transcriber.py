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
    APP_VERSION,
    ASR_LANGUAGE_OPTIONS,
    ASR_MODEL_PROFILES,
    COMPUTE_DEVICE_OPTIONS,
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_DIARIZATION_SENSITIVITY,
    DIARIZATION_CENTROID_SIM_THRESHOLD,
    DIARIZATION_MERGE_GAP_SEC,
    DIARIZATION_MIN_SEGMENT_SEC,
    DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD,
    DIARIZATION_SEGMENTATION_THRESHOLD,
    DIARIZATION_SENSITIVITY_PROFILES,
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
    Applies DC removal and zero-phase 99.5th-percentile gain normalization on quiet
    recordings so isolated clicks/pops never defeat normalization while preserving
    exact timestamps and boosting VAD, diarization, and ASR sensitivity.
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
        dc = float(np.mean(pcm))
        if abs(dc) > 1e-4:
            pcm = pcm - dc
        abs_pcm = np.abs(pcm)
        ref_level = float(np.percentile(abs_pcm, 99.5)) if pcm.size >= 160 else float(np.max(abs_pcm))
        if 0.005 < ref_level < 0.5:
            gain = min(10.0, 0.9 / ref_level)
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
    minor_min_sec: float = 4.5,
) -> list[DiarizationSegment]:
    """
    Absorbs transient noise/laugh/interjection micro-clusters on long recordings
    into the temporally nearest primary speaker, preventing false speaker
    explosions in auto mode without erasing genuine brief participants.
    """
    if len(segments) <= 2:
        return segments

    dur_by_spk: dict[int, float] = {}
    for s in segments:
        dur_by_spk[s.speaker] = dur_by_spk.get(s.speaker, 0.0) + max(0.0, s.end - s.start)

    total_speech = sum(dur_by_spk.values())
    if total_speech < 20.0 or len(dur_by_spk) <= 2:
        return segments

    floor_sec = max(1.5, float(minor_min_sec))
    min_major_dur = min(floor_sec, max(2.0, 0.008 * total_speech))
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
    sim_threshold: Optional[float] = None,
    multi_sim_threshold: Optional[float] = None,
    merge_gap_sec: float = DIARIZATION_MERGE_GAP_SEC,
    return_centroids: bool = False,
    sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
) -> list[DiarizationSegment] | tuple[list[DiarizationSegment], dict[int, np.ndarray]]:
    """
    Two-stage Anchor Average-Linkage (UPGMA) speaker clustering:
      1. Discovers true speaker voiceprint clusters exclusively from reliable
         anchor utterances (>=2.0s) via duration-weighted average-linkage cosine
         similarity with a lower-quantile guard (immune both to Complete-Linkage
         outlier shattering and to single-centroid chaining that collapses
         similar same-gender voices into 2 clusters).
      2. Assigns all segments — including rapid <1.5s answers and backchannels
         ('Ну да', 'Все', 'Добрый день') — unbiasedly to their closest vocal
         centroid, followed by centroid refinement and voiceprint export.
    """
    if not segments:
        return ([], {}) if return_centroids else []
    if len(segments) != len(embeddings):
        raise ValueError("segments and embeddings must have identical length")

    sens_key = (sensitivity or DEFAULT_DIARIZATION_SENSITIVITY).strip().lower()
    sens_prof = DIARIZATION_SENSITIVITY_PROFILES.get(
        sens_key, DIARIZATION_SENSITIVITY_PROFILES[DEFAULT_DIARIZATION_SENSITIVITY]
    )
    resolved_sim = float(sens_prof["sim_threshold"]) if sim_threshold is None else float(sim_threshold)
    resolved_multi = (
        float(sens_prof["multi_sim_threshold"])
        if multi_sim_threshold is None
        else float(multi_sim_threshold)
    )
    minor_min_sec = float(sens_prof.get("minor_min_sec", 4.5))

    mode = (speaker_count or "auto").strip().lower()
    if mode.isdigit() and int(mode) >= 1:
        target_k: Optional[int] = int(mode)
        eff_sim_thr = resolved_sim
    elif mode == "4+":
        target_k = None
        eff_sim_thr = resolved_multi
    else:
        target_k = None
        eff_sim_thr = resolved_sim

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

    if target_k == 1 or not valid_indices:
        single_out = smooth_diarization_segments(
            [DiarizationSegment(s.start, s.end, 0) for s in segments],
            merge_gap_sec=merge_gap_sec,
        )
        if not return_centroids:
            return single_out
        if valid_indices:
            acc = np.zeros_like(norm_embs[valid_indices[0]])
            for vi in valid_indices:
                acc += min(max(durations[vi], 0.25), 8.0) * norm_embs[vi]
            nrm = float(np.linalg.norm(acc))
            cents = {0: acc / max(nrm, 1e-12)} if nrm > 1e-8 else {}
        else:
            cents = {}
        return single_out, cents

    min_needed = max(2, target_k or 2)
    anchor_idx = [i for i in valid_indices if durations[i] >= 2.0]
    if len(anchor_idx) < min_needed:
        anchor_idx = [i for i in valid_indices if durations[i] >= 1.2]
    if len(anchor_idx) < min_needed:
        anchor_idx = list(valid_indices)

    def _compute_centroid(member_ids: list[int]) -> np.ndarray:
        acc = np.zeros_like(norm_embs[member_ids[0]])
        for m_idx in member_ids:
            emb_vec = norm_embs[m_idx]
            if emb_vec is not None:
                acc += min(max(durations[m_idx], 0.25), 8.0) * emb_vec
        nrm = float(np.linalg.norm(acc))
        return acc / max(nrm, 1e-12)

    # Precompute anchor pairwise cosine similarity matrix & duration weights for UPGMA
    anchor_pos = {orig_i: pos for pos, orig_i in enumerate(anchor_idx)}
    anchor_mat = np.stack([norm_embs[i] for i in anchor_idx], axis=0)
    anchor_weights = np.array(
        [min(max(durations[i], 0.25), 8.0) for i in anchor_idx], dtype=np.float32
    )
    pair_sim = anchor_mat @ anchor_mat.T

    clusters: list[list[int]] = [[i] for i in anchor_idx]
    cluster_pos_lists: list[list[int]] = [[anchor_pos[i]] for i in anchor_idx]
    # Weighted similarity sum between clusters for O(1) incremental UPGMA updates
    c_weights = anchor_weights.astype(np.float64).tolist()
    weighted_sum = (pair_sim.astype(np.float64)) * np.outer(anchor_weights, anchor_weights)
    blocked = np.zeros((len(clusters), len(clusters)), dtype=bool)

    while len(clusters) > 1:
        if target_k is not None and len(clusters) <= target_k:
            break

        k_cur = len(clusters)
        w_arr = np.array(c_weights, dtype=np.float64)
        denom = np.outer(w_arr, w_arr)
        avg_mat = np.divide(weighted_sum, np.maximum(denom, 1e-12))
        np.fill_diagonal(avg_mat, -2.0)
        if target_k is None:
            avg_mat = np.where(blocked, -2.0, avg_mat)

        flat_best = int(np.argmax(avg_mat))
        bi, bj = divmod(flat_best, k_cur)
        best_avg_sim = float(avg_mat[bi, bj])

        if target_k is None:
            if mode == "4+" and k_cur <= 4 and best_avg_sim < max(eff_sim_thr, 0.68):
                break
            if best_avg_sim < eff_sim_thr:
                break
            # Lower-quantile guard prevents chaining-drift merges across distinct similar voices
            sub_block = pair_sim[np.ix_(cluster_pos_lists[bi], cluster_pos_lists[bj])]
            low_q_sim = float(np.percentile(sub_block, 20)) if sub_block.size > 1 else best_avg_sim
            if low_q_sim < (eff_sim_thr - 0.14):
                blocked[bi, bj] = True
                blocked[bj, bi] = True
                continue

        if bi > bj:
            bi, bj = bj, bi
        clusters[bi].extend(clusters[bj])
        clusters.pop(bj)
        cluster_pos_lists[bi].extend(cluster_pos_lists[bj])
        cluster_pos_lists.pop(bj)

        # Update weighted_sum and blocked matrices by merging row/col bj into bi and deleting bj
        c_weights[bi] += c_weights[bj]
        c_weights.pop(bj)
        weighted_sum[bi, :] += weighted_sum[bj, :]
        weighted_sum[:, bi] += weighted_sum[:, bj]
        weighted_sum = np.delete(weighted_sum, bj, axis=0)
        weighted_sum = np.delete(weighted_sum, bj, axis=1)
        blocked[bi, :] |= blocked[bj, :]
        blocked[:, bi] |= blocked[:, bj]
        blocked = np.delete(blocked, bj, axis=0)
        blocked = np.delete(blocked, bj, axis=1)

    # Absorb isolated 1-segment overlap/laugh outlier anchors on longer recordings
    if target_k is None and len(clusters) > 1:
        cluster_durs = [sum(durations[i] for i in c) for c in clusters]
        total_anchor_dur = sum(cluster_durs)
        if total_anchor_dur >= 25.0:
            min_keep_dur = min(minor_min_sec, max(2.2, 0.010 * total_anchor_dur))
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
                    if best_m_sim >= 0.42 or cluster_durs[ci] < 2.6:
                        clusters[best_m].extend(clusters[ci])
                        to_drop.add(ci)
                if to_drop:
                    clusters = [c for ci, c in enumerate(clusters) if ci not in to_drop]

    def _assign_to_centroids(cent_mat: np.ndarray) -> list[int]:
        assigned_ids: list[int] = []
        prev_spk: Optional[int] = None
        for emb_vec in norm_embs:
            if emb_vec is None:
                spk = prev_spk if prev_spk is not None else 0
            else:
                sims = cent_mat @ emb_vec
                spk = int(np.argmax(sims))
            assigned_ids.append(spk)
            prev_spk = spk
        return assigned_ids

    cent_mat = np.stack([_compute_centroid(c) for c in clusters], axis=0)
    assigned = _assign_to_centroids(cent_mat)

    # Refinement pass: recompute centroids from >=1.5s assigned segments & check only high-confidence merges
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
            # Slightly higher bar (+0.04) on refined centroids because centroids have higher cosine
            # similarity than raw pairwise embeddings, preventing post-hoc collapse of similar voices.
            refine_merge_thr = min(0.88, eff_sim_thr + 0.04)
            while target_k is None and len(refined_groups) > 1:
                if mode == "4+" and len(refined_groups) <= 4:
                    break
                r_mat = np.stack([_compute_centroid(g) for g in refined_groups], axis=0)
                r_sim = r_mat @ r_mat.T
                np.fill_diagonal(r_sim, -2.0)
                flat_best = int(np.argmax(r_sim))
                bi, bj = divmod(flat_best, r_sim.shape[1])
                if float(r_sim[bi, bj]) < refine_merge_thr:
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
        smoothed = consolidate_minor_speakers(
            smoothed, merge_gap_sec=merge_gap_sec, minor_min_sec=minor_min_sec
        )

    if not return_centroids:
        return smoothed

    surviving_spks = {s.speaker for s in smoothed}
    final_centroids: dict[int, np.ndarray] = {}
    for spk_id in sorted(surviving_spks):
        strong_ids = [
            i
            for i, spk in enumerate(assigned)
            if spk == spk_id and norm_embs[i] is not None and durations[i] >= 1.2
        ]
        if not strong_ids:
            strong_ids = [
                i for i, spk in enumerate(assigned) if spk == spk_id and norm_embs[i] is not None
            ]
        if strong_ids:
            final_centroids[spk_id] = _compute_centroid(strong_ids)
        elif 0 <= spk_id < cent_mat.shape[0]:
            final_centroids[spk_id] = cent_mat[spk_id]

    return smoothed, final_centroids



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
    thread, ~1s for a 1-hour file), filters narrowband dial-tone beeps, and
    subdivides continuous rapid dialogue at local 50ms RMS energy valleys rather
    than hard mid-word duration cuts.
    """
    total_dur = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    if total_dur <= 0.0 or samples.size < 800:
        return []

    raw_chunks = get_speech_timestamps(
        samples,
        vad_options=VadOptions(
            threshold=0.45,
            min_speech_duration_ms=180,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=300,
            speech_pad_ms=180,
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

    return subdivide_long_asr_windows(
        windows, samples, sample_rate=sample_rate, max_window_sec=max_window_sec
    )


def normalize_chunk_level(chunk_samples: np.ndarray) -> np.ndarray:
    """
    Per-chunk quiet-speech leveling: recovers quieter remote participants (e.g.
    -20 dB system audio when local mic is -3 dB) without distorting loud chunks
    or altering timestamps.
    """
    arr = np.ascontiguousarray(chunk_samples, dtype=np.float32)
    if arr.size < 160:
        return arr
    abs_arr = np.abs(arr)
    p99 = float(np.percentile(abs_arr, 99.0))
    if 0.004 < p99 < 0.25:
        gain = min(8.0, 0.65 / p99)
        return np.ascontiguousarray(np.clip(arr * gain, -1.0, 1.0), dtype=np.float32)
    return arr


def _merge_time_intervals(
    intervals: list[tuple[float, float]],
    gap_tol: float = 0.05,
) -> list[tuple[float, float]]:
    valid = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not valid:
        return []
    merged: list[list[float]] = [[valid[0][0], valid[0][1]]]
    for s, e in valid[1:]:
        if s <= merged[-1][1] + gap_tol:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def build_longform_asr_chunks(
    samples: np.ndarray,
    sample_rate: int = 16000,
    diar_segments: Optional[list[DiarizationSegment]] = None,
    max_chunk_sec: float = 20.0,
    overlap_sec: float = 1.5,
    use_silero_vad: bool = True,
) -> list[tuple[float, float, float, float]]:
    """
    Longform chunker engineered for encoder-limited ASR models like GigaAM v3
    (25s encoder cap) and Parakeet TDT:
      1. Builds the UNION of Silero VAD speech regions and diarization segments
         so quiet/missed speech is never dropped.
      2. Splits along natural Silero VAD micro-pauses up to max_chunk_sec (<=20s).
      3. For breathless 1-3 minute monologues with no pause, first checks for an
         acoustic RMS breath valley, and otherwise advances with a sliding window
         of max_chunk_sec with overlap_sec (1.5s) and non-overlapping ownership
         cores [keep_start, keep_end] so no words are lost or duplicated.
    Returns list of (chunk_start, chunk_end, keep_start, keep_end).
    """
    total_dur = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    if total_dur <= 0.0 or samples.size < 800:
        return []

    max_len = max(4.0, float(max_chunk_sec))
    ov = max(0.0, min(float(overlap_sec), max_len * 0.25))

    raw_spans: list[tuple[float, float]] = []
    if diar_segments:
        for seg in diar_segments:
            if seg.end > seg.start:
                raw_spans.append((max(0.0, float(seg.start)), min(total_dur, float(seg.end))))

    if use_silero_vad:
        try:
            vad_chunks = get_speech_timestamps(
                samples,
                vad_options=VadOptions(
                    threshold=0.35,
                    min_speech_duration_ms=150,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=200,
                    speech_pad_ms=160,
                ),
                sampling_rate=sample_rate,
            )
            for ch in vad_chunks:
                s_i = max(0, int(ch["start"]))
                e_i = min(len(samples), int(ch["end"]))
                if e_i <= s_i:
                    continue
                if is_narrowband_tone(samples[s_i:e_i], sample_rate=sample_rate):
                    continue
                raw_spans.append((s_i / float(sample_rate), e_i / float(sample_rate)))
        except Exception:
            pass

    if not raw_spans:
        # Rescue non-silent audio even if both VAD and diarizer returned nothing
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms > 0.003 and not is_narrowband_tone(samples, sample_rate=sample_rate):
            raw_spans.append((0.0, total_dur))
        else:
            return []

    # Merge overlapping spans first, then pack nearby islands up to max_len
    base_spans = _merge_time_intervals(raw_spans, gap_tol=0.12)
    packed: list[tuple[float, float]] = []
    for s, e in base_spans:
        ps = max(0.0, s - 0.14)
        pe = min(total_dur, e + 0.14)
        if not packed:
            packed.append((ps, pe))
        else:
            prev_s, prev_e = packed[-1]
            if ps <= prev_e + 0.55 and (pe - prev_s) <= max_len:
                packed[-1] = (prev_s, max(prev_e, pe))
            else:
                ps = max(ps, prev_e)
                if pe > ps + 0.05:
                    packed.append((ps, pe))

    def _find_true_pause_split(cur_s: float, hard_end_s: float) -> Optional[float]:
        """Returns a genuine RMS pause valley inside [cur + max_len - 6, cur + max_len - 0.5] if present."""
        win_n = max(1, int(0.05 * sample_rate))  # 50 ms frame
        span_lo = max(0, int(cur_s * sample_rate))
        span_hi = min(len(samples), int(min(hard_end_s, cur_s + max_len) * sample_rate))
        if span_hi - span_lo < win_n * 8:
            return None

        # Compute median speech energy in this window to distinguish real pauses from continuous speech
        frame_energies: list[float] = []
        for pos in range(span_lo, span_hi - win_n + 1, win_n):
            f = samples[pos : pos + win_n]
            frame_energies.append(float(np.dot(f, f)) / win_n)
        if not frame_energies:
            return None
        med_energy = float(np.median(frame_energies))
        if med_energy <= 1e-8:
            return None

        search_lo = max(cur_s + max(5.0, max_len - 6.0), cur_s + max_len * 0.55)
        search_hi = min(hard_end_s - 0.8, cur_s + max_len - 0.4)
        if search_hi <= search_lo:
            return None

        lo_i = max(0, int(search_lo * sample_rate))
        hi_i = min(len(samples), int(search_hi * sample_rate))
        best_pos: Optional[int] = None
        best_e = float("inf")
        for pos in range(lo_i, hi_i - win_n + 1, win_n):
            f = samples[pos : pos + win_n]
            e_val = float(np.dot(f, f)) / win_n
            if e_val < best_e:
                best_e = e_val
                best_pos = pos + (win_n // 2)

        # Accept as clean non-overlapping split only if energy clearly dips (<= 22% of median speech energy)
        if best_pos is not None and best_e <= 0.22 * med_energy:
            return best_pos / float(sample_rate)
        return None

    chunks_out: list[tuple[float, float, float, float]] = []
    for island_s, island_e in packed:
        if (island_e - island_s) <= max_len:
            chunks_out.append((island_s, island_e, island_s, island_e))
            continue

        cur = island_s
        keep_lo = island_s
        while cur < island_e - 0.05:
            if (island_e - cur) <= max_len:
                chunks_out.append((cur, island_e, keep_lo, island_e))
                break

            pause_pt = _find_true_pause_split(cur, island_e)
            if pause_pt is not None and pause_pt > cur + 2.0 and (pause_pt - cur) <= max_len:
                chunks_out.append((cur, pause_pt, keep_lo, pause_pt))
                cur = pause_pt
                keep_lo = pause_pt
            else:
                # Breathless continuous speech: sliding window <= max_len with overlap_sec
                c_end = min(island_e, cur + max_len)
                step = max(2.0, (c_end - cur) - ov)
                next_cur = cur + step
                if next_cur >= island_e - 0.25:
                    chunks_out.append((cur, island_e, keep_lo, island_e))
                    break
                mid_bound = 0.5 * (c_end + next_cur)
                chunks_out.append((cur, c_end, keep_lo, mid_bound))
                cur = next_cur
                keep_lo = mid_bound

    return chunks_out


def _augment_windows_with_vad(
    base_windows: list[tuple[float, float]],
    samples: np.ndarray,
    sample_rate: int = 16000,
    max_window_sec: float = 28.5,
) -> list[tuple[float, float]]:
    """
    Rescues any >= 1.0s Silero VAD speech regions missed by diarization so
    Whisper never silently skips quieter remote speakers.
    """
    try:
        vad_wins = build_vad_asr_windows(
            samples,
            sample_rate=sample_rate,
            max_window_sec=min(26.0, max_window_sec),
            merge_gap_sec=1.2,
            pad_sec=0.15,
        )
    except Exception:
        return base_windows

    if not vad_wins:
        return base_windows

    extras: list[tuple[float, float]] = []
    for vs, ve in vad_wins:
        # Compute uncovered sub-intervals of [vs, ve] relative to base_windows
        uncovered = [(vs, ve)]
        for bs, be in base_windows:
            if be <= vs or bs >= ve:
                continue
            next_uncovered: list[tuple[float, float]] = []
            for us, ue in uncovered:
                if be <= us or bs >= ue:
                    next_uncovered.append((us, ue))
                else:
                    if bs > us:
                        next_uncovered.append((us, bs))
                    if be < ue:
                        next_uncovered.append((be, ue))
            uncovered = next_uncovered
            if not uncovered:
                break
        for us, ue in uncovered:
            if (ue - us) >= 1.0:
                extras.append((us, ue))

    if not extras:
        return base_windows

    combined = sorted(base_windows + extras, key=lambda w: (w[0], w[1]))
    return subdivide_long_asr_windows(
        combined, samples, sample_rate=sample_rate, max_window_sec=max_window_sec
    )


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
    cur_start: Optional[float] = None
    cur_last_t: Optional[float] = None
    pending_space = False

    for tok, rel_t in zip(tokens, timestamps):
        if not tok:
            continue
        abs_t = offset_sec + max(0.0, float(rel_t))
        clean_piece = tok.replace("▁", " ")

        # Standalone word-boundary marker (e.g. '▁' or ' ' in GigaAM v3 BPE vocab)
        if not clean_piece.strip():
            if cur_text.strip() and cur_start is not None:
                word_end = min(chunk_end_sec, max(cur_start + 0.08, abs_t))
                units.append(TimedUnit(start=cur_start, end=word_end, text=cur_text))
                cur_text = ""
                cur_start = None
                cur_last_t = None
            pending_space = True
            continue

        starts_new_word = pending_space or tok.startswith(" ") or tok.startswith("▁")
        pending_space = False

        if starts_new_word and cur_text.strip() and cur_start is not None:
            word_end = min(chunk_end_sec, max(cur_start + 0.08, abs_t))
            units.append(TimedUnit(start=cur_start, end=word_end, text=cur_text))
            cur_text = clean_piece if clean_piece.startswith(" ") else f" {clean_piece}"
            cur_start = abs_t
            cur_last_t = abs_t
        else:
            if cur_start is None:
                cur_start = abs_t
                cur_text = clean_piece if (not units or clean_piece.startswith(" ")) else f" {clean_piece}"
            else:
                cur_text += clean_piece
            cur_last_t = abs_t

    if cur_text.strip() and cur_start is not None:
        last_end = min(chunk_end_sec, (cur_last_t or cur_start) + 0.25)
        units.append(TimedUnit(start=cur_start, end=max(cur_start + 0.08, last_end), text=cur_text))

    return units


_COMPOUND_HYPHEN_BEFORE_RE = re.compile(r"([A-Za-zА-Яа-яЁё])\s+-([A-Za-zА-Яа-яЁё])")
_NUMERIC_SUFFIX_HYPHEN_RE = re.compile(r"([0-9])\s+-([A-Za-zА-Яа-яЁё]{1,4})\b")
_ENCLITIC_HYPHEN_AFTER_RE = re.compile(
    r"\b([А-Яа-яЁё]+)-\s+(то|либо|нибудь|таки|ка)\b", re.IGNORECASE
)
_PREFIX_HYPHEN_AFTER_RE = re.compile(
    r"\b(по|во|в|из|кое)-\s+([А-Яа-яЁё]{2,})\b", re.IGNORECASE
)
_MULTI_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!…])\s+(?=[A-ZА-ЯЁ0-9\"'«])")
_SENTENCE_END_PUNCT_RE = re.compile(r"(?:[.?!…]+|[.?!][\"'»\)]*)$")
_STRIP_PUNCT_RE = re.compile(r"^[^\wА-Яа-яЁё]+|[^\wА-Яа-яЁё]+$")

_TURN_STARTER_WORDS = frozenset(
    {
        "да",
        "нет",
        "ну",
        "так",
        "хорошо",
        "ладно",
        "окей",
        "ок",
        "понял",
        "понятно",
        "угу",
        "ага",
        "слушай",
        "слушайте",
        "смотри",
        "смотрите",
        "короче",
        "кстати",
        "тогда",
        "значит",
        "супер",
        "отлично",
        "ясно",
        "скажи",
        "расскажи",
        "какие",
        "какой",
        "какая",
        "а",
        "но",
        "yes",
        "yeah",
        "no",
        "well",
        "okay",
        "ok",
        "right",
        "so",
    }
)

_NON_FINAL_SYNTACTIC_WORDS = frozenset(
    {
        "в",
        "на",
        "к",
        "ко",
        "с",
        "со",
        "у",
        "о",
        "об",
        "от",
        "до",
        "из",
        "за",
        "по",
        "про",
        "без",
        "для",
        "при",
        "не",
        "ни",
        "же",
        "бы",
        "ли",
        "и",
        "а",
        "но",
        "или",
        "что",
        "как",
        "где",
        "когда",
        "который",
        "которая",
        "которые",
        "это",
        "то",
        "все",
        "всё",
        "еще",
        "ещё",
        "очень",
        "прям",
        "просто",
    }
)

_RESPONSE_OPENER_WORDS = frozenset(
    {
        "да",
        "нет",
        "ну",
        "угу",
        "ага",
        "все",
        "всё",
        "конечно",
        "точно",
        "верно",
        "есть",
        "хорошо",
        "ладно",
        "ок",
        "окей",
        "а",
        "понял",
        "понятно",
        "удивительно",
        "отлично",
        "супер",
        "yes",
        "yeah",
        "no",
        "sure",
        "right",
    }
)


def _join_unit_texts(pieces: list[str]) -> str:
    if not pieces:
        return ""
    raw_parts: list[str] = []
    for idx, p in enumerate(pieces):
        if idx == 0 or p.startswith(" ") or (len(p) > 1 and p.startswith("-") and p[1].isalnum()):
            raw_parts.append(p)
        else:
            raw_parts.append(f" {p}")
    raw = "".join(raw_parts)
    # Normalize multiple spaces and spaces before punctuation
    compact = " ".join(raw.split())
    for punct in (".", ",", "!", "?", ":", ";", "…"):
        compact = compact.replace(f" {punct}", punct)
    # Fix broken Whisper compound hyphens ('что -то' -> 'что-то', 'ключ -кандидат' -> 'ключ-кандидат')
    # while preserving spaced sentence dashes ('базовые - есть ещё')
    compact = _COMPOUND_HYPHEN_BEFORE_RE.sub(r"\1-\2", compact)
    compact = _NUMERIC_SUFFIX_HYPHEN_RE.sub(r"\1-\2", compact)
    compact = _ENCLITIC_HYPHEN_AFTER_RE.sub(r"\1-\2", compact)
    compact = _PREFIX_HYPHEN_AFTER_RE.sub(r"\1-\2", compact)
    return clean_hallucinated_text(compact.strip())


@dataclass
class _SpeechClause:
    start: float
    end: float
    text: str


def _clean_word_lower(raw_token: str) -> str:
    return _STRIP_PUNCT_RE.sub("", raw_token.strip()).lower()


def _expand_multi_sentence_units(units: list[TimedUnit]) -> list[TimedUnit]:
    """
    If a coarse segment-level TimedUnit contains multiple punctuated sentences,
    proportionally subdivides its [start, end] span into per-sentence TimedUnits
    so rapid Q&A inside one segment ('...в работе? Ну да.') can be attributed
    per sentence. Word-level TimedUnits pass through unchanged.
    """
    expanded: list[TimedUnit] = []
    for u in units:
        txt = u.text.strip()
        dur = max(0.0, float(u.end) - float(u.start))
        if dur >= 0.8 and ("?" in txt or "." in txt or "!" in txt):
            parts = [p.strip() for p in _MULTI_SENTENCE_SPLIT_RE.split(txt) if p and p.strip()]
            if len(parts) >= 2:
                weights = [max(1, len(p)) for p in parts]
                total_w = float(sum(weights))
                cur_t = float(u.start)
                for i, (part, w) in enumerate(zip(parts, weights)):
                    nxt_t = float(u.end) if i == len(parts) - 1 else (cur_t + dur * (w / total_w))
                    expanded.append(TimedUnit(start=cur_t, end=max(cur_t + 0.04, nxt_t), text=f" {part}"))
                    cur_t = nxt_t
                continue
        expanded.append(u)
    return expanded


def _score_span_overlap(
    s_start: float,
    s_end: float,
    diar_segments: list[DiarizationSegment],
    seg_ends: list[float],
    prev_spk: Optional[int] = None,
) -> tuple[int, dict[int, float], bool]:
    """
    Returns (chosen_speaker, overlap_by_speaker, has_nearby_contention) for a
    temporal span [s_start, s_end].
    """
    u_start = float(s_start)
    u_end = max(u_start + 0.02, float(s_end))

    idx = max(0, bisect.bisect_left(seg_ends, u_start - 1.5) - 1)
    overlap_by_spk: dict[int, float] = {}
    nearby_speakers: set[int] = set()
    nearest_dist = float("inf")
    nearest_spk = diar_segments[0].speaker

    for j in range(idx, len(diar_segments)):
        seg = diar_segments[j]
        if seg.start > u_end + 1.5:
            break

        if seg.end >= (u_start - 0.65) and seg.start <= (u_end + 0.65):
            nearby_speakers.add(seg.speaker)

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
        closest_seg = min(
            diar_segments,
            key=lambda s: min(abs(u_start - s.end), abs(s.start - u_end)),
        )
        chosen_spk = closest_seg.speaker

    has_contention = len(nearby_speakers) >= 2
    if len(overlap_by_spk) >= 2:
        sorted_ovs = sorted(overlap_by_spk.values(), reverse=True)
        if sorted_ovs[1] >= 0.10:
            has_contention = True

    return chosen_spk, overlap_by_spk, has_contention


def _split_unpunctuated_multispeaker_units(
    clause_units: list[TimedUnit],
    diar_segments: list[DiarizationSegment],
    seg_ends: list[float],
    depth: int = 0,
) -> list[list[TimedUnit]]:
    """
    When Whisper omits punctuation across a fast turn transition (e.g. '...сложно
    джойном отнести хорошо так ладно раз мы залезли...'), finds the most natural
    syntactic + acoustic speaker transition point without tearing trailing verbs
    or 1-2 word prefixes away from their clause.
    """
    if depth >= 2 or len(clause_units) < 6 or not diar_segments:
        return [clause_units]

    span_start = float(clause_units[0].start)
    span_end = float(clause_units[-1].end)
    if (span_end - span_start) < 2.4:
        return [clause_units]

    _, ov_map, _ = _score_span_overlap(span_start, span_end, diar_segments, seg_ends)
    substantial_spks = [spk for spk, ov in ov_map.items() if ov >= 0.65]
    if len(substantial_spks) < 2:
        return [clause_units]

    best_k: Optional[int] = None
    best_score = -1e9

    for k in range(2, len(clause_units) - 1):
        left_s = float(clause_units[0].start)
        left_e = float(clause_units[k - 1].end)
        right_s = float(clause_units[k].start)
        right_e = float(clause_units[-1].end)
        left_dur = left_e - left_s
        right_dur = right_e - right_s
        if left_dur < 0.75 or right_dur < 0.75:
            continue

        left_spk, left_ov, _ = _score_span_overlap(left_s, left_e, diar_segments, seg_ends)
        right_spk, right_ov, _ = _score_span_overlap(right_s, right_e, diar_segments, seg_ends)
        if left_spk == right_spk:
            continue

        left_total_ov = sum(left_ov.values()) or max(0.01, left_dur)
        right_total_ov = sum(right_ov.values()) or max(0.01, right_dur)
        left_purity = left_ov.get(left_spk, 0.0) / left_total_ov
        right_purity = right_ov.get(right_spk, 0.0) / right_total_ov
        if left_purity < 0.58 or right_purity < 0.58:
            continue

        prev_raw = clause_units[k - 1].text.rstrip()
        prev_word = _clean_word_lower(clause_units[k - 1].text)
        next_word = _clean_word_lower(clause_units[k].text)
        gap = max(0.0, right_s - left_e)

        ling_bonus = 0.0
        if prev_raw.endswith((",", ":", ";", "—", "-")):
            ling_bonus += 0.42
        if next_word in _TURN_STARTER_WORDS:
            ling_bonus += 0.45
        if prev_word in _NON_FINAL_SYNTACTIC_WORDS and not prev_raw.endswith((",", ":", ";")):
            ling_bonus -= 0.85
        ling_bonus += 0.25 * min(1.0, gap / 0.30)

        score = left_purity + right_purity + ling_bonus
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None or best_score < 1.35:
        return [clause_units]

    left_part = _split_unpunctuated_multispeaker_units(
        clause_units[:best_k], diar_segments, seg_ends, depth=depth + 1
    )
    right_part = _split_unpunctuated_multispeaker_units(
        clause_units[best_k:], diar_segments, seg_ends, depth=depth + 1
    )
    return left_part + right_part


def _build_speech_clauses(
    units: list[TimedUnit],
    diar_segments: list[DiarizationSegment],
    seg_ends: list[float],
) -> list[_SpeechClause]:
    """
    Groups word-level TimedUnits into atomic grammatical clauses so sentences
    like 'А что такое ключ-кандидат?' are never torn mid-phrase across speakers
    during overlapping turn transitions.
    """
    expanded = _expand_multi_sentence_units(units)
    if not expanded:
        return []

    raw_groups: list[list[TimedUnit]] = [[expanded[0]]]
    for cur_u in expanded[1:]:
        prev_u = raw_groups[-1][-1]
        prev_r = prev_u.text.rstrip()
        gap = float(cur_u.start) - float(prev_u.end)

        ends_sentence = bool(_SENTENCE_END_PUNCT_RE.search(prev_r))
        if ends_sentence and prev_r.endswith(".") and not prev_r.endswith(".."):
            prev_alpha = _clean_word_lower(prev_r)
            cur_stripped = cur_u.text.lstrip()
            cur_Starts_lower = bool(cur_stripped) and cur_stripped[0].islower()
            if len(prev_alpha) <= 2 and cur_Starts_lower and gap < 0.25:
                ends_sentence = False

        if ends_sentence or gap >= 0.48:
            raw_groups.append([cur_u])
        else:
            raw_groups[-1].append(cur_u)

    clauses: list[_SpeechClause] = []
    for grp in raw_groups:
        last_txt = grp[-1].text.rstrip()
        is_unpunctuated = not bool(_SENTENCE_END_PUNCT_RE.search(last_txt))
        sub_groups = (
            _split_unpunctuated_multispeaker_units(grp, diar_segments, seg_ends)
            if (is_unpunctuated and diar_segments)
            else [grp]
        )
        for sub in sub_groups:
            joined = _join_unit_texts([u.text for u in sub])
            if not joined:
                continue
            c_start = float(sub[0].start)
            c_end = max(c_start + 0.02, float(max(u.end for u in sub)))
            clauses.append(_SpeechClause(start=c_start, end=c_end, text=joined))

    return clauses


def align_units_to_speakers(
    units: list[TimedUnit],
    diar_segments: list[DiarizationSegment],
    pause_break_sec: float = TRANSCRIPT_PAUSE_BREAK_SEC,
    samples: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    speaker_centroids: Optional[dict[int, np.ndarray]] = None,
    embedding_extractor: Optional[sherpa_onnx.SpeakerEmbeddingExtractor] = None,
    clause_embed_fn: Optional[Callable[[np.ndarray, int, float, float], Optional[np.ndarray]]] = None,
) -> tuple[list[SpeakerTurn], int]:
    """
    Clause-aware + voiceprint-verified speaker attribution:
      1. Groups words into grammatical clauses/sentences so simultaneous turn
         starts ('А что такое ключ-кандидат?') and trailing verbs ('...отнести.')
         are never torn mid-clause by +-250ms overlap boundaries.
      2. Scores each clause against diarization segments with edge-bleed trimming.
      3. When speaker CAM++ voiceprint centroids and waveform samples are
         available, acoustically verifies Q&A turns, rapid replies ('Ну да.',
         'Ну да, есть немного.'), and contested overlap clauses directly against
         speaker centroids, separating ping-pong dialogue even if coarse
         segmentation fused adjacent turns.
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
    clauses = _build_speech_clauses(non_empty_units, diar_segments, seg_ends)
    if not clauses:
        return [], 0

    prev_spk: Optional[int] = None
    assigned_spks: list[int] = []
    clause_contentions: list[bool] = []

    for cl in clauses:
        dur = cl.end - cl.start
        # Trim up to 100ms inward on multi-word clauses so overlap bleed at turn
        # edges doesn't outweigh the main body of the sentence
        trim = min(0.10, 0.15 * dur) if (dur >= 0.65 and " " in cl.text) else 0.0
        chosen_spk, _, contended = _score_span_overlap(
            cl.start + trim,
            cl.end - trim,
            diar_segments,
            seg_ends,
            prev_spk=prev_spk,
        )
        assigned_spks.append(chosen_spk)
        clause_contentions.append(contended)
        prev_spk = chosen_spk

    # Acoustic clause verification via CAM++ voiceprints for Q&A and rapid back-and-forth
    can_verify_acoustically = (
        samples is not None
        and samples.size >= int(0.35 * sample_rate)
        and speaker_centroids is not None
        and len(speaker_centroids) >= 2
        and (embedding_extractor is not None or clause_embed_fn is not None)
    )
    if can_verify_acoustically:
        spk_ids = sorted(speaker_centroids.keys())
        cent_mat = np.stack([speaker_centroids[k] for k in spk_ids], axis=0)
        spk_to_row = {k: i for i, k in enumerate(spk_ids)}

        for idx, cl in enumerate(clauses):
            cur_spk = assigned_spks[idx]
            dur = cl.end - cl.start
            if dur < 0.30 or dur > 8.5 or cur_spk not in spk_to_row:
                continue

            ends_q = cl.text.rstrip().endswith("?")
            prev_ends_q = (
                idx > 0
                and clauses[idx - 1].text.rstrip().endswith("?")
                and (cl.start - clauses[idx - 1].end) <= 2.2
            )
            next_ends_q = (
                idx + 1 < len(clauses)
                and clauses[idx + 1].text.rstrip().endswith("?")
                and (clauses[idx + 1].start - cl.end) <= 1.8
            )
            contended = clause_contentions[idx]
            first_word = _clean_word_lower(cl.text.split()[0]) if cl.text.split() else ""
            is_short_reply = dur <= 2.8 and first_word in _RESPONSE_OPENER_WORDS

            if not (ends_q or prev_ends_q or next_ends_q or contended or is_short_reply):
                continue

            v_trim = min(0.05, 0.10 * dur) if dur >= 0.55 else 0.0
            v_start = cl.start + v_trim
            v_end = cl.end - v_trim
            s_i = max(0, int(v_start * sample_rate))
            e_i = min(len(samples), int(v_end * sample_rate))
            if e_i - s_i < int(0.26 * sample_rate):
                continue
            wav_slice = samples[s_i:e_i]
            rms = float(np.sqrt(np.mean(wav_slice * wav_slice)))
            if rms < 7e-4:
                continue

            if clause_embed_fn is not None:
                emb_vec = clause_embed_fn(samples, sample_rate, v_start, v_end)
            else:
                emb_vec = Transcriber._extract_segment_embedding(
                    embedding_extractor,
                    samples,
                    v_start,
                    v_end,
                    sample_rate=sample_rate,
                    pad_short=False,
                )
            if emb_vec is None:
                continue

            sims = cent_mat @ emb_vec
            best_row = int(np.argmax(sims))
            best_spk = spk_ids[best_row]
            if best_spk == cur_spk:
                continue

            best_sim = float(sims[best_row])
            cur_sim = float(sims[spk_to_row[cur_spk]])
            margin = best_sim - cur_sim

            prev_final_spk = assigned_spks[idx - 1] if idx > 0 else None
            dialogic_turn_prior = (prev_ends_q and prev_final_spk == cur_spk) or (
                ends_q
                and prev_final_spk == cur_spk
                and idx > 0
                and (cl.start - clauses[idx - 1].end) <= 1.5
            )
            if dialogic_turn_prior:
                req_margin = 0.045 if dur >= 0.75 else 0.065
                min_abs_sim = 0.38
            elif contended:
                req_margin = 0.065 if dur >= 0.90 else 0.090
                min_abs_sim = 0.42
            else:
                req_margin = 0.085 if dur >= 0.90 else 0.110
                min_abs_sim = 0.44

            if best_sim >= min_abs_sim and margin >= req_margin:
                assigned_spks[idx] = best_spk

    raw_turns: list[tuple[float, float, int, str]] = []
    cur_spk_raw = assigned_spks[0]
    cur_start = clauses[0].start
    cur_end = clauses[0].end
    cur_pieces: list[str] = [clauses[0].text]

    for cl, raw_spk in zip(clauses[1:], assigned_spks[1:]):
        gap = cl.start - cur_end
        if raw_spk != cur_spk_raw or gap > pause_break_sec:
            text_out = _join_unit_texts(cur_pieces)
            if text_out:
                raw_turns.append((cur_start, cur_end, cur_spk_raw, text_out))
            cur_spk_raw = raw_spk
            cur_start = cl.start
            cur_end = max(cl.start, cl.end)
            cur_pieces = [cl.text]
        else:
            cur_pieces.append(cl.text)
            cur_end = max(cur_end, cl.end)

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
        f"# Транскрибация интервью (Interview Anonymizer v{APP_VERSION})",
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
_cublas_preloaded = False
_cuda_probe_ok = False


def _preload_nvidia_pip_libs() -> None:
    """
    Ensures pip-installed NVIDIA cuBLAS shared libraries (nvidia-cublas-cu12)
    are visible to CTranslate2's dynamic dlopen() both in local virtualenvs and
    inside Docker containers without requiring manual LD_LIBRARY_PATH exports.
    """
    global _nvidia_libs_loaded, _cublas_preloaded
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
                                if "libcublas.so" in os.path.basename(so_path):
                                    _cublas_preloaded = True
                            except OSError:
                                pass


def probe_cuda_available() -> bool:
    """
    Returns True if at least one NVIDIA CUDA GPU and a compatible cuBLAS 12
    runtime are usable by CTranslate2 in the current process.
    """
    global _cuda_probe_ok
    if _cuda_probe_ok:
        return True
    _preload_nvidia_pip_libs()
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        supported = ctranslate2.get_supported_compute_types("cuda")
        if not supported:
            return False
        if sys.platform.startswith("linux") and not _cublas_preloaded:
            ctypes.CDLL("libcublas.so.12")
        _cuda_probe_ok = True
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


_CONSECUTIVE_REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)

# Known Whisper phantom subtitle hallucinations generated on Russian silence/noise
_WHISPER_PHANTOM_RE = re.compile(
    r"^(?:"
    r"продолжение\s+следует|"
    r"редактор\s+субтитров\b.*|"
    r"корректор\b.*|"
    r"субтитры\s+(?:сделал|создавал|подготовил|подогнал|добавил)\b.*|"
    r"спасибо\s+за\s+просмотр\b.*|"
    r"подписывайтесь\s+на\s+(?:наш\s+)?канал\b.*|"
    r"ставьте\s+лайк(?:и)?\b.*|"
    r"dimatorzok\b.*|"
    r"amara\.org\b.*"
    r")[\s.!?,:;…\-—]*$",
    re.IGNORECASE,
)


def is_whisper_phantom_hallucination(
    text: str,
    no_speech_prob: float = 0.0,
    avg_logprob: float = 0.0,
) -> bool:
    """
    Detects well-known Whisper training-data subtitle artifacts (e.g.
    'Продолжение следует...', 'Редактор субтитров ...', 'Субтитры сделал ...')
    that Whisper Large v3 / Turbo emit on silence or low-energy noise.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if not _WHISPER_PHANTOM_RE.match(cleaned):
        return False
    # Always drop unambiguous subtitle credit lines; drop generic sign-offs when confidence is weak
    lowered = cleaned.lower()
    if any(k in lowered for k in ("субтитр", "корректор", "dimatorzok", "amara.org")):
        return True
    return no_speech_prob >= 0.25 or avg_logprob <= -0.45 or len(cleaned) <= 32


def _normalize_prompt_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", (text or "").lower())


def is_whisper_prompt_echo(
    text: str,
    custom_prompt: str,
    no_speech_prob: float = 0.0,
    avg_logprob: float = 0.0,
) -> bool:
    """
    Detects when Whisper hallucinates/echoes the user's initial_prompt glossary
    on a quiet or noisy segment (e.g. repeating 'Собеседование системного аналитика:
    REST, gRPC, HTTP 2.0, SOAP...') so user prompts never leak into transcripts.
    """
    if not custom_prompt or not custom_prompt.strip():
        return False
    seg_toks = _normalize_prompt_tokens(text)
    if len(seg_toks) < 3:
        return False
    prompt_toks = _normalize_prompt_tokens(custom_prompt)
    if len(prompt_toks) < 3:
        return False

    prompt_set = set(prompt_toks)
    overlap_ratio = sum(1 for t in seg_toks if t in prompt_set) / float(len(seg_toks))

    # Check if a contiguous 4-word run from the prompt was echoed verbatim
    if len(seg_toks) >= 4:
        prompt_ngrams = {
            tuple(prompt_toks[i : i + 4]) for i in range(max(0, len(prompt_toks) - 3))
        }
        for i in range(len(seg_toks) - 3):
            if tuple(seg_toks[i : i + 4]) in prompt_ngrams:
                return True

    if overlap_ratio >= 0.85 and len(seg_toks) >= 5:
        return True
    if overlap_ratio >= 0.70 and (no_speech_prob >= 0.30 or avg_logprob <= -0.55):
        return True
    return False


# Strict opt-in phonetic-to-Latin glossary map: each entry activates ONLY when its
# canonical Latin term is explicitly present in the user's custom_prompt.
_GLOSSARY_PHONETIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("gRPC", (r"джи[\s\-]?эр[\s\-]?пи[\s\-]?си", r"гэрпэцэ", r"грпц", r"грпс", r"gpc")),
    ("HTTP 2.0", (r"эйч[\s\-]?ти[\s\-]?ти[\s\-]?пи[\s\-]+(?:2\.0|два(?:\s+ноль)?)", r"хттп[\s\-]+(?:2\.0|два(?:\s+ноль)?)")),
    ("OAuth 2", (r"о[\s\-]?ау[фтс](?:[\s\-]+(?:2(?:\.0)?|два))?", r"оаут(?:[\s\-]+(?:2(?:\.0)?|два))?", r"оауф(?:[\s\-]+(?:2(?:\.0)?|два))?")),
    ("JWT", (r"джей[\s\-]?дабл[\s\-]?ю[\s\-]?ти", r"джей[\s\-]?ви[\s\-]?ти", r"джвт", r"живити")),
    ("Bearer", (r"бирер", r"бэрер", r"берер")),
    ("SOAP", (r"соап", r"соуп")),
    ("REST API", (r"рест[\s\-]+апи", r"рест[\s\-]+эй[\s\-]?пи[\s\-]?ай")),
    ("REST", (r"рест(?:а|у|ом|е)?",)),
    ("PostgreSQL", (r"постгрес[\s\-]?кью[\s\-]?эль", r"постгресс?", r"постгри")),
    ("Redis", (r"редис(?:а|е|ом)?", r"рэдис(?:а|е|ом)?")),
    ("MongoDB", (r"монго[\s\-]?дб", r"монго[\s\-]?ди[\s\-]?би", r"монга")),
    ("Kafka", (r"кафк[аеуи]",)),
    ("RabbitMQ", (r"рэббит[\s\-]?эм[\s\-]?кью", r"реббит[\s\-]?эм[\s\-]?кью", r"рэббит")),
    ("Sequence", (r"сиквенс", r"секвенс", r"сиквэнс", r"sipens")),
    ("Activity", (r"активити",)),
    ("BPMN", (r"би[\s\-]?пи[\s\-]?эм[\s\-]?эн", r"бпмн")),
    ("ERD", (r"и[\s\-]?ар[\s\-]?ди", r"ерд")),
    ("Camunda", (r"камунд[аеуы]",)),
    ("Swagger", (r"сваг+ер(?:а|е|ом)?",)),
    ("Postman", (r"постман(?:а|е|ом)?", r"постмэн(?:а|е|ом)?")),
    ("Confluence", (r"конфлю[еэ]нс(?:а|е|ом)?",)),
    ("Jira", (r"джир[аеуы]", r"жир[аеуы]")),
    ("Jenkins", (r"дженкинс(?:а|е|ом)?",)),
    ("Docs as Code", (r"докс[\s\-]+эз[\s\-]+код", r"докс[\s\-]+аз[\s\-]+код")),
    ("GraphQL", (r"граф[\s\-]?кью[\s\-]?эль", r"графкьюэл")),
    ("OpenAPI", (r"опен[\s\-]?апи", r"оупен[\s\-]?апи")),
    ("Kubernetes", (r"кубернетес(?:а|е|ом)?", r"кубернэтес(?:а|е|ом)?", r"кубер(?:а|е|ом)?")),
    ("Docker", (r"докер(?:а|е|ом)?",)),
    ("GitLab", (r"гит[\s\-]?лаб(?:а|е|ом)?",)),
    ("Git", (r"гит(?:а|е|ом)?",)),
    ("CI/CD", (r"си[\s\-]?ай[\s\-/]+си[\s\-]?ди",)),
    ("Prometheus", (r"прометеус(?:а|е|ом)?",)),
    ("Grafana", (r"графан[аеуы]",)),
    ("Terraform", (r"терраформ(?:а|е|ом)?",)),
    ("TypeScript", (r"тайп[\s\-]?скрипт(?:а|е|ом)?",)),
    ("JavaScript", (r"джава[\s\-]?скрипт(?:а|е|ом)?",)),
    ("React", (r"реакт(?:а|е|ом)?",)),
    ("Playwright", (r"плейрайт(?:а|е|ом)?",)),
    ("WebSocket", (r"веб[\s\-]?сокет(?:а|е|ом|ы|ов)?",)),
]

_COMPILED_GLOSSARY_RULES: list[tuple[str, str, re.Pattern]] = [
    (
        canonical,
        canonical.lower(),
        re.compile(
            r"(?<![A-Za-zА-Яа-яЁё0-9])(?:" + "|".join(variants) + r")(?![A-Za-zА-Яа-яЁё0-9])",
            re.IGNORECASE,
        ),
    )
    for canonical, variants in _GLOSSARY_PHONETIC_RULES
]


def apply_custom_glossary_terms(text: str, custom_prompt: str = "") -> str:
    """
    Strictly opt-in glossary normalization: when the user supplies a custom_prompt
    containing specific Latin/IT terms (e.g. 'gRPC, OAuth 2, PostgreSQL, Swagger'),
    replaces unambiguous Cyrillic phonetic spellings of ONLY those requested terms.
    100% no-op when custom_prompt is empty.
    """
    if not text or not custom_prompt or not custom_prompt.strip():
        return text

    prompt_lower = custom_prompt.lower()
    out = text
    for canonical, key_lower, pattern in _COMPILED_GLOSSARY_RULES:
        if key_lower in prompt_lower:
            out = pattern.sub(canonical, out)
    return out


def _extract_clean_context_tail(text: str, max_chars: int = 120) -> str:
    """
    Extracts a verified, non-repetitive trailing snippet from the previous ASR
    window to preserve punctuation and term continuity across adjacent windows
    without injecting unrelated hardcoded domain prompts or enabling Whisper's
    drift-prone condition_on_previous_text.
    """
    cleaned = clean_hallucinated_text(text or "")
    if len(cleaned) < 16 or _CONSECUTIVE_REPEAT_RE.search(cleaned):
        return ""
    if is_whisper_phantom_hallucination(cleaned):
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    tail = cleaned[-max_chars:]
    space_pos = tail.find(" ")
    if 0 <= space_pos < len(tail) - 8:
        tail = tail[space_pos + 1 :]
    return tail.strip()


def _resolve_whisper_language(
    model,
    valid_windows: list[tuple[float, float, int, int]],
    samples: np.ndarray,
    requested_language: str = DEFAULT_ASR_LANGUAGE,
    profile_id: str = DEFAULT_ASR_MODEL,
) -> tuple[str, bool]:
    """
    Determines (language_code, use_multilingual) for Whisper decoding.
    Explicit language (e.g. 'ru') disables per-window multilingual re-detection
    so Whisper Large v3 never randomly flips Russian chunks into Ukrainian,
    Polish, Korean, or broken Latin. In 'auto' mode, probes up to 3 evenly
    spaced substantive windows and locks the language when they agree.
    """
    lang = (requested_language or DEFAULT_ASR_LANGUAGE).strip().lower()
    profile = ASR_MODEL_PROFILES.get(profile_id, {})

    if lang != "auto":
        return lang, False

    fixed_or_default = profile.get("fixed_language") or profile.get("default_language")
    if fixed_or_default:
        return str(fixed_or_default), False

    if not hasattr(model, "detect_language") or not valid_windows:
        return "ru", True

    substantive = [w for w in valid_windows if (w[1] - w[0]) >= 2.0] or list(valid_windows)
    if len(substantive) <= 3:
        probe_list = substantive
    else:
        probe_list = [
            substantive[0],
            substantive[len(substantive) // 2],
            substantive[-1],
        ]

    votes: list[tuple[str, float]] = []
    for _, _, s_idx, e_idx in probe_list:
        try:
            det_lang, det_prob, _ = model.detect_language(audio=samples[s_idx:e_idx])
            if det_lang and float(det_prob) >= 0.45:
                votes.append((str(det_lang), float(det_prob)))
        except Exception:
            continue

    if not votes:
        return "ru", True

    # Aggregate confidence per detected language
    score_by_lang: dict[str, float] = {}
    for code, prob in votes:
        score_by_lang[code] = score_by_lang.get(code, 0.0) + prob
    dominant_lang = max(score_by_lang.items(), key=lambda kv: kv[1])[0]
    distinct_langs = {code for code, _ in votes}
    avg_dominant_prob = score_by_lang[dominant_lang] / len(votes)

    # Lock language when all probed windows unanimously and confidently agree
    if len(distinct_langs) == 1 and avg_dominant_prob >= 0.80:
        return dominant_lang, False
    return dominant_lang, True


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
        pad_short: bool = True,
        min_start: float = 0.0,
        max_end: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        total_end = (len(samples) / float(sample_rate)) if max_end is None else max_end
        dur = end_sec - start_sec
        if pad_short and dur < 1.2:
            eff_start = max(min_start, start_sec - 0.05)
            eff_end = min(total_end, end_sec + 0.05)
        elif dur >= 1.0:
            # Trim 40ms inward when abutting an adjacent turn to avoid co-articulation bleed
            trim_l = 0.04 if (start_sec - min_start) < 0.15 and min_start > 0.0 else 0.0
            trim_r = (
                0.04
                if (max_end is not None and (max_end - end_sec) < 0.15)
                else 0.0
            )
            eff_start = start_sec + trim_l
            eff_end = end_sec - trim_r
        else:
            eff_start = start_sec
            eff_end = end_sec

        s_idx = max(0, int(eff_start * sample_rate))
        e_idx = min(len(samples), int(eff_end * sample_rate))
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
        return_voiceprints: bool = False,
        sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
    ):
        if cancel_token:
            cancel_token.check()
        if samples.size < 1600:  # < 0.1s
            return ([], {}, None) if return_voiceprints else []

        diar_threads = max(1, min(int(threads), 8))
        emb_workers = min(4, max(1, diar_threads // 2))
        emb_intra_threads = max(1, diar_threads // emb_workers)
        mode = (speaker_count or "auto").strip().lower()
        sens_key = (sensitivity or DEFAULT_DIARIZATION_SENSITIVITY).strip().lower()
        sens_prof = DIARIZATION_SENSITIVITY_PROFILES.get(
            sens_key, DIARIZATION_SENSITIVITY_PROFILES[DEFAULT_DIARIZATION_SENSITIVITY]
        )
        seg_threshold = float(sens_prof.get("seg_threshold", DIARIZATION_SEGMENTATION_THRESHOLD))

        # Pass 1: High-recall speaker-turn boundary detection via pyannote-segmentation-3.0.
        # Fine min_duration_on/off (0.18s) and tight local clustering threshold preserve
        # rapid Q&A replies ('Ну да.', 'Все.', 'Немного?') instead of fusing adjacent speakers.
        diar_key = (
            seg_model_path,
            emb_model_path,
            diar_threads,
            emb_intra_threads,
            seg_threshold,
            0.18,
            0.18,
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
                    threshold=seg_threshold,
                ),
                min_duration_on=0.18,
                min_duration_off=0.18,
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
            return ([], {}, extractor) if return_voiceprints else []

        # Merge only ultra-short intra-speaker pauses (<=0.20s, <=6s span) before CAM++ extraction
        pre_merged: list[DiarizationSegment] = [
            DiarizationSegment(
                speech_segments[0].start, speech_segments[0].end, speech_segments[0].speaker
            )
        ]
        for cur in speech_segments[1:]:
            prev = pre_merged[-1]
            if (
                cur.speaker == prev.speaker
                and (cur.start - prev.end) <= 0.20
                and (cur.end - prev.start) <= 6.0
            ):
                prev.end = max(prev.end, cur.end)
            else:
                pre_merged.append(DiarizationSegment(cur.start, cur.end, cur.speaker))

        if status_cb and len(pre_merged) > 30:
            status_cb("Диаризация спикеров: 90% (сопоставление голосовых отпечатков)...")

        total_dur_s = len(samples) / 16000.0

        def _neighbor_bounds(idx: int) -> tuple[float, float]:
            lo = pre_merged[idx - 1].end if idx > 0 else 0.0
            hi = pre_merged[idx + 1].start if idx + 1 < len(pre_merged) else total_dur_s
            return lo, hi

        if emb_workers <= 1 or len(pre_merged) < 8:
            embeddings: list[Optional[np.ndarray]] = []
            for i, seg in enumerate(pre_merged):
                if cancel_token:
                    cancel_token.check()
                lo_b, hi_b = _neighbor_bounds(i)
                embeddings.append(
                    cls._extract_segment_embedding(
                        extractor,
                        samples,
                        seg.start,
                        seg.end,
                        sample_rate=16000,
                        min_start=lo_b,
                        max_end=hi_b,
                    )
                )
        else:
            embeddings = [None] * len(pre_merged)

            def _embed_idx(idx_seg: tuple[int, DiarizationSegment]) -> tuple[int, Optional[np.ndarray]]:
                i, seg = idx_seg
                if cancel_token:
                    cancel_token.check()
                lo_b, hi_b = _neighbor_bounds(i)
                vec = cls._extract_segment_embedding(
                    extractor,
                    samples,
                    seg.start,
                    seg.end,
                    sample_rate=16000,
                    min_start=lo_b,
                    max_end=hi_b,
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

        if return_voiceprints:
            clustered_segs, centroids = cluster_anchor_embeddings(
                segments=pre_merged,
                embeddings=embeddings,
                speaker_count=mode,
                return_centroids=True,
                sensitivity=sens_key,
            )
            return clustered_segs, centroids, extractor

        return cluster_anchor_embeddings(
            segments=pre_merged,
            embeddings=embeddings,
            speaker_count=mode,
            sensitivity=sens_key,
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
        language: str = DEFAULT_ASR_LANGUAGE,
        custom_prompt: str = "",
    ) -> list[TimedUnit]:
        if cancel_token:
            cancel_token.check()

        if device == "cpu":
            total_threads = max(1, int(threads))
            if profile_id in ("large-v3", "large-v3-ru"):
                max_workers_cap = 1
            elif profile_id == "turbo":
                max_workers_cap = 2
            else:
                max_workers_cap = 3
            if total_threads >= 8 and max_workers_cap > 1:
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
                max_same_speaker_sec=26.5,
                merge_gap_sec=2.0 if device == "cpu" else 1.6,
                split_speaker_turns=(device != "cpu"),
            )
            if windows:
                windows = subdivide_long_asr_windows(
                    windows, samples, sample_rate=sample_rate, max_window_sec=28.5
                )
                windows = _augment_windows_with_vad(
                    windows, samples, sample_rate=sample_rate, max_window_sec=28.5
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

        resolved_lang, use_multilingual = _resolve_whisper_language(
            model=model,
            valid_windows=valid_windows,
            samples=samples,
            requested_language=language,
            profile_id=profile_id,
        )
        cleaned_custom_prompt = " ".join((custom_prompt or "").split())[:320]
        if status_cb:
            mode_desc = "мультиязычный авто" if use_multilingual else f"фиксированный ({resolved_lang})"
            prompt_note = " + пользовательский словарь" if cleaned_custom_prompt else ""
            status_cb(f"Язык распознавания Whisper: {mode_desc}{prompt_note}")

        if device == "cuda":
            beam_size = 5
        elif profile_id in ("large-v3", "large-v3-ru"):
            beam_size = 3
        else:
            beam_size = 2
        temperatures = (0.0, 0.2, 0.4)
        comp_ratio_thr = 2.4
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

        def _decode_single_window(
            win_item: tuple[float, float, int, int],
            rolling_tail: str = "",
        ) -> list[TimedUnit]:
            w_start, w_end, s_idx, e_idx = win_item
            if cancel_token:
                cancel_token.check()

            win_dur = w_end - w_start
            use_vad = win_dur > 30.0
            # Never inject hardcoded domain jargon prompts; combine optional user-provided
            # glossary prompt with verified rolling tail when present.
            if cleaned_custom_prompt and rolling_tail:
                win_prompt: Optional[str] = f"{cleaned_custom_prompt}\n{rolling_tail}"
            elif cleaned_custom_prompt:
                win_prompt = cleaned_custom_prompt
            elif rolling_tail:
                win_prompt = rolling_tail
            else:
                win_prompt = None

            transcribe_kwargs = dict(
                language=resolved_lang,
                task="transcribe",
                multilingual=use_multilingual,
                word_timestamps=True,
                prepend_punctuations="\"'“¿([{",
                vad_filter=use_vad,
                condition_on_previous_text=False,
                initial_prompt=win_prompt,
                no_speech_threshold=0.6,
                compression_ratio_threshold=comp_ratio_thr,
                log_prob_threshold=-1.0,
                repetition_penalty=1.08,
                hallucination_silence_threshold=1.5,
                beam_size=beam_size,
                temperature=temperatures,
            )
            if use_vad:
                transcribe_kwargs["vad_parameters"] = dict(
                    min_silence_duration_ms=350, speech_pad_ms=220
                )

            win_audio = normalize_chunk_level(samples[s_idx:e_idx])
            segments_iter, _ = model.transcribe(win_audio, **transcribe_kwargs)

            win_units: list[TimedUnit] = []
            for seg in segments_iter:
                if cancel_token:
                    cancel_token.check()

                abs_seg_start = w_start + float(seg.start)
                abs_seg_end = min(w_end, w_start + float(seg.end))

                seg_no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                seg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
                seg_comp_ratio = float(getattr(seg, "compression_ratio", 0.0) or 0.0)

                if seg_no_speech > 0.78 and seg_logprob < -1.0:
                    continue
                if seg_comp_ratio > 2.6:
                    continue

                if not used_diar_windows:
                    seg_s = max(0, int(abs_seg_start * sample_rate))
                    seg_e = min(len(samples), int(abs_seg_end * sample_rate))
                    if is_narrowband_tone(samples[seg_s:seg_e], sample_rate=sample_rate):
                        continue

                cleaned_seg_text = clean_hallucinated_text(seg.text or "")
                if not cleaned_seg_text:
                    continue
                if is_whisper_phantom_hallucination(
                    cleaned_seg_text,
                    no_speech_prob=seg_no_speech,
                    avg_logprob=seg_logprob,
                ):
                    continue
                if cleaned_custom_prompt and is_whisper_prompt_echo(
                    cleaned_seg_text,
                    cleaned_custom_prompt,
                    no_speech_prob=seg_no_speech,
                    avg_logprob=seg_logprob,
                ):
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
            prev_tail = ""
            prev_w_end: Optional[float] = None
            for win_item in valid_windows:
                w_s, w_e, _, _ = win_item
                carry = prev_tail if (prev_w_end is not None and (w_s - prev_w_end) <= 2.5) else ""
                w_units = _decode_single_window(win_item, rolling_tail=carry)
                units.extend(w_units)
                if w_units:
                    prev_tail = _extract_clean_context_tail(
                        _join_unit_texts([u.text for u in w_units])
                    )
                else:
                    prev_tail = ""
                prev_w_end = w_e
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
        language: str = DEFAULT_ASR_LANGUAGE,
        custom_prompt: str = "",
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
                language=language,
                custom_prompt=custom_prompt,
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
                            language=language,
                            custom_prompt=custom_prompt,
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
                    language=language,
                    custom_prompt=custom_prompt,
                )
            raise

    @staticmethod
    def _run_sherpa_transducer(
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        diar_segments: Optional[list[DiarizationSegment]] = None,
        profile_id: str = "gigaam-v3",
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[TimedUnit]:
        """
        Runs offline NeMo transducer ONNX models via sherpa-onnx (GigaAM v3 e2e RNN-T
        with feature_dim=64 and Parakeet TDT v3 with feature_dim=80) using Silero VAD
        + diarization union longform chunking (<=20s, 1.5s sliding overlap on pauses-free
        monologues), per-chunk quiet-speech leveling, and fault-isolated retries.
        """
        if cancel_token:
            cancel_token.check()

        profile = ASR_MODEL_PROFILES.get(profile_id, {})
        encoder_file = profile.get("encoder_file", "encoder.int8.onnx")
        decoder_file = profile.get("decoder_file", "decoder.onnx")
        joiner_file = profile.get("joiner_file", "joiner.onnx")
        tokens_file = profile.get("tokens_file", "tokens.txt")
        feature_dim = int(profile.get("feature_dim", 64))
        max_chunk_sec = float(profile.get("max_chunk_sec", 20.0))
        overlap_sec = float(profile.get("overlap_sec", 1.5))
        progress_tag = "Parakeet" if profile_id == "parakeet" else "GigaAM v3"

        asr_threads = max(1, int(threads))
        if profile_id == "parakeet":
            cache_key: tuple = ("sherpa_parakeet", model_dir, asr_threads)
        else:
            cache_key = ("sherpa_transducer", profile_id, model_dir, asr_threads)

        recognizer = _ModelCache.get_or_create(
            cache_key,
            lambda: sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=os.path.join(model_dir, encoder_file),
                decoder=os.path.join(model_dir, decoder_file),
                joiner=os.path.join(model_dir, joiner_file),
                tokens=os.path.join(model_dir, tokens_file),
                num_threads=asr_threads,
                sample_rate=16000,
                feature_dim=feature_dim,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
                provider="cpu",
            ),
        )

        chunks = build_longform_asr_chunks(
            samples=samples,
            sample_rate=sample_rate,
            diar_segments=diar_segments,
            max_chunk_sec=max_chunk_sec,
            overlap_sec=overlap_sec,
            use_silero_vad=True,
        )
        if not chunks:
            return []

        valid_chunks: list[tuple[float, float, float, float, int, int]] = []
        for c_start, c_end, k_start, k_end in chunks:
            s_idx = max(0, int(c_start * sample_rate))
            e_idx = min(len(samples), int(c_end * sample_rate))
            if e_idx - s_idx >= 800:
                valid_chunks.append((c_start, c_end, k_start, k_end, s_idx, e_idx))

        if not valid_chunks:
            return []

        def _extract_stream_units(stream, c_start: float, c_end: float) -> list[TimedUnit]:
            res = stream.result
            out_u: list[TimedUnit] = []
            if res.tokens and res.timestamps and len(res.tokens) == len(res.timestamps):
                chunk_units = group_bpe_tokens_to_units(
                    list(res.tokens), list(res.timestamps), c_start, c_end
                )
                for u in chunk_units:
                    if u.text and u.text.strip() and not is_repetitive_noise_token(u.text):
                        out_u.append(u)
            elif res.text and res.text.strip():
                cleaned = clean_hallucinated_text(res.text.strip())
                if cleaned:
                    out_u.append(TimedUnit(start=c_start, end=c_end, text=cleaned))
            return out_u

        def _decode_single_fallback(c_start: float, c_end: float, s_idx: int, e_idx: int) -> list[TimedUnit]:
            dur = c_end - c_start
            if dur > 10.0:
                mid_s = 0.5 * (c_start + c_end)
                mid_idx = max(s_idx + 800, min(e_idx - 800, int(mid_s * sample_rate)))
                sub_spans = [(c_start, mid_s, s_idx, mid_idx), (mid_s, c_end, mid_idx, e_idx)]
            else:
                sub_spans = [(c_start, c_end, s_idx, e_idx)]
            rescued: list[TimedUnit] = []
            for sub_s, sub_e, sub_si, sub_ei in sub_spans:
                if cancel_token:
                    cancel_token.check()
                if sub_ei - sub_si < 800:
                    continue
                try:
                    st = recognizer.create_stream()
                    st.accept_waveform(sample_rate, normalize_chunk_level(samples[sub_si:sub_ei]))
                    recognizer.decode_stream(st)
                    rescued.extend(_extract_stream_units(st, sub_s, sub_e))
                except Exception:
                    continue
            return rescued

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
            chunk_wavs: list[np.ndarray] = []
            for _, _, _, _, s_idx, e_idx in batch:
                stream = recognizer.create_stream()
                chunk_samples = normalize_chunk_level(samples[s_idx:e_idx])
                chunk_wavs.append(chunk_samples)
                stream.accept_waveform(sample_rate, chunk_samples)
                streams.append(stream)

            batch_failed = False
            if hasattr(recognizer, "decode_streams") and len(streams) > 1:
                try:
                    recognizer.decode_streams(streams)
                except Exception:
                    batch_failed = True
            else:
                batch_failed = True

            if batch_failed:
                for stream in streams:
                    if cancel_token:
                        cancel_token.check()
                    try:
                        recognizer.decode_stream(stream)
                    except Exception:
                        pass

            if cancel_token:
                cancel_token.check()

            for (c_start, c_end, k_start, k_end, s_idx, e_idx), stream, chunk_wav in zip(
                batch, streams, chunk_wavs
            ):
                chunk_units = _extract_stream_units(stream, c_start, c_end)
                # Fault-isolated retry if an energetic speech chunk returned empty output
                if not chunk_units and (c_end - c_start) >= 1.2:
                    chunk_rms = float(np.sqrt(np.mean(chunk_wav * chunk_wav))) if chunk_wav.size else 0.0
                    if chunk_rms > 0.015:
                        chunk_units = _decode_single_fallback(c_start, c_end, s_idx, e_idx)

                has_left_ov = k_start > c_start + 1e-4
                has_right_ov = k_end < c_end - 1e-4
                for u in chunk_units:
                    if has_left_ov or has_right_ov:
                        mid = 0.5 * (u.start + u.end)
                        if has_left_ov and mid < k_start:
                            continue
                        if has_right_ov and mid >= k_end:
                            continue
                    units.append(u)

            done_idx = b_start + len(batch)
            if status_cb and total_chunks > 0:
                pct = min(99, int((done_idx * 100) / total_chunks))
                now = time.time()
                if pct >= last_pct + 20 and (now - last_log_t) >= 2.0:
                    last_pct = pct
                    last_log_t = now
                    status_cb(f"Распознавание речи ({progress_tag}): {pct}% ({done_idx}/{total_chunks})...")

        return units

    @classmethod
    def _run_sherpa_parakeet(
        cls,
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        diar_segments: Optional[list[DiarizationSegment]] = None,
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[TimedUnit]:
        return cls._run_sherpa_transducer(
            samples=samples,
            sample_rate=sample_rate,
            model_dir=model_dir,
            diar_segments=diar_segments,
            profile_id="parakeet",
            threads=threads,
            status_cb=status_cb,
            cancel_token=cancel_token,
        )

    @classmethod
    def _run_asr_engine(
        cls,
        samples: np.ndarray,
        sample_rate: int,
        profile_id: str,
        profile: dict,
        asr_dir: str,
        diar_segments: Optional[list[DiarizationSegment]],
        threads: int,
        compute_device: str,
        language: str,
        custom_prompt: str,
        status_cb: Optional[Callable[[str], None]],
        cancel_token: Optional[CancellationToken],
    ) -> list[TimedUnit]:
        req_dev = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()
        if profile["engine"] == "faster_whisper":
            eff_dev, _ = resolve_compute_device(req_dev)
            dev_label = "GPU CUDA" if eff_dev == "cuda" else "CPU"
            if status_cb:
                status_cb(f"Локальная транскрибация речи ({profile['short_name']}, {dev_label})...")
            return cls._run_faster_whisper(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                profile_id=profile_id,
                diar_segments=diar_segments,
                threads=threads,
                compute_device=req_dev,
                status_cb=status_cb,
                cancel_token=cancel_token,
                language=language,
                custom_prompt=custom_prompt,
            )
        if profile["engine"] in ("sherpa_transducer", "sherpa_parakeet"):
            if req_dev == "cuda" and status_cb:
                status_cb(
                    f"Модель {profile['short_name']} работает на CPU (ONNX INT8); "
                    "для GPU CUDA доступны Whisper Large v3 RU, Turbo и Large v3."
                )
            if status_cb:
                status_cb(f"Локальная транскрибация речи ({profile['short_name']}, CPU)...")
            return cls._run_sherpa_transducer(
                samples=samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                diar_segments=diar_segments,
                profile_id=profile_id,
                threads=threads,
                status_cb=status_cb,
                cancel_token=cancel_token,
            )
        raise ValueError(f"Unsupported engine: {profile['engine']}")

    @classmethod
    def transcribe_units_only(
        cls,
        wav_path: str,
        asr_model: str = DEFAULT_ASR_MODEL,
        compute_device: str = "cuda",
        threads: int = 1,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
        language: str = DEFAULT_ASR_LANGUAGE,
        custom_prompt: str = "",
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
                "custom_prompt": custom_prompt,
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
            language=language,
            custom_prompt=custom_prompt,
        )
        elapsed_asr = time.time() - t_asr
        update(f"[Profile] GPU ASR завершена за {elapsed_asr:.2f}s (единиц текста: {len(units)})")

        return {
            "samples": samples,
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "units": units,
            "profile_id": profile_id,
            "custom_prompt": custom_prompt,
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
        diarization_sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
        custom_prompt: str = "",
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
        sample_rate = int(asr_state.get("sample_rate", 16000))
        duration_sec = float(asr_state.get("duration_sec", 0.0))
        units: list[TimedUnit] = list(asr_state.get("units", []))
        profile_id = str(asr_state.get("profile_id", DEFAULT_ASR_MODEL))
        eff_prompt = custom_prompt if custom_prompt else str(asr_state.get("custom_prompt", ""))
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
            f"Разделение говорящих по ролям (режим спикеров: {speaker_count}, "
            f"чувствительность: {diarization_sensitivity}, потоков CPU: {eff_threads})..."
        )
        t_diar = time.time()
        diar_out = cls._run_diarization(
            samples=samples,
            seg_model_path=seg_onnx,
            emb_model_path=emb_onnx,
            speaker_count=speaker_count,
            threads=eff_threads,
            status_cb=update,
            cancel_token=cancel_token,
            return_voiceprints=True,
            sensitivity=diarization_sensitivity,
        )
        if isinstance(diar_out, tuple):
            diar_segments, speaker_centroids, emb_extractor = diar_out
        else:
            diar_segments, speaker_centroids, emb_extractor = diar_out, None, None

        detected_raw_speakers = len({s.speaker for s in diar_segments})
        update(
            f"[Profile] Диаризация завершена за {time.time() - t_diar:.2f}s "
            f"(сегментов: {len(diar_segments)}, голосов: {detected_raw_speakers})"
        )

        if cancel_token:
            cancel_token.check()

        turns, final_speaker_count = align_units_to_speakers(
            units,
            diar_segments,
            samples=samples,
            sample_rate=sample_rate,
            speaker_centroids=speaker_centroids,
            embedding_extractor=emb_extractor,
        )
        if eff_prompt and eff_prompt.strip():
            turns = [
                SpeakerTurn(t.start, t.end, t.speaker_label, apply_custom_glossary_terms(t.text, eff_prompt))
                for t in turns
            ]
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
        language: str = DEFAULT_ASR_LANGUAGE,
        custom_prompt: str = "",
        diarization_sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
    ) -> dict:
        """
        Runs local speaker diarization + speech recognition on a 16 kHz mono WAV
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
        update(
            f"Разделение говорящих по ролям (режим спикеров: {speaker_count}, "
            f"чувствительность: {diarization_sensitivity})..."
        )
        t_diar = time.time()
        diar_out = cls._run_diarization(
            samples=samples,
            seg_model_path=seg_onnx,
            emb_model_path=emb_onnx,
            speaker_count=speaker_count,
            threads=eff_diar_threads,
            status_cb=update,
            cancel_token=cancel_token,
            return_voiceprints=True,
            sensitivity=diarization_sensitivity,
        )
        if isinstance(diar_out, tuple):
            diar_segments, speaker_centroids, emb_extractor = diar_out
        else:
            diar_segments, speaker_centroids, emb_extractor = diar_out, None, None

        detected_raw_speakers = len({s.speaker for s in diar_segments})
        update(
            f"[Profile] Диаризация завершена за {time.time() - t_diar:.2f}s "
            f"(сегментов: {len(diar_segments)}, голосов: {detected_raw_speakers})"
        )

        # 3. Speech Recognition
        t_asr = time.time()
        units = cls._run_asr_engine(
            samples=samples,
            sample_rate=sample_rate,
            profile_id=profile_id,
            profile=profile,
            asr_dir=asr_dir,
            diar_segments=diar_segments,
            threads=threads,
            compute_device=compute_device,
            language=language,
            custom_prompt=custom_prompt,
            status_cb=update,
            cancel_token=cancel_token,
        )
        update(f"[Profile] ASR завершена за {time.time() - t_asr:.2f}s (единиц текста: {len(units)})")

        if cancel_token:
            cancel_token.check()

        # 4. Align words/units to speaker roles and write UTF-8 .txt
        turns, final_speaker_count = align_units_to_speakers(
            units,
            diar_segments,
            samples=samples,
            sample_rate=sample_rate,
            speaker_centroids=speaker_centroids,
            embedding_extractor=emb_extractor,
        )
        if custom_prompt and custom_prompt.strip():
            turns = [
                SpeakerTurn(t.start, t.end, t.speaker_label, apply_custom_glossary_terms(t.text, custom_prompt))
                for t in turns
            ]
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

    @classmethod
    def transcribe_dual_track_files(
        cls,
        track1_wav_path: str,
        track2_wav_path: str,
        output_txt_path: str,
        asr_model: str = DEFAULT_ASR_MODEL,
        speaker_count: str = "auto",
        threads: int = 4,
        diar_threads: Optional[int] = None,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
        language: str = DEFAULT_ASR_LANGUAGE,
        custom_prompt: str = "",
        diarization_sensitivity: str = DEFAULT_DIARIZATION_SENSITIVITY,
    ) -> dict:
        """
        Dual-track OBS / virtual-mixer transcription pipeline:
          - Track 1 (Candidate microphone): transcribed directly with Silero VAD + ASR
            and assigned deterministically to 'Спикер 1' (100% accuracy, zero diarization confusion).
          - Track 2 (System audio / Interviewers): diarized independently (allocating
            speaker_count - 1 speakers when exact count is given) and mapped to
            'Спикер 2', 'Спикер 3', ...
          - All turns are merged chronologically by timestamp into one unified transcript.
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
        samples1, sr1 = load_wav_16k_mono(track1_wav_path, normalize_gain=True)
        samples2, sr2 = load_wav_16k_mono(track2_wav_path, normalize_gain=True)
        dur1 = len(samples1) / float(sr1) if sr1 > 0 else 0.0
        dur2 = len(samples2) / float(sr2) if sr2 > 0 else 0.0
        duration_sec = max(dur1, dur2)

        os.makedirs(os.path.dirname(os.path.abspath(output_txt_path)), exist_ok=True)
        if samples1.size < 1600 and samples2.size < 1600:
            txt = format_transcript_txt(
                turns=[],
                duration_sec=duration_sec,
                model_name=profile["short_name"],
                num_speakers=0,
                note="[Речь в аудиодорожках не обнаружена]",
            )
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(txt)
            return {"num_speakers": 0, "num_turns": 0, "elapsed": time.time() - t0}

        asr_dir = ModelManager.ensure_asr_model(profile_id, status_cb=update)
        if cancel_token:
            cancel_token.check()

        # Track 1: Deterministic Speaker 1 (Candidate Mic) — no diarization needed!
        t1_turns: list[SpeakerTurn] = []
        if samples1.size >= 1600:
            update("Дорожка 1 (Спикер 1 / микрофон): распознавание речи без диаризации...")
            t1_units = cls._run_asr_engine(
                samples=samples1,
                sample_rate=sr1,
                profile_id=profile_id,
                profile=profile,
                asr_dir=asr_dir,
                diar_segments=None,
                threads=threads,
                compute_device=compute_device,
                language=language,
                custom_prompt=custom_prompt,
                status_cb=update,
                cancel_token=cancel_token,
            )
            t1_turns, _ = align_units_to_speakers(t1_units, diar_segments=[])
            t1_turns = [
                SpeakerTurn(t.start, t.end, "Спикер 1", t.text)
                for t in t1_turns
                if t.text and t.text.strip()
            ]
            update(f"Дорожка 1 готова: {len(t1_turns)} реплик (Спикер 1).")

        if cancel_token:
            cancel_token.check()

        # Track 2: Diarize interviewers independently (no contamination from Speaker 1!)
        t2_turns: list[SpeakerTurn] = []
        t2_spk_count = 0
        if samples2.size >= 1600:
            mode_clean = (speaker_count or "auto").strip().lower()
            if mode_clean.isdigit() and int(mode_clean) >= 2:
                track2_speaker_mode = str(int(mode_clean) - 1)
            else:
                track2_speaker_mode = mode_clean

            seg_onnx, emb_onnx = ModelManager.ensure_diarization_models(status_cb=update)
            if cancel_token:
                cancel_token.check()

            eff_diar_threads = diar_threads if diar_threads is not None else threads
            update(
                f"Дорожка 2 (собеседники): диаризация (режим спикеров: {track2_speaker_mode}, "
                f"чувствительность: {diarization_sensitivity})..."
            )
            diar_out = cls._run_diarization(
                samples=samples2,
                seg_model_path=seg_onnx,
                emb_model_path=emb_onnx,
                speaker_count=track2_speaker_mode,
                threads=eff_diar_threads,
                status_cb=update,
                cancel_token=cancel_token,
                return_voiceprints=True,
                sensitivity=diarization_sensitivity,
            )
            if isinstance(diar_out, tuple):
                diar_segments2, speaker_centroids2, emb_extractor2 = diar_out
            else:
                diar_segments2, speaker_centroids2, emb_extractor2 = diar_out, None, None

            t2_units = cls._run_asr_engine(
                samples=samples2,
                sample_rate=sr2,
                profile_id=profile_id,
                profile=profile,
                asr_dir=asr_dir,
                diar_segments=diar_segments2,
                threads=threads,
                compute_device=compute_device,
                language=language,
                custom_prompt=custom_prompt,
                status_cb=update,
                cancel_token=cancel_token,
            )
            raw_t2_turns, t2_spk_count = align_units_to_speakers(
                t2_units,
                diar_segments2,
                samples=samples2,
                sample_rate=sr2,
                speaker_centroids=speaker_centroids2,
                embedding_extractor=emb_extractor2,
            )
            offset = 1 if t1_turns else 0
            label_remap: dict[str, str] = {}
            for t in raw_t2_turns:
                if t.speaker_label not in label_remap:
                    label_remap[t.speaker_label] = f"Спикер {offset + len(label_remap) + 1}"
                t2_turns.append(SpeakerTurn(t.start, t.end, label_remap[t.speaker_label], t.text))
            update(f"Дорожка 2 готова: {len(t2_turns)} реплик (спикеров: {len(label_remap)}).")

        if cancel_token:
            cancel_token.check()

        merged_turns = sorted(t1_turns + t2_turns, key=lambda t: (t.start, t.end))
        if custom_prompt and custom_prompt.strip():
            merged_turns = [
                SpeakerTurn(
                    t.start, t.end, t.speaker_label, apply_custom_glossary_terms(t.text, custom_prompt)
                )
                for t in merged_turns
            ]
        final_speaker_count = len({t.speaker_label for t in merged_turns})
        track_note = (
            "# Режим дорожек: раздельный (Дорожка 1 = Спикер 1, Дорожка 2 = диаризация собеседников)"
        )
        transcript_text = format_transcript_txt(
            turns=merged_turns,
            duration_sec=duration_sec,
            model_name=profile["short_name"],
            num_speakers=final_speaker_count,
            note=track_note,
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        total_elapsed = time.time() - t0
        update(
            f"[Profile] Двухдорожечный транскрипт сохранён ({len(merged_turns)} реплик, "
            f"спикеров: {final_speaker_count}) за {total_elapsed:.2f}s"
        )
        return {
            "num_speakers": final_speaker_count,
            "num_turns": len(merged_turns),
            "elapsed": total_elapsed,
            "dual_track": True,
        }



