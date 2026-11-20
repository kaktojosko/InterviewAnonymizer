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
import tempfile
import threading
import time
from typing import Callable, Optional
import wave

import ctranslate2
import numpy as np
import onnxruntime as ort
import sherpa_onnx
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

try:
    import onnx_asr
except ImportError:  # pragma: no cover
    onnx_asr = None

from src.cancellation import CancellationToken
from src.config import (
    APP_VERSION,
    ASR_LANGUAGE_OPTIONS,
    ASR_MODEL_PROFILES,
    COMPUTE_DEVICE_OPTIONS,
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_DEEPGRAM_API_KEY,
    DEFAULT_DEEPGRAM_BASE_URL,
    DEFAULT_DIARIZATION_SENSITIVITY,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DENOISE_ENABLED_DEFAULT,
    DIARIZATION_CENTROID_SIM_THRESHOLD,
    DIARIZATION_MERGE_GAP_SEC,
    DIARIZATION_MIN_SEGMENT_SEC,
    DIARIZATION_MULTI_CENTROID_SIM_THRESHOLD,
    DIARIZATION_SEGMENTATION_THRESHOLD,
    DIARIZATION_SENSITIVITY_PROFILES,
    LLM_POLISH_ENABLED_DEFAULT,
    TRANSCRIPT_PAUSE_BREAK_SEC,
)
from src.deepgram_asr import transcribe_with_deepgram_nova3
from src.llm_postprocessor import polish_turns_with_local_llm
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
    unique_count = len(set(letters))
    if len(kept_words) <= 1 and is_repetitive_noise_token(letters):
        return ""
    if len(letters) >= 16 and unique_count <= 5:
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
    minor_min_sec: float = 5.0,
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
    if total_speech >= 120.0:
        max_cap = min(15.0, floor_sec * 3.0)
        min_major_dur = max(floor_sec, min(max_cap, 0.010 * total_speech))
    else:
        min_major_dur = min(floor_sec, max(2.0, 0.010 * total_speech))

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
         centroid, followed by coherence-aware centroid refinement and voiceprint export.
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
    minor_min_sec = float(sens_prof.get("minor_min_sec", 5.0))
    svd_gap_thr = float(sens_prof.get("svd_gap_threshold", 0.095))
    svd_max_cent_sim = float(sens_prof.get("svd_max_cent_sim", 0.82))

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
            if mode == "4+" and k_cur <= 4 and best_avg_sim < max(eff_sim_thr, 0.62):
                break
            if best_avg_sim < eff_sim_thr:
                break
            # Lower-quantile guard prevents chaining-drift merges across distinct similar voices
            sub_block = pair_sim[np.ix_(cluster_pos_lists[bi], cluster_pos_lists[bj])]
            low_q_sim = float(np.percentile(sub_block, 20)) if sub_block.size > 1 else best_avg_sim
            if low_q_sim < (eff_sim_thr - 0.18):
                blocked[bi, bj] = True
                blocked[bj, bi] = True
                continue

        if bi > bj:
            bi, bj = bj, bi
        clusters[bi].extend(clusters[bj])
        clusters.pop(bj)
        cluster_pos_lists[bi].extend(cluster_pos_lists[bj])
        cluster_pos_lists.pop(bj)

        # Update weighted_sum and reset block flags for the newly grown cluster (never propagate
        # stale single-outlier blocks virally across unrelated valid merges).
        c_weights[bi] += c_weights[bj]
        c_weights.pop(bj)
        weighted_sum[bi, :] += weighted_sum[bj, :]
        weighted_sum[:, bi] += weighted_sum[:, bj]
        weighted_sum = np.delete(weighted_sum, bj, axis=0)
        weighted_sum = np.delete(weighted_sum, bj, axis=1)
        blocked[bi, :] = False
        blocked[:, bi] = False
        blocked = np.delete(blocked, bj, axis=0)
        blocked = np.delete(blocked, bj, axis=1)

    # Channel-invariant multimodal cluster splitter (auto mode):
    # When multiple same-gender speakers share a single conference downlink (e.g. 3 female
    # interviewers on Zoom/Telemost), their shared channel+gender vector inflates raw
    # inter-speaker cosine similarity into 0.60-0.75. Subtracting the cluster mean (SVD top
    # singular direction) cancels the shared channel offset: a true single speaker is
    # isotropic (within-between gap <= 0.04, cent_sim >= 0.90), whereas 2+ merged speakers
    # exhibit a sharp coherence gap (>= svd_gap_thr) and clearly separated sub-centroids.
    if target_k is None and clusters:
        min_half_anchors = 5
        min_half_dur = 12.0
        splits_done = 0
        split_changed = True
        while split_changed and splits_done < 4:
            split_changed = False
            next_clusters: list[list[int]] = []
            for grp in clusters:
                grp_dur = sum(durations[i] for i in grp)
                if len(grp) < 2 * min_half_anchors or grp_dur < 2.0 * min_half_dur:
                    next_clusters.append(grp)
                    continue
                pos_arr = np.array([anchor_pos[i] for i in grp], dtype=np.int64)
                sub_vecs = anchor_mat[pos_arr]
                centered = sub_vecs - sub_vecs.mean(axis=0, keepdims=True)
                try:
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                except np.linalg.LinAlgError:
                    next_clusters.append(grp)
                    continue
                proj = centered @ vt[0]
                c_lo = float(np.percentile(proj, 20))
                c_hi = float(np.percentile(proj, 80))
                mask_a = proj <= 0.5 * (c_lo + c_hi)
                for _ in range(12):
                    n_a = int(np.count_nonzero(mask_a))
                    if n_a == 0 or n_a == len(grp):
                        break
                    c_lo = float(proj[mask_a].mean())
                    c_hi = float(proj[~mask_a].mean())
                    mask_a = proj <= 0.5 * (c_lo + c_hi)

                grp_a = [grp[k] for k in range(len(grp)) if mask_a[k]]
                grp_b = [grp[k] for k in range(len(grp)) if not mask_a[k]]
                dur_a = sum(durations[i] for i in grp_a)
                dur_b = sum(durations[i] for i in grp_b)
                if (
                    len(grp_a) < min_half_anchors
                    or len(grp_b) < min_half_anchors
                    or dur_a < min_half_dur
                    or dur_b < min_half_dur
                ):
                    next_clusters.append(grp)
                    continue

                pa = pos_arr[mask_a]
                pb = pos_arr[~mask_a]
                sa = pair_sim[np.ix_(pa, pa)]
                sb = pair_sim[np.ix_(pb, pb)]
                sc = pair_sim[np.ix_(pa, pb)]
                w_a = float((sa.sum() - len(pa)) / max(1, len(pa) * (len(pa) - 1)))
                w_b = float((sb.sum() - len(pb)) / max(1, len(pb) * (len(pb) - 1)))
                btwn = float(sc.mean())
                coherence_gap = min(w_a, w_b) - btwn
                cent_a = _compute_centroid(grp_a)
                cent_b = _compute_centroid(grp_b)
                sub_cent_sim = float(np.dot(cent_a, cent_b))

                if coherence_gap >= svd_gap_thr and sub_cent_sim <= svd_max_cent_sim:
                    next_clusters.append(grp_a)
                    next_clusters.append(grp_b)
                    splits_done += 1
                    split_changed = True
                else:
                    next_clusters.append(grp)
            clusters = next_clusters

    # Absorb transient overlap/laugh/crosstalk micro-clusters into their closest major voiceprint
    if target_k is None and len(clusters) > 1:
        cluster_durs = [sum(durations[i] for i in c) for c in clusters]
        total_anchor_dur = sum(cluster_durs)
        if total_anchor_dur >= 25.0:
            if total_anchor_dur >= 120.0:
                max_cap = min(15.0, minor_min_sec * 3.0)
                min_keep_dur = max(minor_min_sec, min(max_cap, 0.010 * total_anchor_dur))
            else:
                min_keep_dur = min(minor_min_sec, max(2.2, 0.010 * total_anchor_dur))
            major_ids = [ci for ci, c_dur in enumerate(cluster_durs) if c_dur >= min_keep_dur]
            if major_ids and len(major_ids) < len(clusters):
                major_cents = {ci: _compute_centroid(clusters[ci]) for ci in major_ids}
                to_drop: set[int] = set()
                for ci in range(len(clusters)):
                    if ci in major_cents:
                        continue
                    c_vec = _compute_centroid(clusters[ci])
                    best_m = max(major_ids, key=lambda m: float(np.dot(c_vec, major_cents[m])))
                    best_m_sim = float(np.dot(c_vec, major_cents[best_m]))
                    if best_m_sim >= 0.28 or cluster_durs[ci] < minor_min_sec:
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

    # Coherence-aware refinement pass: recomputes centroids from >=1.5s assigned segments and
    # reunites noise-split shards of the SAME speaker (high averaged-centroid similarity and
    # negligible multimodal coherence gap) while strictly refusing to re-merge distinct
    # same-gender voices separated by UPGMA or SVD.
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

        def _within_mean(mat: np.ndarray) -> float:
            n_m = mat.shape[0]
            if n_m <= 1:
                return 1.0
            s_mat = mat @ mat.T
            return float((s_mat.sum() - n_m) / (n_m * (n_m - 1)))

        if len(refined_groups) >= 2:
            refine_high_thr = max(svd_max_cent_sim + 0.02, 0.85)
            refine_mid_thr = max(svd_max_cent_sim - 0.06, 0.72)
            ref_blocked: set[tuple[int, int]] = set()
            while target_k is None and len(refined_groups) > 1:
                if mode == "4+" and len(refined_groups) <= 4:
                    break
                r_mat = np.stack([_compute_centroid(g) for g in refined_groups], axis=0)
                r_sim = r_mat @ r_mat.T
                np.fill_diagonal(r_sim, -2.0)
                for a_i, b_i in ref_blocked:
                    if a_i < r_sim.shape[0] and b_i < r_sim.shape[1]:
                        r_sim[a_i, b_i] = -2.0
                        r_sim[b_i, a_i] = -2.0
                flat_best = int(np.argmax(r_sim))
                bi, bj = divmod(flat_best, r_sim.shape[1])
                best_cent_sim = float(r_sim[bi, bj])
                if best_cent_sim < refine_mid_thr:
                    break
                if bi > bj:
                    bi, bj = bj, bi

                emb_bi = np.stack([norm_embs[m] for m in refined_groups[bi] if norm_embs[m] is not None], axis=0)
                emb_bj = np.stack([norm_embs[m] for m in refined_groups[bj] if norm_embs[m] is not None], axis=0)
                cross_mean = float(np.mean(emb_bi @ emb_bj.T))
                both_substantive = emb_bi.shape[0] >= 3 and emb_bj.shape[0] >= 3
                coherence_gap = (
                    (min(_within_mean(emb_bi), _within_mean(emb_bj)) - cross_mean)
                    if both_substantive
                    else 0.0
                )

                should_merge = False
                if best_cent_sim >= 0.88 and cross_mean >= max(0.35, eff_sim_thr - 0.22):
                    should_merge = True
                elif (
                    best_cent_sim >= refine_high_thr
                    and cross_mean >= max(0.38, eff_sim_thr - 0.18)
                    and coherence_gap < svd_gap_thr
                ):
                    should_merge = True
                elif (
                    best_cent_sim >= refine_mid_thr
                    and cross_mean >= max(0.42, eff_sim_thr - 0.12)
                    and coherence_gap < (svd_gap_thr * 0.75)
                ):
                    should_merge = True

                if not should_merge:
                    ref_blocked.add((bi, bj))
                    continue
                refined_groups[bi].extend(refined_groups[bj])
                refined_groups.pop(bj)
                ref_blocked.clear()

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
                    threshold=0.28,
                    min_speech_duration_ms=140,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=220,
                    speech_pad_ms=220,
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

    # Merge overlapping spans first, then pack nearby islands up to max_len with +-220ms edge pad
    outer_pad = 0.22
    base_spans = _merge_time_intervals(raw_spans, gap_tol=0.14)
    packed: list[tuple[float, float]] = []
    for s, e in base_spans:
        ps = max(0.0, s - outer_pad)
        pe = min(total_dur, e + outer_pad)
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

    pause_ctx_ov = min(0.50, ov * 0.5) if ov > 0.0 else 0.0

    def _find_true_pause_split(cur_s: float, hard_end_s: float) -> Optional[float]:
        """
        Returns a genuine >=150ms RMS silence valley inside
        [cur + max_len - 6, cur + max_len - pause_ctx_ov - 0.15] if present.
        Using a 150ms (3x50ms) window avoids mistaking single 50ms unvoiced stop
        consonants ('п', 'т', 'к', 'с') for pauses.
        """
        win_n = max(1, int(0.05 * sample_rate))  # 50 ms step
        valley_n = win_n * 3                     # 150 ms valley width
        span_lo = max(0, int(cur_s * sample_rate))
        span_hi = min(len(samples), int(min(hard_end_s, cur_s + max_len) * sample_rate))
        if span_hi - span_lo < win_n * 10:
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
        search_hi = min(hard_end_s - 0.8, cur_s + max_len - pause_ctx_ov - 0.15)
        if search_hi <= search_lo:
            return None

        lo_i = max(0, int(search_lo * sample_rate))
        hi_i = min(len(samples), int(search_hi * sample_rate))
        best_pos: Optional[int] = None
        best_e = float("inf")
        for pos in range(lo_i, hi_i - valley_n + 1, win_n):
            f = samples[pos : pos + valley_n]
            e_val = float(np.dot(f, f)) / valley_n
            if e_val < best_e:
                best_e = e_val
                best_pos = pos + (valley_n // 2)

        # Require a true silence valley (<= 10% of median speech energy over 150ms)
        if best_pos is not None and best_e <= 0.10 * med_energy:
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
                # Always supply +-pause_ctx_ov (0.50s) acoustic context across the pause seam
                # while keeping ownership cores [keep_lo, pause_pt] strictly non-overlapping.
                c_end = min(island_e, cur + max_len, pause_pt + pause_ctx_ov)
                chunks_out.append((cur, c_end, keep_lo, pause_pt))
                cur = max(island_s, pause_pt - pause_ctx_ov)
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

    # Add +-0.40s acoustic context across tightly adjacent packed-island boundaries (< 0.65s gap,
    # e.g. where packing hit max_len) without ever exceeding max_len or overlapping keep cores.
    seam_ov = min(0.40, ov * 0.4) if ov > 0.0 else 0.0
    if seam_ov > 0.0 and len(chunks_out) >= 2:
        patched = [list(c) for c in chunks_out]
        for idx in range(len(patched) - 1):
            c1_s, c1_e, k1_s, k1_e = patched[idx]
            c2_s, c2_e, k2_s, k2_e = patched[idx + 1]
            # Only patch non-overlapping packed seams that are closely adjacent
            if c1_e <= c2_s + 1e-4 and (c2_s - c1_e) < 0.65:
                mid = 0.5 * (c1_e + c2_s)
                new_c1_e = min(total_dur, c1_s + max_len, max(c1_e, mid + seam_ov))
                new_c2_s = max(0.0, c2_e - max_len, min(c2_s, mid - seam_ov))
                if new_c1_e > c1_e + 0.02 or new_c2_s < c2_s - 0.02:
                    patched[idx][1] = new_c1_e
                    patched[idx][3] = max(k1_e, mid)
                    patched[idx + 1][0] = new_c2_s
                    patched[idx + 1][2] = min(k2_s, mid)
        chunks_out = [(a, b, c, d) for a, b, c, d in patched]

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
        # Split ANY clause spanning >=2 substantial speakers (>=0.65s overlap each),
        # even when GigaAM/Whisper placed a period at the very end of a glued 2-speaker turn.
        sub_groups = (
            _split_unpunctuated_multispeaker_units(grp, diar_segments, seg_ends)
            if diar_segments
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
         'Ну да, есть немного.'), contested overlap clauses, and turn-transition
         boundaries directly against speaker centroids, preventing phrase bleed
         across adjacent speakers.
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

    # Acoustic clause verification via CAM++ voiceprints for Q&A, overlaps, and turn boundaries
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
        initial_spks = list(assigned_spks)

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
            is_turn_boundary = dur >= 0.50 and (
                (idx > 0 and initial_spks[idx - 1] != initial_spks[idx])
                or (idx + 1 < len(clauses) and initial_spks[idx + 1] != initial_spks[idx])
            )

            if not (ends_q or prev_ends_q or next_ends_q or contended or is_short_reply or is_turn_boundary):
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
            elif is_turn_boundary and not (ends_q or prev_ends_q or next_ends_q or is_short_reply):
                req_margin = 0.095 if dur >= 0.90 else 0.115
                min_abs_sim = 0.45
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
    denoiser_key: Optional[tuple] = None
    denoiser_engine = None

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

    @classmethod
    def get_or_create_denoiser(cls, key: tuple, factory: Callable):
        with cls.lock:
            if cls.denoiser_key == key and cls.denoiser_engine is not None:
                return cls.denoiser_engine
            cls.denoiser_engine = factory()
            cls.denoiser_key = key
            return cls.denoiser_engine


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
_onnx_cuda_probe_ok: Optional[bool] = None


def _preload_nvidia_pip_libs() -> None:
    """
    Ensures pip-installed NVIDIA shared libraries (nvidia-cublas-cu12,
    nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12, nvidia-cufft-cu12, etc.)
    are visible to CTranslate2 and ONNX Runtime CUDAExecutionProvider both in
    local virtualenvs and inside Docker containers without manual LD_LIBRARY_PATH.
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

    nvidia_subdirs = (
        "cuda_runtime",
        "nvjitlink",
        "cublas",
        "cuda_nvrtc",
        "cufft",
        "curand",
        "cusparse",
        "cudnn",
    )
    so_patterns = (
        "libcudart.so*",
        "libnvJitLink.so*",
        "libcublasLt.so*",
        "libcublas.so*",
        "libnvrtc.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libcusparse.so*",
        "libcudnn.so*",
        "libcudnn_ops.so*",
        "libcudnn_cnn.so*",
        "libcudnn_graph.so*",
        "libcudnn_engines*.so*",
        "libcudnn_heuristic.so*",
    )

    for root in search_roots:
        for sub in nvidia_subdirs:
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
                    for pat in so_patterns:
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


def probe_onnx_cuda_available() -> bool:
    """
    Returns True if onnxruntime-gpu's CUDAExecutionProvider is installed AND
    an NVIDIA GPU is accessible in the current environment (enabling GPU CUDA
    acceleration for GigaAM v3 FP32 RNN-T / CTC via onnx-asr).
    """
    global _onnx_cuda_probe_ok
    if _onnx_cuda_probe_ok is not None:
        return _onnx_cuda_probe_ok
    _preload_nvidia_pip_libs()
    try:
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            _onnx_cuda_probe_ok = False
            return False
        # Confirm an actual CUDA GPU device is present (e.g. inside Docker --gpus all)
        if ctranslate2.get_cuda_device_count() <= 0:
            _onnx_cuda_probe_ok = False
            return False
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        _onnx_cuda_probe_ok = True
        return True
    except Exception:
        _onnx_cuda_probe_ok = False
        return False


def resolve_onnx_provider(
    requested_device: Optional[str] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str, list[str]]:
    """
    Resolves user-selected compute device into ('cuda' | 'cpu', providers_list)
    for ONNX Runtime / onnx-asr models (GigaAM v3 FP32, GigaAM v3 CTC).
    """
    req = (requested_device or os.environ.get("ASR_DEVICE") or DEFAULT_COMPUTE_DEVICE).strip().lower()
    if req not in COMPUTE_DEVICE_OPTIONS and req != "auto":
        req = DEFAULT_COMPUTE_DEVICE

    if req == "cpu":
        return "cpu", ["CPUExecutionProvider"]

    if probe_onnx_cuda_available():
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if req == "cuda" and status_cb:
        status_cb(
            "ONNX Runtime CUDA недоступен (установите GPU-сборку docker-compose.gpu.yml или onnxruntime-gpu) — "
            "выполнение GigaAM v3 на CPU..."
        )
    return "cpu", ["CPUExecutionProvider"]


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
    REST, gRPC, HTTP 2.0, SOAP...') so user prompts never leak into transcripts,
    while preserving genuine confident tech-stack enumerations spoken by candidates.
    """
    if not custom_prompt or not custom_prompt.strip():
        return False
    seg_toks = _normalize_prompt_tokens(text)
    if len(seg_toks) < 3:
        return False
    prompt_toks = _normalize_prompt_tokens(custom_prompt)
    if len(prompt_toks) < 3:
        return False

    # 1. Verbatim echo of the prompt's descriptive lead-in header (text before ':')
    if ":" in custom_prompt:
        header_toks = _normalize_prompt_tokens(custom_prompt.split(":", 1)[0])
        h_len = len(header_toks)
        if h_len >= 2 and len(seg_toks) >= h_len:
            header_tuple = tuple(header_toks)
            for i in range(len(seg_toks) - h_len + 1):
                if tuple(seg_toks[i : i + h_len]) == header_tuple:
                    return True

    prompt_set = set(prompt_toks)
    overlap_ratio = sum(1 for t in seg_toks if t in prompt_set) / float(len(seg_toks))
    low_confidence = no_speech_prob >= 0.25 or avg_logprob <= -0.50

    # 2. Contiguous verbatim 4-gram run from the prompt: drop on weak acoustic confidence,
    #    or when an entire long utterance (>=6 tokens, >=90% overlap) verbatim-echoes the list.
    if len(seg_toks) >= 4:
        prompt_ngrams = {
            tuple(prompt_toks[i : i + 4]) for i in range(max(0, len(prompt_toks) - 3))
        }
        has_4gram = any(
            tuple(seg_toks[i : i + 4]) in prompt_ngrams for i in range(len(seg_toks) - 3)
        )
        if has_4gram and (low_confidence or (len(seg_toks) >= 6 and overlap_ratio >= 0.90)):
            return True

    if overlap_ratio >= 0.85 and len(seg_toks) >= 5 and (low_confidence or len(seg_toks) >= 8):
        return True
    if overlap_ratio >= 0.70 and (no_speech_prob >= 0.30 or avg_logprob <= -0.55):
        return True
    return False


# Strict opt-in phonetic-to-Latin glossary map: each entry activates ONLY when its
# canonical Latin term appears as a whole token in the user's custom_prompt.
# Ambiguous homonyms (e.g. 'GPT-токены', 'жиры', 'редис', bare 'о аут', bare 'C+') are
# intentionally excluded from blind regex substitution and left to contextual LLM polish.
_GLOSSARY_PHONETIC_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("gRPC", "по gRPC", (r"поджирописью", r"под[\s\-]?жирописью")),
    ("gRPC", "gRPC", (r"джи[\s\-]?эр[\s\-]?пи[\s\-]?си", r"гэрпэцэ", r"грпц", r"грпс", r"gpc", r"джиэрпис[ию]?")),
    ("HTTP 2.0", "HTTP 2.0", (r"эйч[\s\-]?ти[\s\-]?ти[\s\-]?пи[\s\-]+(?:2\.0|два(?:\s+ноль)?)", r"хттп[\s\-]+(?:2\.0|два(?:\s+ноль)?)")),
    ("OAuth 2", "OAuth 2", (r"оау[фтс](?:[\s\-]+(?:2(?:\.0)?|два))?", r"о[\s\-]+ау[фтс][\s\-]+(?:2(?:\.0)?|два)", r"ауф[\s\-]+(?:2(?:\.0)?|два)", r"аут[\s\-]+(?:2(?:\.0)?|два)")),
    ("JWT", "JWT", (r"джей[\s\-]?дабл[\s\-]?ю[\s\-]?ти", r"джей[\s\-]?ви[\s\-]?ти", r"джвт", r"живити", r"гвт", r"gwt")),
    ("Bearer", "Bearer-токены", (r"бирутокены", r"бир[\s\-]?токены")),
    ("Bearer", "Bearer-токен", (r"бирутокен", r"бир[\s\-]?токен")),
    ("Bearer", "Bearer-токенов", (r"бирутокенов", r"бир[\s\-]?токенов")),
    ("Bearer", "Bearer-токена", (r"бирутокена", r"бир[\s\-]?токена")),
    ("Bearer", "Bearer", (r"бирер", r"бэрер", r"берер")),
    ("API-шлюз", "API-шлюз", (r"appi[\s\-]+plues", r"апи[\s\-]+шлюз")),
    ("SOAP", "SOAP", (r"соап", r"соуп")),
    ("JSON", "JSON", (r"джейсон(?:а|е|ом)?", r"джисон(?:а|е|ом)?", r"джесон(?:а|е|ом)?", r"жсон(?:а|е|ом)?", r"geson")),
    ("REST API", "REST API", (r"рест[\s\-]+апи", r"рест[\s\-]+эй[\s\-]?пи[\s\-]?ай")),
    ("REST", "REST", (r"рест(?:а|у|ом|е)?", r"рист(?:а|у|ы|ом|е)?", r"рыст(?:а|у|ом|е)?")),
    ("PostgreSQL", "PostgreSQL", (r"постгрес[\s\-]?кью[\s\-]?эль", r"постгресс?", r"постгри")),
    ("Redis", "Redis", (r"рэдис(?:а|е|ом)?",)),
    ("MongoDB", "MongoDB", (r"монго[\s\-]?дб", r"монго[\s\-]?ди[\s\-]?би", r"монга")),
    ("Kafka", "Kafka", (r"кафк[аеуи]",)),
    ("RabbitMQ", "RabbitMQ", (r"рэббит[\s\-]?эм[\s\-]?кью", r"реббит[\s\-]?эм[\s\-]?кью", r"рэббит")),
    ("Sequence", "Sequence", (r"сиквенс", r"секвенс", r"сиквэнс", r"sipens", r"siven")),
    ("Activity", "Activity", (r"активити",)),
    ("BPMN", "BPMN", (r"би[\s\-]?пи[\s\-]?эм[\s\-]?эн", r"бпмн", r"bpen", r"бпен", r"vipmencum")),
    ("ERD", "ERD", (r"и[\s\-]?ар[\s\-]?ди", r"ерд")),
    ("Camunda", "Camunda", (r"камунд[аеуы]", r"комунд[аеуы]")),
    ("PlantUML", "PlantUML", (r"плант[\s\-]?юэмэль", r"плант[\s\-]?юмл", r"план[\s\-]+юля")),
    ("draw.io", "draw.io", (r"дроу?[\s\-]+ио", r"дру[\s\-]+ио")),
    ("UI-kit", "UI-kit", (r"юай[\s\-]?кит(?:а|е|ом)?", r"ui[\s\-]?гид(?:а|е|ом)?")),
    ("use cases", "use cases", (r"юз[\s\-]?кейс(?:ы|ов|ами)?", r"user[\s\-]+kase")),
    ("Spec-Driven Development", "Spec-Driven Development", (r"specdrive[\s\-]+and[\s\-]+development", r"спек[\s\-]+дривен[\s\-]+девелопмент")),
    ("Swagger", "Swagger", (r"сваг+ер(?:а|е|ом)?",)),
    ("Postman", "Postman", (r"постман(?:а|е|ом)?", r"постмэн(?:а|е|ом)?")),
    ("Confluence", "Confluence", (r"конфлю[еэ]нс(?:а|е|ом)?",)),
    ("Jira", "в Jira", (r"в\s+джерри",)),
    ("Jira", "Jira", (r"джир[аеуы]",)),
    ("Jenkins", "Jenkins", (r"дженкинс(?:а|е|ом)?",)),
    ("Docs as Code", "Docs as Code", (r"докс[\s\-]+эз[\s\-]+код", r"докс[\s\-]+аз[\s\-]+код", r"докс[\s\-]+ас[\s\-]+код")),
    ("SFA", "SFA", (r"цфа(?=\-систем)",)),
    ("OTLP", "OTLP", (r"от[\s\-]+лп", r"о[\s\-]?ти[\s\-]?эл[\s\-]?пи")),
    ("C++", "C++", (r"си[\s\-]+плюс[\s\-]+плюс", r"(?:(?<=на\s)|(?<=с\s)|(?<=из\s))c\+(?!\+)")),
    ("GraphQL", "GraphQL", (r"граф[\s\-]?кью[\s\-]?эль", r"графкьюэл")),
    ("OpenAPI", "OpenAPI", (r"опен[\s\-]?апи", r"оупен[\s\-]?апи")),
    ("Kubernetes", "Kubernetes", (r"кубернетес(?:а|е|ом)?", r"кубернэтес(?:а|е|ом)?", r"кубер(?:а|е|ом)?")),
    ("Docker", "Docker", (r"докер(?:а|е|ом)?",)),
    ("GitLab", "GitLab", (r"гит[\s\-]?лаб(?:а|е|ом)?",)),
    ("Git", "Git", (r"гит(?:а|е|ом)?",)),
    ("CI/CD", "CI/CD", (r"си[\s\-]?ай[\s\-/]+си[\s\-]?ди",)),
    ("Prometheus", "Prometheus", (r"прометеус(?:а|е|ом)?",)),
    ("Grafana", "Grafana", (r"графан[аеуы]",)),
    ("Terraform", "Terraform", (r"терраформ(?:а|е|ом)?",)),
    ("TypeScript", "TypeScript", (r"тайп[\s\-]?скрипт(?:а|е|ом)?",)),
    ("JavaScript", "JavaScript", (r"джава[\s\-]?скрипт(?:а|е|ом)?",)),
    ("React", "React", (r"реакт(?:а|е|ом)?",)),
    ("Playwright", "Playwright", (r"плейрайт(?:а|е|ом)?",)),
    ("WebSocket", "WebSocket", (r"веб[\s\-]?сокет(?:а|е|ом|ы|ов)?",)),
]

_COMPILED_GLOSSARY_RULES: list[tuple[str, re.Pattern, re.Pattern]] = [
    (
        replacement,
        re.compile(
            r"(?<![A-Za-zА-Яа-яЁё0-9])" + re.escape(trigger) + r"(?![A-Za-zА-Яа-яЁё0-9])",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-zА-Яа-яЁё0-9])(?:" + "|".join(variants) + r")(?![A-Za-zА-Яа-яЁё0-9])",
            re.IGNORECASE,
        ),
    )
    for trigger, replacement, variants in _GLOSSARY_PHONETIC_RULES
]


def apply_custom_glossary_terms(text: str, custom_prompt: str = "") -> str:
    """
    Strictly opt-in glossary normalization: when the user supplies a custom_prompt
    containing specific Latin/IT terms (e.g. 'gRPC, OAuth 2, PostgreSQL, Swagger'),
    replaces unambiguous Cyrillic phonetic spellings of ONLY those requested terms
    using whole-token trigger matching. 100% no-op when custom_prompt is empty.
    """
    if not text or not custom_prompt or not custom_prompt.strip():
        return text

    out = text
    for canonical, trigger_pattern, pattern in _COMPILED_GLOSSARY_RULES:
        if trigger_pattern.search(custom_prompt):
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


_PYANNOTE_POWERSET_3_2 = np.array(
    [
        [0, 0, 0],  # 0: non-speech
        [1, 0, 0],  # 1: speaker #0
        [0, 1, 0],  # 2: speaker #1
        [0, 0, 1],  # 3: speaker #2
        [1, 1, 0],  # 4: speakers #0 + #1
        [1, 0, 1],  # 5: speakers #0 + #2
        [0, 1, 1],  # 6: speakers #1 + #2
    ],
    dtype=np.uint8,
)
_PYANNOTE_PERM3 = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def decode_pyannote_powerset_windows(
    window_outputs: list[tuple[int, np.ndarray]],
    hop_frames: int,
    sample_rate: int = 16000,
    rf_size: int = 991,
    rf_shift: int = 270,
    total_dur_s: float = 0.0,
    min_duration_on: float = 0.18,
    min_duration_off: float = 0.18,
    max_turn_sec: float = 6.0,
) -> list[DiarizationSegment]:
    """
    Decodes sliding-window 7-class powerset outputs from pyannote-segmentation-3.0
    into clean single-speaker segments with center-slice ownership and local-speaker
    overlap stitching across adjacent windows (eliminating Pass-1 duplicate CAM++ work).
    """
    if not window_outputs:
        return []

    n_win = len(window_outputs)
    own_start = max(0, int(hop_frames) // 2)
    next_track_id = 0
    prev_act: Optional[np.ndarray] = None
    prev_track_ids = [0, 1, 2]
    intervals_by_track: dict[int, list[tuple[float, float]]] = {}

    for w_idx, (win_start_sample, logits_2d) in enumerate(window_outputs):
        arr = np.asarray(logits_2d)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"Expected pyannote powerset logits (T, 7), got shape {arr.shape}")
        num_frames = int(arr.shape[0])
        if num_frames == 0:
            continue

        cls_idx = np.argmax(arr, axis=-1)
        act = _PYANNOTE_POWERSET_3_2[cls_idx]

        if prev_act is None:
            cur_track_ids = [next_track_id, next_track_id + 1, next_track_id + 2]
            next_track_id += 3
        else:
            ov_len = min(int(prev_act.shape[0]) - int(hop_frames), num_frames)
            if ov_len > 0:
                prev_ov = prev_act[int(hop_frames) : int(hop_frames) + ov_len].astype(np.int32)
                cur_ov = act[:ov_len].astype(np.int32)
                ov_mat = prev_ov.T @ cur_ov
                best_perm = max(
                    _PYANNOTE_PERM3,
                    key=lambda p: int(ov_mat[0, p[0]] + ov_mat[1, p[1]] + ov_mat[2, p[2]]),
                )
                act = act[:, list(best_perm)]
                cur_track_ids = []
                for s in range(3):
                    if int(ov_mat[s, best_perm[s]]) >= 3:
                        cur_track_ids.append(prev_track_ids[s])
                    else:
                        cur_track_ids.append(next_track_id)
                        next_track_id += 1
            else:
                cur_track_ids = [next_track_id, next_track_id + 1, next_track_id + 2]
                next_track_id += 3

        if n_win == 1:
            own_s, own_e = 0, num_frames
        elif w_idx == 0:
            own_s, own_e = 0, min(num_frames, own_start + int(hop_frames))
        elif w_idx == n_win - 1:
            own_s, own_e = min(own_start, num_frames), num_frames
        else:
            own_s, own_e = min(own_start, num_frames), min(num_frames, own_start + int(hop_frames))

        if own_e > own_s:
            for s in range(3):
                col = act[own_s:own_e, s]
                if not np.any(col):
                    continue
                padded = np.empty(col.size + 2, dtype=np.int8)
                padded[0] = 0
                padded[1:-1] = col
                padded[-1] = 0
                diff = np.diff(padded)
                run_starts = np.flatnonzero(diff == 1) + own_s
                run_ends = np.flatnonzero(diff == -1) + own_s
                tid = cur_track_ids[s]
                track_list = intervals_by_track.setdefault(tid, [])
                for f_s, f_e in zip(run_starts, run_ends):
                    t_s = max(0.0, (win_start_sample + 0.5 * rf_size + int(f_s) * rf_shift) / float(sample_rate))
                    t_e = (win_start_sample + 0.5 * rf_size + int(f_e) * rf_shift) / float(sample_rate)
                    if total_dur_s > 0.0:
                        t_e = min(total_dur_s, t_e)
                    if t_e > t_s:
                        track_list.append((t_s, t_e))

        prev_act = act
        prev_track_ids = cur_track_ids

    segments: list[DiarizationSegment] = []
    for tid, ivs in intervals_by_track.items():
        if not ivs:
            continue
        ivs.sort(key=lambda pair: (pair[0], pair[1]))
        merged: list[list[float]] = [[ivs[0][0], ivs[0][1]]]
        for st, en in ivs[1:]:
            if st - merged[-1][1] <= min_duration_off:
                if en > merged[-1][1]:
                    merged[-1][1] = en
            else:
                merged.append([st, en])

        for st, en in merged:
            dur = en - st
            if dur < min_duration_on:
                continue
            if max_turn_sec > 0.0 and dur > max_turn_sec:
                n_parts = int(math.ceil(dur / max_turn_sec))
                step = dur / float(n_parts)
                for p in range(n_parts):
                    p_st = st + p * step
                    p_en = en if p == n_parts - 1 else (st + (p + 1) * step)
                    if p_en - p_st >= min_duration_on:
                        segments.append(DiarizationSegment(start=p_st, end=p_en, speaker=tid))
            else:
                segments.append(DiarizationSegment(start=st, end=en, speaker=tid))

    segments.sort(key=lambda seg: (seg.start, seg.end))
    return segments


def _run_direct_pyannote_segmentation(
    seg_session: ort.InferenceSession,
    samples: np.ndarray,
    sample_rate: int = 16000,
    status_cb: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> list[DiarizationSegment]:
    meta = {}
    try:
        meta = seg_session.get_modelmeta().custom_metadata_map or {}
    except Exception:
        meta = {}

    window_size = int(meta.get("window_size", 160000) or 160000)
    rf_size = int(meta.get("receptive_field_size", 991) or 991)
    rf_shift = int(meta.get("receptive_field_shift", 270) or 270)
    hop_frames = max(1, (window_size // 2) // max(1, rf_shift))
    hop_samples = hop_frames * rf_shift

    inp_info = seg_session.get_inputs()[0]
    inp_name = inp_info.name
    inp_rank = len(inp_info.shape or [1, 1, window_size])
    out_name = seg_session.get_outputs()[0].name

    total_samples = int(samples.size)
    total_dur_s = total_samples / float(sample_rate)
    starts: list[int] = [0]
    cur_start = 0
    while cur_start + window_size < total_samples:
        cur_start += hop_samples
        starts.append(cur_start)

    total_windows = len(starts)
    window_outputs: list[tuple[int, np.ndarray]] = []
    last_pct = 0
    last_log_t = 0.0
    buf = np.zeros(window_size, dtype=np.float32)

    for idx, win_s in enumerate(starts):
        if cancel_token:
            cancel_token.check()

        chunk = samples[win_s : min(total_samples, win_s + window_size)]
        if chunk.size == window_size:
            win_arr = np.ascontiguousarray(chunk, dtype=np.float32)
        else:
            buf.fill(0.0)
            buf[: chunk.size] = chunk
            win_arr = buf

        feed = win_arr.reshape(1, 1, window_size) if inp_rank == 3 else win_arr.reshape(1, window_size)
        out = seg_session.run([out_name], {inp_name: feed})[0]
        logits = np.asarray(out, dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[0]
        window_outputs.append((win_s, logits))

        if status_cb and total_windows > 0:
            pct = int(((idx + 1) * 80) / total_windows)
            now = time.time()
            if pct >= last_pct + 10 and (now - last_log_t) >= 1.5:
                last_pct = pct
                last_log_t = now
                status_cb(f"Диаризация спикеров: {pct}%...")

    return decode_pyannote_powerset_windows(
        window_outputs=window_outputs,
        hop_frames=hop_frames,
        sample_rate=sample_rate,
        rf_size=rf_size,
        rf_shift=rf_shift,
        total_dur_s=total_dur_s,
        min_duration_on=0.18,
        min_duration_off=0.18,
        max_turn_sec=6.0,
    )


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
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
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
        req_dev = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()

        raw_objs: Optional[list[DiarizationSegment]] = None
        extractor = None

        # Fast single-pass path: run pyannote-segmentation-3-0/model.onnx directly via
        # onnxruntime (5s hop, optional GPU CUDA, zero redundant Pass-1 CAM++ extraction).
        # Falls back transparently to sherpa-onnx OfflineSpeakerDiarization on any error
        # or when seg_model_path is a mocked path.
        if seg_model_path and os.path.isfile(seg_model_path):
            eff_seg_dev, seg_providers = resolve_onnx_provider(req_dev)
            direct_key = (
                "onnx_pyannote_v1",
                seg_model_path,
                emb_model_path,
                eff_seg_dev,
                diar_threads,
                emb_intra_threads,
            )

            def _create_direct_engines():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = diar_threads
                opts.inter_op_num_threads = 1
                seg_sess = ort.InferenceSession(
                    seg_model_path, sess_options=opts, providers=seg_providers
                )
                ext_inst = sherpa_onnx.SpeakerEmbeddingExtractor(
                    sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=emb_model_path,
                        num_threads=emb_intra_threads,
                        provider="cpu",
                    )
                )
                return seg_sess, ext_inst

            try:
                seg_session, extractor = _DiarizationCache.get_or_create(
                    direct_key, _create_direct_engines
                )
                raw_objs = _run_direct_pyannote_segmentation(
                    seg_session=seg_session,
                    samples=samples,
                    sample_rate=16000,
                    status_cb=status_cb,
                    cancel_token=cancel_token,
                )
            except Exception:
                if cancel_token and cancel_token.is_cancelled():
                    raise
                raw_objs = None
                extractor = None

        if raw_objs is None:
            # Safe fallback: sherpa-onnx OfflineSpeakerDiarization with 0.25 window shift ratio
            diar_key = (
                "sherpa_sd",
                seg_model_path,
                emb_model_path,
                diar_threads,
                emb_intra_threads,
                seg_threshold,
                0.18,
                0.18,
            )

            def _create_diar_engines():
                try:
                    pyannote_cfg = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=seg_model_path,
                        window_shift_ratio=0.25,
                    )
                except TypeError:
                    pyannote_cfg = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=seg_model_path
                    )
                cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                        pyannote=pyannote_cfg,
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
            if profile_id == "breeze-asr-25":
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

        profile_cfg = ASR_MODEL_PROFILES.get(profile_id, {})
        decode_cfg = profile_cfg.get("decode_params", {})
        max_win_sec = float(decode_cfg.get("max_window_sec", 28.5))

        ordered_diar: Optional[list[DiarizationSegment]] = None
        if diar_segments:
            ordered_diar = sorted(
                (s for s in diar_segments if s.end > s.start),
                key=lambda s: (s.start, s.end),
            )
            windows = build_speaker_asr_windows(
                ordered_diar,
                total_dur,
                max_same_speaker_sec=min(26.5, max_win_sec - 1.5),
                merge_gap_sec=2.0 if device == "cpu" else 1.6,
                split_speaker_turns=(device != "cpu"),
            )
            if windows:
                windows = subdivide_long_asr_windows(
                    windows, samples, sample_rate=sample_rate, max_window_sec=max_win_sec
                )
                windows = _augment_windows_with_vad(
                    windows, samples, sample_rate=sample_rate, max_window_sec=max_win_sec
                )
            used_diar_windows = bool(windows)
        else:
            windows = build_vad_asr_windows(
                samples, sample_rate=sample_rate, max_window_sec=min(26.0, max_win_sec)
            )
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
        cleaned_custom_prompt = " ".join((custom_prompt or "").split())[:420]
        # Extract clean comma-separated terms for CTranslate2 beam-boosting hotwords
        # (strips descriptive lead-ins like 'Собеседование системного аналитика:')
        hotwords_str: Optional[str] = None
        if cleaned_custom_prompt and bool(decode_cfg.get("use_hotwords", True)):
            hw_src = (
                cleaned_custom_prompt.split(":", 1)[1].strip()
                if ":" in cleaned_custom_prompt
                else cleaned_custom_prompt
            )
            if hw_src:
                hotwords_str = hw_src[:220]

        if status_cb:
            mode_desc = "мультиязычный авто" if use_multilingual else f"фиксированный ({resolved_lang})"
            prompt_note = " + пользовательский словарь (hotwords)" if hotwords_str else (
                " + пользовательский словарь" if cleaned_custom_prompt else ""
            )
            status_cb(f"Язык распознавания Whisper: {mode_desc}{prompt_note}")

        default_cpu_beam = 3 if profile_id == "breeze-asr-25" else 2
        if device == "cuda":
            beam_size = int(decode_cfg.get("beam_size_cuda", 5))
        else:
            beam_size = int(decode_cfg.get("beam_size_cpu", default_cpu_beam))
        patience = float(decode_cfg.get("patience", 1.0))
        rep_penalty = float(decode_cfg.get("repetition_penalty", 1.08))
        no_repeat_ngram = int(decode_cfg.get("no_repeat_ngram_size", 4))
        no_speech_thr = float(decode_cfg.get("no_speech_threshold", 0.6))
        comp_ratio_thr = float(decode_cfg.get("compression_ratio_threshold", 2.4))
        log_prob_thr = float(decode_cfg.get("log_prob_threshold", -1.0))
        hall_silence_thr = float(decode_cfg.get("hallucination_silence_threshold", 1.5))
        temperatures = tuple(decode_cfg.get("temperatures", (0.0, 0.2, 0.4)))
        seg_comp_cap = max(2.4, comp_ratio_thr + 0.2)
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
                suppress_blank=True,
                no_speech_threshold=no_speech_thr,
                compression_ratio_threshold=comp_ratio_thr,
                log_prob_threshold=log_prob_thr,
                repetition_penalty=rep_penalty,
                no_repeat_ngram_size=no_repeat_ngram,
                hallucination_silence_threshold=hall_silence_thr,
                beam_size=beam_size,
                best_of=beam_size,
                patience=patience,
                temperature=temperatures,
            )
            if hotwords_str:
                transcribe_kwargs["hotwords"] = hotwords_str
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

                if seg_no_speech > 0.78 and seg_logprob < log_prob_thr:
                    continue
                if seg_comp_ratio > seg_comp_cap:
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

    @classmethod
    def _denoise_speech(
        cls,
        samples: np.ndarray,
        sample_rate: int = 16000,
        denoiser_path: Optional[str] = None,
        threads: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> np.ndarray:
        """
        Optional lightweight GTCRN neural speech enhancement (~535 KB ONNX via
        sherpa-onnx). Cleans background hum, fan noise, and laptop-mic hiss in
        15s blocks with 80ms linear overlap-add crossfading while strictly
        preserving exact sample count (zero boundary clicks, zero timestamp drift).
        """
        if cancel_token:
            cancel_token.check()
        if samples.size < 800:
            return samples
        try:
            model_path = denoiser_path or ModelManager.ensure_denoiser_model(status_cb=status_cb)
            eff_threads = max(1, min(int(threads), 4))
            t0 = time.time()
            if status_cb:
                status_cb("Шумоподавление речи (GTCRN ONNX)...")

            def _create_denoiser():
                cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(
                    model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                        gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model_path),
                        num_threads=eff_threads,
                        provider="cpu",
                    )
                )
                return sherpa_onnx.OfflineSpeechDenoiser(cfg)

            denoiser = _ModelCache.get_or_create_denoiser(
                ("gtcrn", model_path, eff_threads),
                _create_denoiser,
            )

            block_n = max(sample_rate, 15 * sample_rate)
            fade_n = min(block_n // 8, max(0, int(0.08 * sample_rate)))
            step_n = max(1, block_n - fade_n)
            cleaned = np.zeros(samples.size, dtype=np.float32)

            pos = 0
            while pos < samples.size:
                if cancel_token:
                    cancel_token.check()
                end = min(samples.size, pos + block_n)
                blk = np.ascontiguousarray(samples[pos:end], dtype=np.float32)
                res = denoiser(blk, int(sample_rate))
                arr = np.asarray(res.samples, dtype=np.float32).reshape(-1)
                if arr.size > blk.size:
                    arr = arr[: blk.size]
                elif arr.size < blk.size:
                    arr = np.pad(arr, (0, blk.size - arr.size), mode="constant")

                if pos == 0:
                    cleaned[0:end] = arr
                else:
                    ov = min(fade_n, arr.size, samples.size - pos)
                    if ov > 0:
                        w = np.linspace(0.0, 1.0, ov, dtype=np.float32)
                        cleaned[pos : pos + ov] = cleaned[pos : pos + ov] * (1.0 - w) + arr[:ov] * w
                    if arr.size > ov:
                        cleaned[pos + ov : end] = arr[ov:]

                if end >= samples.size:
                    break
                pos += step_n

            cleaned = np.ascontiguousarray(cleaned, dtype=np.float32)
            if status_cb:
                status_cb(f"[Profile] Шумоподавление GTCRN завершено за {time.time() - t0:.2f}s")
            return cleaned
        except Exception as exc:
            if status_cb:
                status_cb(f"Предупреждение: шумоподавление GTCRN пропущено ({exc}), используется исходный звук.")
            return samples

    @classmethod
    def _run_onnx_asr(
        cls,
        samples: np.ndarray,
        sample_rate: int,
        model_dir: str,
        profile_id: str = "gigaam-v3",
        diar_segments: Optional[list[DiarizationSegment]] = None,
        threads: int = 4,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
        language: str = DEFAULT_ASR_LANGUAGE,
    ) -> list[TimedUnit]:
        """
        Runs full-precision FP32 ONNX ASR models via onnx-asr
        (GigaAM v3 E2E RNN-T FP32 and GigaAM v3 E2E CTC FP32)
        on NVIDIA GPU (CUDAExecutionProvider) or CPU (CPUExecutionProvider)
        using <=20s Silero VAD + sliding overlap chunking with token timestamps.
        """
        if cancel_token:
            cancel_token.check()
        if onnx_asr is None:
            raise RuntimeError("Пакет onnx-asr не установлен. Выполните: pip install onnx-asr")

        profile = ASR_MODEL_PROFILES.get(profile_id, {})
        onnx_model_name = str(profile.get("onnx_asr_model", "gigaam-v3-e2e-rnnt"))
        quantization = profile.get("quantization", None)
        max_chunk_sec = float(profile.get("max_chunk_sec", 20.0))
        overlap_sec = float(profile.get("overlap_sec", 1.5))
        progress_tag = profile.get("short_name", profile_id)

        req_dev = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()
        if not profile.get("supports_cuda", False):
            eff_dev, providers = "cpu", ["CPUExecutionProvider"]
        else:
            eff_dev, providers = resolve_onnx_provider(req_dev, status_cb=status_cb)

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

        def _load_engine(dev_name: str, prov_list: list[str]):
            asr_threads = max(1, int(threads)) if dev_name == "cpu" else min(2, max(1, int(threads)))
            cache_key = ("onnx_asr", profile_id, model_dir, dev_name, asr_threads)

            def _factory():
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = asr_threads
                sess_opts.inter_op_num_threads = 1
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                loaded = onnx_asr.load_model(
                    onnx_model_name,
                    path=model_dir,
                    quantization=quantization,
                    sess_options=sess_opts,
                    providers=prov_list,
                )
                return loaded.with_timestamps()

            return _ModelCache.get_or_create(cache_key, _factory)

        def _split_chunk_text_proportionally(text: str, c_start: float, c_end: float) -> list[TimedUnit]:
            cleaned = clean_hallucinated_text(text.strip())
            if not cleaned:
                return []
            parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", cleaned) if p.strip()]
            if len(parts) <= 1:
                return [TimedUnit(start=c_start, end=c_end, text=f" {cleaned}")]
            total_chars = max(1, sum(len(p) for p in parts))
            span = max(0.05, c_end - c_start)
            cur = c_start
            out_parts: list[TimedUnit] = []
            for idx, p in enumerate(parts):
                p_end = c_end if idx == len(parts) - 1 else min(c_end, cur + span * (len(p) / total_chars))
                out_parts.append(TimedUnit(start=cur, end=max(cur + 0.02, p_end), text=f" {p}"))
                cur = p_end
            return out_parts

        def _extract_result_units(res, c_start: float, c_end: float) -> list[TimedUnit]:
            out_u: list[TimedUnit] = []
            if isinstance(res, str):
                if res.strip():
                    out_u.extend(_split_chunk_text_proportionally(res, c_start, c_end))
                return out_u
            toks = getattr(res, "tokens", None)
            ts = getattr(res, "timestamps", None)
            txt = getattr(res, "text", None)
            if toks and ts and len(toks) == len(ts):
                chunk_units = group_bpe_tokens_to_units(list(toks), list(ts), c_start, c_end)
                for u in chunk_units:
                    if u.text and u.text.strip() and not is_repetitive_noise_token(u.text):
                        out_u.append(u)
            elif txt and str(txt).strip():
                out_u.extend(_split_chunk_text_proportionally(str(txt), c_start, c_end))
            return out_u

        def _decode_with_device(dev_name: str, prov_list: list[str]) -> list[TimedUnit]:
            model = _load_engine(dev_name, prov_list)
            if dev_name == "cuda":
                batch_size = max(1, int(profile.get("batch_size_cuda", 8)))
            else:
                batch_size = max(1, int(profile.get("batch_size_cpu", 4)))
            total_chunks = len(valid_chunks)
            out_units: list[TimedUnit] = []
            last_pct = -20
            last_log_t = time.time()

            def _single_retry(c_start: float, c_end: float, s_idx: int, e_idx: int) -> list[TimedUnit]:
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
                        sub_wav = normalize_chunk_level(samples[sub_si:sub_ei])
                        sub_res = model.recognize(sub_wav, sample_rate=sample_rate)
                        rescued.extend(_extract_result_units(sub_res, sub_s, sub_e))
                    except Exception:
                        continue
                return rescued

            for b_start in range(0, total_chunks, batch_size):
                if cancel_token:
                    cancel_token.check()

                batch = valid_chunks[b_start : b_start + batch_size]
                chunk_wavs = [normalize_chunk_level(samples[si:ei]) for _, _, _, _, si, ei in batch]

                batch_results: list = []
                try:
                    if len(chunk_wavs) == 1:
                        batch_results = [
                            model.recognize(chunk_wavs[0], sample_rate=sample_rate)
                        ]
                    else:
                        raw_res = model.recognize(chunk_wavs, sample_rate=sample_rate)
                        batch_results = list(raw_res) if isinstance(raw_res, (list, tuple)) else [raw_res]
                except Exception as batch_exc:
                    if dev_name == "cuda" and b_start == 0:
                        raise batch_exc
                    batch_results = []
                    for wav_arr in chunk_wavs:
                        if cancel_token:
                            cancel_token.check()
                        try:
                            batch_results.append(
                                model.recognize(wav_arr, sample_rate=sample_rate)
                            )
                        except Exception:
                            batch_results.append(None)

                if cancel_token:
                    cancel_token.check()

                for (c_start, c_end, k_start, k_end, s_idx, e_idx), res, wav_arr in zip(
                    batch, batch_results, chunk_wavs
                ):
                    chunk_units = _extract_result_units(res, c_start, c_end) if res is not None else []
                    if not chunk_units and (c_end - c_start) >= 1.2:
                        chunk_rms = float(np.sqrt(np.mean(wav_arr * wav_arr))) if wav_arr.size else 0.0
                        if chunk_rms > 0.010:
                            chunk_units = _single_retry(c_start, c_end, s_idx, e_idx)

                    has_left_ov = k_start > c_start + 1e-4
                    has_right_ov = k_end < c_end - 1e-4
                    first_kept_on_seam = has_left_ov
                    for u in chunk_units:
                        if has_left_ov or has_right_ov:
                            mid = 0.5 * (u.start + u.end)
                            if has_left_ov and mid < k_start:
                                continue
                            if has_right_ov and mid >= k_end:
                                continue
                        if first_kept_on_seam and out_units:
                            first_kept_on_seam = False
                            prev_u = out_units[-1]
                            if (
                                abs(u.start - prev_u.start) <= 0.35
                                and _clean_word_lower(u.text)
                                and _clean_word_lower(u.text) == _clean_word_lower(prev_u.text)
                            ):
                                continue
                        out_units.append(u)

                done_idx = b_start + len(batch)
                if status_cb and total_chunks > 0:
                    pct = min(99, int((done_idx * 100) / total_chunks))
                    now = time.time()
                    if pct >= last_pct + 20 and (now - last_log_t) >= 2.0:
                        last_pct = pct
                        last_log_t = now
                        status_cb(f"Распознавание речи ({progress_tag}): {pct}% ({done_idx}/{total_chunks})...")

            return out_units

        try:
            return _decode_with_device(eff_dev, providers)
        except Exception as gpu_err:
            if eff_dev == "cuda":
                if status_cb:
                    status_cb(
                        f"Ошибка ONNX Runtime CUDA ({gpu_err}) — автоматическое переключение {progress_tag} на CPU..."
                    )
                return _decode_with_device("cpu", ["CPUExecutionProvider"])
            raise

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
        if profile["engine"] == "onnx_asr":
            if profile.get("supports_cuda", False):
                eff_dev, _ = resolve_onnx_provider(req_dev)
            else:
                eff_dev = "cpu"
                if req_dev == "cuda" and status_cb:
                    status_cb(
                        f"Модель {profile['short_name']} работает на CPU; "
                        "для GPU CUDA выберите GigaAM v3 FP32, GigaAM v3 CTC FP32, Breeze-ASR-25 или Whisper Turbo."
                    )
            dev_label = "GPU CUDA" if eff_dev == "cuda" else "CPU"
            if status_cb:
                status_cb(f"Локальная транскрибация речи ({profile['short_name']}, {dev_label})...")
            return cls._run_onnx_asr(
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
            )
        raise ValueError(f"Unsupported engine: {profile['engine']}")

    @staticmethod
    def _finalize_and_write_transcript(
        turns: list[SpeakerTurn],
        duration_sec: float,
        model_name: str,
        num_speakers: int,
        output_txt_path: str,
        custom_prompt: str = "",
        base_note: Optional[str] = None,
        llm_polish: bool = LLM_POLISH_ENABLED_DEFAULT,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> dict:
        final_turns = list(turns)
        notes: list[str] = []
        if base_note:
            notes.append(base_note)

        llm_applied = False
        raw_backup_path: Optional[str] = None

        if llm_polish and final_turns:
            # Pass raw ASR turns directly to the LLM (avoid blind regex corruption before context-aware editing)
            raw_text = format_transcript_txt(
                turns=final_turns,
                duration_sec=duration_sec,
                model_name=model_name,
                num_speakers=num_speakers,
                note="\n".join(notes) if notes else None,
            )
            stem, ext = os.path.splitext(output_txt_path)
            raw_backup_path = f"{stem}_raw{ext or '.txt'}"
            try:
                with open(raw_backup_path, "w", encoding="utf-8") as rf:
                    rf.write(raw_text)
            except OSError:
                raw_backup_path = None

            polished_turns, llm_stats = polish_turns_with_local_llm(
                turns=final_turns,
                base_url=llm_base_url,
                model=llm_model,
                custom_prompt=custom_prompt,
                api_key=llm_api_key,
                status_cb=status_cb,
                cancel_token=cancel_token,
            )
            if llm_stats.get("applied"):
                final_turns = polished_turns
                llm_applied = True
                notes.append(
                    f"# Постобработка: LLM-корректура ({llm_model}, исправлено реплик: {llm_stats.get('changed_turns', 0)})"
                )

        # Opt-in phonetic regex glossary runs ONLY when LLM polish is off or did not apply
        if not llm_applied and custom_prompt and custom_prompt.strip():
            final_turns = [
                SpeakerTurn(
                    t.start,
                    t.end,
                    t.speaker_label,
                    apply_custom_glossary_terms(t.text, custom_prompt),
                )
                for t in final_turns
            ]

        transcript_text = format_transcript_txt(
            turns=final_turns,
            duration_sec=duration_sec,
            model_name=model_name,
            num_speakers=num_speakers,
            note="\n".join(notes) if notes else None,
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        return {
            "turns": final_turns,
            "llm_applied": llm_applied,
            "raw_transcript_path": raw_backup_path,
        }

    @staticmethod
    def _write_wav_16k_mono(path: str, samples: np.ndarray, sample_rate: int = 16000) -> None:
        pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype("<i2")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm_i16.tobytes())

    @classmethod
    def _run_deepgram_cloud(
        cls,
        wav_path: str,
        samples: np.ndarray,
        sample_rate: int,
        denoise: bool,
        threads: int,
        api_key: str,
        language: str,
        custom_prompt: str,
        diarize: bool,
        status_cb: Optional[Callable[[str], None]] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> tuple[list[TimedUnit], list[DiarizationSegment]]:
        upload_path = wav_path
        tmp_denoised_path: Optional[str] = None
        if denoise:
            denoised = cls._denoise_speech(
                samples,
                sample_rate=sample_rate,
                threads=threads,
                status_cb=status_cb,
                cancel_token=cancel_token,
            )
            fd, tmp_denoised_path = tempfile.mkstemp(prefix="dg_denoised_", suffix=".wav")
            os.close(fd)
            cls._write_wav_16k_mono(tmp_denoised_path, denoised, sample_rate=sample_rate)
            upload_path = tmp_denoised_path

        try:
            eff_key = (api_key or "").strip() or DEFAULT_DEEPGRAM_API_KEY
            return transcribe_with_deepgram_nova3(
                wav_path=upload_path,
                api_key=eff_key,
                language=language,
                custom_prompt=custom_prompt,
                diarize=diarize,
                base_url=DEFAULT_DEEPGRAM_BASE_URL,
                cancel_token=cancel_token,
                status_cb=status_cb,
            )
        finally:
            if tmp_denoised_path:
                try:
                    os.remove(tmp_denoised_path)
                except OSError:
                    pass

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
        denoise: bool = DENOISE_ENABLED_DEFAULT,
        asr_api_key: str = DEFAULT_DEEPGRAM_API_KEY,
    ) -> dict:
        """
        Phase A of hardware-decoupled GPU mode: runs Silero VAD + GPU ASR
        (Whisper/Breeze on CTranslate2 CUDA OR GigaAM v3 FP32/CTC on ONNX Runtime CUDA)
        immediately during CPU video blurring, without competing with CPU video workers
        by deferring CPU speaker diarization until Phase B.
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

        if profile["engine"] == "deepgram_cloud":
            t_asr = time.time()
            units, _ = cls._run_deepgram_cloud(
                wav_path=wav_path,
                samples=samples,
                sample_rate=sample_rate,
                denoise=denoise,
                threads=max(1, min(2, threads)),
                api_key=asr_api_key,
                language=language,
                custom_prompt=custom_prompt,
                diarize=False,
                status_cb=update,
                cancel_token=cancel_token,
            )
            elapsed_asr = time.time() - t_asr
            update(f"[Profile] Cloud ASR завершена за {elapsed_asr:.2f}s (единиц текста: {len(units)})")
            return {
                "samples": samples,
                "sample_rate": sample_rate,
                "duration_sec": duration_sec,
                "units": units,
                "profile_id": profile_id,
                "custom_prompt": custom_prompt,
                "elapsed_asr": elapsed_asr,
            }

        asr_samples = (
            cls._denoise_speech(
                samples,
                sample_rate=sample_rate,
                threads=max(1, min(2, threads)),
                status_cb=update,
                cancel_token=cancel_token,
            )
            if denoise
            else samples
        )

        asr_dir = ModelManager.ensure_asr_model(profile_id, status_cb=update)
        if cancel_token:
            cancel_token.check()

        req_dev = (compute_device or "cuda").strip().lower()
        t_asr = time.time()
        if profile["engine"] == "faster_whisper":
            eff_dev, eff_ctype = resolve_compute_device(req_dev)
            dev_label = f"GPU CUDA ({eff_ctype})" if eff_dev == "cuda" else "CPU"
            update(f"Локальная транскрибация речи ({profile['short_name']}, {dev_label})...")
            units = cls._run_faster_whisper(
                samples=asr_samples,
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
        elif profile["engine"] == "onnx_asr":
            eff_dev, _ = resolve_onnx_provider(req_dev)
            dev_label = "GPU CUDA (FP32)" if eff_dev == "cuda" else "CPU"
            update(f"Локальная транскрибация речи ({profile['short_name']}, {dev_label})...")
            units = cls._run_onnx_asr(
                samples=asr_samples,
                sample_rate=sample_rate,
                model_dir=asr_dir,
                profile_id=profile_id,
                diar_segments=None,
                threads=threads,
                compute_device=req_dev,
                status_cb=update,
                cancel_token=cancel_token,
                language=language,
            )
        else:
            units = cls._run_asr_engine(
                samples=asr_samples,
                sample_rate=sample_rate,
                profile_id=profile_id,
                profile=profile,
                asr_dir=asr_dir,
                diar_segments=None,
                threads=threads,
                compute_device=req_dev,
                language=language,
                custom_prompt=custom_prompt,
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
            "custom_prompt": custom_prompt,
            "compute_device": req_dev,
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
        llm_polish: bool = LLM_POLISH_ENABLED_DEFAULT,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
    ) -> dict:
        """
        Phase B of hardware-decoupled GPU mode: runs speaker diarization
        using uncontested resources after video blurring completes, then aligns
        the pre-computed GPU ASR words to speaker turns and writes .txt.
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
        diar_device = str(asr_state.get("compute_device", DEFAULT_COMPUTE_DEVICE))
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
            compute_device=diar_device,
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
        fin = cls._finalize_and_write_transcript(
            turns=turns,
            duration_sec=duration_sec,
            model_name=profile["short_name"],
            num_speakers=final_speaker_count,
            output_txt_path=output_txt_path,
            custom_prompt=eff_prompt,
            llm_polish=llm_polish,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            status_cb=update,
            cancel_token=cancel_token,
        )

        total_elapsed = elapsed_asr + (time.time() - t0)
        update(
            f"[Profile] Транскрипт сохранён ({len(fin['turns'])} реплик, спикеров: {final_speaker_count}) "
            f"за {total_elapsed:.2f}s"
        )
        return {
            "num_speakers": final_speaker_count,
            "num_turns": len(fin["turns"]),
            "elapsed": total_elapsed,
            "llm_applied": fin["llm_applied"],
            "raw_transcript_path": fin["raw_transcript_path"],
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
        denoise: bool = DENOISE_ENABLED_DEFAULT,
        llm_polish: bool = LLM_POLISH_ENABLED_DEFAULT,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
        asr_api_key: str = DEFAULT_DEEPGRAM_API_KEY,
    ) -> dict:
        """
        Runs speaker diarization + speech recognition (+ optional GTCRN
        denoising and LLM polish) on a 16 kHz mono WAV file and writes
        the formatted role-split transcript to output_txt_path.
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

        if profile["engine"] == "deepgram_cloud":
            want_diarize = (speaker_count or "auto").strip().lower() != "1"
            t_cloud = time.time()
            units, diar_segments = cls._run_deepgram_cloud(
                wav_path=wav_path,
                samples=samples,
                sample_rate=sample_rate,
                denoise=denoise,
                threads=threads,
                api_key=asr_api_key,
                language=language,
                custom_prompt=custom_prompt,
                diarize=want_diarize,
                status_cb=update,
                cancel_token=cancel_token,
            )
            speaker_centroids, emb_extractor = None, None
            detected_raw_speakers = len({s.speaker for s in diar_segments})
            update(
                f"[Profile] Deepgram Nova-3 завершена за {time.time() - t_cloud:.2f}s "
                f"(единиц текста: {len(units)}, сегментов диаризации: {len(diar_segments)}, "
                f"голосов: {detected_raw_speakers})"
            )
            # Graceful fallback to local CAM++ only if cloud returned words with zero speaker tags
            if want_diarize and units and not diar_segments:
                update("Deepgram не вернул метки спикеров — резервный запуск локальной диаризации...")
                seg_onnx, emb_onnx = ModelManager.ensure_diarization_models(status_cb=update)
                if cancel_token:
                    cancel_token.check()
                eff_diar_threads = diar_threads if diar_threads is not None else threads
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
                    compute_device=compute_device,
                )
                if isinstance(diar_out, tuple):
                    diar_segments, speaker_centroids, emb_extractor = diar_out
                else:
                    diar_segments, speaker_centroids, emb_extractor = diar_out, None, None
        else:
            # 1. Ensure local models are cached & verified
            seg_onnx, emb_onnx = ModelManager.ensure_diarization_models(status_cb=update)
            if cancel_token:
                cancel_token.check()
            asr_dir = ModelManager.ensure_asr_model(profile_id, status_cb=update)
            if cancel_token:
                cancel_token.check()

            # 2. Speaker Diarization (always on original voice timbre for clean CAM++ embeddings)
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
                compute_device=compute_device,
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

            # Optional neural speech denoising before ASR (original timbre stays intact for diarization)
            asr_samples = (
                cls._denoise_speech(
                    samples,
                    sample_rate=sample_rate,
                    threads=threads,
                    status_cb=update,
                    cancel_token=cancel_token,
                )
                if denoise
                else samples
            )

            # 3. Speech Recognition
            t_asr = time.time()
            units = cls._run_asr_engine(
                samples=asr_samples,
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

        # 4. Align words/units to speaker roles, apply optional Stage 4 LLM polish, write UTF-8 .txt
        turns, final_speaker_count = align_units_to_speakers(
            units,
            diar_segments,
            samples=samples,
            sample_rate=sample_rate,
            speaker_centroids=speaker_centroids,
            embedding_extractor=emb_extractor,
        )
        fin = cls._finalize_and_write_transcript(
            turns=turns,
            duration_sec=duration_sec,
            model_name=profile["short_name"],
            num_speakers=final_speaker_count,
            output_txt_path=output_txt_path,
            custom_prompt=custom_prompt,
            llm_polish=llm_polish,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            status_cb=update,
            cancel_token=cancel_token,
        )

        total_elapsed = time.time() - t0
        update(
            f"[Profile] Транскрипт сохранён ({len(fin['turns'])} реплик, спикеров: {final_speaker_count}) "
            f"за {total_elapsed:.2f}s"
        )
        return {
            "num_speakers": final_speaker_count,
            "num_turns": len(fin["turns"]),
            "elapsed": total_elapsed,
            "llm_applied": fin["llm_applied"],
            "raw_transcript_path": fin["raw_transcript_path"],
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
        denoise: bool = DENOISE_ENABLED_DEFAULT,
        llm_polish: bool = LLM_POLISH_ENABLED_DEFAULT,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
        asr_api_key: str = DEFAULT_DEEPGRAM_API_KEY,
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

        is_deepgram = profile["engine"] == "deepgram_cloud"
        asr_dir = "" if is_deepgram else ModelManager.ensure_asr_model(profile_id, status_cb=update)
        if cancel_token:
            cancel_token.check()

        # Track 1: Deterministic Speaker 1 (Candidate Mic) — no diarization needed!
        t1_turns: list[SpeakerTurn] = []
        if samples1.size >= 1600:
            if is_deepgram:
                update("Дорожка 1 (Спикер 1 / микрофон): облачное распознавание Deepgram Nova-3...")
                t1_units, _ = cls._run_deepgram_cloud(
                    wav_path=track1_wav_path,
                    samples=samples1,
                    sample_rate=sr1,
                    denoise=denoise,
                    threads=threads,
                    api_key=asr_api_key,
                    language=language,
                    custom_prompt=custom_prompt,
                    diarize=False,
                    status_cb=update,
                    cancel_token=cancel_token,
                )
            else:
                asr_samples1 = (
                    cls._denoise_speech(samples1, sample_rate=sr1, threads=threads, status_cb=update, cancel_token=cancel_token)
                    if denoise
                    else samples1
                )
                update("Дорожка 1 (Спикер 1 / микрофон): распознавание речи без диаризации...")
                t1_units = cls._run_asr_engine(
                    samples=asr_samples1,
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
        if samples2.size >= 1600:
            mode_clean = (speaker_count or "auto").strip().lower()
            if mode_clean.isdigit() and int(mode_clean) >= 2:
                track2_speaker_mode = str(int(mode_clean) - 1)
            else:
                track2_speaker_mode = mode_clean

            if is_deepgram:
                want_diarize2 = track2_speaker_mode != "1"
                update(
                    f"Дорожка 2 (собеседники): облачное распознавание Deepgram Nova-3 "
                    f"(diarize={'on' if want_diarize2 else 'off'})..."
                )
                t2_units, diar_segments2 = cls._run_deepgram_cloud(
                    wav_path=track2_wav_path,
                    samples=samples2,
                    sample_rate=sr2,
                    denoise=denoise,
                    threads=threads,
                    api_key=asr_api_key,
                    language=language,
                    custom_prompt=custom_prompt,
                    diarize=want_diarize2,
                    status_cb=update,
                    cancel_token=cancel_token,
                )
                speaker_centroids2, emb_extractor2 = None, None
                if want_diarize2 and t2_units and not diar_segments2:
                    seg_onnx, emb_onnx = ModelManager.ensure_diarization_models(status_cb=update)
                    if cancel_token:
                        cancel_token.check()
                    eff_diar_threads = diar_threads if diar_threads is not None else threads
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
                        compute_device=compute_device,
                    )
                    if isinstance(diar_out, tuple):
                        diar_segments2, speaker_centroids2, emb_extractor2 = diar_out
                    else:
                        diar_segments2, speaker_centroids2, emb_extractor2 = diar_out, None, None
            else:
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
                    compute_device=compute_device,
                )
                if isinstance(diar_out, tuple):
                    diar_segments2, speaker_centroids2, emb_extractor2 = diar_out
                else:
                    diar_segments2, speaker_centroids2, emb_extractor2 = diar_out, None, None

                asr_samples2 = (
                    cls._denoise_speech(samples2, sample_rate=sr2, threads=threads, status_cb=update, cancel_token=cancel_token)
                    if denoise
                    else samples2
                )
                t2_units = cls._run_asr_engine(
                    samples=asr_samples2,
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
            raw_t2_turns, _ = align_units_to_speakers(
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
        final_speaker_count = len({t.speaker_label for t in merged_turns})
        track_note = (
            "# Режим дорожек: раздельный (Дорожка 1 = Спикер 1, Дорожка 2 = диаризация собеседников)"
        )
        fin = cls._finalize_and_write_transcript(
            turns=merged_turns,
            duration_sec=duration_sec,
            model_name=profile["short_name"],
            num_speakers=final_speaker_count,
            output_txt_path=output_txt_path,
            custom_prompt=custom_prompt,
            base_note=track_note,
            llm_polish=llm_polish,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            status_cb=update,
            cancel_token=cancel_token,
        )

        total_elapsed = time.time() - t0
        update(
            f"[Profile] Двухдорожечный транскрипт сохранён ({len(fin['turns'])} реплик, "
            f"спикеров: {final_speaker_count}) за {total_elapsed:.2f}s"
        )
        return {
            "num_speakers": final_speaker_count,
            "num_turns": len(fin["turns"]),
            "elapsed": total_elapsed,
            "dual_track": True,
            "llm_applied": fin["llm_applied"],
            "raw_transcript_path": fin["raw_transcript_path"],
        }




