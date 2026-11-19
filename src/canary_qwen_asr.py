"""
Pure ONNX Runtime inference engine for NVIDIA Canary-Qwen 2.5B
(`OpenVoiceOS/canary-qwen-2.5b-onnx` export of `nvidia/canary-qwen-2.5b`).

Architecture (NeMo SALM, English-only SOTA #1 Open ASR Leaderboard):
  - Preprocessor: NeMo 128-bin log-mel (`nemo128` via `onnx-asr`)
  - Encoder: FastConformer + projection (`encoder_int8.onnx` or `encoder.onnx`)
  - Token embeddings: Qwen3-1.7B (`embed_tokens_int8.onnx` or `embed_tokens.onnx`)
  - Decoder: Merged prefill/decode causal Qwen3-1.7B with KV cache
    (`decoder_int8.onnx` or `decoder.onnx`)

Zero PyTorch / NeMo dependency — runs locally on ONNX Runtime.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import onnxruntime as ort

from src.cancellation import CancellationToken

try:
    from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor as _OnnxPreprocessor
except Exception:  # pragma: no cover - optional fallback
    _OnnxPreprocessor = None

try:
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy as _NemoPreprocessorNumpy
except Exception:  # pragma: no cover - optional fallback
    _NemoPreprocessorNumpy = None


def _build_gpt2_unicode_to_bytes() -> dict[str, int]:
    """Inverse of GPT-2 / Qwen byte-level BPE byte-to-unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_GPT2_UNICODE_TO_BYTE: dict[str, int] = _build_gpt2_unicode_to_bytes()


def decode_gpt2_byte_bpe(
    token_ids: list[int],
    id_to_token: dict[int, str],
    skip_token_ids: set[int],
) -> str:
    """Decodes Qwen/GPT-2 byte-level BPE token IDs to UTF-8 text without external tokenizers."""
    raw = bytearray()
    for tid in token_ids:
        if tid in skip_token_ids:
            continue
        tok = id_to_token.get(tid, "")
        if not tok:
            continue
        if tok.startswith("<|") and tok.endswith("|>"):
            continue
        for ch in tok:
            byte_val = _GPT2_UNICODE_TO_BYTE.get(ch)
            if byte_val is not None:
                raw.append(byte_val)
            else:
                raw.extend(ch.encode("utf-8"))
    return raw.decode("utf-8", errors="replace").strip()


def detect_repetition_loop_trim(token_ids: list[int]) -> Optional[int]:
    """
    Detects degenerate greedy repetition loops (common if non-English speech or noise
    is fed into an English-only Speech-LLM).

    Returns the index to truncate `token_ids` to (keeping one clean copy before the loop),
    or None if no loop is detected.
    """
    n = len(token_ids)
    # 1. Single-token stutter (6+ identical tokens in a row)
    if n >= 6 and len(set(token_ids[-6:])) == 1:
        return n - 5

    # 2. Phrase / n-gram loop: period p in [2..24] repeating 3 consecutive times at tail
    max_p = min(24, n // 3)
    for p in range(2, max_p + 1):
        tail = token_ids[n - p : n]
        prev1 = token_ids[n - 2 * p : n - p]
        prev2 = token_ids[n - 3 * p : n - 2 * p]
        if tail == prev1 == prev2:
            # Keep at most one occurrence of the repeated phrase
            return n - 2 * p

    return None


def _resolve_onnx_artifact(model_dir: str, stem: str, prefer_int8: bool = True) -> str:
    candidates = [f"{stem}_int8.onnx", f"{stem}.onnx"] if prefer_int8 else [f"{stem}.onnx", f"{stem}_int8.onnx"]
    for fname in candidates:
        full = os.path.join(model_dir, fname)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(
        f"Отсутствует ONNX-файл модели Canary-Qwen 2.5B ({candidates[0]}) в каталоге {model_dir}"
    )


def _create_nemo128_preprocessor(preprocessor_name: str = "nemo128"):
    if _OnnxPreprocessor is not None:
        try:
            return _OnnxPreprocessor(preprocessor_name, {})
        except Exception:
            pass
    if _NemoPreprocessorNumpy is not None:
        return _NemoPreprocessorNumpy(preprocessor_name)
    raise RuntimeError(
        "Не найден препроцессор nemo128 из пакета onnx-asr. Выполните: pip install onnx-asr"
    )


class CanaryQwenSpeechLlm:
    """
    Self-contained ONNX Runtime runner for NVIDIA Canary-Qwen 2.5B (`speech-llm` graph layout).
    """

    def __init__(
        self,
        model_dir: str,
        threads: int = 4,
        providers: Optional[list[str]] = None,
        prefer_int8: bool = True,
        repetition_penalty: float = 1.05,
        edge_pad_sec: float = 0.12,
    ):
        self.model_dir = model_dir
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self.edge_pad_sec = max(0.0, float(edge_pad_sec))

        cfg_path = os.path.join(model_dir, "config.json")
        vocab_path = os.path.join(model_dir, "vocab.json")
        if not os.path.isfile(cfg_path) or not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"Отсутствуют config.json / vocab.json в {model_dir}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_map = json.load(f)
        self.id_to_token: dict[int, str] = {int(idx): str(tok) for tok, idx in vocab_map.items()}

        self.preprocessor_name = str(self.config.get("preprocessor", "nemo128"))
        self.num_layers = int(self.config.get("num_layers", 28))
        self.num_kv_heads = int(self.config.get("num_key_value_heads", 8))
        self.head_dim = int(self.config.get("head_dim", 128))
        self.max_sequence_length = int(self.config.get("max_sequence_length", 448))
        self.eos_token_ids: set[int] = {int(x) for x in self.config.get("eos_token_ids", [151643, 151645])}
        self.prompt_prefix_ids: list[int] = [
            int(x) for x in self.config.get("prompt_prefix_ids", [151644, 872, 198, 3167, 3114, 279, 2701, 25, 220])
        ]
        self.prompt_suffix_ids: list[int] = [
            int(x) for x in self.config.get("prompt_suffix_ids", [151645, 198, 151644, 77091, 198])
        ]

        self.preprocessor = _create_nemo128_preprocessor(self.preprocessor_name)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, int(threads))
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.enable_mem_pattern = True

        enc_path = _resolve_onnx_artifact(model_dir, "encoder", prefer_int8=prefer_int8)
        emb_path = _resolve_onnx_artifact(model_dir, "embed_tokens", prefer_int8=prefer_int8)
        dec_path = _resolve_onnx_artifact(model_dir, "decoder", prefer_int8=prefer_int8)

        self.encoder_sess = ort.InferenceSession(enc_path, sess_options=sess_opts, providers=self.providers)
        self.embed_sess = ort.InferenceSession(emb_path, sess_options=sess_opts, providers=self.providers)
        self.decoder_sess = ort.InferenceSession(dec_path, sess_options=sess_opts, providers=self.providers)

        # Precompute fixed prompt prefix & suffix token embeddings once
        self._prefix_embeds = self._embed_ids(self.prompt_prefix_ids)
        self._suffix_embeds = self._embed_ids(self.prompt_suffix_ids)

    @staticmethod
    def _select_next_token(
        logits_row: np.ndarray,
        recent_tokens: list[int],
        repetition_penalty: float = 1.0,
    ) -> int:
        if repetition_penalty <= 1.0 or not recent_tokens:
            return int(np.argmax(logits_row))
        scores = np.array(logits_row, dtype=np.float32, copy=True)
        vocab_n = scores.shape[0]
        for tid in set(recent_tokens[-32:]):
            if 0 <= tid < vocab_n:
                val = scores[tid]
                scores[tid] = val / repetition_penalty if val > 0.0 else val * repetition_penalty
        return int(np.argmax(scores))

    def _embed_ids(self, ids: list[int]) -> np.ndarray:
        arr = np.asarray([ids], dtype=np.int64)
        out = self.embed_sess.run(["inputs_embeds"], {"input_ids": arr})[0]
        return np.asarray(out, dtype=np.float32)

    def _encode_audio(self, wav_16k: np.ndarray) -> np.ndarray:
        wav_2d = np.asarray(wav_16k, dtype=np.float32).reshape(1, -1)
        wav_lens = np.array([wav_2d.shape[1]], dtype=np.int64)
        feats, feat_lens = self.preprocessor(wav_2d, wav_lens)
        feats = np.asarray(feats, dtype=np.float32)
        feat_lens = np.asarray(feat_lens, dtype=np.int64).reshape(1)
        # Clamp declared frame length to actual tensor time axis (NeMo floor vs ceil frame guard)
        feat_lens = np.minimum(feat_lens, np.array([feats.shape[2]], dtype=np.int64))
        audio_embeds, audio_lens = self.encoder_sess.run(
            ["audio_embeds", "audio_embeds_lens"],
            {"input_features": feats, "input_features_lens": feat_lens},
        )
        valid_len = max(1, min(int(audio_lens[0]), int(audio_embeds.shape[1])))
        return np.asarray(audio_embeds[:, :valid_len, :], dtype=np.float32)

    def recognize_chunk(
        self,
        wav_16k: np.ndarray,
        sample_rate: int = 16000,
        cancel_token: Optional[CancellationToken] = None,
    ) -> str:
        """Transcribes a single <=35s 16 kHz mono float32 English speech chunk."""
        if cancel_token:
            cancel_token.check()
        wav = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
        if wav.size < 800:
            return ""

        audio_dur_sec = wav.size / float(max(1, sample_rate))
        # Pad very short (<1.5s) utterances slightly so 8x FastConformer subsampling preserves edge phonemes
        if self.edge_pad_sec > 0.0 and audio_dur_sec < 1.5:
            pad_n = int(self.edge_pad_sec * max(1, sample_rate))
            if pad_n > 0:
                wav = np.pad(wav, (pad_n, pad_n), mode="constant")

        audio_embeds = self._encode_audio(wav)
        if cancel_token:
            cancel_token.check()

        prefill_embeds = np.concatenate([self._prefix_embeds, audio_embeds, self._suffix_embeds], axis=1)
        seq_len = int(prefill_embeds.shape[1])

        # Causal additive float32 attention bias: 0.0 on j <= i, -1e9 on j > i
        causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        attn_bias = causal_mask[None, None, :, :]
        position_ids = np.arange(0, seq_len, dtype=np.int64)[None, :]

        feed: dict[str, np.ndarray] = {
            "inputs_embeds": prefill_embeds,
            "attn_bias": attn_bias,
            "position_ids": position_ids,
        }
        empty_kv = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
        for i in range(self.num_layers):
            feed[f"past_key_values.{i}.key"] = empty_kv
            feed[f"past_key_values.{i}.value"] = empty_kv

        output_names = ["logits"]
        for i in range(self.num_layers):
            output_names.append(f"present.{i}.key")
            output_names.append(f"present.{i}.value")

        outs = self.decoder_sess.run(output_names, feed)
        logits = outs[0]
        kv_cache = outs[1:]
        past_len = seq_len

        generated: list[int] = []
        next_token = self._select_next_token(logits[0, -1, :], generated, self.repetition_penalty)

        # Adaptive token ceiling (~10 tokens/sec + headroom, capped by model max_sequence_length)
        max_new_tokens = min(self.max_sequence_length, max(64, int(audio_dur_sec * 10.0) + 48))

        while next_token not in self.eos_token_ids and len(generated) < max_new_tokens:
            generated.append(next_token)

            trim_idx = detect_repetition_loop_trim(generated)
            if trim_idx is not None:
                generated = generated[:trim_idx]
                break

            if cancel_token and (len(generated) % 8 == 0):
                cancel_token.check()

            step_embeds = self._embed_ids([next_token])
            step_feed: dict[str, np.ndarray] = {
                "inputs_embeds": step_embeds,
                "attn_bias": np.zeros((1, 1, 1, past_len + 1), dtype=np.float32),
                "position_ids": np.array([[past_len]], dtype=np.int64),
            }
            for i in range(self.num_layers):
                step_feed[f"past_key_values.{i}.key"] = kv_cache[2 * i]
                step_feed[f"past_key_values.{i}.value"] = kv_cache[2 * i + 1]

            step_outs = self.decoder_sess.run(output_names, step_feed)
            logits = step_outs[0]
            kv_cache = step_outs[1:]
            past_len += 1
            next_token = self._select_next_token(logits[0, -1, :], generated, self.repetition_penalty)

        return decode_gpt2_byte_bpe(generated, self.id_to_token, self.eos_token_ids)
