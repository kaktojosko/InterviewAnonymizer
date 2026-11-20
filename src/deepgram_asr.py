"""
Optional Cloud ASR engine for Deepgram Nova-3 (`POST https://api.deepgram.com/v1/listen`).

Uses pure `httpx` (zero extra SDK dependencies) with:
  - `model=nova-3`, `smart_format=true`, `punctuate=true`, `utterances=true`, `diarize=true`
  - Repeated `keyterm=` prompting automatically extracted from `custom_prompt`
    with transparent HTTP 400 fallback if a language mode disallows keyterms
  - Word-level `TimedUnit` timestamps + native cloud `DiarizationSegment` extraction
  - Strict API-key hygiene: never logged, never echoed in error strings
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Callable, Optional

import httpx

from src.cancellation import CancellationToken
from src.config import DEFAULT_DEEPGRAM_BASE_URL


def extract_deepgram_keyterms(custom_prompt: str, max_terms: int = 80) -> list[str]:
    """
    Extracts clean deduplicated domain terms from a user glossary prompt
    (stripping descriptive prefixes like 'Собеседование системного аналитика:').
    """
    if not custom_prompt or not custom_prompt.strip():
        return []
    src = custom_prompt.split(":", 1)[1].strip() if ":" in custom_prompt else custom_prompt.strip()
    if not src:
        return []
    raw_items = re.split(r"[,;\n]+", src)
    seen: set[str] = set()
    terms: list[str] = []
    for item in raw_items:
        cleaned = " ".join(item.split()).strip(" .\"'«»()[]")
        if not cleaned or len(cleaned) < 2 or len(cleaned) > 64:
            continue
        if len(cleaned) < 3 and not any(ch in "+/#0123456789" for ch in cleaned):
            continue
        low = cleaned.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    return terms


def _create_deepgram_httpx_client(timeout_sec: float) -> httpx.Client:
    try:
        return httpx.Client(timeout=timeout_sec, trust_env=True)
    except httpx.InvalidURL:
        proxy_url = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or None
        )
        return httpx.Client(timeout=timeout_sec, trust_env=False, proxy=proxy_url)


def _prepare_upload_audio(wav_path: str) -> tuple[bytes, str, Optional[str]]:
    """
    Reads WAV bytes directly for clips <= 45 MB (~23 min at 16 kHz mono WAV).
    For longer recordings, compresses via ffmpeg to 64 kbps mono MP3 to cut
    upload time by ~4x, falling back safely to raw WAV if ffmpeg fails.
    Returns (payload_bytes, content_type, optional_temp_path_to_remove).
    """
    file_size = os.path.getsize(wav_path)
    if file_size > 45 * 1024 * 1024:
        fd, tmp_mp3 = tempfile.mkstemp(prefix="dg_nova3_", suffix=".mp3")
        os.close(fd)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    wav_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-b:a",
                    "64k",
                    tmp_mp3,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
            if proc.returncode == 0 and os.path.isfile(tmp_mp3) and os.path.getsize(tmp_mp3) > 1024:
                with open(tmp_mp3, "rb") as f:
                    return f.read(), "audio/mpeg", tmp_mp3
        except Exception:
            pass
        try:
            os.remove(tmp_mp3)
        except OSError:
            pass

    with open(wav_path, "rb") as f:
        return f.read(), "audio/wav", None


def transcribe_with_deepgram_nova3(
    wav_path: str,
    api_key: str,
    language: str = "ru",
    custom_prompt: str = "",
    diarize: bool = True,
    base_url: str = DEFAULT_DEEPGRAM_BASE_URL,
    timeout_sec: float = 300.0,
    cancel_token: Optional[CancellationToken] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> tuple[list, list]:
    """
    Sends 16 kHz mono audio to Deepgram Nova-3 (`/v1/listen`) and returns
    `(units: list[TimedUnit], diar_segments: list[DiarizationSegment])`.
    """
    from src.transcriber import DiarizationSegment, TimedUnit, smooth_diarization_segments

    if cancel_token:
        cancel_token.check()

    clean_key = (api_key or "").strip()
    if not clean_key:
        raise RuntimeError(
            "Для облачной модели Deepgram Nova-3 укажите Deepgram API-ключ "
            "(в поле «API-ключ ASR» или через переменную окружения DEEPGRAM_API_KEY)."
        )

    endpoint = (base_url or DEFAULT_DEEPGRAM_BASE_URL).strip().rstrip("/")
    if not endpoint.endswith("/listen"):
        endpoint = f"{endpoint}/listen"

    lang_clean = (language or "ru").strip().lower()
    if lang_clean == "en":
        dg_lang = "en"
    elif lang_clean == "auto":
        dg_lang = "multi"
    else:
        dg_lang = "ru"

    raw_keyterms = extract_deepgram_keyterms(custom_prompt)
    # Deepgram Nova-3 rejects keyterm on language=multi (HTTP 400); skip upfront to avoid double audio upload.
    keyterms = [] if dg_lang == "multi" else raw_keyterms

    def _build_params(include_keyterms: bool) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = [
            ("model", "nova-3"),
            ("language", dg_lang),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("utterances", "true"),
            ("diarize", "true" if diarize else "false"),
        ]
        if include_keyterms and keyterms:
            for term in keyterms:
                params.append(("keyterm", term))
        return params

    if status_cb:
        kt_note = f", keyterms: {len(keyterms)}" if keyterms else ""
        status_cb(
            f"Облачная транскрибация Deepgram Nova-3 (язык: {dg_lang}, diarize={'on' if diarize else 'off'}{kt_note})..."
        )

    audio_bytes, content_type, tmp_path = _prepare_upload_audio(wav_path)
    try:
        if cancel_token:
            cancel_token.check()

        headers = {
            "Authorization": f"Token {clean_key}",
            "Content-Type": content_type,
        }

        with _create_deepgram_httpx_client(timeout_sec) as client:
            params = _build_params(include_keyterms=bool(keyterms))
            try:
                resp = client.post(endpoint, params=params, headers=headers, content=audio_bytes)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if status_code == 400 and keyterms:
                    # Transparent one-shot fallback when a language mode rejects keyterm
                    if status_cb:
                        status_cb("Deepgram Nova-3: повтор запроса без keyterms (HTTP 400 для выбранного языка)...")
                    resp = client.post(
                        endpoint,
                        params=_build_params(include_keyterms=False),
                        headers=headers,
                        content=audio_bytes,
                    )
                    resp.raise_for_status()
                elif status_code in (401, 403):
                    raise RuntimeError(
                        f"Deepgram API отклонил авторизацию (HTTP {status_code}). "
                        "Проверьте Deepgram API-ключ."
                    ) from None
                elif status_code == 402:
                    raise RuntimeError(
                        "Deepgram API: недостаточно средств на балансе (HTTP 402)."
                    ) from None
                else:
                    raise RuntimeError(
                        f"Ошибка Deepgram Nova-3 API (HTTP {status_code})."
                    ) from None
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Сетевая ошибка при обращении к Deepgram Nova-3 API: {type(exc).__name__}"
                ) from None

            data = resp.json()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if cancel_token:
        cancel_token.check()

    channels = (data or {}).get("results", {}).get("channels", [])
    if not channels:
        return [], []
    alts = channels[0].get("alternatives", [])
    if not alts:
        return [], []
    alt0 = alts[0] or {}
    words = alt0.get("words") or []

    units: list[TimedUnit] = []
    raw_diar: list[DiarizationSegment] = []

    for w in words:
        raw_txt = str(w.get("punctuated_word") or w.get("word") or "").strip()
        if not raw_txt:
            continue
        w_start = max(0.0, float(w.get("start", 0.0) or 0.0))
        w_end = max(w_start + 0.02, float(w.get("end", w_start + 0.08) or (w_start + 0.08)))
        units.append(TimedUnit(start=w_start, end=w_end, text=f" {raw_txt}"))
        if diarize and w.get("speaker") is not None:
            try:
                spk_id = int(w["speaker"])
                raw_diar.append(DiarizationSegment(start=w_start, end=w_end, speaker=spk_id))
            except (TypeError, ValueError):
                pass

    if not units:
        fallback_txt = str(alt0.get("transcript") or "").strip()
        if fallback_txt:
            meta_dur = float((data or {}).get("metadata", {}).get("duration", 0.0) or 0.0)
            units.append(TimedUnit(start=0.0, end=max(0.1, meta_dur), text=f" {fallback_txt}"))

    diar_segments = (
        smooth_diarization_segments(raw_diar, merge_gap_sec=0.45, min_segment_sec=0.05)
        if raw_diar
        else []
    )
    return units, diar_segments
