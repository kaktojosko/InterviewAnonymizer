import difflib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from src.cancellation import CancellationToken

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_META_LEAK_RE = re.compile(
    r"^(?:исправленный\s+текст|вот\s+исправленный|конечно[,!]|готово[,!]|примечание:)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")

_CYR_TO_LAT_TABLE = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "ch",
        "ш": "sh",
        "щ": "sh",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


@dataclass
class _WorkChunk:
    work_id: int
    orig_idx: int
    part_idx: int
    total_parts: int
    speaker_label: str
    text: str


def _normalize_chat_endpoint(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _is_local_endpoint(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except Exception:
        return False
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"):
        return True
    if host.endswith(".local") or host.startswith("192.168.") or host.startswith("10."):
        return True
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) == 4 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
            return True
    return False


def _is_deepseek_endpoint(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except Exception:
        return False
    return host == "deepseek.com" or host.endswith(".deepseek.com")


def _translit_for_similarity(text: str) -> str:
    """
    Normalizes Russian + Latin text into a script-neutral phonetic lowercase
    form so legitimate Cyrillic->Latin IT term conversions (e.g. 'постгрес,
    рэдис, конфлюенс' -> 'PostgreSQL, Redis, Confluence') keep high similarity
    while semantic hallucinations / complete rewrites drop sharply.
    """
    low = (text or "").lower().translate(_CYR_TO_LAT_TABLE)
    return re.sub(r"[^a-z0-9]+", " ", low).strip()


def _split_long_turn_into_chunks(
    text: str,
    max_words: int = 155,
    max_chars: int = 1050,
) -> list[str]:
    """
    Splits very long monologues (e.g. 5-7 minute ~1000-word answers) into
    focused ~100-155 word chunks along sentence boundaries before LLM editing.
    Prevents fast cloud/local LLMs from losing attention mid-monologue while
    preserving exact whitespace-normalized content for seamless reassembly.
    """
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return [""]

    eff_words = max(30, int(max_words))
    eff_chars = max(240, int(max_chars))
    if len(cleaned) <= eff_chars and len(cleaned.split()) <= eff_words:
        return [cleaned]

    def _split_by_words(piece: str) -> list[str]:
        words = piece.split()
        out: list[str] = []
        buf: list[str] = []
        buf_chars = 0
        for w in words:
            add_chars = len(w) + (1 if buf else 0)
            if buf and (len(buf) >= eff_words or (buf_chars + add_chars) > eff_chars):
                out.append(" ".join(buf))
                buf = [w]
                buf_chars = len(w)
            else:
                buf.append(w)
                buf_chars += add_chars
        if buf:
            out.append(" ".join(buf))
        return out

    # Step 1: split on sentence endings, falling back to clauses / words only for oversized sentences
    atomic_pieces: list[str] = []
    for sent in _SENTENCE_SPLIT_RE.split(cleaned):
        s = sent.strip()
        if not s:
            continue
        if len(s) <= eff_chars and len(s.split()) <= eff_words:
            atomic_pieces.append(s)
            continue
        for clause in _CLAUSE_SPLIT_RE.split(s):
            c = clause.strip()
            if not c:
                continue
            if len(c) <= eff_chars and len(c.split()) <= eff_words:
                atomic_pieces.append(c)
            else:
                atomic_pieces.extend(_split_by_words(c))

    if not atomic_pieces:
        return [cleaned]

    # Step 2: pack atomic pieces into balanced chunks within word/char budgets
    chunks: list[str] = []
    cur_parts: list[str] = []
    cur_words = 0
    cur_chars = 0

    for piece in atomic_pieces:
        p_words = len(piece.split())
        p_chars = len(piece) + (1 if cur_parts else 0)
        if cur_parts and ((cur_words + p_words) > eff_words or (cur_chars + p_chars) > eff_chars):
            chunks.append(" ".join(cur_parts))
            cur_parts = [piece]
            cur_words = p_words
            cur_chars = len(piece)
        else:
            cur_parts.append(piece)
            cur_words += p_words
            cur_chars += p_chars
    if cur_parts:
        chunks.append(" ".join(cur_parts))

    # Avoid stranding a tiny 1-2 sentence tail without context if it fits comfortably into the previous chunk
    if len(chunks) >= 2:
        tail = chunks[-1]
        prev = chunks[-2]
        if (
            len(tail.split()) < 22
            and len(tail) < 170
            and (len(prev.split()) + len(tail.split())) <= int(eff_words * 1.25)
            and (len(prev) + 1 + len(tail)) <= int(eff_chars * 1.25)
        ):
            chunks[-2] = f"{prev} {tail}"
            chunks.pop()

    return chunks


def _sample_transcript_for_glossary(turns: list, max_chars: int = 44000) -> str:
    """
    Builds a representative sample of the full transcript for Pass-0 global
    glossary extraction. Fits an entire ~1-hour interview verbatim on cloud
    models (44k chars) and evenly samples beginning/middle/end on longer calls
    or smaller local context windows.
    """
    lines: list[str] = []
    for t in turns:
        txt = " ".join((getattr(t, "text", "") or "").split()).strip()
        if not txt:
            continue
        spk = getattr(t, "speaker_label", "") or "Спикер"
        lines.append(f"{spk}: {txt}")

    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    # Evenly sample slices across the full recording so late-meeting terms are captured too
    n_slices = 6
    slice_budget = max(400, (max_chars - 64) // n_slices)
    total_len = len(full_text)
    slices: list[str] = []
    for idx in range(n_slices):
        if idx == n_slices - 1:
            start = max(0, total_len - slice_budget)
        else:
            start = int((idx * (total_len - slice_budget)) / max(1, n_slices - 1))
        chunk = full_text[start : start + slice_budget].strip()
        if chunk:
            slices.append(chunk)
    return "\n...\n".join(slices)[:max_chars]


def _build_glossary_extraction_messages(
    sample_text: str,
    custom_prompt: str = "",
) -> list[dict[str, str]]:
    ref_block = ""
    if custom_prompt and custom_prompt.strip():
        ref_block = (
            f"\nПользователь уже указал базовый словарь: {custom_prompt.strip()}\n"
            "Сохраняй его каноническое написание и дополни ТОЛЬКО недостающими сущностями, "
            "продуктами, компаниями, инструментами, аббревиатурами и именами из этой записи.\n"
        )

    system_msg = (
        "Ты — редактор расшифровок интервью. Проанализируй черновую расшифровку (ASR) целиком и "
        "составь единый глоссарий ключевых сущностей этой записи, чтобы во всех частях текста "
        "использовалось ОДНО И ТО ЖЕ каноническое написание.\n"
        "Включи в список (до 45 пунктов):\n"
        "1. Названия продуктов, проектов и компаний (если ASR искажала одно название по-разному "
        "в разных местах записи, выбери ОДНО наиболее вероятное каноническое написание).\n"
        "2. Специфические профессиональные термины, инструменты, протоколы, методологии и аббревиатуры, "
        "которые реально обсуждаются в этой записи.\n"
        "3. Имена участников, если они отчётливо и однозначно встречаются в тексте.\n"
        "СТРОГО ЗАПРЕЩЕНО включать обычные общеупотребительные русские слова или выдумывать термины, "
        "которых нет в расшифровке.\n"
        "Верни ТОЛЬКО валидный JSON вида {\"terms\": [\"Термин 1\", \"Термин 2\", ...]}."
        f"{ref_block}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": sample_text},
    ]


def _strip_json_wrappers(raw_content: str) -> str:
    cleaned = str(raw_content or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    if "<think>" in cleaned.lower():
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    return cleaned


def _parse_json_lenient(cleaned: str):
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        candidates = []
        if 0 <= obj_start < obj_end:
            candidates.append(cleaned[obj_start : obj_end + 1])
        if 0 <= arr_start < arr_end:
            candidates.append(cleaned[arr_start : arr_end + 1])
        for cand in candidates:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
    return None


def _parse_glossary_terms(raw_content: str, max_terms: int = 50) -> list[str]:
    cleaned = _strip_json_wrappers(raw_content)
    parsed = _parse_json_lenient(cleaned)
    if parsed is None:
        return []

    raw_items = None
    if isinstance(parsed, list):
        raw_items = parsed
    elif isinstance(parsed, dict):
        for key in ("terms", "glossary", "entities", "items", "keywords"):
            if isinstance(parsed.get(key), list):
                raw_items = parsed[key]
                break

    if not isinstance(raw_items, list):
        return []

    terms: list[str] = []
    seen_keys: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            val = item.get("term") or item.get("name") or item.get("text") or ""
        elif isinstance(item, str):
            val = item
        else:
            continue
        cleaned_term = " ".join(str(val).split()).strip(" \t\r\n,;:.•-*\"'«»")
        if len(cleaned_term) < 2 or len(cleaned_term) > 60:
            continue
        if len(cleaned_term.split()) > 5:
            continue
        norm_key = re.sub(r"[^a-zа-яё0-9]+", "", cleaned_term.lower())
        if len(norm_key) < 2 or norm_key in seen_keys:
            continue
        seen_keys.add(norm_key)
        terms.append(cleaned_term)
        if len(terms) >= max_terms:
            break
    return terms


def _merge_glossaries(custom_prompt: str, auto_terms: list[str]) -> str:
    """
    Merges user's explicit custom_prompt with Pass-0 auto-extracted terms.
    User's explicit spelling ALWAYS takes priority and duplicates are filtered out.
    """
    base = (custom_prompt or "").strip()
    if not auto_terms:
        return base

    existing_keys: set[str] = set()
    if base:
        for raw_tok in re.split(r"[,;:\n]+", base):
            tok_key = re.sub(r"[^a-zа-яё0-9]+", "", raw_tok.strip().lower())
            if tok_key:
                existing_keys.add(tok_key)

    new_terms: list[str] = []
    seen_new: set[str] = set()
    for term in auto_terms:
        t_clean = " ".join(str(term).split()).strip()
        if not t_clean:
            continue
        t_key = re.sub(r"[^a-zа-яё0-9]+", "", t_clean.lower())
        if not t_key or t_key in existing_keys or t_key in seen_new:
            continue
        if base and re.search(
            r"(?<![A-Za-zА-Яа-яЁё0-9])" + re.escape(t_clean) + r"(?![A-Za-zА-Яа-яЁё0-9])",
            base,
            re.IGNORECASE,
        ):
            continue
        seen_new.add(t_key)
        new_terms.append(t_clean)

    if not new_terms:
        return base

    auto_block = (
        "Ключевые сущности, продукты, имена и термины из этой записи "
        "(используй ЕДИНОЕ написание во всём тексте): " + ", ".join(new_terms)
    )
    if base:
        return f"{base}\n{auto_block}"
    return auto_block


def _build_turn_batches(
    turns: list,
    batch_size: int = 12,
    max_batch_chars: int = 3200,
) -> list[tuple[int, int]]:
    """
    Splits turns (or sub-turn chunks) into contextual batches bounded by BOTH
    item count and character budget so long consecutive monologues never
    overflow LLM output token limits or degrade editing attention.
    """
    eff_size = max(1, min(int(batch_size), 25))
    eff_chars = max(600, int(max_batch_chars))
    batches: list[tuple[int, int]] = []
    n = len(turns)
    i = 0
    while i < n:
        start = i
        chars = 0
        while i < n and (i - start) < eff_size:
            t_len = len(getattr(turns[i], "text", "") or "")
            if i > start and (chars + t_len) > eff_chars:
                break
            chars += t_len
            i += 1
        batches.append((start, i))
    return batches


def _create_httpx_client(endpoint: str, timeout_sec: float) -> httpx.Client:
    """
    Creates an httpx.Client suitable for both local (Ollama / LM Studio) and
    cloud (DeepSeek / Qwen DashScope) OpenAI-compatible endpoints:
      - Local endpoints use trust_env=False so private transcripts never pass
        through system proxies and never fail on IPv6 NO_PROXY=[::1].
      - Cloud endpoints respect HTTPS_PROXY / HTTP_PROXY, falling back gracefully
        if httpx rejects an IPv6 NO_PROXY entry.
    """
    if _is_local_endpoint(endpoint):
        return httpx.Client(timeout=timeout_sec, trust_env=False)
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


def _build_polish_messages(
    batch_turns: list[tuple[int, str, str]],
    prev_context: list[tuple[str, str]],
    custom_prompt: str = "",
) -> list[dict[str, str]]:
    glossary_block = ""
    if custom_prompt and custom_prompt.strip():
        glossary_block = (
            f"\nПрофильный словарь и глоссарий записи (используй именно их каноническое написание, "
            f"если по звучанию и контексту имелся в виду соответствующий термин): {custom_prompt.strip()}\n"
        )

    system_msg = (
        "Ты — профессиональный редактор расшифровок (ASR) интервью.\n"
        "Твоя задача — внимательно вычитать КАЖДУЮ реплику от первого до последнего слова и исправить "
        "ошибки распознавания речи, не искажая смысл и не додумывая от себя.\n"
        "Правила редактирования:\n"
        "1. Фонетические искажения и «русские обманки» (ВАЖНО): распознаватель речи часто превращает "
        "незнакомые термины, инструменты, протоколы и названия продуктов либо в искажённую латиницу/кириллицу, "
        "либо в грамматически правильные обычные русские слова, нелепые в данном контексте "
        "(например: «поджирописью» → «по gRPC», «из революционных баз» → «из реляционных баз», "
        "«по листам» в контексте API → «по RESTам», «оауф 2» → «OAuth 2»). "
        "Активно проверяй КАЖДОЕ подозрительное слово в контексте фразы и переданного словаря.\n"
        "2. СТРОГОЕ ПРАВИЛО ФОНЕТИЧЕСКОГО СХОДСТВА (анти-галлюцинации): замена допустима ТОЛЬКО если "
        "исправленное слово или термин реально близко ПО ЗВУЧАНИЮ к исходным слогам ASR И логично по контексту. "
        "ЗАПРЕЩЕНО выдумывать фонетически непохожие длинные слова, угадывать другой «близкий по теме» термин "
        "или менять осмысленные нормальные слова (например, не превращай «GPT» в «JWT», если речь об ИИ/LLM, "
        "но исправь «GPT» → «JWT», если в этой фразе речь явно об авторизации/Bearer/OAuth-токенах). "
        "Если фрагмент ASR сильно оборван или искажён и не восстанавливается однозначно по звучанию — "
        "ОСТАВЬ исходный фрагмент без выдумывания.\n"
        "3. Выпавшие аббревиатуры и склейки: если ASR «проглотила» короткую аббревиатуру перед дефисом "
        "или склеила/разорвала слово, восстанови её только когда она однозначно следует из контекста или словаря.\n"
        "4. Единообразие названий: используй ОДНО И ТО ЖЕ каноническое написание продуктов, компаний, "
        "инструментов и имён из переданного словаря на протяжении всех реплик.\n"
        "5. Сохраняй исходный язык каждой реплики (не переводи EN↔RU), разговорный стиль и полноту текста. "
        "Поправляй очевидную пунктуацию. СТРОГО ЗАПРЕЩЕНО переписывать реплики своими словами, сжимать/сокращать "
        "смысл, добавлять пояснения, объединять или удалять элементы массива.\n"
        "6. Верни ТОЛЬКО валидный JSON-объект вида {\"turns\": [{\"id\": 0, \"text\": \"...\"}, ...]} "
        "с тем же самым списком id и в том же порядке."
        f"{glossary_block}"
    )

    payload = {
        "context_before": [{"speaker": spk, "text": txt} for spk, txt in prev_context],
        "turns": [{"id": idx, "speaker": spk, "text": txt} for idx, spk, txt in batch_turns],
    }
    user_msg = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _extract_turns_list(parsed) -> Optional[list]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("turns", "items", "results", "segments"):
            val = parsed.get(key)
            if isinstance(val, list):
                return val
        nested = parsed.get("data")
        if isinstance(nested, dict) and isinstance(nested.get("turns"), list):
            return nested["turns"]
    return None


def _validate_batch_response_detailed(
    raw_content: str,
    batch_turns: list[tuple[int, str, str]],
    finish_reason: str = "",
    has_reasoning: bool = False,
) -> tuple[Optional[dict[int, str]], str, int]:
    """
    Validates an LLM JSON batch response with per-turn guardrails so one
    suspect turn never discards valid fixes for the rest of the batch.
    Returns (validated_map_or_None, error_reason, fallback_turn_count).
    """
    if not raw_content or not str(raw_content).strip():
        if finish_reason == "length" and has_reasoning:
            return None, "thinking исчерпал max_tokens (пустой content, finish_reason=length)", 0
        if finish_reason == "length":
            return None, "пустой ответ (finish_reason=length)", 0
        return None, "пустой ответ / пробелы JSON-режима", 0

    cleaned = _strip_json_wrappers(raw_content)
    parsed = _parse_json_lenient(cleaned)

    if parsed is None:
        if finish_reason == "length":
            return None, "обрезанный JSON (finish_reason=length)", 0
        return None, "невалидный JSON", 0

    items = _extract_turns_list(parsed)
    if not isinstance(items, list) or not items:
        return None, "в JSON отсутствует список turns", 0

    orig_by_id = {idx: txt for idx, _, txt in batch_turns}
    accepted_map: dict[int, str] = {}
    rejected_reasons: list[str] = []
    seen_ids: set[int] = set()

    for item in items:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            rejected_reasons.append("bad_item")
            continue
        try:
            tid = int(item["id"])
        except (TypeError, ValueError):
            rejected_reasons.append("bad_id")
            continue
        if tid not in orig_by_id or tid in seen_ids:
            rejected_reasons.append("unknown_or_dup_id")
            continue
        seen_ids.add(tid)

        new_txt = " ".join(str(item["text"]).split()).strip()
        orig_txt = orig_by_id[tid].strip()
        if not new_txt:
            rejected_reasons.append("empty_turn")
            continue
        if _META_LEAK_RE.match(new_txt):
            rejected_reasons.append("meta_leak")
            continue

        # Per-turn guardrail against aggressive summarization or runaway hallucination
        if len(orig_txt) >= 24:
            ratio = len(new_txt) / float(max(1, len(orig_txt)))
            if ratio < 0.45 or ratio > 1.75:
                rejected_reasons.append(f"length_ratio={ratio:.2f}")
                continue

        # Script-neutral rewrite-drift guardrail on substantive turns (>= 60 chars):
        # catches invented hallucinations while allowing dense Cyrillic->Latin IT term transliteration.
        if len(orig_txt) >= 60:
            raw_sim = difflib.SequenceMatcher(None, orig_txt.lower(), new_txt.lower()).ratio()
            translit_sim = difflib.SequenceMatcher(
                None,
                _translit_for_similarity(orig_txt),
                _translit_for_similarity(new_txt),
            ).ratio()
            best_sim = max(raw_sim, translit_sim)
            if best_sim < 0.42:
                rejected_reasons.append(f"rewrite_drift={best_sim:.2f}")
                continue

        accepted_map[tid] = new_txt

    if not accepted_map:
        top_reason = ", ".join(rejected_reasons[:3]) if rejected_reasons else "нет совпавших id"
        return None, f"все реплики отклонены проверкой ({top_reason})", 0

    # Fill any omitted or individually rejected turns with their original ASR text
    final_map: dict[int, str] = {}
    fallback_count = 0
    for idx, _, orig_txt in batch_turns:
        if idx in accepted_map:
            final_map[idx] = accepted_map[idx]
        else:
            final_map[idx] = orig_txt
            fallback_count += 1

    return final_map, "", fallback_count


def _parse_and_validate_batch_response(
    raw_content: str,
    batch_turns: list[tuple[int, str, str]],
) -> Optional[dict[int, str]]:
    """
    Strictly validates the LLM JSON response so timestamps, speaker labels,
    turn count, and meaning can never be corrupted or hallucinated.
    Returns mapping {turn_id: polished_text} or None to trigger safe fallback.
    """
    result_map, _, _ = _validate_batch_response_detailed(raw_content, batch_turns)
    return result_map


def _sleep_with_cancel(
    delay_sec: float,
    cancel_token: Optional[CancellationToken] = None,
    abort_event: Optional[threading.Event] = None,
) -> None:
    end_t = time.time() + max(0.0, float(delay_sec))
    while True:
        if cancel_token:
            cancel_token.check()
        if abort_event is not None and abort_event.is_set():
            return
        rem = end_t - time.time()
        if rem <= 0.0:
            return
        time.sleep(min(0.25, rem))


def polish_turns_with_local_llm(
    turns: list,
    base_url: str,
    model: str,
    custom_prompt: str = "",
    api_key: str = "",
    batch_size: int = 12,
    timeout_sec: float = 90.0,
    status_cb: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
    max_batch_chars: int = 3200,
    max_turn_words: int = 155,
    max_turn_chars: int = 1050,
    extract_global_glossary: bool = True,
    chunk_fallback_fn: Optional[Callable[[str], str]] = None,
) -> tuple[list, dict]:
    """
    Stage 4 optional LLM post-editor (cloud DeepSeek / Qwen DashScope OR local
    Ollama / LM Studio / vLLM OpenAI-compatible API). Splits very long monologues
    into focused sentence-aligned chunks, optionally runs a Pass-0 cross-file
    global glossary extraction for canonical entity consistency, and polishes
    ASR text in contextual batches while strictly preserving original timestamps,
    speaker labels, and turn structure. Any error safely preserves original turns.
    """
    if not turns:
        return turns, {
            "applied": False,
            "batches_ok": 0,
            "batches_total": 0,
            "changed_turns": 0,
            "failed_indices": [],
            "auto_glossary_count": 0,
            "auto_glossary_terms": [],
            "split_turns": 0,
            "chunks_total": 0,
        }

    endpoint = _normalize_chat_endpoint(base_url)
    model_name = (model or "").strip()
    if not endpoint or not model_name:
        if status_cb:
            status_cb("LLM-корректура пропущена: не указан URL или имя модели.")
        return turns, {
            "applied": False,
            "batches_ok": 0,
            "batches_total": 0,
            "changed_turns": 0,
            "failed_indices": list(range(len(turns))),
            "auto_glossary_count": 0,
            "auto_glossary_terms": [],
            "split_turns": 0,
            "chunks_total": len(turns),
        }

    is_local = _is_local_endpoint(endpoint)
    is_deepseek = _is_deepseek_endpoint(endpoint)
    mode_tag = "локально" if is_local else "облачный API"

    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    orig_turns = list(turns)
    work_items: list[_WorkChunk] = []
    work_ids_by_orig: list[list[int]] = []
    split_turns_count = 0

    for orig_idx, turn in enumerate(orig_turns):
        parts = _split_long_turn_into_chunks(
            getattr(turn, "text", "") or "",
            max_words=max_turn_words,
            max_chars=max_turn_chars,
        )
        if len(parts) > 1:
            split_turns_count += 1
        ids_for_turn: list[int] = []
        for p_idx, p_txt in enumerate(parts):
            wid = len(work_items)
            work_items.append(
                _WorkChunk(
                    work_id=wid,
                    orig_idx=orig_idx,
                    part_idx=p_idx,
                    total_parts=len(parts),
                    speaker_label=turn.speaker_label,
                    text=p_txt,
                )
            )
            ids_for_turn.append(wid)
        work_ids_by_orig.append(ids_for_turn)

    polished_chunks: list[str] = [w.text for w in work_items]
    batches = _build_turn_batches(
        work_items,
        batch_size=batch_size,
        max_batch_chars=max_batch_chars,
    )
    total_batches = len(batches)
    total_chars = sum(len(getattr(t, "text", "") or "") for t in orig_turns)
    batches_ok = 0
    done_batches = 0
    failed_work_ids: set[int] = set()
    use_json_response_format = True
    use_deepseek_thinking_disable = is_deepseek
    state_lock = threading.Lock()
    abort_event = threading.Event()
    cloud_workers = 1 if (is_local or total_batches <= 2) else min(3, total_batches - 1)

    if status_cb:
        mode_extra = ", non-thinking" if use_deepseek_thinking_disable else ""
        conc_extra = f", потоков: {cloud_workers}" if cloud_workers > 1 else ""
        split_extra = f", разбито длинных реплик: {split_turns_count}" if split_turns_count > 0 else ""
        status_cb(
            f"Шаг 4: LLM-корректура ({mode_tag}: {model_name}{mode_extra} @ {endpoint}, "
            f"батчей: {total_batches}{conc_extra}{split_extra})..."
        )

    _TRANSIENT_HTTP_CODES = (429, 500, 502, 503, 504)

    def _post_with_retry(client: httpx.Client, cur_body: dict) -> httpx.Response:
        max_retries = 2
        for attempt in range(max_retries + 1):
            if cancel_token:
                cancel_token.check()
            try:
                resp = client.post(endpoint, headers=headers, json=cur_body)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                sc = exc.response.status_code if exc.response is not None else 0
                if sc in _TRANSIENT_HTTP_CODES and attempt < max_retries and not abort_event.is_set():
                    retry_after = 0.0
                    if exc.response is not None:
                        ra_hdr = exc.response.headers.get("Retry-After", "")
                        try:
                            retry_after = float(ra_hdr)
                        except (TypeError, ValueError):
                            retry_after = 0.0
                    delay = min(8.0, max(0.5, retry_after)) if retry_after > 0.0 else min(4.0, 1.0 * (2 ** attempt))
                    _sleep_with_cancel(delay, cancel_token=cancel_token, abort_event=abort_event)
                    continue
                raise
            except httpx.ReadTimeout:
                if attempt < max_retries and not abort_event.is_set():
                    _sleep_with_cancel(min(4.0, 1.0 * (2 ** attempt)), cancel_token=cancel_token, abort_event=abort_event)
                    continue
                raise

    def _send_completion(client: httpx.Client, req_body: dict) -> tuple[str, str, bool]:
        nonlocal use_json_response_format, use_deepseek_thinking_disable
        cur_body = dict(req_body)
        try:
            resp = _post_with_retry(client, cur_body)
        except httpx.HTTPStatusError as rf_exc:
            sc = rf_exc.response.status_code if rf_exc.response is not None else 0
            if sc in (400, 422) and ("response_format" in cur_body or "thinking" in cur_body):
                # Step 1: strip response_format FIRST while keeping thinking={"type": "disabled"} intact!
                if "response_format" in cur_body:
                    with state_lock:
                        use_json_response_format = False
                    cur_body.pop("response_format", None)
                    try:
                        resp = _post_with_retry(client, cur_body)
                    except httpx.HTTPStatusError as rf_exc2:
                        sc2 = rf_exc2.response.status_code if rf_exc2.response is not None else 0
                        if sc2 in (400, 422) and "thinking" in cur_body:
                            with state_lock:
                                use_deepseek_thinking_disable = False
                            cur_body.pop("thinking", None)
                            resp = _post_with_retry(client, cur_body)
                        else:
                            raise
                else:
                    with state_lock:
                        use_deepseek_thinking_disable = False
                    cur_body.pop("thinking", None)
                    resp = _post_with_retry(client, cur_body)
            else:
                raise

        data = resp.json() or {}
        choice0 = (data.get("choices") or [{}])[0] or {}
        msg0 = choice0.get("message") or {}
        raw_content = msg0.get("content") or ""
        finish_reason = str(choice0.get("finish_reason") or "").strip().lower()
        has_reasoning = bool(msg0.get("reasoning_content"))
        return raw_content, finish_reason, has_reasoning

    effective_prompt = (custom_prompt or "").strip()
    auto_terms: list[str] = []

    def _run_batch(client: httpx.Client, b_idx: int, start_i: int, end_i: int) -> None:
        nonlocal batches_ok, done_batches
        if cancel_token:
            cancel_token.check()
        if abort_event.is_set():
            with state_lock:
                failed_work_ids.update(range(start_i, end_i))
            return

        batch_spec = [
            (i, work_items[i].speaker_label, work_items[i].text)
            for i in range(start_i, end_i)
        ]
        ctx_start = max(0, start_i - 2)
        prev_ctx = [
            (
                work_items[i].speaker_label,
                polished_chunks[i] if is_local else work_items[i].text,
            )
            for i in range(ctx_start, start_i)
        ]

        messages = _build_polish_messages(batch_spec, prev_ctx, custom_prompt=effective_prompt)
        batch_chars = sum(len(txt) for _, _, txt in batch_spec)
        base_max_tokens = min(16384, max(2048, batch_chars * 3))
        with state_lock:
            want_thinking_disable = use_deepseek_thinking_disable
            want_json_fmt = use_json_response_format

        body: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.05,
            "max_tokens": base_max_tokens,
            "stream": False,
        }
        if want_thinking_disable:
            # DeepSeek deepseek-flash / deepseek-v4-pro default to thinking=enabled (high),
            # which spends max_tokens inside reasoning_content and leaves content empty/truncated.
            body["thinking"] = {"type": "disabled"}
        else:
            body["top_p"] = 0.9
        if want_json_fmt:
            body["response_format"] = {"type": "json_object"}

        batch_applied = False
        try:
            raw_content, finish_reason, has_reasoning = _send_completion(client, body)
            validated, fail_reason, fallback_cnt = _validate_batch_response_detailed(
                raw_content,
                batch_spec,
                finish_reason=finish_reason,
                has_reasoning=has_reasoning,
            )

            # One-shot smart recovery for DeepSeek JSON-mode blank spaces OR max_tokens truncation
            if validated is None:
                retry_body: Optional[dict] = None
                if not str(raw_content).strip() and "response_format" in body:
                    retry_body = dict(body)
                    retry_body.pop("response_format", None)
                    if finish_reason == "length":
                        retry_body["max_tokens"] = min(32768, int(body["max_tokens"]) * 2)
                elif finish_reason == "length":
                    retry_body = dict(body)
                    retry_body["max_tokens"] = min(32768, int(body["max_tokens"]) * 2)

                if retry_body is not None:
                    if cancel_token:
                        cancel_token.check()
                    raw_content, finish_reason, has_reasoning = _send_completion(client, retry_body)
                    validated, fail_reason, fallback_cnt = _validate_batch_response_detailed(
                        raw_content,
                        batch_spec,
                        finish_reason=finish_reason,
                        has_reasoning=has_reasoning,
                    )

            if validated is not None:
                with state_lock:
                    for idx, new_text in validated.items():
                        polished_chunks[idx] = new_text
                    batches_ok += 1
                batch_applied = True
                if fallback_cnt > 0 and status_cb:
                    status_cb(
                        f"LLM-корректура: батч {b_idx + 1}/{total_batches} применён "
                        f"(оставлено без правок по guard: {fallback_cnt}/{len(batch_spec)})."
                    )
            else:
                if status_cb:
                    status_cb(
                        f"LLM-корректура: батч {b_idx + 1}/{total_batches} не прошёл валидацию "
                        f"({fail_reason}) — оставлен исходный ASR."
                    )
        except httpx.HTTPStatusError as http_exc:
            status_code = http_exc.response.status_code if http_exc.response is not None else 0
            if status_code in (401, 403):
                first_abort = not abort_event.is_set()
                abort_event.set()
                if first_abort and status_cb:
                    status_cb(
                        f"LLM-корректура: ошибка авторизации (HTTP {status_code}) — проверьте API-ключ; оставлен исходный ASR."
                    )
            else:
                if status_cb:
                    status_cb(
                        f"LLM-корректура: ошибка HTTP {status_code} на батче {b_idx + 1}/{total_batches} — оставлен исходный ASR."
                    )
        except Exception as exc:
            if cancel_token and cancel_token.is_cancelled():
                raise
            is_conn_fail = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
            first_abort = True
            if is_conn_fail:
                first_abort = not abort_event.is_set()
                abort_event.set()
            if first_abort and status_cb:
                status_cb(
                    f"LLM-корректура: ошибка на батче {b_idx + 1}/{total_batches} ({exc}) — оставлен исходный ASR."
                )

        with state_lock:
            if not batch_applied:
                failed_work_ids.update(range(start_i, end_i))
            done_batches += 1
            cur_done = done_batches

        if status_cb and total_batches > 1 and not abort_event.is_set():
            status_cb(f"LLM-корректура: обработано {cur_done}/{total_batches} батчей...")

    t0 = time.time()
    with _create_httpx_client(endpoint, timeout_sec) as client:
        # Pass 0: cross-file global meeting glossary extraction on multi-batch recordings (>= 1500 chars)
        # so product/company/tool/person names stay 100% consistent across all batches.
        if extract_global_glossary and total_chars >= 1500 and total_batches >= 2:
            if cancel_token:
                cancel_token.check()
            sample_text = _sample_transcript_for_glossary(
                orig_turns,
                max_chars=12000 if is_local else 44000,
            )
            g_msgs = _build_glossary_extraction_messages(sample_text, custom_prompt=effective_prompt)
            g_body: dict = {
                "model": model_name,
                "messages": g_msgs,
                "temperature": 0.0,
                "max_tokens": 1024,
                "stream": False,
            }
            if use_deepseek_thinking_disable:
                g_body["thinking"] = {"type": "disabled"}
            if use_json_response_format:
                g_body["response_format"] = {"type": "json_object"}
            try:
                g_raw, _, _ = _send_completion(client, g_body)
                auto_terms = _parse_glossary_terms(g_raw, max_terms=50)
                if auto_terms:
                    effective_prompt = _merge_glossaries(effective_prompt, auto_terms)
                    if status_cb:
                        preview = ", ".join(auto_terms[:10]) + ("..." if len(auto_terms) > 10 else "")
                        status_cb(
                            f"LLM-корректура: извлечён глобальный глоссарий записи ({len(auto_terms)} терм.): {preview}"
                        )
            except httpx.HTTPStatusError as http_exc:
                status_code = http_exc.response.status_code if http_exc.response is not None else 0
                if status_code in (401, 403):
                    abort_event.set()
                    if status_cb:
                        status_cb(
                            f"LLM-корректура: ошибка авторизации (HTTP {status_code}) — проверьте API-ключ; оставлен исходный ASR."
                        )
                else:
                    if status_cb:
                        status_cb(
                            f"LLM-корректура: автоглоссарий пропущен (HTTP {status_code}) — продолжаем корректуру."
                        )
            except Exception as exc:
                if cancel_token and cancel_token.is_cancelled():
                    raise
                if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                    abort_event.set()
                    if status_cb:
                        status_cb(
                            f"LLM-корректура: ошибка подключения ({exc}) — оставлен исходный ASR."
                        )
                elif status_cb:
                    status_cb(f"LLM-корректура: автоглоссарий пропущен ({exc}) — продолжаем корректуру.")

        if abort_event.is_set():
            failed_work_ids.update(range(len(work_items)))
        else:
            # Always probe batch 0 synchronously first: validates auth/connectivity in 1 request
            # and negotiates response_format / thinking flags cleanly before any fan-out.
            _run_batch(client, 0, batches[0][0], batches[0][1])

            if total_batches > 1 and not abort_event.is_set():
                if cloud_workers <= 1:
                    for b_idx in range(1, total_batches):
                        if abort_event.is_set():
                            failed_work_ids.update(range(batches[b_idx][0], batches[b_idx][1]))
                            break
                        _run_batch(client, b_idx, batches[b_idx][0], batches[b_idx][1])
                else:
                    pool = ThreadPoolExecutor(max_workers=cloud_workers)
                    try:
                        futs = [
                            pool.submit(_run_batch, client, b_idx, batches[b_idx][0], batches[b_idx][1])
                            for b_idx in range(1, total_batches)
                        ]
                        for fut in as_completed(futs):
                            fut.result()
                    except BaseException:
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    else:
                        pool.shutdown(wait=True)
            elif total_batches > 1 and abort_event.is_set():
                for b_idx in range(1, total_batches):
                    failed_work_ids.update(range(batches[b_idx][0], batches[b_idx][1]))

    # Reassemble sub-turn chunks seamlessly under the exact original (start, end, speaker_label)
    polished_turns: list = []
    failed_indices: list[int] = []
    changed_turns = 0

    for orig_idx, orig_turn in enumerate(orig_turns):
        wids = work_ids_by_orig[orig_idx]
        all_failed = all(wid in failed_work_ids for wid in wids)
        if all_failed:
            failed_indices.append(orig_idx)
            polished_turns.append(orig_turn)
            continue

        pieces: list[str] = []
        for wid in wids:
            piece = polished_chunks[wid]
            if batches_ok > 0 and wid in failed_work_ids and chunk_fallback_fn is not None:
                piece = chunk_fallback_fn(piece)
            if piece and piece.strip():
                pieces.append(piece.strip())

        joined_text = " ".join(pieces).strip() or orig_turn.text
        if joined_text != orig_turn.text:
            changed_turns += 1
        turn_cls = type(orig_turn)
        polished_turns.append(
            turn_cls(
                start=orig_turn.start,
                end=orig_turn.end,
                speaker_label=orig_turn.speaker_label,
                text=joined_text,
            )
        )

    elapsed = time.time() - t0
    if status_cb:
        status_cb(
            f"[Profile] LLM-корректура завершена за {elapsed:.2f}s "
            f"(успешных батчей: {batches_ok}/{total_batches}, исправлено реплик: {changed_turns})"
        )
    return polished_turns, {
        "applied": batches_ok > 0,
        "batches_ok": batches_ok,
        "batches_total": total_batches,
        "changed_turns": changed_turns,
        "failed_indices": sorted(failed_indices),
        "elapsed": elapsed,
        "is_cloud": not is_local,
        "auto_glossary_count": len(auto_terms),
        "auto_glossary_terms": auto_terms,
        "split_turns": split_turns_count,
        "chunks_total": len(work_items),
    }
