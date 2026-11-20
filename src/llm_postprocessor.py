import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from src.cancellation import CancellationToken

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_META_LEAK_RE = re.compile(
    r"^(?:исправленный\s+текст|вот\s+исправленный|конечно[,!]|готово[,!]|примечание:)",
    re.IGNORECASE,
)


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


def _build_turn_batches(
    turns: list,
    batch_size: int = 12,
    max_batch_chars: int = 3600,
) -> list[tuple[int, int]]:
    """
    Splits turns into contextual batches bounded by BOTH turn count and character
    budget so long consecutive monologues never overflow LLM output token limits.
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
            f"\nПрофильный словарь / термины (используй именно их каноническое написание, "
            f"если по звучанию и контексту имелся в виду соответствующий термин): {custom_prompt.strip()}\n"
        )

    system_msg = (
        "Ты — профессиональный редактор расшифровок (ASR) интервью.\n"
        "Твоя задача — аккуратно исправить ТОЛЬКО очевидные ошибки распознавания речи:\n"
        "1. Восстанови фонетически искажённые термины, названия инструментов и аббревиатуры, опираясь строго на контекст диалога и переданный профильный словарь (например: «GPC» → «gRPC», «оауф 2» → «OAuth 2»). НЕ подменяй осмысленные обычные слова или валидные термины (например, не превращай реальное «GPT» в «JWT», если речь действительно про GPT/LLM).\n"
        "2. Если ASR «проглотила» короткую аббревиатуру перед дефисом и она однозначно следует из контекста или словаря, аккуратно восстанови её; если контекст неоднозначен — оставь как есть.\n"
        "3. Поправь очевидные омофоны, склейки слов и пунктуацию, строго сохраняя исходный язык каждой реплики "
        "(не переводи английские реплики на русский и русские на английский) и разговорный стиль спикера.\n"
        "4. СТРОГО ЗАПРЕЩЕНО: переписывать реплики от себя, сжимать/сокращать смысл, добавлять пояснения, "
        "объединять или удалять элементы массива.\n"
        "5. Верни ТОЛЬКО валидный JSON-объект вида {\"turns\": [{\"id\": 0, \"text\": \"...\"}, ...]} "
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

    cleaned = str(raw_content).strip()
    # Strip optional reasoning tags (e.g. Qwen3 <think>...</think>) before fence/JSON parsing
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    if "<think>" in cleaned.lower():
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
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
                parsed = json.loads(cand)
                break
            except json.JSONDecodeError:
                continue

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
    max_batch_chars: int = 3600,
) -> tuple[list, dict]:
    """
    Stage 4 optional LLM post-editor (cloud DeepSeek / Qwen DashScope OR local
    Ollama / LM Studio / vLLM OpenAI-compatible API). Polishes ASR text in
    contextual batches while strictly preserving original timestamps, speaker
    labels, and turn structure. Any error safely preserves original turns.
    """
    if not turns:
        return turns, {
            "applied": False,
            "batches_ok": 0,
            "batches_total": 0,
            "changed_turns": 0,
            "failed_indices": [],
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
        }

    is_local = _is_local_endpoint(endpoint)
    is_deepseek = _is_deepseek_endpoint(endpoint)
    mode_tag = "локально" if is_local else "облачный API"

    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    orig_turns = list(turns)
    polished_turns = list(turns)
    batches = _build_turn_batches(
        orig_turns,
        batch_size=batch_size,
        max_batch_chars=max_batch_chars,
    )
    total_batches = len(batches)
    batches_ok = 0
    changed_turns = 0
    done_batches = 0
    failed_indices: set[int] = set()
    use_json_response_format = True
    use_deepseek_thinking_disable = is_deepseek
    state_lock = threading.Lock()
    abort_event = threading.Event()
    cloud_workers = 1 if (is_local or total_batches <= 2) else min(3, total_batches - 1)

    if status_cb:
        mode_extra = ", non-thinking" if use_deepseek_thinking_disable else ""
        conc_extra = f", потоков: {cloud_workers}" if cloud_workers > 1 else ""
        status_cb(
            f"Шаг 4: LLM-корректура ({mode_tag}: {model_name}{mode_extra} @ {endpoint}, "
            f"батчей: {total_batches}{conc_extra})..."
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

    def _run_batch(client: httpx.Client, b_idx: int, start_i: int, end_i: int) -> None:
        nonlocal batches_ok, changed_turns, done_batches
        if cancel_token:
            cancel_token.check()
        if abort_event.is_set():
            with state_lock:
                failed_indices.update(range(start_i, end_i))
            return

        ctx_source = polished_turns if is_local else orig_turns
        batch_spec = [
            (i, orig_turns[i].speaker_label, orig_turns[i].text)
            for i in range(start_i, end_i)
        ]
        ctx_start = max(0, start_i - 2)
        prev_ctx = [
            (ctx_source[i].speaker_label, ctx_source[i].text)
            for i in range(ctx_start, start_i)
        ]

        messages = _build_polish_messages(batch_spec, prev_ctx, custom_prompt=custom_prompt)
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
                turn_cls = type(orig_turns[start_i])
                local_changed = 0
                new_objs: list[tuple[int, object]] = []
                for idx, new_text in validated.items():
                    orig = orig_turns[idx]
                    if new_text != orig.text:
                        local_changed += 1
                    new_objs.append(
                        (
                            idx,
                            turn_cls(
                                start=orig.start,
                                end=orig.end,
                                speaker_label=orig.speaker_label,
                                text=new_text,
                            ),
                        )
                    )
                with state_lock:
                    for idx, obj in new_objs:
                        polished_turns[idx] = obj
                    batches_ok += 1
                    changed_turns += local_changed
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
                failed_indices.update(range(start_i, end_i))
            done_batches += 1
            cur_done = done_batches

        if status_cb and total_batches > 1 and not abort_event.is_set():
            status_cb(f"LLM-корректура: обработано {cur_done}/{total_batches} батчей...")

    t0 = time.time()
    with _create_httpx_client(endpoint, timeout_sec) as client:
        # Always probe batch 0 synchronously first: validates auth/connectivity in 1 request
        # and negotiates response_format / thinking flags cleanly before any fan-out.
        _run_batch(client, 0, batches[0][0], batches[0][1])

        if total_batches > 1 and not abort_event.is_set():
            if cloud_workers <= 1:
                for b_idx in range(1, total_batches):
                    if abort_event.is_set():
                        failed_indices.update(range(batches[b_idx][0], batches[b_idx][1]))
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
                failed_indices.update(range(batches[b_idx][0], batches[b_idx][1]))

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
    }
