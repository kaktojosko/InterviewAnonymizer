import json
import os
import re
import time
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
        "Ты — профессиональный редактор расшифровок (ASR) IT-интервью.\n"
        "Твоя задача — аккуратно исправить ТОЛЬКО ошибки распознавания речи:\n"
        "1. Восстанови искажённые кириллицей или созвучием английские IT-термины, протоколы и инструменты по контексту "
        "(типичные искажения ASR: «поджирописью»/«GPC» → «по gRPC», «ристы»/«РС»/«РСТ»/«рысту» → «REST», "
        "«Geson» → «JSON», «пользова два.0» → «HTTP/2.0», «SWAP» → «SOAP», «GPT-токены»/«джвт» → «JWT-токены», "
        "«бирутокены» → «Bearer-токены», «ауф два»/«оауф два» → «OAuth 2», «Appi Plues» → «API-шлюз», "
        "«в Комунде» → «в Camunda», «BPen» → «BPMN», «ARD»/«продишки» → «ERD»/«ERD-шки», «Siven»/«сиквенс» → «Sequence», "
        "«План Юля» → «PlantUML», «дру Ио» → «draw.io», «в Джерри» → «в Jira», «UI-гид» → «UI-kit», "
        "«user Kase» → «use cases», «только со Скотой»/«докс эз код» → «Docs as Code», «C+» → «C++»).\n"
        "2. Если ASR «проглотила» короткую английскую аббревиатуру перед русским дефисом и смысл однозначен из контекста, "
        "аккуратно восстанови её (например: «в нашу-систему» → «в нашу CRM-систему», «нашего-специалиста» → «нашего HR-специалиста»).\n"
        "3. Поправь очевидные омофоны («революционных баз» → «реляционных баз», «воспитательную лабораторию» → «испытательную лабораторию»), "
        "склейки и пунктуацию, строго сохраняя исходный язык каждой реплики "
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


def _parse_and_validate_batch_response(
    raw_content: str,
    batch_turns: list[tuple[int, str, str]],
) -> Optional[dict[int, str]]:
    """
    Strictly validates the LLM JSON response so timestamps, speaker labels,
    turn count, and meaning can never be corrupted or hallucinated.
    Returns mapping {turn_id: polished_text} or None to trigger safe fallback.
    """
    if not raw_content or not raw_content.strip():
        return None

    cleaned = raw_content.strip()
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    # Strip optional reasoning tags (e.g. Qwen3 <think>...</think>) before JSON parsing
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting outermost JSON object or array
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
        return None

    items = parsed.get("turns") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list) or len(items) != len(batch_turns):
        return None

    orig_by_id = {idx: txt for idx, _, txt in batch_turns}
    result_map: dict[int, str] = {}

    for item in items:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            return None
        try:
            tid = int(item["id"])
        except (TypeError, ValueError):
            return None
        if tid not in orig_by_id or tid in result_map:
            return None

        new_txt = " ".join(str(item["text"]).split()).strip()
        orig_txt = orig_by_id[tid].strip()
        if not new_txt:
            return None
        if _META_LEAK_RE.match(new_txt):
            return None

        # Guardrail against aggressive summarization or runaway hallucination
        if len(orig_txt) >= 24:
            ratio = len(new_txt) / float(max(1, len(orig_txt)))
            if ratio < 0.55 or ratio > 1.55:
                return None

        result_map[tid] = new_txt

    if len(result_map) != len(batch_turns):
        return None
    return result_map


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
) -> tuple[list, dict]:
    """
    Stage 4 optional LLM post-editor (local Ollama / LM Studio / vLLM OR cloud
    DeepSeek / Qwen DashScope OpenAI-compatible API). Polishes ASR text in
    contextual batches while strictly preserving original timestamps, speaker
    labels, and turn structure. Any error safely preserves original turns.
    """
    if not turns:
        return turns, {"applied": False, "batches_ok": 0, "batches_total": 0, "changed_turns": 0}

    endpoint = _normalize_chat_endpoint(base_url)
    model_name = (model or "").strip()
    if not endpoint or not model_name:
        if status_cb:
            status_cb("LLM-корректура пропущена: не указан URL или имя модели.")
        return turns, {"applied": False, "batches_ok": 0, "batches_total": 0, "changed_turns": 0}

    is_local = _is_local_endpoint(endpoint)
    mode_tag = "локально" if is_local else "облачный API"

    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    eff_batch_size = max(1, min(int(batch_size), 25))
    total_turns = len(turns)
    total_batches = (total_turns + eff_batch_size - 1) // eff_batch_size
    batches_ok = 0
    changed_turns = 0
    polished_turns = list(turns)
    use_json_response_format = True

    if status_cb:
        status_cb(
            f"Шаг 4: LLM-корректура ({mode_tag}: {model_name} @ {endpoint}, батчей: {total_batches})..."
        )

    t0 = time.time()
    with _create_httpx_client(endpoint, timeout_sec) as client:
        for b_idx in range(total_batches):
            if cancel_token:
                cancel_token.check()

            start_i = b_idx * eff_batch_size
            end_i = min(total_turns, start_i + eff_batch_size)
            batch_spec = [
                (i, polished_turns[i].speaker_label, polished_turns[i].text)
                for i in range(start_i, end_i)
            ]
            ctx_start = max(0, start_i - 2)
            prev_ctx = [
                (polished_turns[i].speaker_label, polished_turns[i].text)
                for i in range(ctx_start, start_i)
            ]

            messages = _build_polish_messages(batch_spec, prev_ctx, custom_prompt=custom_prompt)
            batch_chars = sum(len(txt) for _, _, txt in batch_spec)
            body: dict = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.05,
                "top_p": 0.9,
                "max_tokens": min(4096, max(1024, batch_chars * 2)),
                "stream": False,
            }
            if use_json_response_format:
                body["response_format"] = {"type": "json_object"}

            try:
                try:
                    resp = client.post(endpoint, headers=headers, json=body)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as rf_exc:
                    sc = rf_exc.response.status_code if rf_exc.response is not None else 0
                    if use_json_response_format and sc in (400, 422):
                        use_json_response_format = False
                        body.pop("response_format", None)
                        resp = client.post(endpoint, headers=headers, json=body)
                        resp.raise_for_status()
                    else:
                        raise

                data = resp.json()
                raw_content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                validated = _parse_and_validate_batch_response(raw_content, batch_spec)
                if validated is not None:
                    turn_cls = type(polished_turns[start_i])
                    for idx, new_text in validated.items():
                        orig = polished_turns[idx]
                        if new_text != orig.text:
                            changed_turns += 1
                        polished_turns[idx] = turn_cls(
                            start=orig.start,
                            end=orig.end,
                            speaker_label=orig.speaker_label,
                            text=new_text,
                        )
                    batches_ok += 1
                else:
                    if status_cb:
                        status_cb(
                            f"LLM-корректура: батч {b_idx + 1}/{total_batches} не прошёл валидацию — оставлен исходный ASR."
                        )
            except httpx.HTTPStatusError as http_exc:
                status_code = http_exc.response.status_code if http_exc.response is not None else 0
                if status_code in (401, 403):
                    if status_cb:
                        status_cb(
                            f"LLM-корректура: ошибка авторизации (HTTP {status_code}) — проверьте API-ключ; оставлен исходный ASR."
                        )
                    break
                if status_cb:
                    status_cb(
                        f"LLM-корректура: ошибка HTTP {status_code} на батче {b_idx + 1}/{total_batches} — оставлен исходный ASR."
                    )
            except Exception as exc:
                if status_cb:
                    status_cb(
                        f"LLM-корректура: ошибка на батче {b_idx + 1}/{total_batches} ({exc}) — оставлен исходный ASR."
                    )
                # Abort remaining batches immediately if the LLM server is unreachable
                if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                    break

            if status_cb and total_batches > 1:
                status_cb(f"LLM-корректура: обработано {b_idx + 1}/{total_batches} батчей...")

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
        "elapsed": elapsed,
        "is_cloud": not is_local,
    }
