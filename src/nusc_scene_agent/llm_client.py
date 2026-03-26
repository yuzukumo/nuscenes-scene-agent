from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib import error, request


DEFAULT_TIMEOUT_S = 90.0


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls) -> Optional["LLMConfig"]:
        base_url = (
            os.getenv("NUSC_SCENE_AGENT_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or ""
        ).strip()
        api_key = (
            os.getenv("NUSC_SCENE_AGENT_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        model = (
            os.getenv("NUSC_SCENE_AGENT_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or ""
        ).strip()
        if not (base_url and api_key and model):
            return None
        return cls(base_url=base_url, api_key=api_key, model=model)


def _normalized_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _candidate_response_urls(base_url: str) -> List[str]:
    root = _normalized_base_url(base_url)
    if root.endswith("/v1"):
        return [root + "/responses"]
    return [root + "/v1/responses", root + "/responses"]


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout_s: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer {0}".format(api_key),
        },
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _extract_output_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = payload.get("output") or []
    for item in outputs:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content") or []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and str(part.get("text") or "").strip():
                return str(part.get("text")).strip()

    raise ValueError("Responses API payload did not include output_text.")


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1].strip()
    return stripped


def _extract_json_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    texts: List[str] = []
    try:
        texts.append(_extract_output_text(payload))
    except Exception:  # noqa: BLE001
        pass

    outputs = payload.get("output") or []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and str(part.get("text") or "").strip():
                texts.append(str(part.get("text")).strip())

    last_error: Optional[Exception] = None
    for text in texts:
        try:
            return json.loads(_extract_json_text(text))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError("No JSON object could be extracted from Responses payload: {0}".format(last_error))


def responses_json(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    max_retries: int = 3,
) -> Dict[str, Any]:
    input_parts: List[Dict[str, Any]] = []
    if system_prompt.strip():
        input_parts.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        )
    input_parts.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        }
    )

    payload: Dict[str, Any] = {
        "model": config.model,
        "input": input_parts,
        "temperature": float(temperature),
        "reasoning": {"effort": reasoning_effort},
    }
    if max_tokens is not None:
        payload["max_output_tokens"] = int(max_tokens)

    last_error: Optional[Exception] = None
    for url in _candidate_response_urls(config.base_url):
        for _ in range(max_retries):
            try:
                response = _post_json(url, payload, config.api_key, config.timeout_s)
                return _extract_json_payload(response)
            except error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError("HTTP {0} from {1}: {2}".format(exc.code, url, error_text))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
    raise RuntimeError("LLM request failed: {0}".format(last_error))
