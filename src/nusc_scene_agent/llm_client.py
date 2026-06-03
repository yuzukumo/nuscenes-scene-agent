from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request


DEFAULT_TIMEOUT_S = 90.0


@dataclass
class LLMConfig:
    base_url: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls) -> Optional["LLMConfig"]:
        base_url = (
            os.getenv("NUSC_SCENE_AGENT_OLLAMA_BASE_URL")
            or ""
        ).strip()
        model = (
            os.getenv("NUSC_SCENE_AGENT_OLLAMA_MODEL")
            or ""
        ).strip()
        if not (base_url and model):
            return None
        return cls(base_url=base_url, model=model)


def _normalized_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _candidate_ollama_urls(base_url: str) -> List[str]:
    root = _normalized_base_url(base_url)
    if root.endswith("/api/chat"):
        return [root]
    return [root + "/api/chat"]


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    req = request.Request(
        url=url,
        data=body,
        method="POST",
        headers=headers,
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


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


def llm_json(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    options: Dict[str, Any] = {"temperature": float(temperature)}
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": options,
    }

    last_error: Optional[Exception] = None
    for url in _candidate_ollama_urls(config.base_url):
        for _ in range(max_retries):
            try:
                response = _post_json(url, payload, config.timeout_s)
                message = response.get("message")
                text = str(message.get("content") or "") if isinstance(message, dict) else ""
                if not text:
                    text = str(response.get("response") or "")
                if not text.strip():
                    raise ValueError("Ollama response did not include message content.")
                return json.loads(_extract_json_text(text))
            except error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError("HTTP {0} from {1}: {2}".format(exc.code, url, error_text))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
    raise RuntimeError("Ollama request failed: {0}".format(last_error))
