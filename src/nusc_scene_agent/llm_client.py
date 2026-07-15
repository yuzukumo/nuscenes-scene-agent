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
    digest: str = ""
    resolved_digest: str = ""
    digest_verified: bool = False
    require_digest: bool = False

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
        digest = (os.getenv("NUSC_SCENE_AGENT_OLLAMA_DIGEST") or "").strip()
        require_digest = (os.getenv("NUSC_SCENE_AGENT_OLLAMA_REQUIRE_DIGEST") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(base_url=base_url, model=model, digest=digest, require_digest=require_digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": _ollama_root_url(self.base_url),
            "model": self.model,
            "digest": self.digest,
            "resolved_digest": self.resolved_digest,
            "digest_verified": bool(self.digest_verified),
            "require_digest": bool(self.require_digest),
            "mutable_tag": self.model.endswith(":latest") or ":" not in self.model,
            "timeout_s": float(self.timeout_s),
        }


def _normalized_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _candidate_ollama_urls(base_url: str) -> List[str]:
    root = _normalized_base_url(base_url)
    if root.endswith("/api/chat"):
        return [root]
    return [root + "/api/chat"]


def _ollama_root_url(base_url: str) -> str:
    root = _normalized_base_url(base_url)
    for suffix in ["/api/chat", "/api/show", "/api/tags"]:
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


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


def _get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    req = request.Request(url=url, method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def inspect_ollama_model(config: LLMConfig) -> Dict[str, Any]:
    root = _ollama_root_url(config.base_url)
    show_url = root + "/api/show"
    tags_url = root + "/api/tags"
    show_payload = _post_json(show_url, {"model": config.model}, config.timeout_s)

    tag_record: Dict[str, Any] = {}
    tags_error = ""
    try:
        tags_payload = _get_json(tags_url, config.timeout_s)
        for item in tags_payload.get("models", []):
            if str(item.get("name") or "") == config.model:
                tag_record = dict(item)
                break
    except Exception as exc:  # noqa: BLE001
        tags_error = str(exc)

    resolved_digest = str(tag_record.get("digest") or show_payload.get("digest") or "")
    digest_matches = not config.digest or config.digest == resolved_digest
    config.resolved_digest = resolved_digest
    config.digest_verified = bool(config.digest and digest_matches)
    return {
        "schema": "ollama_model_metadata_v1",
        "base_url": root,
        "model": config.model,
        "show": show_payload,
        "tag": tag_record,
        "digest": resolved_digest,
        "expected_digest": config.digest,
        "digest_matches": digest_matches,
        "model_identifier": resolved_digest or config.model,
        "reproducible": bool(config.digest and digest_matches),
        "tags_error": tags_error,
        "reproducibility_note": (
            "Record the Ollama model digest when available. A mutable tag such as latest is not a stable "
            "bit-level model identifier."
        ),
    }


def verify_ollama_model(config: LLMConfig) -> Dict[str, Any]:
    if config.require_digest and not config.digest:
        raise RuntimeError(
            "A stable Ollama digest is required for this experiment. Run inspect-ollama-model, "
            "then set NUSC_SCENE_AGENT_OLLAMA_DIGEST to the recorded digest."
        )
    metadata = inspect_ollama_model(config)
    if config.digest and not metadata["digest"]:
        raise RuntimeError(
            "Ollama did not report a digest for {0}; expected {1}.".format(config.model, config.digest)
        )
    if config.digest and not metadata["digest_matches"]:
        raise RuntimeError(
            "Ollama model digest mismatch for {0}: expected {1}, resolved {2}.".format(
                config.model,
                config.digest,
                metadata["digest"],
            )
        )
    config.resolved_digest = str(metadata.get("digest") or "")
    config.digest_verified = bool(config.digest and metadata.get("digest_matches"))
    return metadata


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
    if (config.digest or config.require_digest) and not config.digest_verified:
        verify_ollama_model(config)
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
