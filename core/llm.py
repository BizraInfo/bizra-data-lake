from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from core.model_family import ModelFamily, PinnedArtifact, llm_endpoints
from urllib.parse import urlparse


class LLMCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMCompletion:
    provider: str
    model_name: str
    model_identifier: str
    latency_ms: float
    text: str
    raw: Dict[str, Any]


def _join_url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    path = (path or "").lstrip("/")
    return f"{base}/{path}"


def _ensure_lmstudio_v1(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        return base_url
    try:
        parsed = urlparse(base_url)
        current_path = (parsed.path or "").rstrip("/")
        if current_path.endswith("/v1") or current_path == "v1":
            return base_url
        new_path = (current_path + "/v1") if current_path else "/v1"
        return parsed._replace(path=new_path).geturl().rstrip("/")
    except Exception:
        if base_url.endswith("/v1"):
            return base_url
        return base_url + "/v1"


def _model_identifier(artifact: PinnedArtifact) -> str:
    if artifact.provider == "lmstudio":
        return artifact.model_id or artifact.name
    return artifact.name


def _timeout_s(model_family: ModelFamily, provider: str) -> float:
    raw = model_family.raw.get("resource_policy") if isinstance(model_family.raw, dict) else None
    if isinstance(raw, dict):
        provider_cfg = raw.get(provider)
        if isinstance(provider_cfg, dict):
            val = provider_cfg.get("request_timeout_s")
            try:
                if val is not None:
                    return float(val)
            except Exception:
                pass
        global_cfg = raw.get("global")
        if isinstance(global_cfg, dict):
            val2 = global_cfg.get("request_timeout_s")
            try:
                if val2 is not None:
                    return float(val2)
            except Exception:
                pass
    return 30.0


def _ollama_options(artifact: PinnedArtifact) -> Dict[str, Any]:
    params = artifact.raw.get("params") if isinstance(artifact.raw, dict) else None
    if not isinstance(params, dict):
        return {}
    out: Dict[str, Any] = {}
    if "temperature" in params:
        try:
            out["temperature"] = float(params["temperature"])
        except Exception:
            pass
    if "num_ctx" in params:
        try:
            out["num_ctx"] = int(params["num_ctx"])
        except Exception:
            pass
    return out


def _lmstudio_params(artifact: PinnedArtifact) -> Dict[str, Any]:
    params = artifact.raw.get("params") if isinstance(artifact.raw, dict) else None
    if not isinstance(params, dict):
        return {}
    out: Dict[str, Any] = {}
    if "temperature" in params:
        try:
            out["temperature"] = float(params["temperature"])
        except Exception:
            pass
    return out


async def ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_s: float,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = _join_url(base_url, "/api/chat")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        raise LLMCallError(f"ollama_http_{resp.status_code}: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception as e:
        raise LLMCallError(f"ollama_invalid_json: {e}") from e


async def lmstudio_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_s: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_url = _ensure_lmstudio_v1(base_url)
    url = _join_url(base_url, "/chat/completions")
    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if extra:
        payload.update(extra)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        raise LLMCallError(f"lmstudio_http_{resp.status_code}: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception as e:
        raise LLMCallError(f"lmstudio_invalid_json: {e}") from e


def _extract_text(provider: str, payload: Dict[str, Any]) -> str:
    if provider == "ollama":
        msg = payload.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
        resp = payload.get("response")
        if isinstance(resp, str):
            return resp
        return json.dumps(payload, ensure_ascii=False)[:2000]

    # lmstudio (openai compat)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    return json.dumps(payload, ensure_ascii=False)[:2000]


async def chat_with_routing(
    *,
    model_family: ModelFamily,
    slot: str,
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 3,
) -> Tuple[LLMCompletion, List[Dict[str, Any]]]:
    models = model_family.route_models(slot)
    if not models:
        raise LLMCallError(f"no_models_for_slot: {slot}")

    ollama_base, lmstudio_base = llm_endpoints()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    attempts: List[Dict[str, Any]] = []
    last_error: Optional[str] = None

    for model_name in models[: max(1, int(max_attempts))]:
        artifact = model_family.artifact(model_name)
        provider = artifact.provider
        model_identifier = _model_identifier(artifact)
        timeout_s = _timeout_s(model_family, provider)
        t0 = time.monotonic()

        try:
            if provider == "ollama":
                raw = await ollama_chat(
                    base_url=ollama_base,
                    model=model_identifier,
                    messages=messages,
                    timeout_s=timeout_s,
                    options=_ollama_options(artifact),
                )
            elif provider == "lmstudio":
                raw = await lmstudio_chat(
                    base_url=lmstudio_base,
                    model=model_identifier,
                    messages=messages,
                    timeout_s=timeout_s,
                    extra=_lmstudio_params(artifact),
                )
            else:
                raise LLMCallError(f"unsupported_provider: {provider}")

            latency_ms = (time.monotonic() - t0) * 1000.0
            text = _extract_text(provider, raw)
            attempts.append(
                {
                    "model": model_name,
                    "provider": provider,
                    "status": "ok",
                    "latency_ms": round(latency_ms, 2),
                }
            )
            return (
                LLMCompletion(
                    provider=provider,
                    model_name=model_name,
                    model_identifier=model_identifier,
                    latency_ms=latency_ms,
                    text=text,
                    raw=raw,
                ),
                attempts,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000.0
            last_error = str(e)
            attempts.append(
                {
                    "model": model_name,
                    "provider": provider,
                    "status": "error",
                    "latency_ms": round(latency_ms, 2),
                    "error": last_error,
                }
            )

    raise LLMCallError(f"all_models_failed_for_slot={slot}: {last_error}")
