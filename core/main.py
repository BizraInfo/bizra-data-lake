from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

from core.fate import FateEngine, FateSeal, get_fate_engine
from core.llm import LLMCallError, chat_with_routing
from core.model_family import ModelFamily, load_model_family
from core.sape import SapeExecuteRequest, SapeExecuteResponse, SapePlanRequest, SapePlanResponse, compile_sape_plan, sha256_text
from core.wisdom import HouseOfWisdom
from tools.ecosystem.config import load_ecosystem_config
from tools.ecosystem.indexer import build_manifest
from tools.ecosystem.sealer import seal_manifest, write_manifest
from core.meta_prompt.models import MetaPromptRequest, MetaPromptResponse
from core.meta_prompt.engine import MetaPromptEngine


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_snippet(value: Any, *, limit: int = 8000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…[truncated]"


_HEALTHZ_CACHE: Optional[tuple[float, Dict[str, Any], int]] = None
_HEALTHZ_LOCK: Optional[asyncio.Lock] = None


def _healthz_lock() -> asyncio.Lock:
    global _HEALTHZ_LOCK
    if _HEALTHZ_LOCK is None:
        _HEALTHZ_LOCK = asyncio.Lock()
    return _HEALTHZ_LOCK


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_name(primary: str, alias: str, default: str = "") -> str:
    return (os.getenv(primary) or os.getenv(alias) or default).strip()


def _parse_neo4j_auth(raw: str) -> Optional[tuple[str, str]]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "none":
        return None
    if "/" in raw:
        user, password = raw.split("/", 1)
    elif ":" in raw:
        user, password = raw.split(":", 1)
    else:
        return None
    user = user.strip()
    password = password.strip()
    if not user or not password:
        return None
    return user, password


def _sanitize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        p = urlparse(raw)
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        scheme = p.scheme or ""
        return f"{scheme}://{host}{port}"
    except Exception:
        return raw


def _redis_ping_sync(url: str, timeout_s: float) -> None:
    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    port = int(p.port or 6379)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(b"*1\r\n$4\r\nPING\r\n")
        data = sock.recv(1024)
    if b"PONG" not in data:
        raise RuntimeError(f"unexpected_redis_response: {data[:64]!r}")


def _tcp_connect_sync(url: str, timeout_s: float) -> None:
    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    if p.port:
        port = int(p.port)
    else:
        if (p.scheme or "").lower() == "https":
            port = 443
        else:
            port = 80
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(b"")  # connection success is the signal


def _neo4j_verify_sync(*, uri: str, auth: tuple[str, str], timeout_s: float) -> None:
    from neo4j import GraphDatabase  # type: ignore

    try:
        driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=timeout_s)
    except TypeError:
        driver = GraphDatabase.driver(uri, auth=auth)
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


REQUESTS_TOTAL = Counter("bizra_kernel_requests_total", "Total requests", ["path", "method", "status"])
REQUEST_LATENCY = Histogram("bizra_kernel_request_latency_seconds", "Request latency", ["path", "method"])
FATE_VERDICTS = Counter("bizra_kernel_fate_verdict_total", "FATE verdict totals", ["verdict"])
GRAPH_QUERIES = Counter("bizra_kernel_graph_queries_total", "Graph query totals", ["status"])
GRAPH_QUERY_LATENCY = Histogram("bizra_kernel_graph_query_latency_seconds", "Graph query latency")

SAPE_REQUESTS = Counter("bizra_kernel_sape_requests_total", "Total SAPE requests", ["endpoint", "status"])
SAPE_LLM_CALLS = Counter("bizra_kernel_sape_llm_calls_total", "Total SAPE LLM calls", ["provider", "model", "status"])
SAPE_LLM_LATENCY = Histogram("bizra_kernel_sape_llm_latency_seconds", "SAPE LLM call latency", ["provider", "model"])


def _require_token(expected: str, token: str) -> None:
    if not expected:
        raise HTTPException(status_code=503, detail="kernel misconfigured: BIZRA_API_TOKEN not set")
    if token != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _extract_token(authorization: Optional[str], x_bizra_token: Optional[str]) -> str:
    if x_bizra_token and x_bizra_token.strip():
        return x_bizra_token.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def verify_token(
    authorization: Optional[str] = Header(default=None),
    x_bizra_token: Optional[str] = Header(default=None),
) -> str:
    expected = os.getenv("BIZRA_API_TOKEN", "").strip()
    token = _extract_token(authorization, x_bizra_token)
    _require_token(expected, token)
    return token


def _receipt_dir() -> Path:
    configured = os.getenv("BIZRA_KERNEL_RECEIPT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "docs" / "evidence" / "receipts").resolve()


def _sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _write_receipt(payload: Dict[str, Any]) -> Optional[Path]:
    if os.getenv("BIZRA_KERNEL_RECEIPTS", "1").strip().lower() in {"0", "false", "no"}:
        return None
    base = _receipt_dir()
    base.mkdir(parents=True, exist_ok=True)
    folder = base / f"kernel_request_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_{payload['request_id']}"
    folder.mkdir(parents=True, exist_ok=True)

    # Optional evidence artifact emission (keeps receipts small + auditable).
    # If payload already contains evidence hashes/paths, we still write evidence.json to provide
    # a stable evidence bundle with its own sha256.
    try:
        evidence = payload.get("evidence")
        if isinstance(evidence, list) and evidence:
            evidence_bytes = _canonical_json_bytes(evidence)
            evidence_sha = _sha256_hex_bytes(evidence_bytes)
            (folder / "evidence.json").write_bytes(evidence_bytes + b"\n")
            payload.setdefault(
                "evidence_artifact",
                {
                    "file": "evidence.json",
                    "sha256": f"sha256:{evidence_sha}",
                    "count": len(evidence),
                },
            )
    except Exception:
        # Evidence artifacts are best-effort; receipt write must not fail because of them.
        pass

    # Deterministic integrity hash (self-sealing receipts).
    # Only set if not already present so callers can override.
    if not payload.get("integrity_hash"):
        canonical = _canonical_json_bytes({k: v for k, v in payload.items() if k != "integrity_hash"})
        payload["integrity_hash"] = f"sha256:{_sha256_hex_bytes(canonical)}"

    out = folder / "receipt.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


app = FastAPI(
    title="BIZRA Node0 Sovereign Kernel",
    version="1.0.0",
    description="Token-gated API gateway for the House of Wisdom (Neo4j), enforcing Ihsan/Adl/Amanah via FATE.",
    docs_url="/docs",
    redoc_url=None,
)

cors_origins = [o.strip() for o in os.getenv("BIZRA_CORS_ORIGINS", "http://localhost,http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-BIZRA-TOKEN", "Content-Type"],
)

fate_level = _env_name("BIZRA_FATE_STRICT", "FATE_LEVEL", "1")
fate = FateEngine(strict_mode=fate_level.strip() != "0" and fate_level.strip().lower() not in {"lenient", "off"})
wisdom = HouseOfWisdom()

_MODEL_FAMILY: Optional[ModelFamily] = None
_MODEL_FAMILY_ERROR: Optional[str] = None


def _get_model_family() -> ModelFamily:
    global _MODEL_FAMILY, _MODEL_FAMILY_ERROR
    if _MODEL_FAMILY is not None:
        return _MODEL_FAMILY
    try:
        _MODEL_FAMILY = load_model_family()
        _MODEL_FAMILY_ERROR = None
        return _MODEL_FAMILY
    except Exception as e:
        _MODEL_FAMILY_ERROR = str(e)
        raise HTTPException(status_code=503, detail=f"model_family_unavailable: {_MODEL_FAMILY_ERROR}") from e


def _choose_sape_slot(req: SapePlanRequest, mf: ModelFamily) -> str:
    if req.slot and req.slot.strip():
        return req.slot.strip()
    # High-stakes default: deterministic cold core
    if req.stakes == "H" and "cold_core" in mf.capability_slots:
        return "cold_core"
    # Otherwise prefer primary_reasoning if present
    if "primary_reasoning" in mf.capability_slots:
        return "primary_reasoning"
    # Fall back to any available slot (stable order)
    return sorted(mf.capability_slots.keys())[0]


def _candidate_models(mf: ModelFamily, slot: str) -> List[Dict[str, Any]]:
    models = mf.route_models(slot)
    out: List[Dict[str, Any]] = []
    ollama_base = (os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    lmstudio_base = (os.getenv("LMSTUDIO_URL") or os.getenv("BIZRA_LMSTUDIO_URL") or "http://127.0.0.1:1234/v1").strip()
    for name in models:
        a = mf.artifact(name)
        out.append(
            {
                "name": name,
                "provider": a.provider,
                "digest": a.digest,
                "modelfile_sha256": a.modelfile_sha256,
                "model_id": a.model_id,
                "endpoint_base_url": ollama_base if a.provider == "ollama" else lmstudio_base,
            }
        )
    return out


def _collect_graph_evidence(req: SapePlanRequest) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    evidence: List[Dict[str, Any]] = []
    topics = [t for t in (req.evidence_topics or []) if isinstance(t, str) and t.strip()]
    if not topics:
        topics = [req.domain]

    seen: set[str] = set()
    for topic in topics:
        try:
            rows = wisdom.query_knowledge(topic=topic, limit=req.evidence_limit)
        except Exception as e:
            warnings.append(f"graph_evidence_unavailable: {e}")
            return [], warnings
        for r in rows:
            h = str(r.get("hash") or "")
            if h and h in seen:
                continue
            if h:
                seen.add(h)
            evidence.append(r)
            if len(evidence) >= req.evidence_limit:
                return evidence, warnings

    if not evidence:
        warnings.append("graph_evidence_empty: no_matching_artifacts")
    return evidence, warnings


@app.middleware("http")
async def metrics_middleware(request, call_next):
    path = request.url.path
    method = request.method
    with REQUEST_LATENCY.labels(path=path, method=method).time():
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            REQUESTS_TOTAL.labels(path=path, method=method, status=status if "status" in locals() else "500").inc()


class AgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    intent: str = Field(..., min_length=1)
    context: str = ""
    limit: int = 10


class KernelResponse(BaseModel):
    status: str
    seal: FateSeal
    data: Dict[str, Any]
    processing_time_ms: float
    request_id: str


@app.get("/healthz")
async def healthz():
    budget_s = max(0.2, min(_env_float("BIZRA_HEALTHZ_BUDGET_S", 0.85), 1.0))
    redis_timeout_s = max(0.05, min(_env_float("BIZRA_HEALTHZ_REDIS_TIMEOUT_S", 0.35), budget_s))
    neo4j_timeout_s = max(0.05, min(_env_float("BIZRA_HEALTHZ_NEO4J_TIMEOUT_S", 0.55), budget_s))
    llm_timeout_s = max(0.05, min(_env_float("BIZRA_HEALTHZ_LLM_TIMEOUT_S", 0.25), budget_s))
    cache_ttl_s = max(0.0, min(_env_float("BIZRA_HEALTHZ_CACHE_TTL_S", 0.5), 5.0))

    global _HEALTHZ_CACHE
    t0 = time.monotonic()
    if cache_ttl_s > 0 and _HEALTHZ_CACHE and (t0 - _HEALTHZ_CACHE[0]) <= cache_ttl_s:
        cached_body, cached_status = _HEALTHZ_CACHE[1], _HEALTHZ_CACHE[2]
        body = dict(cached_body)
        body["cached"] = True
        body["cache_age_ms"] = round((t0 - _HEALTHZ_CACHE[0]) * 1000.0, 2)
        body["time"] = utc_now_iso()
        return JSONResponse(content=body, status_code=cached_status)

    async def _run() -> tuple[Dict[str, Any], int]:
        checks: Dict[str, Any] = {}
        healthy = True

        token_set = bool(os.getenv("BIZRA_API_TOKEN", "").strip())
        checks["kernel_config"] = {
            "ok": token_set,
            "detail": "BIZRA_API_TOKEN is set" if token_set else "BIZRA_API_TOKEN missing",
        }
        if not token_set:
            healthy = False

        fate_strict = fate.strict_mode
        fate_ok = (not fate_strict) or fate.policy.loaded
        checks["fate"] = {
            "ok": fate_ok,
            "strict_mode": fate_strict,
            "policy_loaded": fate.policy.loaded,
            "constitution_sha256": fate.policy.constitution_sha256,
        }
        if not fate_ok:
            healthy = False

        try:
            mf = _get_model_family()
            checks["model_family"] = {
                "ok": True,
                "sealed": mf.sealed,
                "sealed_at_utc": mf.sealed_at_utc,
                "manifest_path": str(mf.path),
                "slots": sorted(mf.capability_slots.keys()),
            }
        except Exception as e:
            checks["model_family"] = {"ok": False, "error": str(e)}
            healthy = False

        synapse_url = os.getenv("SYNAPSE_URL", "").strip()
        wisdom_url = os.getenv("WISDOM_URL", "").strip()
        neo4j_auth_raw = os.getenv("NEO4J_AUTH", "").strip()
        ollama_cfg = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL") or os.getenv("BIZRA_OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "").strip()
        lmstudio_cfg = (os.getenv("LMSTUDIO_BASE_URL") or os.getenv("LMSTUDIO_URL") or os.getenv("BIZRA_LMSTUDIO_URL") or "").strip()

        def _llm_provider_status(provider: str, *, enabled: bool, details: Dict[str, Any]) -> Dict[str, Any]:
            if not enabled:
                return {"enabled": False, "status": "disabled"}
            raw = details.get(provider)
            if not isinstance(raw, dict):
                return {"enabled": True, "ok": False, "status": "degraded", "error": "missing_result"}
            ok = bool(raw.get("ok"))
            payload: Dict[str, Any] = {
                "enabled": True,
                "ok": ok,
                "status": "ok" if ok else "degraded",
            }
            if raw.get("url"):
                payload["url"] = raw.get("url")
            if raw.get("latency_ms") is not None:
                payload["latency_ms"] = raw.get("latency_ms")
            if not ok and raw.get("error"):
                payload["error"] = raw.get("error")
            return payload

        # Explicit per-provider LLM checks for observability (DEGRADED is allowed for readiness).
        checks["ollama"] = {"enabled": False, "status": "disabled"}
        checks["lmstudio"] = {"enabled": False, "status": "disabled"}

        async def _llm_check() -> Dict[str, Any]:
            if not ollama_cfg and not lmstudio_cfg:
                return {"enabled": False}

            async def _tcp(url: str) -> Dict[str, Any]:
                start = time.monotonic()
                try:
                    timeout = min(llm_timeout_s, max(0.05, budget_s - (time.monotonic() - t0)))
                    await asyncio.wait_for(asyncio.to_thread(_tcp_connect_sync, url, timeout), timeout=timeout)
                    return {"ok": True, "url": _sanitize_url(url), "latency_ms": round((time.monotonic() - start) * 1000.0, 2)}
                except Exception as e:
                    return {
                        "ok": False,
                        "url": _sanitize_url(url),
                        "latency_ms": round((time.monotonic() - start) * 1000.0, 2),
                        "error": str(e),
                    }

            tasks_llm = []
            names_llm = []
            if ollama_cfg:
                tasks_llm.append(asyncio.create_task(_tcp(ollama_cfg)))
                names_llm.append("ollama")
            if lmstudio_cfg:
                tasks_llm.append(asyncio.create_task(_tcp(lmstudio_cfg)))
                names_llm.append("lmstudio")

            results = await asyncio.gather(*tasks_llm, return_exceptions=False) if tasks_llm else []
            details = {k: v for k, v in zip(names_llm, results, strict=True)}
            all_ok = all(v.get("ok") for v in details.values()) if details else True
            return {
                "enabled": True,
                "ok": all_ok,
                "status": "OK" if all_ok else "DEGRADED",
                "details": details,
            }

        async def _redis_check() -> Dict[str, Any]:
            start = time.monotonic()
            try:
                timeout = min(redis_timeout_s, max(0.05, budget_s - (time.monotonic() - t0)))
                await asyncio.wait_for(asyncio.to_thread(_redis_ping_sync, synapse_url, timeout), timeout=timeout)
                return {"enabled": True, "ok": True, "url": _sanitize_url(synapse_url), "latency_ms": round((time.monotonic() - start) * 1000.0, 2)}
            except Exception as e:
                return {
                    "enabled": True,
                    "ok": False,
                    "url": _sanitize_url(synapse_url),
                    "latency_ms": round((time.monotonic() - start) * 1000.0, 2),
                    "error": str(e),
                }

        async def _neo4j_check() -> Dict[str, Any]:
            start = time.monotonic()
            auth = _parse_neo4j_auth(neo4j_auth_raw)
            if not auth:
                return {"enabled": True, "ok": False, "url": _sanitize_url(wisdom_url), "error": "invalid NEO4J_AUTH (expected user/pass)"}
            try:
                timeout = min(neo4j_timeout_s, max(0.05, budget_s - (time.monotonic() - t0)))
                await asyncio.wait_for(
                    asyncio.to_thread(_neo4j_verify_sync, uri=wisdom_url, auth=auth, timeout_s=timeout),
                    timeout=timeout,
                )
                return {"enabled": True, "ok": True, "url": _sanitize_url(wisdom_url), "latency_ms": round((time.monotonic() - start) * 1000.0, 2)}
            except Exception as e:
                return {
                    "enabled": True,
                    "ok": False,
                    "url": _sanitize_url(wisdom_url),
                    "latency_ms": round((time.monotonic() - start) * 1000.0, 2),
                    "error": str(e),
                }

        tasks = []
        task_names = []
        if synapse_url:
            tasks.append(asyncio.create_task(_redis_check()))
            task_names.append("synapse")
        else:
            checks["synapse"] = {"enabled": False}

        if wisdom_url and neo4j_auth_raw and neo4j_auth_raw.lower() != "none":
            tasks.append(asyncio.create_task(_neo4j_check()))
            task_names.append("wisdom")
        else:
            checks["wisdom"] = {"enabled": False}

        if ollama_cfg or lmstudio_cfg:
            tasks.append(asyncio.create_task(_llm_check()))
            task_names.append("llm")
        else:
            checks["llm"] = {"enabled": False}

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for name, result in zip(task_names, results, strict=True):
                checks[name] = result
                if result.get("enabled") and not result.get("ok"):
                    # LLM reachability is DEGRADE, not a readiness failure.
                    if name in {"synapse", "wisdom"}:
                        healthy = False

        llm_check = checks.get("llm") or {}
        llm_details = llm_check.get("details") if isinstance(llm_check, dict) else None
        if not isinstance(llm_details, dict):
            llm_details = {}
        checks["ollama"] = _llm_provider_status("ollama", enabled=bool(ollama_cfg), details=llm_details)
        checks["lmstudio"] = _llm_provider_status("lmstudio", enabled=bool(lmstudio_cfg), details=llm_details)

        # If high-stakes SAPE is configured to require Neo4j evidence, readiness must fail when Neo4j is down.
        require_neo4j_for_h = _env_bool("BIZRA_SAPE_REQUIRE_NEO4J_EVIDENCE_H", True)
        checks["sape_policy"] = {"require_neo4j_evidence_for_H": require_neo4j_for_h}
        if require_neo4j_for_h:
            wisdom_check = checks.get("wisdom") or {}
            if wisdom_check.get("enabled") and not wisdom_check.get("ok"):
                healthy = False

        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
        body = {
            "status": "ok" if healthy else "unhealthy",
            "time": utc_now_iso(),
            "elapsed_ms": elapsed_ms,
            "budget_ms": round(budget_s * 1000.0, 2),
            "checks": checks,
        }
        return body, (200 if healthy else 503)

    lock = _healthz_lock()
    remaining = max(0.01, budget_s - (time.monotonic() - t0))
    try:
        await asyncio.wait_for(lock.acquire(), timeout=remaining)
    except asyncio.TimeoutError:
        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
        return JSONResponse(
            content={
                "status": "unhealthy",
                "time": utc_now_iso(),
                "elapsed_ms": elapsed_ms,
                "budget_ms": round(budget_s * 1000.0, 2),
                "error": "healthz_lock_timeout",
            },
            status_code=503,
        )

    try:
        now2 = time.monotonic()
        if cache_ttl_s > 0 and _HEALTHZ_CACHE and (now2 - _HEALTHZ_CACHE[0]) <= cache_ttl_s:
            cached_body, cached_status = _HEALTHZ_CACHE[1], _HEALTHZ_CACHE[2]
            body = dict(cached_body)
            body["cached"] = True
            body["cache_age_ms"] = round((now2 - _HEALTHZ_CACHE[0]) * 1000.0, 2)
            body["time"] = utc_now_iso()
            return JSONResponse(content=body, status_code=cached_status)

        try:
            remaining = max(0.01, budget_s - (time.monotonic() - t0))
            body, status_code = await asyncio.wait_for(_run(), timeout=remaining)
        except asyncio.TimeoutError:
            elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
            body = {
                "status": "unhealthy",
                "time": utc_now_iso(),
                "elapsed_ms": elapsed_ms,
                "budget_ms": round(budget_s * 1000.0, 2),
                "error": "healthz_timeout",
            }
            status_code = 503

        if cache_ttl_s > 0:
            _HEALTHZ_CACHE = (time.monotonic(), body, status_code)
        return JSONResponse(content=body, status_code=status_code)
    finally:
        lock.release()


@app.get("/livez")
async def livez():
    return {"status": "live", "time": utc_now_iso()}


@app.get("/", dependencies=[Depends(verify_token)])
async def system_heartbeat():
    mf_status: Dict[str, Any]
    try:
        mf = _get_model_family()
        mf_status = {
            "status": "ONLINE",
            "sealed": mf.sealed,
            "sealed_at_utc": mf.sealed_at_utc,
            "manifest_path": str(mf.path),
            "slots": sorted(mf.capability_slots.keys()),
        }
    except Exception as e:
        mf_status = {"status": "OFFLINE", "error": str(e)}

    return {
        "system": "BIZRA Node0",
        "state": "OPERATIONAL",
        "env": _env_name("BIZRA_ENV", "ENV", "development"),
        "ethics_engine": "ACTIVE" if fate.policy.loaded else "DEGRADED",
        "knowledge_graph": wisdom.get_stats(),
        "model_family": mf_status,
    }


@app.get("/metrics", dependencies=[Depends(verify_token)])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/agent/query", response_model=KernelResponse, dependencies=[Depends(verify_token)])
async def agent_query_knowledge(request: AgentRequest):
    start = time.monotonic()
    request_id = uuid.uuid4().hex

    seal = fate.audit_request(intent=request.intent, context=request.context)
    FATE_VERDICTS.labels(verdict=seal.verdict).inc()

    if seal.verdict == "REJECTED":
        payload = {
            "schema": "bizra_kernel_request_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "endpoint": "/v1/agent/query",
            "agent_id": request.agent_id,
            "status": "BLOCKED_BY_FATE",
            "fate_seal": seal.model_dump(),
        }
        _write_receipt(payload)
        return KernelResponse(
            status="BLOCKED_BY_FATE",
            seal=seal,
            data={"results": [], "count": 0},
            processing_time_ms=(time.monotonic() - start) * 1000.0,
            request_id=request_id,
        )

    try:
        t0 = time.monotonic()
        results = wisdom.query_knowledge(topic=request.intent, limit=request.limit)
        GRAPH_QUERY_LATENCY.observe(time.monotonic() - t0)
        GRAPH_QUERIES.labels(status="ok").inc()
    except Exception as e:
        GRAPH_QUERIES.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=f"knowledge_graph_offline: {e}") from e

    payload = {
        "schema": "bizra_kernel_request_receipt_v1",
        "generated_at": utc_now_iso(),
        "truth_label": "MEASURED",
        "request_id": request_id,
        "endpoint": "/v1/agent/query",
        "agent_id": request.agent_id,
        "status": "SUCCESS",
        "fate_seal": seal.model_dump(),
        "result_count": len(results),
    }
    _write_receipt(payload)

    return KernelResponse(
        status="SUCCESS",
        seal=seal,
        data={"results": results, "count": len(results)},
        processing_time_ms=(time.monotonic() - start) * 1000.0,
        request_id=request_id,
    )


@app.post("/v1/sape/plan", response_model=SapePlanResponse, dependencies=[Depends(verify_token)])
async def sape_plan(request: SapePlanRequest):
    start = time.monotonic()
    request_id = uuid.uuid4().hex
    plan_id = uuid.uuid4().hex

    seal = fate.audit_request(intent=request.objective, context=f"{request.domain}\n{request.constraints}".strip())
    FATE_VERDICTS.labels(verdict=seal.verdict).inc()

    if seal.verdict == "REJECTED":
        SAPE_REQUESTS.labels(endpoint="/v1/sape/plan", status="BLOCKED_BY_FATE").inc()
        payload = {
            "schema": "bizra_sape_plan_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "plan_id": plan_id,
            "endpoint": "/v1/sape/plan",
            "status": "BLOCKED_BY_FATE",
            "fate_seal": seal.model_dump(),
        }
        _write_receipt(payload)
        return SapePlanResponse(
            status="BLOCKED_BY_FATE",
            seal=seal,
            plan_id=plan_id,
            generated_at=utc_now_iso(),
            slot=request.slot or "",
            candidate_models=[],
            system_prompt="",
            user_prompt="",
            prompt_sha256=sha256_text(""),
            evidence=[],
            warnings=["blocked_by_fate"],
            request_id=request_id,
        )

    mf = _get_model_family()
    slot = _choose_sape_slot(request, mf)

    evidence: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if request.require_graph_evidence:
        evidence, warnings = _collect_graph_evidence(request)
        if request.stakes == "H" and not evidence:
            SAPE_REQUESTS.labels(endpoint="/v1/sape/plan", status="BLOCKED_BY_EVIDENCE").inc()
            payload = {
                "schema": "bizra_sape_plan_receipt_v1",
                "generated_at": utc_now_iso(),
                "truth_label": "MEASURED",
                "request_id": request_id,
                "plan_id": plan_id,
                "endpoint": "/v1/sape/plan",
                "status": "BLOCKED_BY_EVIDENCE",
                "slot": slot,
                "fate_seal": seal.model_dump(),
                "warnings": warnings,
            }
            _write_receipt(payload)
            return SapePlanResponse(
                status="BLOCKED_BY_EVIDENCE",
                seal=seal,
                plan_id=plan_id,
                generated_at=utc_now_iso(),
                slot=slot,
                candidate_models=_candidate_models(mf, slot),
                system_prompt="",
                user_prompt="",
                prompt_sha256=sha256_text(""),
                evidence=[],
                warnings=warnings,
                request_id=request_id,
            )

    compiled = compile_sape_plan(request, evidence=evidence)
    SAPE_REQUESTS.labels(endpoint="/v1/sape/plan", status="PLANNED").inc()

    payload = {
        "schema": "bizra_sape_plan_receipt_v1",
        "generated_at": utc_now_iso(),
        "truth_label": "MEASURED",
        "request_id": request_id,
        "plan_id": plan_id,
        "endpoint": "/v1/sape/plan",
        "status": "PLANNED",
        "slot": slot,
        "candidate_models": _candidate_models(mf, slot),
        "prompt_sha256": compiled.prompt_sha256,
        "graph_evidence_count": len(evidence),
        "warnings": compiled.warnings + warnings,
        "fate_seal": seal.model_dump(),
    }
    if os.getenv("BIZRA_KERNEL_RECEIPTS_INCLUDE_PROMPTS", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        payload["system_prompt"] = compiled.system_prompt
        payload["user_prompt"] = compiled.user_prompt
    _write_receipt(payload)

    return SapePlanResponse(
        status="PLANNED",
        seal=seal,
        plan_id=plan_id,
        generated_at=utc_now_iso(),
        slot=slot,
        candidate_models=_candidate_models(mf, slot),
        system_prompt=compiled.system_prompt,
        user_prompt=compiled.user_prompt,
        prompt_sha256=compiled.prompt_sha256,
        evidence=evidence,
        warnings=compiled.warnings + warnings,
        request_id=request_id,
    )


@app.post("/v1/sape/execute", response_model=SapeExecuteResponse, dependencies=[Depends(verify_token)])
async def sape_execute(request: SapeExecuteRequest):
    start = time.monotonic()
    request_id = uuid.uuid4().hex
    plan_id = uuid.uuid4().hex

    seal = fate.audit_request(intent=request.objective, context=f"{request.domain}\n{request.constraints}".strip())
    FATE_VERDICTS.labels(verdict=seal.verdict).inc()

    if seal.verdict == "REJECTED":
        SAPE_REQUESTS.labels(endpoint="/v1/sape/execute", status="BLOCKED_BY_FATE").inc()
        payload = {
            "schema": "bizra_sape_execute_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "plan_id": plan_id,
            "endpoint": "/v1/sape/execute",
            "status": "BLOCKED_BY_FATE",
            "fate_seal": seal.model_dump(),
        }
        _write_receipt(payload)
        return SapeExecuteResponse(
            status="BLOCKED_BY_FATE",
            seal=seal,
            plan_id=plan_id,
            executed_at=utc_now_iso(),
            slot=request.slot or "",
            model_used=None,
            provider_used=None,
            attempts=[],
            output_text="",
            processing_time_ms=(time.monotonic() - start) * 1000.0,
            prompt_sha256=sha256_text(""),
            system_prompt=None,
            user_prompt=None,
            evidence=[],
            warnings=["blocked_by_fate"],
            request_id=request_id,
        )

    mf = _get_model_family()
    slot = _choose_sape_slot(request, mf)

    evidence: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if request.require_graph_evidence:
        evidence, warnings = _collect_graph_evidence(request)
        if request.stakes == "H" and not evidence:
            SAPE_REQUESTS.labels(endpoint="/v1/sape/execute", status="BLOCKED_BY_EVIDENCE").inc()
            payload = {
                "schema": "bizra_sape_execute_receipt_v1",
                "generated_at": utc_now_iso(),
                "truth_label": "MEASURED",
                "request_id": request_id,
                "plan_id": plan_id,
                "endpoint": "/v1/sape/execute",
                "status": "BLOCKED_BY_EVIDENCE",
                "slot": slot,
                "fate_seal": seal.model_dump(),
                "warnings": warnings,
            }
            _write_receipt(payload)
            return SapeExecuteResponse(
                status="BLOCKED_BY_EVIDENCE",
                seal=seal,
                plan_id=plan_id,
                executed_at=utc_now_iso(),
                slot=slot,
                model_used=None,
                provider_used=None,
                attempts=[],
                output_text="",
                processing_time_ms=(time.monotonic() - start) * 1000.0,
                prompt_sha256=sha256_text(""),
                system_prompt=None,
                user_prompt=None,
                evidence=[],
                warnings=warnings,
                request_id=request_id,
            )

    compiled = compile_sape_plan(request, evidence=evidence)

    try:
        completion, attempts = await chat_with_routing(
            model_family=mf,
            slot=slot,
            system_prompt=compiled.system_prompt,
            user_prompt=compiled.user_prompt,
            max_attempts=request.max_model_attempts,
        )
        for a in attempts:
            SAPE_LLM_CALLS.labels(provider=a.get("provider", "unknown"), model=a.get("model", "unknown"), status=a.get("status", "unknown")).inc()
            try:
                if a.get("status") == "ok":
                    SAPE_LLM_LATENCY.labels(provider=a.get("provider", "unknown"), model=a.get("model", "unknown")).observe(float(a.get("latency_ms", 0.0)) / 1000.0)
            except Exception:
                pass
    except LLMCallError as e:
        SAPE_REQUESTS.labels(endpoint="/v1/sape/execute", status="ERROR").inc()
        payload = {
            "schema": "bizra_sape_execute_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "plan_id": plan_id,
            "endpoint": "/v1/sape/execute",
            "status": "ERROR",
            "slot": slot,
            "prompt_sha256": compiled.prompt_sha256,
            "fate_seal": seal.model_dump(),
            "error": str(e),
        }
        _write_receipt(payload)
        return SapeExecuteResponse(
            status="ERROR",
            seal=seal,
            plan_id=plan_id,
            executed_at=utc_now_iso(),
            slot=slot,
            model_used=None,
            provider_used=None,
            attempts=[],
            output_text="",
            processing_time_ms=(time.monotonic() - start) * 1000.0,
            prompt_sha256=compiled.prompt_sha256,
            system_prompt=compiled.system_prompt if request.include_prompts_in_response else None,
            user_prompt=compiled.user_prompt if request.include_prompts_in_response else None,
            evidence=evidence,
            warnings=compiled.warnings + warnings + [str(e)],
            request_id=request_id,
        )

    SAPE_REQUESTS.labels(endpoint="/v1/sape/execute", status="SUCCESS").inc()
    payload = {
        "schema": "bizra_sape_execute_receipt_v1",
        "generated_at": utc_now_iso(),
        "truth_label": "MEASURED",
        "request_id": request_id,
        "plan_id": plan_id,
        "endpoint": "/v1/sape/execute",
        "status": "SUCCESS",
        "slot": slot,
        "prompt_sha256": compiled.prompt_sha256,
        "model_used": completion.model_name,
        "provider_used": completion.provider,
        "attempts": attempts,
        "graph_evidence_count": len(evidence),
        "warnings": compiled.warnings + warnings,
        "fate_seal": seal.model_dump(),
    }
    if os.getenv("BIZRA_KERNEL_RECEIPTS_INCLUDE_PROMPTS", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        payload["system_prompt"] = compiled.system_prompt
        payload["user_prompt"] = compiled.user_prompt
    _write_receipt(payload)

    return SapeExecuteResponse(
        status="SUCCESS",
        seal=seal,
        plan_id=plan_id,
        executed_at=utc_now_iso(),
        slot=slot,
        model_used=completion.model_name,
        provider_used=completion.provider,
        attempts=attempts,
        output_text=completion.text,
        processing_time_ms=(time.monotonic() - start) * 1000.0,
        prompt_sha256=compiled.prompt_sha256,
        system_prompt=compiled.system_prompt if request.include_prompts_in_response else None,
        user_prompt=compiled.user_prompt if request.include_prompts_in_response else None,
        evidence=evidence,
        warnings=compiled.warnings + warnings,
        request_id=request_id,
    )


# ============================================================================
# QUALITY RADAR ENDPOINT
# ============================================================================

class QualityRadarResponse(BaseModel):
    """Quality Radar assessment response."""
    id: str
    timestamp: str
    overall_score: float = Field(..., description="Overall quality score 0-10")
    ihsan_composite: float = Field(..., description="Ihsān composite score 0-1")
    ihsan_snr: float = Field(..., description="SNR value 7-9")
    ihsan_tier: str = Field(..., description="SNR tier T1-T6")
    math_rigor_score: float = Field(..., description="Mathematical rigor 0-1")
    trend_direction: str = Field(..., description="improving/stable/declining/unknown")
    probes: Dict[str, Any] = Field(default_factory=dict)
    ihsan_vector: Dict[str, float] = Field(default_factory=dict)
    invariants_passed: int = 0
    invariants_total: int = 0
    evidence_count: int = 0
    warnings: list = Field(default_factory=list)


@app.get("/v1/quality/radar", response_model=QualityRadarResponse, dependencies=[Depends(verify_token)])
async def quality_radar():
    """
    Real-time quality radar assessment.
    
    Returns comprehensive quality metrics with Ihsān 8-dimension vector,
    SNR-tier classification, and mathematical invariant verification.
    """
    request_id = uuid.uuid4().hex[:12]

    repo_root = Path(__file__).resolve().parents[1]
    evidence_dir = repo_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    script_path = repo_root / "scripts" / "quality_radar_elite.py"
    if not script_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Quality radar script missing: {script_path}",
        )

    out_prefix = evidence_dir / f"radar_{request_id}"
    json_path = out_prefix.with_suffix(".json")

    def _decode(b: bytes, limit: int = 4000) -> str:
        try:
            s = b.decode("utf-8", errors="replace")
        except Exception:
            s = str(b)
        return s if len(s) <= limit else (s[:limit] + "...<truncated>")

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(script_path),
            "--skip-tests",  # Real-time endpoint: keep fast
            "--json",
            "-o",
            str(out_prefix),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=120.0)

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Quality radar subprocess failed (exit={proc.returncode}). "
                    f"stdout={_decode(stdout_b)} stderr={_decode(stderr_b)}"
                ),
            )

        if not json_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Quality report generation failed (missing {json_path})",
            )

        data = json.loads(json_path.read_text(encoding="utf-8"))

        return QualityRadarResponse(
            id=data.get("id", request_id),
            timestamp=data.get("timestamp", utc_now_iso()),
            overall_score=data.get("overall_score", 0.0),
            ihsan_composite=data.get("ihsan", {}).get("composite", 0.0),
            ihsan_snr=data.get("ihsan", {}).get("snr", 7.0),
            ihsan_tier=data.get("ihsan", {}).get("tier", "T1"),
            math_rigor_score=data.get("math_rigor_score", 0.0),
            trend_direction=data.get("trend", {}).get("direction", "unknown"),
            probes=data.get("probes", {}),
            ihsan_vector=data.get("ihsan", {}).get("vector", {}),
            invariants_passed=sum(1 for i in data.get("invariants", []) if i.get("passed")),
            invariants_total=len(data.get("invariants", [])),
            evidence_count=data.get("evidence_count", 0),
            warnings=data.get("warnings", []),
        )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Quality radar timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality radar error: {e}")
    finally:
        # Cleanup generated artifacts even on errors
        try:
            for p in evidence_dir.glob(f"radar_{request_id}*"):
                if p.is_file():
                    p.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/v1/quality/prometheus", dependencies=[Depends(verify_token)])
async def quality_prometheus():
    """
    Export quality metrics in Prometheus format.
    """
    prom_path = Path(__file__).parent.parent / "evidence" / "quality_radar_elite.prom"
    
    if prom_path.exists():
        content = prom_path.read_text(encoding="utf-8")
        return Response(content, media_type="text/plain; charset=utf-8")
    else:
        return Response("# No quality metrics available\n", media_type="text/plain")


# =========================================================================
# ECOSYSTEM INDEX + SEAL ENDPOINT
# =========================================================================


class EcosystemSealRequest(BaseModel):
    config_path: str = Field(
        default="tools/ecosystem/ecosystem_config.yaml",
        description="Path to ecosystem_config.yaml (relative to repo root or absolute)",
    )
    out_path: str = Field(
        default="BIZRA_ECOSYSTEM_MANIFEST.json",
        description="Output manifest path (relative to repo root or absolute)",
    )
    max_projects: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Max projects to index (safety bound)",
    )
    seal_note: str = Field(
        default="BIZRA Ecosystem Manifest sealed",
        description="Seal note included in the receipt",
    )


class EcosystemSealResponse(BaseModel):
    status: str
    generated_at: str
    manifest_path: str
    manifest_sha256: str
    receipt: Dict[str, Any]
    request_id: str


@app.post("/v1/ecosystem/seal", response_model=EcosystemSealResponse, dependencies=[Depends(verify_token)])
async def ecosystem_seal(req: EcosystemSealRequest):
    """Generate the ecosystem manifest and emit a SHA-256 seal receipt."""
    request_id = uuid.uuid4().hex[:12]
    repo_root = Path(__file__).resolve().parents[1]

    try:
        config_path = Path(req.config_path)
        if not config_path.is_absolute():
            config_path = (repo_root / config_path).resolve()
        out_path = Path(req.out_path)
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()

        cfg = load_ecosystem_config(config_path)
        manifest = build_manifest(repo_root=repo_root, cfg=cfg, max_projects=req.max_projects)
        write_manifest(manifest, out_path=out_path)

        receipt = seal_manifest(manifest_path=out_path, seal_note=req.seal_note)

        payload = {
            "schema": "bizra_ecosystem_seal_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "endpoint": "/v1/ecosystem/seal",
            "manifest_path": str(out_path),
            "manifest_sha256": receipt.get("manifest_sha256"),
            "ecosystem_receipt": receipt,
        }
        _write_receipt(payload)

        return EcosystemSealResponse(
            status="SUCCESS",
            generated_at=utc_now_iso(),
            manifest_path=str(out_path),
            manifest_sha256=str(receipt.get("manifest_sha256")),
            receipt=receipt,
            request_id=request_id,
        )
    except Exception as e:
        payload = {
            "schema": "bizra_ecosystem_seal_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "endpoint": "/v1/ecosystem/seal",
            "status": "ERROR",
            "error": str(e),
        }
        _write_receipt(payload)
        raise HTTPException(status_code=500, detail=f"ecosystem_seal_failed: {e}")


_META_PROMPT_ENGINE: Optional[MetaPromptEngine] = None

def _extract_meta_prompt_evidence(results: List[MetaPromptResult]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    seen = set()
    for result in results:
        if result.source_agent != "OntologyArchitect":
            continue
        content = result.content
        if not isinstance(content, dict):
            continue
        items = content.get("evidence")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            entry: Dict[str, Any] = {}
            hash_value = item.get("hash")
            path_value = item.get("path")
            if hash_value:
                entry["hash"] = hash_value
            if path_value:
                entry["path"] = path_value
            if not entry:
                continue
            key = (entry.get("hash"), entry.get("path"))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(entry)
    return evidence

def _get_meta_prompt_engine() -> MetaPromptEngine:
    global _META_PROMPT_ENGINE
    if _META_PROMPT_ENGINE is None:
        _META_PROMPT_ENGINE = MetaPromptEngine(wisdom=wisdom)
    return _META_PROMPT_ENGINE

@app.post("/v1/meta-prompt/query", response_model=MetaPromptResponse, dependencies=[Depends(verify_token)])
async def meta_prompt_query(req: MetaPromptRequest):
    """
    Execute a Meta Prompt Generator workflow.
    """
    request_id = str(uuid.uuid4())
    endpoint = "/v1/meta-prompt/query"

    fate_engine = get_fate_engine()

    # Preflight: block unsafe/low-Ihsān intent before running the workflow.
    pre_intent = f"meta_prompt_query: {req.query}"
    pre_context = _safe_json_snippet({"context": req.context, "preferences": req.preferences})
    pre_seal, pre_feedback = fate_engine.audit_request_with_feedback(intent=pre_intent, context=pre_context)
    if pre_seal.verdict == "REJECTED":
        payload = {
            "schema": "bizra_meta_prompt_fate_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "request_id": request_id,
            "endpoint": endpoint,
            "stage": "preflight",
            "status": "BLOCKED_BY_FATE",
            "fate_seal": pre_seal.model_dump(),
            "feedback": pre_feedback.to_dict() if pre_feedback else None,
        }
        _write_receipt(payload)
        raise HTTPException(
            status_code=403,
            detail={
                "error": "meta_prompt_blocked_by_fate_preflight",
                "reason": pre_seal.reason,
                "seal_id": pre_seal.id,
                "feedback": pre_feedback.to_dict() if pre_feedback else None,
            },
        )

    engine = _get_meta_prompt_engine()
    response = await engine.run_knowledge_expansion(req)
    meta_prompt_evidence = _extract_meta_prompt_evidence(response.results)

    # Postflight: gate the generated output as well (fail-closed).
    post_intent = f"meta_prompt_response: {response.explanation}"
    post_context = _safe_json_snippet(
        {
            "query": req.query,
            "explanation": response.explanation,
            "confidence": response.confidence,
            "results": [r.model_dump() for r in response.results],
        }
    )
    post_seal, post_feedback = fate_engine.audit_request_with_feedback(intent=post_intent, context=post_context)

    payload = {
        "schema": "bizra_meta_prompt_fate_receipt_v1",
        "generated_at": utc_now_iso(),
        "truth_label": "MEASURED",
        "request_id": request_id,
        "endpoint": endpoint,
        "stage": "postflight",
        "status": "APPROVED" if post_seal.verdict == "APPROVED" else "BLOCKED_BY_FATE",
        "preflight": {
            "seal": pre_seal.model_dump(),
            "feedback": pre_feedback.to_dict() if pre_feedback else None,
        },
        "postflight": {
            "seal": post_seal.model_dump(),
            "feedback": post_feedback.to_dict() if post_feedback else None,
        },
        "workflow_id": str(response.workflow_id),
        "evidence_count": len(meta_prompt_evidence),
        "evidence": meta_prompt_evidence,
    }
    _write_receipt(payload)

    if post_seal.verdict == "REJECTED":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "meta_prompt_blocked_by_fate_postflight",
                "reason": post_seal.reason,
                "seal_id": post_seal.id,
                "feedback": post_feedback.to_dict() if post_feedback else None,
            },
        )

    return response


@app.on_event("shutdown")
def shutdown_event():
    wisdom.close()


def main() -> int:
    import uvicorn

    host = os.getenv("BIZRA_KERNEL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("BIZRA_KERNEL_PORT", "8010"))
    # When running as `python -m core.main`, avoid importing this module twice (which would duplicate metrics).
    uvicorn.run(app, host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
