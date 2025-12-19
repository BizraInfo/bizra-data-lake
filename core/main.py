from __future__ import annotations

import asyncio
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

from core.fate import FateEngine, FateSeal
from core.llm import LLMCallError, chat_with_routing
from core.model_family import ModelFamily, load_model_family
from core.sape import SapeExecuteRequest, SapeExecuteResponse, SapePlanRequest, SapePlanResponse, compile_sape_plan, sha256_text
from core.wisdom import HouseOfWisdom


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_receipt(payload: Dict[str, Any]) -> Optional[Path]:
    if os.getenv("BIZRA_KERNEL_RECEIPTS", "1").strip().lower() in {"0", "false", "no"}:
        return None
    base = _receipt_dir()
    base.mkdir(parents=True, exist_ok=True)
    folder = base / f"kernel_request_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_{payload['request_id']}"
    folder.mkdir(parents=True, exist_ok=True)
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
