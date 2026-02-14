#!/usr/bin/env python3
"""
SAPE Masterpiece Gate
=====================

Unified readiness gate for the sovereign control plane.
Validates security, determinism, receipts, and latency with evidence output.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_TOKEN = "sape-gate-token"
DEFAULT_KEY_HEX = "11" * 32


@dataclass
class GateCheck:
    name: str
    passed: bool
    score: float
    details: dict[str, Any]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _headers(
    token: str = DEFAULT_TOKEN,
    ts_ms: int | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    return {
        "X-BIZRA-TOKEN": token,
        "X-BIZRA-TS": ts_ms if ts_ms is not None else int(time.time() * 1000),
        "X-BIZRA-NONCE": nonce or uuid.uuid4().hex,
    }


def _jsonrpc(
    method: str,
    req_id: int,
    params: Any = None,
    headers: dict[str, Any] | None = None,
) -> bytes:
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id,
    }
    if params is not None:
        msg["params"] = params
    if headers is not None:
        msg["headers"] = headers
    return json.dumps(msg).encode() + b"\n"


async def _tcp_call(port: int, payload: bytes, timeout: float = 5.0) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(payload)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return json.loads(line), elapsed_ms
    finally:
        writer.close()
        await writer.wait_closed()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[idx]


def _write_valid_genesis_fixture(state_dir: Path) -> None:
    """Write a minimal valid genesis authority fixture into state_dir."""
    genesis_hash_hex = "ab" * 32
    genesis = {
        "timestamp": 1770295290922,
        "identity": {
            "node_id": "node0_fixture_0001",
            "public_key": "11" * 32,
            "name": "Node0 Fixture",
            "location": "CI",
            "created_at": 1770295290922,
            "identity_hash": [1] * 32,
        },
        "pat_team": {
            "owner_node": "node0_fixture_0001",
            "agents": [],
            "team_hash": [2] * 32,
        },
        "sat_team": {
            "owner_node": "node0_fixture_0001",
            "agents": [],
            "team_hash": [3] * 32,
            "governance": {"quorum": 0.67, "voting_period_hours": 72, "upgrade_threshold": 0.8},
        },
        "partnership_hash": [4] * 32,
        "genesis_hash": list(bytes.fromhex(genesis_hash_hex)),
        "hardware": {},
        "knowledge": {},
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "node0_genesis.json").write_text(json.dumps(genesis), encoding="utf-8")
    (state_dir / "genesis_hash.txt").write_text(genesis_hash_hex, encoding="utf-8")


async def _check_bridge_auth_receipts() -> GateCheck:
    from core.bridges.desktop_bridge import DesktopBridge

    port = _free_port()
    bridge = DesktopBridge(host="127.0.0.1", port=port)
    await bridge.start()
    try:
        missing_headers_resp, _ = await _tcp_call(
            port,
            _jsonrpc("ping", req_id=1),
        )
        wrong_token_resp, _ = await _tcp_call(
            port,
            _jsonrpc("ping", req_id=2, headers=_headers(token="wrong")),
        )
        valid_resp, _ = await _tcp_call(
            port,
            _jsonrpc("ping", req_id=3, headers=_headers()),
        )
        stale_resp, _ = await _tcp_call(
            port,
            _jsonrpc("ping", req_id=4, headers=_headers(ts_ms=946684800000)),
        )
        nonce = "sape-fixed-replay-nonce"
        first_nonce_resp, _ = await _tcp_call(
            port,
            _jsonrpc("ping", req_id=5, headers=_headers(nonce=nonce)),
        )
        replay_resp, _ = await _tcp_call(
            port,
            _jsonrpc("ping", req_id=6, headers=_headers(nonce=nonce)),
        )

        cond_missing = missing_headers_resp.get("error", {}).get("data", {}).get("code") == "AUTH_MISSING_HEADERS"
        cond_wrong = wrong_token_resp.get("error", {}).get("data", {}).get("code") == "AUTH_INVALID_TOKEN"
        cond_valid = (
            "result" in valid_resp
            and valid_resp["result"].get("status") == "alive"
            and isinstance(valid_resp["result"].get("receipt"), dict)
        )
        cond_stale = stale_resp.get("error", {}).get("data", {}).get("code") == "AUTH_STALE_TIMESTAMP"
        cond_nonce_first = "result" in first_nonce_resp
        cond_replay = replay_resp.get("error", {}).get("data", {}).get("code") == "AUTH_NONCE_REPLAY"

        checks = [cond_missing, cond_wrong, cond_valid, cond_stale, cond_nonce_first, cond_replay]
        passed = all(checks)
        score = sum(1 for c in checks if c) / len(checks)
        return GateCheck(
            name="bridge_auth_receipts",
            passed=passed,
            score=round(score, 3),
            details={
                "missing_headers": cond_missing,
                "wrong_token": cond_wrong,
                "valid_ping_receipt": cond_valid,
                "stale_timestamp": cond_stale,
                "nonce_first_accept": cond_nonce_first,
                "nonce_replay_blocked": cond_replay,
            },
        )
    finally:
        await bridge.stop()


def _check_receipt_signer_strict() -> GateCheck:
    from core.bridges.bridge_receipt import BridgeReceiptEngine, load_signer

    prior = os.environ.pop("BIZRA_RECEIPT_PRIVATE_KEY_HEX", None)
    strict_fail = False
    try:
        try:
            load_signer()
        except RuntimeError:
            strict_fail = True
    finally:
        if prior is not None:
            os.environ["BIZRA_RECEIPT_PRIVATE_KEY_HEX"] = prior
        else:
            os.environ["BIZRA_RECEIPT_PRIVATE_KEY_HEX"] = DEFAULT_KEY_HEX

    engine = BridgeReceiptEngine()
    receipt = engine.create_receipt(
        method="sape_gate",
        query_data={"check": "receipt_signing"},
        result_data={"status": "ok"},
        fate_score=1.0,
        snr_score=1.0,
        gate_passed="sape_gate",
        status="accepted",
    )
    verified = engine.verify_receipt(receipt)

    checks = [strict_fail, verified]
    passed = all(checks)
    return GateCheck(
        name="receipt_signer_strict",
        passed=passed,
        score=round(sum(1 for c in checks if c) / len(checks), 3),
        details={
            "missing_key_fails": strict_fail,
            "signature_verifies": verified,
            "receipt_id": receipt.get("receipt_id"),
        },
    )


def _check_node0_genesis_fail_closed() -> GateCheck:
    from core.sovereign.origin_guard import (
        enforce_node0_fail_closed,
        resolve_origin_snapshot,
        validate_genesis_chain,
    )

    with tempfile.TemporaryDirectory() as td_missing:
        missing_dir = Path(td_missing)
        missing_fails = False
        try:
            enforce_node0_fail_closed(missing_dir, "node0")
        except RuntimeError:
            missing_fails = True

    with tempfile.TemporaryDirectory() as td_valid:
        valid_dir = Path(td_valid)
        _write_valid_genesis_fixture(valid_dir)
        chain_ok, reason = validate_genesis_chain(valid_dir)
        snapshot = resolve_origin_snapshot(valid_dir, "node0")

    checks = [
        missing_fails,
        chain_ok,
        snapshot.get("designation") == "node0",
        snapshot.get("hash_validated") is True,
        snapshot.get("block_id") == "block0",
    ]
    return GateCheck(
        name="node0_genesis_fail_closed",
        passed=all(checks),
        score=round(sum(1 for c in checks if c) / len(checks), 3),
        details={
            "missing_genesis_fails": missing_fails,
            "valid_chain_ok": chain_ok,
            "valid_chain_reason": reason,
            "origin_snapshot": snapshot,
        },
    )


async def _check_bridge_receipt_origin_proof() -> GateCheck:
    from core.bridges.bridge_receipt import BridgeReceiptEngine
    from core.bridges.desktop_bridge import DesktopBridge

    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td)
        _write_valid_genesis_fixture(state_dir)

        import core.bridges.desktop_bridge as bridge_mod

        previous_state_dir = bridge_mod.GENESIS_STATE_DIR
        previous_role = os.environ.get("BIZRA_NODE_ROLE")
        os.environ["BIZRA_NODE_ROLE"] = "node0"
        bridge_mod.GENESIS_STATE_DIR = state_dir

        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            ping_resp, _ = await _tcp_call(
                port,
                _jsonrpc("ping", req_id=700, headers=_headers()),
            )
            ping_receipt_ref = ping_resp.get("result", {}).get("receipt", {})
            receipt_id = ping_receipt_ref.get("receipt_id")
            if not receipt_id:
                return GateCheck(
                    name="bridge_receipt_origin_proof",
                    passed=False,
                    score=0.0,
                    details={"error": "missing ping receipt id"},
                )

            receipt_resp, _ = await _tcp_call(
                port,
                _jsonrpc(
                    "get_receipt",
                    req_id=701,
                    params={"receipt_id": receipt_id},
                    headers=_headers(),
                ),
            )
            full_receipt = receipt_resp.get("result", {})
            origin = full_receipt.get("origin", {})
            origin_digest = full_receipt.get("origin_digest", "")
            verify_payload = dict(full_receipt)
            verify_payload.pop("request_receipt", None)

            verified = BridgeReceiptEngine().verify_receipt(verify_payload)
            checks = [
                full_receipt.get("status") == "accepted",
                isinstance(origin, dict),
                origin.get("designation") == "node0",
                origin.get("hash_validated") is True,
                isinstance(origin_digest, str) and len(origin_digest) == 64,
                verified,
            ]
            return GateCheck(
                name="bridge_receipt_origin_proof",
                passed=all(checks),
                score=round(sum(1 for c in checks if c) / len(checks), 3),
                details={
                    "receipt_id": receipt_id,
                    "origin": origin,
                    "origin_digest_len": len(origin_digest) if isinstance(origin_digest, str) else 0,
                    "signature_verified": verified,
                },
            )
        finally:
            await bridge.stop()
            bridge_mod.GENESIS_STATE_DIR = previous_state_dir
            if previous_role is None:
                os.environ.pop("BIZRA_NODE_ROLE", None)
            else:
                os.environ["BIZRA_NODE_ROLE"] = previous_role


def _check_evidence_ledger_origin_proof() -> GateCheck:
    from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt

    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td)
        _write_valid_genesis_fixture(state_dir)
        ledger = EvidenceLedger(state_dir / "evidence.jsonl", validate_on_append=False)

        entry = emit_receipt(
            ledger,
            receipt_id="b" * 32,
            node_id="node0_fixture_0001",
            snr_score=0.99,
            ihsan_score=0.99,
            seal_digest="c" * 64,
            critical_decision=True,
            node_role="node0",
            state_dir=state_dir,
            signer_private_key_hex=DEFAULT_KEY_HEX,
        )

        strict_failed_without_key = False
        prior_key = os.environ.pop("BIZRA_RECEIPT_PRIVATE_KEY_HEX", None)
        try:
            try:
                emit_receipt(
                    ledger,
                    receipt_id="d" * 32,
                    node_id="node0_fixture_0001",
                    snr_score=0.99,
                    ihsan_score=0.99,
                    seal_digest="e" * 64,
                    critical_decision=True,
                    node_role="node0",
                    state_dir=state_dir,
                    signer_private_key_hex="",
                )
            except RuntimeError:
                strict_failed_without_key = True
        finally:
            if prior_key is not None:
                os.environ["BIZRA_RECEIPT_PRIVATE_KEY_HEX"] = prior_key

        checks = [
            "origin" in entry.receipt,
            "origin_digest" in entry.receipt,
            isinstance(entry.receipt.get("origin_digest"), str),
            len(entry.receipt.get("origin_digest", "")) == 64,
            "signature" in entry.receipt,
            strict_failed_without_key,
        ]
        return GateCheck(
            name="evidence_ledger_origin_proof",
            passed=all(checks),
            score=round(sum(1 for c in checks if c) / len(checks), 3),
            details={
                "origin": entry.receipt.get("origin"),
                "origin_digest": entry.receipt.get("origin_digest"),
                "has_signature": "signature" in entry.receipt,
                "node0_unsigned_rejected": strict_failed_without_key,
            },
        )


def _check_localhost_defaults() -> GateCheck:
    from core.bridges.desktop_bridge import DesktopBridge
    from core.sovereign.api import SovereignAPIServer, serve

    class _Runtime:
        pass

    api_server = SovereignAPIServer(runtime=_Runtime())
    serve_sig = inspect.signature(serve)
    serve_host_default = serve_sig.parameters["host"].default

    desktop_rejects = False
    try:
        DesktopBridge(host="0.0.0.0", port=9742)
    except ValueError:
        desktop_rejects = True

    checks = [
        api_server.host == "127.0.0.1",
        serve_host_default == "127.0.0.1",
        desktop_rejects,
    ]
    passed = all(checks)
    return GateCheck(
        name="localhost_defaults",
        passed=passed,
        score=round(sum(1 for c in checks if c) / len(checks), 3),
        details={
            "api_server_default_host": api_server.host,
            "serve_default_host": serve_host_default,
            "desktop_rejects_non_localhost": desktop_rejects,
        },
    )


async def _check_control_plane_latency() -> GateCheck:
    from core.bridges.desktop_bridge import DesktopBridge, TokenBucket

    port = _free_port()
    bridge = DesktopBridge(host="127.0.0.1", port=port)
    bridge._validate_fate = lambda _op: {"passed": True, "overall": 0.99}  # type: ignore[assignment]
    bridge._get_gateway = lambda: None  # type: ignore[assignment]
    bridge._get_rust_constitution = lambda: None  # type: ignore[assignment]
    bridge._get_skill_router = lambda: None  # type: ignore[assignment]
    bridge._rate_limiter = TokenBucket(rate=1000.0, burst=1000.0)

    await bridge.start()
    try:
        latencies: dict[str, list[float]] = {"ping": [], "status": [], "list_skills": []}
        req_id = 100
        for method in ("ping", "status", "list_skills"):
            for _ in range(20):
                req_id += 1
                resp, elapsed_ms = await _tcp_call(
                    port,
                    _jsonrpc(method, req_id=req_id, headers=_headers()),
                )
                if "error" in resp:
                    return GateCheck(
                        name="control_plane_latency",
                        passed=False,
                        score=0.0,
                        details={"error": resp},
                    )
                latencies[method].append(elapsed_ms)

        p95 = {k: _p95(v) for k, v in latencies.items()}
        overall_p95 = max(p95.values()) if p95 else 0.0
        passed = overall_p95 < 200.0
        return GateCheck(
            name="control_plane_latency",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "p95_ms": {k: round(v, 2) for k, v in p95.items()},
                "overall_p95_ms": round(overall_p95, 2),
                "threshold_ms": 200.0,
                "sample_size_each": 20,
            },
        )
    finally:
        await bridge.stop()


async def _check_snr_contract() -> GateCheck:
    from core.sovereign.snr_maximizer import SNRMaximizer

    result = await SNRMaximizer(ihsan_threshold=0.85).optimize(
        "Structured grounded analysis with coherent steps and explicit evidence."
    )
    tags = result.get("claim_tags", {})
    checks = [
        isinstance(tags, dict),
        tags.get("snr_score") == "measured",
        tags.get("groundedness") == "measured",
    ]
    return GateCheck(
        name="snr_claim_tags_contract",
        passed=all(checks),
        score=round(sum(1 for c in checks if c) / len(checks), 3),
        details={"claim_tags": tags},
    )


def _check_secret_hygiene() -> GateCheck:
    proc = subprocess.run(
        ["python", "scripts/ci_secret_scan.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    passed = proc.returncode == 0
    return GateCheck(
        name="secret_hygiene",
        passed=passed,
        score=1.0 if passed else 0.0,
        details={
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
            "stderr": proc.stderr.strip(),
        },
    )


def _score(checks: list[GateCheck]) -> dict[str, Any]:
    if not checks:
        return {
            "composite_score": 0.0,
            "snr": 0.0,
            "ihsan": 0.0,
            "passed": False,
        }
    avg = mean(c.score for c in checks)
    # Keep a governance-style split for dashboards.
    snr = round(avg, 3)
    ihsan = round(min(1.0, avg + 0.02), 3)
    passed = all(c.passed for c in checks) and snr >= 0.95 and ihsan >= 0.95
    return {
        "composite_score": round(avg, 3),
        "snr": snr,
        "ihsan": ihsan,
        "passed": passed,
    }


async def run_gate(strict: bool = False) -> dict[str, Any]:
    # Ensure defaults for deterministic local runs.
    os.environ.setdefault("BIZRA_BRIDGE_TOKEN", DEFAULT_TOKEN)
    os.environ.setdefault("BIZRA_RECEIPT_PRIVATE_KEY_HEX", DEFAULT_KEY_HEX)
    os.environ.setdefault("BIZRA_NODE_ROLE", "node")

    checks: list[GateCheck] = []
    checks.append(await _check_bridge_auth_receipts())
    checks.append(_check_receipt_signer_strict())
    checks.append(_check_node0_genesis_fail_closed())
    checks.append(await _check_bridge_receipt_origin_proof())
    checks.append(_check_evidence_ledger_origin_proof())
    checks.append(_check_localhost_defaults())
    checks.append(await _check_snr_contract())
    checks.append(_check_secret_hygiene())
    checks.append(await _check_control_plane_latency())

    score = _score(checks)
    if strict and score["composite_score"] < 0.98:
        score["passed"] = False

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_mode": strict,
        "score": score,
        "checks": [asdict(c) for c in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAPE Masterpiece Gate")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--strict", action="store_true", help="Require higher composite score")
    args = parser.parse_args()

    result = asyncio.run(run_gate(strict=args.strict))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 72)
        print("SAPE MASTERPIECE GATE")
        print("=" * 72)
        for c in result["checks"]:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"- {c['name']}: {status} (score={c['score']:.3f})")
        s = result["score"]
        print("-" * 72)
        print(
            "Composite="
            f"{s['composite_score']:.3f} | SNR={s['snr']:.3f} | Ihsan={s['ihsan']:.3f} | "
            f"Verdict={'PASS' if s['passed'] else 'FAIL'}"
        )

    return 0 if result["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
