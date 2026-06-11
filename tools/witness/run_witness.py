#!/usr/bin/env python3
"""NODE0 mission replay persist witness — vendored runner (W3/Z1-W0).

Vendored from the v3 inline runner. ONLY change vs source: absolute paths are
parameterized (--out / --repo / env, with repo-relative defaults inferred from
this file's location) so an external witness can run it from a fresh clone.
No behavior change otherwise. data_lake_commit embedding was already present.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Set in main() from args/env/inferred defaults. Functions read these as globals.
PROOF: Path
GW: Path
DEMA: Path
DL: Path
COMMIT: str
BASE = "http://127.0.0.1:7421"


def infer_repo_root() -> Path:
    """Repo root. Vendored at <repo>/tools/witness/run_witness.py -> parents[2]."""
    env = os.environ.get("BIZRA_DATA_LAKE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NODE0 mission replay persist witness")
    ap.add_argument(
        "--out",
        default=os.environ.get("BIZRA_WITNESS_OUT", ""),
        help="output/proof directory (default: <repo>/artifacts/witness/run)",
    )
    ap.add_argument(
        "--repo",
        default="",
        help="data-lake repo root (default: $BIZRA_DATA_LAKE_ROOT or inferred from this file)",
    )
    return ap.parse_args()


def gateway_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"^{GW}$"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.splitlines() if x.strip()]


def stop_gateway() -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in gateway_pids():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(1)
        if not gateway_pids():
            break
    for _ in range(20):
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=1)
        except Exception:
            return
        time.sleep(0.5)
    raise RuntimeError("gateway still reachable on :7421")


def curl_json(url: str, out: Path) -> dict:
    raw = urllib.request.urlopen(url, timeout=5).read()
    out.write_bytes(raw)
    return json.loads(raw)


def start_gateway(log: Path, runtime: Path, extra_env: dict[str, str]) -> subprocess.Popen:
    runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(extra_env)
    log.parent.mkdir(parents=True, exist_ok=True)
    f = log.open("a")
    return subprocess.Popen(
        [str(GW)],
        cwd=runtime,
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_scenario(name: str, extra_env: dict[str, str] | None = None) -> dict:
    extra_env = extra_env or {}
    d = PROOF / name
    d.mkdir(parents=True, exist_ok=True)
    runtime = d / "runtime"
    stop_gateway()

    gw1 = start_gateway(d / "gw1.log", runtime, extra_env)
    (d / "gw1.pid").write_text(str(gw1.pid))
    time.sleep(3)
    curl_json(f"{BASE}/health", d / "health1.json")
    before = curl_json(f"{BASE}/chain", d / "chain-before.json")
    activate_out = subprocess.run(
        [str(DEMA), "activate", "--json"], check=True, capture_output=True, text=True
    ).stdout
    (d / "activate.json").write_text(activate_out)
    after = curl_json(f"{BASE}/chain", d / "chain-after-mission.json")
    head = after["head"]
    curl_json(f"{BASE}/chain/{head}", d / "receipt-before-stop.json")

    gw1.send_signal(signal.SIGTERM)
    try:
        gw1.wait(timeout=5)
    except subprocess.TimeoutExpired:
        gw1.kill()
    stop_gateway()
    (d / "stop-verified.txt").write_text("stopped_ok\n")

    gw2 = start_gateway(d / "gw2.log", runtime, extra_env)
    (d / "gw2.pid").write_text(str(gw2.pid))
    time.sleep(3)
    gw2_ok = False
    restart = None
    rh_after = None
    try:
        curl_json(f"{BASE}/health", d / "health2.json")
        gw2_ok = True
        restart = curl_json(f"{BASE}/chain", d / "chain-after-restart.json")
        try:
            rh_after = curl_json(f"{BASE}/chain/{head}", d / "receipt-after-restart.json")
        except urllib.error.HTTPError:
            (d / "receipt-after-restart.json").write_text('{"not_found":true}\n')
            rh_after = {"not_found": True}
    except Exception:
        (d / "gw2-status.txt").write_text("gw2_failed_health\n")

    gw2.send_signal(signal.SIGTERM)
    try:
        gw2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        gw2.kill()

    cache_present = False
    cache_records = 0
    if name == "dema_cache_root":
        cache_file = PROOF / "dema_cache_root" / "cache" / "dema_cache" / "receipt_history.json"
        if cache_file.exists():
            cache_present = True
            cache_records = len(json.loads(cache_file.read_text()).get("records", []))

    return {
        "gw2_started_cleanly": gw2_ok,
        "chain_before_length": before.get("length"),
        "chain_after_mission": {"length": after["length"], "head": after["head"]},
        "chain_after_restart": {"length": restart["length"], "head": restart["head"]} if restart else None,
        "head_receipt_before_stop_ok": True,
        "head_receipt_after_restart_ok": isinstance(rh_after, dict) and rh_after.get("id") == head,
        "receipt_history_cache_present": cache_present,
        "receipt_history_cache_records": cache_records,
        "persist_survives_restart": bool(
            restart and gw2_ok and restart["length"] == after["length"] and restart["head"] == after["head"]
        ),
    }


def main() -> None:
    global PROOF, GW, DEMA, DL, COMMIT
    args = parse_args()
    DL = Path(args.repo).resolve() if args.repo else infer_repo_root()
    GW = DL / "bizra-omega" / "target" / "release" / "bizra-cognition-gateway"
    DEMA = DL / "bizra-omega" / "target" / "release" / "dema"
    PROOF = Path(args.out).resolve() if args.out else (DL / "artifacts" / "witness" / "run")
    COMMIT = subprocess.check_output(["git", "-C", str(DL), "rev-parse", "HEAD"], text=True).strip()

    PROOF.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "default_in_memory": run_scenario("default_in_memory"),
    }
    cache_root = PROOF / "dema_cache_root" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    scenarios["dema_cache_root"] = run_scenario(
        "dema_cache_root", {"BIZRA_DEMA_CACHE_ROOT": str(cache_root)}
    )
    store_root = PROOF / "receipt_store_path" / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    scenarios["receipt_store_path"] = run_scenario(
        "receipt_store_path",
        {"BIZRA_RECEIPT_STORE_PATH": str(store_root)},
    )
    default_root = PROOF / "receipt_store_default_token" / "lake"
    default_root.mkdir(parents=True, exist_ok=True)
    scenarios["receipt_store_default_token"] = run_scenario(
        "receipt_store_default_token",
        {
            "BIZRA_RECEIPT_STORE_PATH": "default",
            "BIZRA_DATA_LAKE_ROOT": str(default_root),
        },
    )
    stop_gateway()

    label = (
        "NODE0_MISSION_REPLAY_PERSIST_WITNESS_COMPLETE"
        if any(s["persist_survives_restart"] for s in scenarios.values())
        else "NODE0_MISSION_REPLAY_PERSIST_BLOCKED_WITH_EVIDENCE"
    )
    out = {
        "schema": "bizra.node0.mission_replay_persist_witness.v0",
        "success_label": label,
        "witness_harness": "v3-python — exe-scoped stop; verifies :7421 unreachable between stop and second start",
        "data_lake_commit": COMMIT,
        "proof_dir": str(PROOF),
        "scenarios": scenarios,
        "root_cause": {
            "component": "ReceiptChain payload store at gateway bootstrap",
            "primary_code": "bizra-omega/bizra-cognition-gateway/src/main.rs:668 — ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()))",
            "get_chain_reads": "main.rs:896-912 — GET /chain length/head from in-memory rt.chain only",
            "sled_unwired": "bizra-omega/bizra-cognition/src/receipt_freeze_v1.rs — SledPayloadStore behind feature sled-store; gateway README: Cycle-6 Arc 3",
            "dema_cache_limit": "BIZRA_DEMA_CACHE_ROOT persists derived receipt_history.json but does not rehydrate rt.chain on boot (main.rs:758 logs only)",
            "sovereign_fallthrough": "GET /chain/:hash can read Python sovereign_state envelopes when BIZRA_SOVEREIGN_STATE_PATH set — separate from Rust POST /mission in-memory chain",
        },
        "smallest_follow_up_go": "GO: CYCLE6_ARC3_GATEWAY_SLED_PAYLOAD_STORE — select SledPayloadStore (or full chain snapshot restore) in bootstrap_runtime when BIZRA_RECEIPT_STORE_PATH is set; add gateway test: mission -> stop -> start -> GET /chain length/head unchanged",
    }
    witness = PROOF / "NODE0_MISSION_REPLAY_PERSIST_WITNESS.json"
    witness.write_text(json.dumps(out, indent=2) + "\n")
    print(label)
    print(witness)
    print(json.dumps(scenarios, indent=2))


if __name__ == "__main__":
    main()
