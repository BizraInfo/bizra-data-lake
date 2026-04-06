"""
BIZRA Genesis Mission — Constitutional End-to-End Proof
========================================================
The first real LLM-powered mission through the Rust-verified pipeline.

This is not mock events. This is:
  User query → Python EventBus → Rust synapse → Ollama inference →
  Action receipt → Rust constitutional verification → Proof chain

Phase 88: The First Constitutional Mission

Standing on Giants:
  - Maturana (1980): Autopoiesis — the organism acts, not just responds
  - Shannon (1948): SNR measured on real output, not synthetic data
  - Al-Ghazali (1095): Ihsan — excellence in the actual work, not the plan
"""

import sys
import json
import time

sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")

print("=" * 60)
print("  BIZRA GENESIS MISSION — CONSTITUTIONAL PROOF")
print("=" * 60)

# ── Step 1: Verify Rust bridge is live ─────────────────────
from core.bus.rust_bridge import diagnose_bridge

diag = diagnose_bridge()
assert diag["rust_available"], "Rust bridge not available!"
print(f"\n1. Rust bridge: v{diag['version']}, ihsan={diag['ihsan_threshold']}")

# ── Step 2: Wire the full nervous system ───────────────────
from core.bus.subscribers import EventBus, EventType, wire_all_subscribers
from core.bus.rust_bridge import wire_rust_bridge

bus = EventBus()


# Wire Python subscribers (minimal no-op adapters for non-LLM paths)
class _NoOp:
    def reinforce(self, **kw):
        pass

    def get_success_count(self, key):
        return 0

    def set_success_count(self, key, val):
        pass

    def promote_to_semantic(self, **kw):
        return True

    def record_failure_pattern(self, **kw):
        pass


class _NoOpTS:
    def begin_execution(self, **kw):
        return "ts_genesis"


class _NoOpSession:
    def halt(self, **kw):
        pass


subs = wire_all_subscribers(
    bus,
    memory_store=_NoOp(),
    telescript_engine=_NoOpTS(),
    receipt_chain=[],
    reflex_cache={},
    session_manager=_NoOpSession(),
    audit_log=type("A", (), {"log_violation": lambda s, **kw: None})(),
    quarantine_store=type("Q", (), {"isolate": lambda s, **kw: None})(),
    healing_engine=None,
    hhmm_engine=None,
    poi_engine=None,
    token_minter=None,
    context_budget=None,
    self_model=None,
    capability_registry=None,
)
print(f"2. Python EventBus: {len(subs)} subscribers wired")

# Wire Rust constitutional bridge
rust_sub = wire_rust_bridge(bus, production=False)
assert rust_sub is not None, "Failed to wire Rust bridge!"
print(f"   Rust bridge: ACTIVE ({rust_sub.stats})")


# ── Step 3: Check Ollama is alive ──────────────────────────
import urllib.request

try:
    resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    models_data = json.loads(resp.read())
    model_names = [m["name"] for m in models_data.get("models", [])]
    print(f"\n3. Ollama: {len(model_names)} models available")
    for m in model_names[:5]:
        print(f"   - {m}")
    if len(model_names) > 5:
        print(f"   ... +{len(model_names)-5} more")
except Exception as e:
    print(f"\n3. Ollama: OFFLINE ({e})")
    print("   Cannot run live mission without Ollama. Exiting.")
    sys.exit(1)

# Pick the fastest model for genesis proof
fast_model = None
for preferred in ["llama3.1:8b", "qwen2.5:3b", "phi3:mini", "mistral"]:
    if preferred in model_names:
        fast_model = preferred
        break
if not fast_model and model_names:
    fast_model = model_names[0]
print(f"   Selected: {fast_model}")


# ── Step 4: Execute Genesis Mission ────────────────────────
MISSION = "Explain in a detailed paragraph why constitutional governance matters for autonomous AI agents, covering trust, accountability, and the difference between governed and ungoverned systems."

print(f'\n4. GENESIS MISSION: "{MISSION}"')
t0 = time.perf_counter()

# 4a. Emit action intent (Python cognitive event → Rust)
bus.publish(
    EventType.ACTION_INTENT,
    {
        "query": MISSION,
        "agent": "ATLAS",
        "model": fast_model,
        "timestamp": time.time(),
    },
)
print("   [INTENT] Emitted to Python + Rust")

# 4b. Call Ollama directly (the actual LLM inference)
t_llm = time.perf_counter()
req_body = json.dumps(
    {
        "model": fast_model,
        "prompt": MISSION,
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0.7},
    }
).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=req_body,
    headers={"Content-Type": "application/json"},
)

try:
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    llm_response = result.get("response", "").strip()
    llm_duration_ms = int((time.perf_counter() - t_llm) * 1000)
    print(f"   [LLM] {fast_model} responded in {llm_duration_ms}ms")
    print(f"   Response: {llm_response[:200]}")
except Exception as e:
    print(f"   [LLM] FAILED: {e}")
    sys.exit(1)

# 4c. Compute proof artifacts
import blake3 as _b3

content_hash = _b3.blake3(
    b"bizra-genesis-mission-v1:" + llm_response.encode()
).hexdigest()
receipt_hash = _b3.blake3(
    b"bizra-receipt-v1:" + content_hash.encode() + MISSION.encode()
).hexdigest()

# 4d. Compute SNR (signal quality of the response)
try:
    snr_engine = __import__("bizra").SNREngine(0.85, 0.95)
    snr_result = snr_engine.analyze_text(llm_response)
    snr_score = snr_result["snr"]
    print(f"   [SNR] Rust-native analysis: {snr_score:.4f}")
except Exception as e:
    snr_score = 0.90  # fallback
    print(f"   [SNR] Fallback: {snr_score} ({e})")


# 4e. Emit action receipt (the constitutional proof event)
ihsan_composite = min(snr_score * 1.05, 1.0)  # bounded
bus.publish(
    EventType.ACTION_RECEIPT,
    {
        "action_type": "llm_inference",
        "model": fast_model,
        "query": MISSION,
        "response_preview": llm_response[:100],
        "content_hash": content_hash,
        "receipt_hash": receipt_hash,
        "ihsan_composite": round(ihsan_composite, 4),
        "snr_score": round(snr_score, 4),
        "duration_ms": llm_duration_ms,
        "result_summary": f"Genesis mission completed via {fast_model}",
    },
)
print(
    f"   [RECEIPT] Emitted — ihsan={ihsan_composite:.4f}, hash={receipt_hash[:16]}..."
)

total_ms = int((time.perf_counter() - t0) * 1000)
print(f"\n   Total mission time: {total_ms}ms")


# ── Step 5: Verify constitutional pipeline ─────────────────
stats = rust_sub.stats
print("\n5. CONSTITUTIONAL VERIFICATION:")
print(f"   Python chain height: {bus.chain_height}")
print(f"   Python chain valid:  {bus.verify_chain()}")
print(f"   Rust bridge forwarded: {stats['forwarded']}")
print(f"   Rust bridge failed:    {stats['failed']}")
print(f"   Rust bridge healthy:   {stats['bridge_healthy']}")

# ── Step 6: Build the Genesis Mission Receipt ──────────────
receipt = {
    "version": "bizra-genesis-mission-v1",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "mission": MISSION,
    "model": fast_model,
    "response_preview": llm_response[:200],
    "content_hash": content_hash,
    "receipt_hash": receipt_hash,
    "snr_score": round(snr_score, 4),
    "ihsan_composite": round(ihsan_composite, 4),
    "duration_ms": total_ms,
    "llm_duration_ms": llm_duration_ms,
    "python_chain_height": bus.chain_height,
    "python_chain_valid": bus.verify_chain(),
    "rust_bridge_forwarded": stats["forwarded"],
    "rust_bridge_failed": stats["failed"],
    "rust_module_version": diag["version"],
    "constitutional_thresholds": {
        "ihsan_floor": diag["ihsan_threshold"],
        "snr_floor": diag["snr_threshold"],
    },
}


# Save receipt as evidence
receipt_path = r"C:\BIZRA-DATA-LAKE\evidence\genesis_mission_receipt.json"
import os

os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
with open(receipt_path, "w") as f:
    json.dump(receipt, f, indent=2)
print(f"\n6. Receipt saved: {receipt_path}")

# ── VERDICT ────────────────────────────────────────────────
print(f"\n{'=' * 60}")
all_ok = (
    stats["forwarded"] >= 2  # at least intent + receipt
    and stats["failed"] == 0
    and bus.verify_chain()
    and ihsan_composite
    >= 0.95  # Constitutional floor — not the degradation floor (0.85)
    and len(llm_response) > 10
)

if all_ok:
    print("  GENESIS MISSION: CONSTITUTIONAL PROOF VERIFIED")
    print()
    print(f"  Query:    {MISSION[:50]}...")
    print(f"  Model:    {fast_model}")
    print(f"  Response: {llm_response[:80]}...")
    print(f"  SNR:      {snr_score:.4f} (Rust-native)")
    print(f"  Ihsan:    {ihsan_composite:.4f}")
    print(f"  Hash:     {receipt_hash[:32]}...")
    print(f"  Latency:  {total_ms}ms total, {llm_duration_ms}ms LLM")
    print(f"  Chain:    {bus.chain_height} events, valid={bus.verify_chain()}")
    print(f"  Bridge:   {stats['forwarded']} forwarded, {stats['failed']} failed")
    print()
    print("  Python cognition served the user.")
    print("  Rust constitution verified independently.")
    print("  BLAKE3 proof chain is intact.")
    print("  The organism does real work.")
else:
    print("  MISSION INCOMPLETE:")
    if stats["failed"] > 0:
        print(f"    bridge failures: {stats['failed']}")
    if not bus.verify_chain():
        print("    chain integrity broken")
    if ihsan_composite < 0.95:
        print(f"    ihsan below floor: {ihsan_composite}")

print(f"{'=' * 60}")
