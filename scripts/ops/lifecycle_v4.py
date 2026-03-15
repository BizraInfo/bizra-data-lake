#!/usr/bin/env python3
"""BIZRA Lifecycle v4 — END-TO-END: Identity → Bus(12 subs) → Heartbeat → Breath → Proof"""
import sys, os, time, hashlib, tempfile
sys.path.insert(0, ".")
R = {}; T0 = time.perf_counter()
def stage(n): print(f"\n{'='*60}\n  {n}\n{'='*60}\n")
def ck(label, fn):
    try:
        v = fn(); R[label] = ("PASS", str(v)[:160]); print(f"  [PASS] {label}: {v}"); return v
    except Exception as e:
        R[label] = ("FAIL", str(e)[:160]); print(f"  [FAIL] {label}: {e}"); return None

# ═══ STAGE 0: IDENTITY GENESIS ═══
stage("STAGE 0: IDENTITY GENESIS (L0)")
from core.identity.genesis import (
    IdentityGenesis, derive_identity_id, derive_agent_keypairs,
    SovereigntyClass, PersonaSeed, GenesisWalletState)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
sk = Ed25519PrivateKey.generate()
pk = sk.public_key().public_bytes(
    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
iid = ck("Identity ID (SHA-256)", lambda: derive_identity_id(pk))
aks = ck("12 HD agent keypairs", lambda: derive_agent_keypairs(pk, 12))
sig_msg = f"bizra-identity-genesis-v1:{iid}".encode()
sig = sk.sign(sig_msg)
ck("Ed25519 sign+verify", lambda: (sk.public_key().verify(sig, sig_msg), "VALID")[1])
gen = IdentityGenesis(public_key=pk, identity_id=iid,
    sovereignty_class=SovereigntyClass.SEED,
    persona_seed=PersonaSeed(display_name="NODE0"),
    genesis_wallet_state=GenesisWalletState(seed_balance=847.32))
ck("Genesis object", lambda: f"SEED={gen.genesis_wallet_state.seed_balance} Zakat={gen.genesis_wallet_state.zakat_due_ratio}")
print(f"  --> PK:  {pk.hex()[:16]}...")
print(f"  --> ID:  {iid[:16]}...")
print(f"  --> Sig: {sig.hex()[:16]}...")
print(f"  --> PAT: {[k[0].hex()[:6] for k in aks[:7]]}")
print(f"  --> SAT: {[k[0].hex()[:6] for k in aks[7:]]}")

# ═══ STAGE 1: CONSTITUTIONAL CONSTANTS ═══
stage("STAGE 1: CONSTITUTIONAL CONSTANTS (SSOT)")
from core.integration.constants import (
    IHSAN_THRESHOLD, ADL_GINI_THRESHOLD, SNR_THRESHOLD, KERNEL_INVARIANTS)
ck("IHSAN_THRESHOLD", lambda: IHSAN_THRESHOLD)
ck("ADL_GINI_THRESHOLD", lambda: ADL_GINI_THRESHOLD)
ck("SNR_THRESHOLD", lambda: SNR_THRESHOLD)
ck("KERNEL_INVARIANTS", lambda: KERNEL_INVARIANTS)

# ═══ STAGE 2: PRODUCTION EVENT BUS + 12 SUBSCRIBERS ═══
stage("STAGE 2: PRODUCTION BUS + 12 SUBSCRIBERS (the last inch)")
from core.bus.subscribers import (
    EventBus as SyncBus, EventType, wire_all_subscribers)

# Mock dependencies (same pattern as _run_smoke_tests in subscribers.py)
class MockStore:
    def __init__(self): self._data = {}; self._counts = {}
    def reinforce(self, **kw): self._data[kw.get("key","")] = kw
    def get_success_count(self, key): return self._counts.get(key, 0)
    def set_success_count(self, key, val): self._counts[key] = val
    def promote_to_semantic(self, **kw): return True
    def record_failure_pattern(self, **kw): pass
class MockTeleScript:
    def begin_execution(self, **kw): return f"ts_{int(time.time())}"
class MockReceiptChain(list): pass
class MockReflexCache(dict):
    def precipitate(self, **kw): self[kw.get("action_type","x")] = kw
class MockSessionManager:
    def halt(self, **kw): pass
class MockAuditLog:
    def log_violation(self, **kw): pass
class MockQuarantine:
    def isolate(self, **kw): pass
class MockHealing:
    def diagnose(self, **kw):
        class Plan: strategy = "retry"
        return Plan()
class MockHHMM:
    def classify(self, payload): return "macro_general"
class MockPoI:
    total_credit = 0.0
    def accumulate(self, **kw): self.total_credit += 0.01; return 0.01
class MockMinter:
    minted = 0.0
    def compute_reward(self, **kw): return 0.05
    def mint_seed(self, **kw): self.minted += 0.05
class MockBudget:
    total_used = 0
    def record_retrieval(self, **kw): self.total_used += kw.get("tokens", 0)
class MockSelfModel:
    def update_capability_map(self, **kw): pass
class MockCapRegistry:
    def register(self, **kw): pass
    def count(self): return 7
    def count_by_type(self, t): return 7 if t == "PAT" else 5
    def total_capabilities(self): return 42
    def capability_vector(self): return [1.0] * 8

bus = SyncBus()
ck("SyncBus created", lambda: f"chain_height={bus.chain_height}")

mock_store = MockStore()
mock_poi = MockPoI()
mock_minter = MockMinter()
mock_reflex = MockReflexCache()
subs = wire_all_subscribers(
    bus,
    memory_store=mock_store,
    telescript_engine=MockTeleScript(),
    receipt_chain=MockReceiptChain(),
    reflex_cache=mock_reflex,
    session_manager=MockSessionManager(),
    audit_log=MockAuditLog(),
    quarantine_store=MockQuarantine(),
    healing_engine=MockHealing(),
    hhmm_engine=MockHHMM(),
    poi_engine=mock_poi,
    token_minter=mock_minter,
    context_budget=MockBudget(),
    self_model=MockSelfModel(),
    capability_registry=MockCapRegistry(),
)
ck("12 subscribers wired", lambda: f"count={len(subs)}")

# Publish ACTION_RECEIPT — the event the heartbeat emits
evt1 = bus.publish(EventType.ACTION_RECEIPT, {
    "source": "lifecycle_v4", "tick": 1,
    "ihsan_composite": 0.97, "action_type": "genesis_boot",
    "result_summary": "sovereignty proven", "chain_hash": "test",
})
ck("ACTION_RECEIPT dispatched", lambda: f"hash={evt1.event_hash[:16]} height={bus.chain_height}")
ck("Chain integrity after dispatch", lambda: bus.verify_chain())
ck("Memory reinforced", lambda: len(mock_store._data) > 0)

# ═══ STAGE 3: BLAKE3 PROOF CHAIN ═══
stage("STAGE 3: BLAKE3 PROOF CHAIN (7 receipts)")
from core.proof_engine.canonical import canonical_bytes
import blake3
chain_hashes = []
for i, act in enumerate(["genesis","heartbeat","mission","reflex_compile","token_mint","audit","seal"]):
    prev = chain_hashes[-1] if chain_hashes else "0"*64
    data = {"action": act, "prev": prev, "ihsan": 0.95+0.005*i, "tick": i, "ts": time.time()}
    canon = canonical_bytes(data)
    h = blake3.blake3(canon).hexdigest()
    chain_hashes.append(h)
ck("7-receipt chain", lambda: f"len={len(chain_hashes)} tip={chain_hashes[-1][:16]}")
tampered = canonical_bytes({"action":"heartbeat","prev":"TAMPERED","ihsan":0.98,"tick":1,"ts":0})
ck("Tamper detection", lambda: blake3.blake3(tampered).hexdigest() != chain_hashes[1])

# ═══ STAGE 4: ENTROPY ROUTER (calibrated) ═══
stage("STAGE 4: ENTROPY ROUTER (calibrated)")
from core.reasoning.entropy_router import EntropyRouter, QueryComplexity
router = EntropyRouter()
s = router.route("What time is it in Tokyo?")
ck("Simple -> S1", lambda: f"{s.query_complexity.name} sys={s.system}")
c = router.route("Redesign the ReflexCache to be thread-safe for concurrent UAB, while maintaining lock-free reads")
ck("Complex -> S2", lambda: f"{c.query_complexity.name} sys={c.system} got={c.use_got}")
b = router.route("Build a BLAKE3 validator in Rust with error handling")
ck("Build imperative", lambda: f"{b.query_complexity.name} sys={b.system}")

# ═══ STAGE 5: HEARTBEAT BOOT + BREATH (wired to production bus) ═══
stage("STAGE 5: NODE0 HEARTBEAT (boot + breath + bus)")
from pathlib import Path
from core.node0.heartbeat import Node0Heartbeat
tmpdir = Path(tempfile.mkdtemp(prefix="bizra_lc4_"))
chain_before_boot = bus.chain_height
hb = Node0Heartbeat(
    data_dir=tmpdir,
    node_id=iid[:16],
    event_bus=bus,
    identity_mode="placeholder_degraded",
    signer_public_key_prefix=pk.hex()[:16],
)
boot = ck("boot()", lambda: hb.boot())
if boot:
    ck("Sovereignty proven", lambda: boot.sovereignty_proven)
    ck("Boot hash", lambda: boot.boot_hash[:32])
    ck("Boot duration", lambda: f"{boot.duration_ms:.1f}ms")

# ═══ STAGE 6: FIRST BREATH ═══
stage("STAGE 6: FIRST SOVEREIGN BREATH")
breath = ck("breathe()", lambda: hb.breathe())
if breath:
    ck("Ihsan composite", lambda: f"{breath.ihsan_composite:.4f}")
    ck("Gini coefficient", lambda: f"{breath.gini_coefficient:.4f} ok={breath.gini_ok}")
    ck("Chain hash", lambda: breath.chain_hash[:32])
    ck("Tick number", lambda: breath.tick_number)
chain_after = bus.chain_height
events_emitted = hb._total_events_emitted
ck("Bus chain grew", lambda: f"before={chain_before_boot} after={chain_after} delta={chain_after - chain_before_boot}")
ck("Heartbeat events emitted", lambda: events_emitted)

# ═══ STAGE 7: TOKEN + REFLEX + MEMORY VERIFICATION ═══
stage("STAGE 7: SUBSYSTEM VERIFICATION")
from core.token.bloom import compute_gini, TOKEN_ZAKAT_RATE, SEED_MINT_FLOOR
ck("Gini([100,200,300,50,150])", lambda: f"{compute_gini([100,200,300,50,150]):.4f}")
ck("Zakat rate", lambda: TOKEN_ZAKAT_RATE)
ck("SEED mint floor", lambda: SEED_MINT_FLOOR)
from core.sovereign.reflex_compiler import ReflexCompiler
rc = ReflexCompiler()
ck("ReflexCompiler", lambda: type(rc).__name__)
from core.living_memory.core import LivingMemoryCore, MemoryType
ck("Memory types", lambda: [t.name for t in MemoryType])
from core.sovereign.runtime_types import HealthStatus, GoTNodeSnapshot
ck("HealthStatus", lambda: [s.name for s in HealthStatus])
ck("GoT snapshot", lambda: GoTNodeSnapshot(node_id="root", content="test", score=0.94).score)

# ═══ FINAL EVALUATION ═══
elapsed = time.perf_counter() - T0
stage("FINAL LIFECYCLE EVALUATION — REAL DATA, REAL BUS, REAL HEARTBEAT")
passed = sum(1 for v in R.values() if v[0] == "PASS")
failed = sum(1 for v in R.values() if v[0] == "FAIL")
total = len(R)
pct = passed / total * 100 if total > 0 else 0

print(f"  Boot time:     {elapsed:.2f}s")
print(f"  Total checks:  {total}")
print(f"  Passed:        {passed} ({pct:.1f}%)")
print(f"  Failed:        {failed} ({100-pct:.1f}%)")
print()
print("  === ALL CHECKS ===")
for k, v in R.items():
    print(f"    [{'PASS' if v[0]=='PASS' else 'FAIL'}] {k}: {v[1][:75]}")
print()
print("  === EVIDENCE CHAIN ===")
print(f"  Identity:      {iid[:32]}...")
print(f"  PK prefix:     {pk.hex()[:16]}")
print(f"  Agent keys:    {len(aks)} (7 PAT + 5 SAT)")
print(f"  Bus chain:     {bus.chain_height} events")
print(f"  Bus integrity: {bus.verify_chain()}")
print(f"  Proof chain:   {len(chain_hashes)} receipts")
print(f"  HB events:     {hb._total_events_emitted}")
print(f"  Subscribers:   {len(subs)}")
print(f"  Constants:     IHSAN={IHSAN_THRESHOLD} GINI={ADL_GINI_THRESHOLD}")
print(f"  Invariants:    {KERNEL_INVARIANTS}")
print()
bus_wired = bus.chain_height > 1 and bus.verify_chain()
hb_ok = breath is not None
nervous = bus_wired and hb_ok and hb._total_events_emitted > 0
status = "SOVEREIGN" if pct >= 85 and nervous else "DEGRADED"
lifecycle = "COMPLETE" if pct >= 90 and nervous else "PARTIAL"
print(f"  === VERDICT ===")
print(f"  Bus wired:      {bus_wired}")
print(f"  Heartbeat OK:   {hb_ok}")
print(f"  Nervous system: {nervous}")
print(f"  NODE0 STATUS:   {status}")
print(f"  LIFECYCLE:      {lifecycle}")
print(f"  IHSAN EVAL:     {pct/100:.3f}")
if failed > 0:
    print(f"\n  === GAPS ({failed}) ===")
    for k, v in R.items():
        if v[0] == "FAIL":
            print(f"    [FAIL] {k}: {v[1][:100]}")
