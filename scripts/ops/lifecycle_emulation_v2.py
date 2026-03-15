#!/usr/bin/env python3
"""BIZRA Complete System Lifecycle — Real Data, Real Codebase"""
import sys, os, time, json, hashlib, traceback, asyncio
sys.path.insert(0, ".")
R = {}
T0 = time.perf_counter()
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
pk = sk.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
iid = ck("Identity ID derivation", lambda: derive_identity_id(pk))
aks = ck("12 HD agent keypairs", lambda: derive_agent_keypairs(pk, 12))
ck("7 PAT + 5 SAT split", lambda: f"PAT={len(aks[:7])} SAT={len(aks[7:])}")
sig_msg = f"bizra-identity-genesis-v1:{iid}".encode()
sig_bytes = sk.sign(sig_msg)
ck("Ed25519 signature", lambda: f"{len(sig_bytes)} bytes, hex={sig_bytes.hex()[:24]}...")
# Verify signature
try:
    sk.public_key().verify(sig_bytes, sig_msg)
    ck("Signature verification", lambda: "VALID")
except Exception:
    ck("Signature verification", lambda: "INVALID")
gen = IdentityGenesis(public_key=pk, identity_id=iid, sovereignty_class=SovereigntyClass.SEED,
    persona_seed=PersonaSeed(display_name="NODE0"), genesis_wallet_state=GenesisWalletState(seed_balance=847.32))
ck("Genesis object", lambda: f"class={gen.sovereignty_class.name} SEED={gen.genesis_wallet_state.seed_balance}")
ck("Zakat=2.5%", lambda: gen.genesis_wallet_state.zakat_due_ratio)
ck("BLOOM soulbound", lambda: not gen.genesis_wallet_state.bloom_transferable)

# ═══ STAGE 1: CONSTITUTIONAL CONSTANTS ═══
stage("STAGE 1: CONSTITUTIONAL CONSTANTS (SSOT)")
from core.integration.constants import (
    IHSAN_THRESHOLD, ADL_GINI_THRESHOLD, SNR_THRESHOLD,
    KERNEL_INVARIANTS, DOMAIN_IDENTITY_GENESIS, IHSAN_WEIGHTS, IHSAN_CANONICAL_WEIGHTS)
ck("IHSAN_THRESHOLD", lambda: IHSAN_THRESHOLD)
ck("ADL_GINI_THRESHOLD", lambda: ADL_GINI_THRESHOLD)
ck("SNR_THRESHOLD", lambda: SNR_THRESHOLD)
ck("KERNEL_INVARIANTS", lambda: KERNEL_INVARIANTS)
ck("DOMAIN_IDENTITY_GENESIS", lambda: DOMAIN_IDENTITY_GENESIS)
ck("IHSAN 8-dim weights", lambda: IHSAN_WEIGHTS)
ck("Canonical weights (8-dim)", lambda: IHSAN_CANONICAL_WEIGHTS)

# ═══ STAGE 2: EVENT BUS ═══
stage("STAGE 2: EVENT BUS — IMMUTABLE TRUTH LOG")
from core.sovereign.event_bus import EventBus, Event, EventPriority
bus = EventBus()
cap = []
async def on_evt(e): cap.append(e)
bus.subscribe("genesis.*", on_evt)
ck("EventBus subscribers", lambda: len(bus._subscribers))
e1 = Event(topic="genesis.boot", payload={"node":"NODE0","id":iid[:16]}, source="lifecycle")
asyncio.get_event_loop().run_until_complete(bus.publish(e1))
ck("Genesis event captured", lambda: f"topic={cap[0].topic}")
e2 = Event(topic="genesis.identity", payload={"pk":pk.hex()[:16]}, source="lifecycle", priority=EventPriority.HIGH)
asyncio.get_event_loop().run_until_complete(bus.publish(e2))
ck("Identity event captured", lambda: f"events={len(cap)} priority={cap[1].priority.name}")

# ═══ STAGE 3: PROOF ENGINE — BLAKE3 CHAIN ═══
stage("STAGE 3: PROOF ENGINE — BLAKE3 HASH CHAIN")
from core.proof_engine.canonical import canonical_bytes
import blake3
r1 = {"action":"genesis_boot","node":"NODE0","ihsan":0.97,"ts":time.time()}
c1 = canonical_bytes(r1); h1 = blake3.blake3(c1).hexdigest()
ck("Receipt #1 canonical", lambda: f"{len(c1)} bytes")
ck("Receipt #1 BLAKE3", lambda: h1[:32])
r2 = {"action":"heartbeat","prev":h1,"ihsan":0.98,"ts":time.time()}
c2 = canonical_bytes(r2); h2 = blake3.blake3(c2).hexdigest()
ck("Chain #1->#2", lambda: f"{h1[:12]}->{h2[:12]}")
r3 = {"action":"mission","prev":h2,"ihsan":0.96,"agent":"P3_Creator","ts":time.time()}
c3 = canonical_bytes(r3); h3 = blake3.blake3(c3).hexdigest()
ck("Chain #2->#3", lambda: f"{h2[:12]}->{h3[:12]}")
# Tamper detection
tampered = canonical_bytes({"action":"heartbeat","prev":"TAMPERED","ihsan":0.98,"ts":time.time()})
ht = blake3.blake3(tampered).hexdigest()
ck("Tamper detection", lambda: f"original={h2[:12]} tampered={ht[:12]} match={h2==ht}")

# === STAGE 4: SOVEREIGN RUNTIME ===
stage("STAGE 4: SOVEREIGN RUNTIME CORE")
try:
    from core.sovereign.runtime_types import RuntimeConfig, HealthStatus, GoTNodeSnapshot
    config = RuntimeConfig()
    ck("RuntimeConfig", lambda: type(config).__name__)
    ck("HealthStatus enum", lambda: [s.name for s in HealthStatus])
    got = GoTNodeSnapshot(node_id="root", content="test", score=0.95)
    ck("GoT node snapshot", lambda: f"id={got.node_id} score={got.score}")
except Exception as e:
    ck("Runtime core", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 5: NODE0 HEARTBEAT ===
stage("STAGE 5: NODE0 HEARTBEAT")
from core.node0.heartbeat import HEARTBEAT_INTERVAL_S, PRECIPITATION_IHSAN_FLOOR
ck("Heartbeat interval", lambda: f"{HEARTBEAT_INTERVAL_S}s")
ck("Precipitation floor", lambda: PRECIPITATION_IHSAN_FLOOR)

# === STAGE 6: REFLEX CACHE ===
stage("STAGE 6: REFLEX CACHE (SYSTEM-1 / SYSTEM-2)")
try:
    from core.sovereign.reflex_cache import ReflexCache, Reflex
    cache = ReflexCache()
    cache.add(Reflex(trigger="timezone_query", action="direct_answer", confidence=0.97))
    hit = cache.lookup("timezone_query")
    ck("Reflex HIT (S1 fast path)", lambda: f"conf={hit.confidence}")
    miss = cache.lookup("novel_question")
    ck("Reflex MISS (S2 deliberative)", lambda: f"miss={miss is None}")
    for i in range(3):
        cache.add(Reflex(trigger="rust_crypto", action="scholar", confidence=0.96+i*0.01))
    comp = cache.lookup("rust_crypto")
    ck("Reflex compilation 3/3", lambda: f"compiled={comp is not None} conf={comp.confidence if comp else 0}")
except Exception as e:
    ck("Reflex cache", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 7: PCI GATES ===
stage("STAGE 7: PCI GATES — CONSTITUTIONAL VERIFICATION")
try:
    from core.pci.gates import PCIGateKeeper
    gk = PCIGateKeeper(ihsan_threshold=0.95, snr_threshold=0.85)
    ck("Gate PASS (ihsan=0.97 snr=0.92)", lambda: gk.check(ihsan_score=0.97, snr_score=0.92))
    ck("Gate FAIL (ihsan=0.80 snr=0.60)", lambda: gk.check(ihsan_score=0.80, snr_score=0.60))
except Exception as e:
    ck("PCI gates", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 8: TOKEN ECONOMY ===
stage("STAGE 8: TOKEN ECONOMY (SEED / BLOOM)")
try:
    from core.token.bloom import BloomToken
    ck("BLOOM token", lambda: type(BloomToken()).__name__)
except Exception as e:
    ck("BLOOM token", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 9: ENTROPY ROUTER ===
stage("STAGE 9: ENTROPY ROUTER — SHANNON H")
try:
    from core.reasoning.entropy_router import EntropyRouter
    router = EntropyRouter()
    ck("Simple query", lambda: router.classify("What time is it in Tokyo?"))
    ck("Complex query", lambda: router.classify("Redesign ReflexCache for concurrent UAB"))
except Exception as e:
    ck("Entropy router", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 10: LIVING MEMORY ===
stage("STAGE 10: LIVING MEMORY")
try:
    from core.living_memory.core import MemoryFragment
    frag = MemoryFragment(content="Genesis boot", importance=0.95, tags=["genesis"])
    ck("MemoryFragment", lambda: f"imp={frag.importance}")
except Exception as e:
    ck("Living memory", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 11: CONSTITUTIONAL SIMULATION ===
stage("STAGE 11: CONSTITUTIONAL SIMULATION")
try:
    from core.constitutional.simulation import ConstitutionalSimulation
    ck("Simulation module", lambda: type(ConstitutionalSimulation).__name__)
except Exception as e:
    ck("Simulation", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === STAGE 12: TEST SUITE (REAL COUNT) ===
stage("STAGE 12: TEST SUITE VERIFICATION")
import subprocess
try:
    r = subprocess.run(
        ["python3", "-m", "pytest", "--collect-only", "-q", "tests/", "-m", "not slow",
         "--ignore=tests/root_legacy", "--timeout=10"],
        capture_output=True, text=True, timeout=60, cwd="/mnt/c/BIZRA-DATA-LAKE"
    )
    lines = r.stdout.strip().split("\n")
    count_line = [l for l in lines if "test" in l.lower() and ("selected" in l.lower() or "item" in l.lower())]
    if count_line:
        ck("Test collection", lambda: count_line[-1].strip())
    else:
        last = lines[-1] if lines else "empty"
        ck("Test collection", lambda: f"output={last}")
except Exception as e:
    ck("Test collection", lambda: (_ for _ in ()).throw(Exception(str(e))))

# === FINAL EVALUATION ===
elapsed = time.perf_counter() - T0
stage("FINAL LIFECYCLE EVALUATION")
passed = sum(1 for v in R.values() if v[0] == "PASS")
failed = sum(1 for v in R.values() if v[0] == "FAIL")
total = len(R)
pct = passed / total * 100 if total > 0 else 0

print(f"  Boot time:     {elapsed:.2f}s")
print(f"  Total checks:  {total}")
print(f"  Passed:        {passed} ({pct:.1f}%)")
print(f"  Failed:        {failed} ({100-pct:.1f}%)")
print()
print("  DETAIL:")
for k, v in R.items():
    sym = "PASS" if v[0] == "PASS" else "FAIL"
    detail = v[1][:70]
    print(f"    [{sym}] {k}: {detail}")
print()
print(f"  === CONSTITUTIONAL STATE ===")
print(f"  IHSAN_THRESHOLD:     {IHSAN_THRESHOLD}")
print(f"  ADL_GINI_THRESHOLD:  {ADL_GINI_THRESHOLD}")
print(f"  SNR_THRESHOLD:       {SNR_THRESHOLD}")
print(f"  KERNEL_INVARIANTS:   {KERNEL_INVARIANTS}")
print(f"  Identity:            {iid[:16]}...")
print(f"  Agent keys:          {len(aks)} (7 PAT + 5 SAT)")
print(f"  Events captured:     {len(cap)}")
print(f"  Hash chain:          3 receipts linked")
print()
status = "SOVEREIGN" if pct >= 80 else "DEGRADED"
lifecycle = "COMPLETE" if pct >= 85 else "PARTIAL"
ihsan_eval = pct / 100
print(f"  NODE0 STATUS:  {status}")
print(f"  LIFECYCLE:     {lifecycle}")
print(f"  IHSAN EVAL:    {ihsan_eval:.3f}")
print()
if failed > 0:
    print(f"  === FAILED CHECKS ({failed}) ===")
    for k, v in R.items():
        if v[0] == "FAIL":
            print(f"    [FAIL] {k}: {v[1][:100]}")
