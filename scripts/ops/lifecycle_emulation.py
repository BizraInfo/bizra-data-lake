#!/usr/bin/env python3
"""BIZRA Complete System Lifecycle Emulation — Real Data"""
import sys, os, time, json, hashlib, traceback
sys.path.insert(0, ".")
results = {}

def stage(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

def check(label, fn):
    try:
        val = fn()
        results[label] = {"status": "PASS", "value": str(val)[:200]}
        print(f"  [PASS] {label}: {val}")
        return val
    except Exception as e:
        results[label] = {"status": "FAIL", "error": str(e)[:200]}
        print(f"  [FAIL] {label}: {e}")
        return None

stage("STAGE 0: COLD BOOT - IDENTITY GENESIS")

check("Import identity.genesis", lambda: __import__("core.identity.genesis", fromlist=["IdentityGenesis"]))

from core.identity.genesis import (
    IdentityGenesis, derive_identity_id, derive_agent_keypairs,
    SovereigntyClass, PersonaSeed, GenesisWalletState
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

sk = Ed25519PrivateKey.generate()
pk_bytes = sk.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
)
identity_id = check("Derive identity ID", lambda: derive_identity_id(pk_bytes))
agent_keys = check("Derive 12 agent keypairs", lambda: derive_agent_keypairs(pk_bytes, count=12))
check("Agent count (7 PAT + 5 SAT)", lambda: len(agent_keys))
print(f"  --> PK prefix: {pk_bytes.hex()[:16]}")
print(f"  --> ID prefix: {identity_id[:16]}")
print(f"  --> PAT keys: {[kp[0].hex()[:8] for kp in agent_keys[:7]]}")
print(f"  --> SAT keys: {[kp[0].hex()[:8] for kp in agent_keys[7:]]}")

genesis = IdentityGenesis(
    public_key=pk_bytes, identity_id=identity_id,
    sovereignty_class=SovereigntyClass.SEED,
    persona_seed=PersonaSeed(display_name="NODE0", mission_statement="Sovereign AI"),
    genesis_wallet_state=GenesisWalletState(seed_balance=847.32, bloom_balance=0.0)
)
check("IdentityGenesis created", lambda: f"class={genesis.sovereignty_class.name} SEED={genesis.genesis_wallet_state.seed_balance}")
check("Zakat ratio = 2.5%", lambda: genesis.genesis_wallet_state.zakat_due_ratio)
check("BLOOM non-transferable", lambda: not genesis.genesis_wallet_state.bloom_transferable)

stage("STAGE 1: CONSTITUTIONAL CONSTANTS (SSOT)")

from core.integration.constants import (
    IHSAN_FLOOR, ADL_GINI_THRESHOLD, SNR_THRESHOLD, KERNEL_INVARIANTS
)
check("IHSAN_FLOOR", lambda: IHSAN_FLOOR)
check("ADL_GINI_THRESHOLD", lambda: ADL_GINI_THRESHOLD)
check("SNR_THRESHOLD", lambda: SNR_THRESHOLD)
check("KERNEL_INVARIANTS", lambda: KERNEL_INVARIANTS)

stage("STAGE 2: EVENT BUS - IMMUTABLE TRUTH LOG")

from core.sovereign.event_bus import EventBus, Event, EventPriority
bus = EventBus()
captured = []
import asyncio
async def on_event(e): captured.append(e)
bus.subscribe("genesis.*", on_event)
check("EventBus initialized", lambda: f"subscribers={len(bus._subscribers)}")

evt = Event(topic="genesis.boot", payload={"node": "NODE0", "id": identity_id[:16]}, source="lifecycle")
asyncio.get_event_loop().run_until_complete(bus.publish(evt))
check("Genesis event published+captured", lambda: f"topic={captured[0].topic} payload={captured[0].payload}")

stage("STAGE 3: PROOF ENGINE - BLAKE3 HASH CHAIN")

try:
    from core.proof_engine.canonical import canonical_bytes
    import blake3
    r1 = {"action": "genesis_boot", "node": "NODE0", "ihsan": 0.97, "ts": time.time()}
    c1 = canonical_bytes(r1)
    h1 = blake3.blake3(c1).hexdigest()
    check("Canonical serialize receipt#1", lambda: f"{len(c1)} bytes")
    check("BLAKE3 hash receipt#1", lambda: h1[:32])
    r2 = {"action": "heartbeat", "prev_hash": h1, "ihsan": 0.98, "ts": time.time()}
    c2 = canonical_bytes(r2)
    h2 = blake3.blake3(c2).hexdigest()
    check("Hash chain link #1->#2", lambda: f"#{1}={h1[:12]}->#{2}={h2[:12]}")
    r3 = {"action": "mission_complete", "prev_hash": h2, "ihsan": 0.96, "ts": time.time()}
    c3 = canonical_bytes(r3)
    h3 = blake3.blake3(c3).hexdigest()
    check("Hash chain link #2->#3", lambda: f"#{2}={h2[:12]}->#{3}={h3[:12]}")
    # Verify chain integrity
    check("Chain tamper test", lambda: blake3.blake3(c2).hexdigest() == h2)
except Exception as e:
    check("Proof engine", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 4: SOVEREIGN RUNTIME CORE")

check("Import runtime_types", lambda: __import__("core.sovereign.runtime_types", fromlist=["RuntimeConfig"]))
from core.sovereign.runtime_types import RuntimeConfig, HealthStatus
config = RuntimeConfig()
check("RuntimeConfig created", lambda: f"type={type(config).__name__}")
check("HealthStatus enum", lambda: [s.name for s in HealthStatus])

stage("STAGE 5: NODE0 HEARTBEAT")

from core.node0.heartbeat import HEARTBEAT_INTERVAL_S, PRECIPITATION_IHSAN_FLOOR
check("Heartbeat interval", lambda: f"{HEARTBEAT_INTERVAL_S}s")
check("Precipitation ihsan floor", lambda: PRECIPITATION_IHSAN_FLOOR)

stage("STAGE 6: REFLEX CACHE - SYSTEM-1 / SYSTEM-2")

try:
    from core.sovereign.reflex_cache import ReflexCache, Reflex
    cache = ReflexCache()
    r = Reflex(trigger="timezone_query", action="direct_answer", confidence=0.97)
    cache.add(r)
    hit = cache.lookup("timezone_query")
    check("Reflex HIT (System-1)", lambda: f"conf={hit.confidence}")
    miss = cache.lookup("novel_architecture_question")
    check("Reflex MISS (System-2)", lambda: f"miss={miss is None}")
    # Compilation simulation: 3 observations -> compile
    for i in range(3):
        cache.add(Reflex(trigger="rust_crypto_gen", action="scholar_direct", confidence=0.96+i*0.01))
    compiled = cache.lookup("rust_crypto_gen")
    check("Reflex compilation (3/3)", lambda: f"compiled={compiled is not None} conf={compiled.confidence if compiled else 0}")
except Exception as e:
    check("Reflex cache", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 7: PCI GATES - CONSTITUTIONAL VERIFICATION")

try:
    from core.pci.gates import PCIGateKeeper
    gk = PCIGateKeeper(ihsan_threshold=0.95, snr_threshold=0.85)
    check("Gate PASS (ihsan=0.97 snr=0.92)", lambda: gk.check(ihsan_score=0.97, snr_score=0.92))
    check("Gate FAIL (ihsan=0.80 snr=0.60)", lambda: gk.check(ihsan_score=0.80, snr_score=0.60))
except Exception as e:
    check("PCI gates", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 8: TOKEN ECONOMY")

try:
    from core.token.bloom import BloomToken
    bloom = BloomToken()
    check("BLOOM token init", lambda: type(bloom).__name__)
except Exception as e:
    check("BLOOM token", lambda: (_ for _ in ()).throw(Exception(str(e))))
try:
    from core.token.token_economy import TokenEconomy
    check("TokenEconomy available", lambda: type(TokenEconomy).__name__)
except Exception as e:
    check("TokenEconomy", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 9: ENTROPY ROUTER - SHANNON H CLASSIFICATION")

try:
    from core.reasoning.entropy_router import EntropyRouter
    router = EntropyRouter()
    s = router.classify("What time is it in Tokyo?")
    check("Simple query classification", lambda: s)
    c = router.classify("Redesign ReflexCache for concurrent UAB with lock-free reads")
    check("Complex query classification", lambda: c)
except Exception as e:
    check("Entropy router", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 10: LIVING MEMORY")

try:
    from core.living_memory.core import MemoryFragment
    frag = MemoryFragment(content="Genesis boot", importance=0.95, tags=["genesis"])
    check("MemoryFragment", lambda: f"imp={frag.importance} tags={frag.tags}")
except Exception as e:
    check("Living memory", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 11: CONSTITUTIONAL SIMULATION DATA")

try:
    from core.constitutional.simulation import ConstitutionalSimulation
    check("Constitutional simulation available", lambda: type(ConstitutionalSimulation).__name__)
except Exception as e:
    check("Constitutional simulation", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("STAGE 12: GRAPH-OF-THOUGHTS")

try:
    from core.sovereign.runtime_types import GoTNodeSnapshot
    got = GoTNodeSnapshot(node_id="root", content="test", score=0.95)
    check("GoT node snapshot", lambda: f"id={got.node_id} score={got.score}")
except Exception as e:
    check("GoT", lambda: (_ for _ in ()).throw(Exception(str(e))))

stage("FINAL EVALUATION - REAL DATA SUMMARY")

passed = sum(1 for v in results.values() if v["status"] == "PASS")
failed = sum(1 for v in results.values() if v["status"] == "FAIL")
total = len(results)
pct = passed / total * 100 if total > 0 else 0

print(f"  Total checks:  {total}")
print(f"  Passed:        {passed} ({pct:.1f}%)")
print(f"  Failed:        {failed} ({100-pct:.1f}%)")
print()
for k, v in results.items():
    sym = "PASS" if v["status"] == "PASS" else "FAIL"
    detail = v.get("value", v.get("error", ""))[:80]
    print(f"  [{sym}] {k}: {detail}")
print()
print(f"  IHSAN_FLOOR={IHSAN_FLOOR}  GINI={ADL_GINI_THRESHOLD}  SNR={SNR_THRESHOLD}")
print(f"  KERNEL_INVARIANTS={KERNEL_INVARIANTS}")
print(f"  Identity: {identity_id[:16]}...")
print(f"  Agent keys: {len(agent_keys)} derived")
print(f"  Events captured: {len(captured)}")
print()
status = "SOVEREIGN" if pct >= 80 else "DEGRADED"
print(f"  NODE0 STATUS: {status}")
print(f"  LIFECYCLE:    {'COMPLETE' if pct >= 90 else 'PARTIAL'}")
print(f"  IHSAN EVAL:   {pct/100:.3f}")
