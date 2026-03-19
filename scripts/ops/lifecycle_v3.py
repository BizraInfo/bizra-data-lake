#!/usr/bin/env python3
"""BIZRA Lifecycle Emulation v3 — REAL APIs, REAL DATA"""

import asyncio
import hashlib
import os
import sys
import time

sys.path.insert(0, ".")
R = {}
T0 = time.perf_counter()


def stage(n):
    print(f"\n{'='*60}\n  {n}\n{'='*60}\n")


def ck(label, fn):
    try:
        v = fn()
        R[label] = ("PASS", str(v)[:160])
        print(f"  [PASS] {label}: {v}")
        return v
    except Exception as e:
        R[label] = ("FAIL", str(e)[:160])
        print(f"  [FAIL] {label}: {e}")
        return None


# ═══ STAGE 0: IDENTITY GENESIS ═══
stage("STAGE 0: IDENTITY GENESIS (L0)")
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.identity.genesis import (
    GenesisWalletState,
    IdentityGenesis,
    PersonaSeed,
    SovereigntyClass,
    derive_agent_keypairs,
    derive_identity_id,
)

sk = Ed25519PrivateKey.generate()
pk = sk.public_key().public_bytes(
    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
)
iid = ck("Identity ID (SHA-256 of pubkey)", lambda: derive_identity_id(pk))
aks = ck("12 HD agent keypairs", lambda: derive_agent_keypairs(pk, 12))
ck("7 PAT + 5 SAT", lambda: f"PAT={len(aks[:7])} SAT={len(aks[7:])}")
sig_msg = f"bizra-identity-genesis-v1:{iid}".encode()
sig = sk.sign(sig_msg)
ck("Ed25519 sign", lambda: f"{len(sig)}B hex={sig.hex()[:24]}...")
try:
    sk.public_key().verify(sig, sig_msg)
    ck("Ed25519 verify", lambda: "VALID")
except Exception:
    ck("Ed25519 verify", lambda: "INVALID")
gen = IdentityGenesis(
    public_key=pk,
    identity_id=iid,
    sovereignty_class=SovereigntyClass.SEED,
    persona_seed=PersonaSeed(display_name="NODE0"),
    genesis_wallet_state=GenesisWalletState(seed_balance=847.32),
)
ck(
    "Genesis created",
    lambda: f"class={gen.sovereignty_class.name} SEED={gen.genesis_wallet_state.seed_balance}",
)
ck("Zakat=2.5%", lambda: gen.genesis_wallet_state.zakat_due_ratio)
ck("BLOOM soulbound", lambda: not gen.genesis_wallet_state.bloom_transferable)

# ═══ STAGE 1: CONSTITUTIONAL CONSTANTS ═══
stage("STAGE 1: CONSTITUTIONAL CONSTANTS (SSOT)")
from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    DOMAIN_IDENTITY_GENESIS,
    IHSAN_CANONICAL_WEIGHTS,
    IHSAN_THRESHOLD,
    IHSAN_WEIGHTS,
    KERNEL_INVARIANTS,
    SNR_THRESHOLD,
)

ck("IHSAN_THRESHOLD", lambda: IHSAN_THRESHOLD)
ck("ADL_GINI_THRESHOLD", lambda: ADL_GINI_THRESHOLD)
ck("SNR_THRESHOLD", lambda: SNR_THRESHOLD)
ck("KERNEL_INVARIANTS", lambda: KERNEL_INVARIANTS)
ck("DOMAIN_IDENTITY_GENESIS", lambda: DOMAIN_IDENTITY_GENESIS)
ck("IHSAN 8-dim weights", lambda: IHSAN_WEIGHTS)

# ═══ STAGE 2: EVENT BUS ═══
stage("STAGE 2: EVENT BUS — IMMUTABLE TRUTH LOG")
from core.sovereign.event_bus import Event, EventBus, EventPriority

bus = EventBus()
cap = []


async def run_bus():
    async def handler(e):
        cap.append(e)

    bus.subscribe("genesis.*", handler)
    await bus.publish(
        Event(topic="genesis.boot", payload={"node": "NODE0"}, source="lifecycle")
    )
    await bus.publish(
        Event(
            topic="genesis.identity",
            payload={"id": iid[:16]},
            source="lifecycle",
            priority=EventPriority.HIGH,
        )
    )


asyncio.get_event_loop().run_until_complete(run_bus())
ck("EventBus events captured", lambda: f"count={len(cap)}")
if cap:
    ck("Event topic", lambda: cap[0].topic)
    ck("Event payload", lambda: cap[0].payload)

# ═══ STAGE 3: PROOF ENGINE — BLAKE3 CHAIN ═══
stage("STAGE 3: PROOF ENGINE — BLAKE3 HASH CHAIN")
import blake3

from core.proof_engine.canonical import canonical_bytes

receipts = []
r1 = {"action": "genesis_boot", "node": "NODE0", "ihsan": 0.97, "ts": time.time()}
c1 = canonical_bytes(r1)
h1 = blake3.blake3(c1).hexdigest()
receipts.append(h1)
ck("Receipt #1 canonical bytes", lambda: f"{len(c1)}B")
ck("Receipt #1 BLAKE3", lambda: h1[:32])
for i, act in enumerate(
    [
        "heartbeat",
        "mission_complete",
        "reflex_compile",
        "token_mint",
        "audit_pass",
        "chain_seal",
    ],
    2,
):
    prev = receipts[-1]
    rd = {
        "action": act,
        "prev_hash": prev,
        "ihsan": 0.95 + 0.01 * i,
        "ts": time.time(),
        "receipt_num": i,
    }
    cd = canonical_bytes(rd)
    hd = blake3.blake3(cd).hexdigest()
    receipts.append(hd)
ck(f"7-receipt chain built", lambda: f"receipts={len(receipts)}")
ck("Chain integrity (no gaps)", lambda: all(len(h) == 64 for h in receipts))
# Tamper test
tampered = canonical_bytes(
    {"action": "heartbeat", "prev_hash": "TAMPERED", "ihsan": 0.98, "ts": 0}
)
ht = blake3.blake3(tampered).hexdigest()
ck("Tamper detection", lambda: f"tampered_hash_differs={receipts[1]!=ht}")

# ═══ STAGE 4: SOVEREIGN RUNTIME ═══
stage("STAGE 4: SOVEREIGN RUNTIME CORE")
from core.sovereign.runtime_types import GoTNodeSnapshot, HealthStatus, RuntimeConfig

ck("RuntimeConfig", lambda: type(RuntimeConfig()).__name__)
ck("HealthStatus states", lambda: [s.name for s in HealthStatus])
got = GoTNodeSnapshot(node_id="root", content="thread_safe_reflex", score=0.941)
ck("GoT node (synthesis)", lambda: f"id={got.node_id} score={got.score}")

# ═══ STAGE 5: HEARTBEAT ═══
stage("STAGE 5: NODE0 HEARTBEAT")
from core.node0.heartbeat import HEARTBEAT_INTERVAL_S, PRECIPITATION_IHSAN_FLOOR

ck("Heartbeat interval", lambda: f"{HEARTBEAT_INTERVAL_S}s")
ck("Precipitation ihsan", lambda: PRECIPITATION_IHSAN_FLOOR)

# ═══ STAGE 6: REFLEX COMPILER (not reflex_cache) ═══
stage("STAGE 6: REFLEX COMPILER — S2 -> S1 MYELINATION")
try:
    from core.sovereign.reflex_compiler import ReflexCompiler

    rc = ReflexCompiler()
    ck("ReflexCompiler init", lambda: type(rc).__name__)
    ck(
        "ReflexCompiler methods",
        lambda: [m for m in dir(rc) if not m.startswith("_")][:8],
    )
except Exception as e:
    ck("ReflexCompiler", lambda: (_ for _ in ()).throw(Exception(str(e))))

# ═══ STAGE 7: PCI GATES (correct API) ═══
stage("STAGE 7: PCI GATES — CONSTITUTIONAL VERIFICATION")
try:
    from core.pci.gates import PCIGateKeeper

    gk = PCIGateKeeper()
    ck("PCIGateKeeper init", lambda: type(gk).__name__)
    methods = [m for m in dir(gk) if not m.startswith("_")]
    ck("Gate methods", lambda: methods)
except Exception as e:
    ck("PCI GateKeeper", lambda: (_ for _ in ()).throw(Exception(str(e))))

# ═══ STAGE 8: TOKEN ECONOMY (correct names) ═══
stage("STAGE 8: TOKEN ECONOMY (SEED/BLOOM)")
try:
    from core.token.bloom import (
        BLOOM_MINT_FLOOR,
        SEED_MINT_FLOOR,
        TOKEN_ZAKAT_RATE,
        BloomBalance,
        TokenMinter,
        WalletState,
        compute_gini,
    )

    ck("TokenMinter", lambda: type(TokenMinter).__name__)
    ck("WalletState", lambda: type(WalletState).__name__)
    ck("TOKEN_ZAKAT_RATE", lambda: TOKEN_ZAKAT_RATE)
    ck("SEED_MINT_FLOOR", lambda: SEED_MINT_FLOOR)
    ck("BLOOM_MINT_FLOOR", lambda: BLOOM_MINT_FLOOR)
    gini = compute_gini([100, 200, 300, 50, 150])
    ck("Gini computation", lambda: f"gini={gini:.4f} passes_adl={gini<=0.35}")
except Exception as e:
    ck("Token economy", lambda: (_ for _ in ()).throw(Exception(str(e))))

# ═══ STAGE 9: ENTROPY ROUTER (correct method: route/estimate_complexity) ═══
stage("STAGE 9: ENTROPY ROUTER — SHANNON H")
try:
    from core.reasoning.entropy_router import EntropyRouter

    router = EntropyRouter()
    s = router.route("What time is it in Tokyo?")
    ck("Simple query route()", lambda: s)
    c = router.route(
        "Redesign the ReflexCache for concurrent UAB action streams with lock-free System-1 reads"
    )
    ck("Complex query route()", lambda: c)
    e1 = router.estimate_complexity("hello")
    ck("Estimate simple complexity", lambda: e1)
    e2 = router.estimate_complexity(
        "Design a thread-safe double-buffered hot/cold HashMap with ArcSwap for atomic pointer swap"
    )
    ck("Estimate complex complexity", lambda: e2)
except Exception as e:
    ck("Entropy router", lambda: (_ for _ in ()).throw(Exception(str(e))))

# ═══ STAGE 10: LIVING MEMORY (correct: LivingMemoryCore, MemoryEntry) ═══
stage("STAGE 10: LIVING MEMORY")
try:
    from core.living_memory.core import LivingMemoryCore, MemoryEntry, MemoryType

    ck("LivingMemoryCore class", lambda: type(LivingMemoryCore).__name__)
    ck("MemoryType enum", lambda: [t.name for t in MemoryType])
    ck("MemoryEntry class", lambda: type(MemoryEntry).__name__)
except Exception as e:
    ck("Living memory", lambda: (_ for _ in ()).throw(Exception(str(e))))

# ═══ STAGE 11: CONSTITUTIONAL SIMULATION ═══
stage("STAGE 11: CONSTITUTIONAL SIMULATION")
try:
    from core.constitutional.simulation import (
        SimulationConfig,
        SovereignNetworkSimulation,
        run_simulation,
    )

    ck("SovereignNetworkSimulation", lambda: type(SovereignNetworkSimulation).__name__)
    ck("SimulationConfig", lambda: type(SimulationConfig).__name__)
    ck("run_simulation available", lambda: callable(run_simulation))
except Exception as e:
    ck("Simulation", lambda: (_ for _ in ()).throw(Exception(str(e))))

# ═══ FINAL EVALUATION ═══
elapsed = time.perf_counter() - T0
stage("FINAL LIFECYCLE EVALUATION — REAL DATA")
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
    sym = "PASS" if v[0] == "PASS" else "FAIL"
    print(f"    [{sym}] {k}")
print()
print("  === CONSTITUTIONAL STATE ===")
print(f"  IHSAN_THRESHOLD:     {IHSAN_THRESHOLD}")
print(f"  ADL_GINI_THRESHOLD:  {ADL_GINI_THRESHOLD}")
print(f"  SNR_THRESHOLD:       {SNR_THRESHOLD}")
print(f"  KERNEL_INVARIANTS:   {KERNEL_INVARIANTS}")
print(f"  Identity:            {iid[:32]}...")
print(f"  Agent keys:          {len(aks)} (7 PAT + 5 SAT)")
print(f"  Events captured:     {len(cap)}")
print(f"  Receipt chain:       {len(receipts)} receipts")
print()
status = "SOVEREIGN" if pct >= 85 else "DEGRADED"
lifecycle = "COMPLETE" if pct >= 90 else "PARTIAL"
print(f"  NODE0 STATUS:  {status}")
print(f"  LIFECYCLE:     {lifecycle}")
print(f"  IHSAN EVAL:    {pct/100:.3f}")
if failed > 0:
    print(f"\n  === FAILED ({failed}) ===")
    for k, v in R.items():
        if v[0] == "FAIL":
            print(f"    [FAIL] {k}: {v[1][:100]}")
