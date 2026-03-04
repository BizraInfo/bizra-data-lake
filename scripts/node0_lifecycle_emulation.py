"""
Node0 lifecycle emulation harness.

Turns the narrative lifecycle into a deterministic, executable trace:
genesis -> system-2 action -> learning -> myelination -> system-1 action
-> autopoiesis -> network contribution -> convergence summary.

This harness is intentionally local/offline and uses existing constitutional
components (IhsanComputer, IhsanGate, ThermodynamicIhsanGate, EvidenceLedger).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from core.pci.crypto import PrivateKeyWrapper
from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt
from core.proof_engine.ihsan_computer import IhsanComputer
from core.proof_engine.ihsan_gate import IhsanGate
from core.proof_engine.thermodynamic_gate import ThermodynamicIhsanGate


@dataclass
class EmulationConfig:
    user_name: str = "Dr. Sarah Chen"
    start_time: datetime = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
    initial_impt: float = 100.0
    compile_threshold: int = 5
    compile_cost_phase3: float = 50.0
    compile_cost_phase5: float = 80.0
    thermal_threshold: float = 0.35
    strict_signing: bool = False


def _receipt_id(seq: int) -> str:
    return f"{seq:032x}"[-32:]


def _sigmoid_reward(ihsan: float, efficiency_bonus: float) -> float:
    base = 1.0 + max(0.0, min(1.0, ihsan))
    return round(base + efficiency_bonus, 3)


def _identity_projection(user_name: str, start_time: datetime) -> dict[str, str]:
    seed = f"{user_name}|{start_time.isoformat()}".encode("utf-8")
    digest = hashlib.blake2b(seed, digest_size=32).hexdigest()
    pub = hashlib.blake2b((digest + ":pk").encode("utf-8"), digest_size=32).hexdigest()
    return {
        "public_key": f"ed25519:{pub}",
        "identity_hash": f"blake3:{digest}",
    }


def _resolve_receipt_signer(strict_signing: bool) -> tuple[str, str]:
    """Resolve signer keys from env or generate ephemeral key for local emulation."""
    env_priv = os.getenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "").strip()
    env_pub = os.getenv("BIZRA_RECEIPT_PUBLIC_KEY_HEX", "").strip()

    if env_priv:
        signer = PrivateKeyWrapper(env_priv)
        derived_pub = signer.public_key_hex
        if env_pub and env_pub.lower() != derived_pub:
            raise RuntimeError(
                "BIZRA_RECEIPT_PUBLIC_KEY_HEX does not match private key for lifecycle emulation"
            )
        return env_priv, (env_pub or derived_pub)

    if strict_signing:
        raise RuntimeError(
            "Strict signing enabled but BIZRA_RECEIPT_PRIVATE_KEY_HEX is not set"
        )

    # Local dev fallback: generate ephemeral signer to keep receipts signed.
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    priv_hex = private_bytes.hex()
    pub_hex = PrivateKeyWrapper(priv_hex).public_key_hex
    return priv_hex, pub_hex


def run_lifecycle_emulation(
    *,
    state_dir: Path,
    config: EmulationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EmulationConfig()
    state_dir.mkdir(parents=True, exist_ok=True)

    strict_signing = cfg.strict_signing or (
        os.getenv("BIZRA_PHASE65_STRICT_SIGNING", "0").lower()
        in {"1", "true", "yes", "on"}
    )
    signer_priv, signer_pub = _resolve_receipt_signer(strict_signing)

    ledger_path = state_dir / "lifecycle_receipts.jsonl"
    ledger = EvidenceLedger(ledger_path)

    ihsan_gate = IhsanGate(threshold=0.75)
    ihsan_computer = IhsanComputer(enable_thermal_mode=False)
    thermal_gate = ThermodynamicIhsanGate(threshold=cfg.thermal_threshold)

    now = cfg.start_time
    impt = float(cfg.initial_impt)
    events: list[dict[str, Any]] = []

    # Phase 0: Genesis
    identity = _identity_projection(cfg.user_name, cfg.start_time)
    events.append(
        {
            "phase": "genesis",
            "timestamp": now.isoformat(),
            "state": "ROOTED",
            "identity": identity,
            "ledger_height": ledger.count(),
        }
    )

    # Shared action evaluator
    def execute_action(
        *,
        action_seq: int,
        intent: str,
        content: str,
        snr_score: float,
        latency_ms: float,
        thermal_step: int,
        previous_total_energy: float | None,
        mode: str,
    ) -> dict[str, Any]:
        nonlocal impt, now
        ctx: dict[str, Any] = {
            "risk_score": 0.1,
            "thermal_step": thermal_step,
        }
        if previous_total_energy is not None:
            ctx["previous_total_energy"] = previous_total_energy

        components = ihsan_computer.compute(
            content=content,
            snr_score=snr_score,
            query_text=intent,
            context=ctx,
        )
        gate_result = ihsan_gate.evaluate(components)
        thermal_result = thermal_gate.evaluate(
            content=content,
            snr_score=snr_score,
            query_text=intent,
            context=ctx,
            previous_energy=previous_total_energy,
            step=min(thermal_step, 2),
        )
        ihsan_score = gate_result.score
        decision = (
            "APPROVED"
            if gate_result.decision == "APPROVED" and thermal_result.approved
            else "REJECTED"
        )

        receipt = emit_receipt(
            ledger,
            receipt_id=_receipt_id(action_seq),
            node_id="node0",
            status="accepted" if decision == "APPROVED" else "rejected",
            decision=decision,
            reason_codes=(
                []
                if decision == "APPROVED"
                else ["IHSAN_BELOW_THRESHOLD", "THERMODYNAMIC_GATE_REJECTED"]
            ),
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            ihsan_threshold=max(ihsan_gate.threshold, cfg.thermal_threshold),
            seal_digest=hashlib.sha256(
                f"{intent}|{action_seq}".encode("utf-8")
            ).hexdigest(),
            duration_ms=latency_ms,
            signer_private_key_hex=signer_priv,
            signer_public_key_hex=signer_pub,
            critical_decision=True,
            node_role="node0",
            state_dir=state_dir,
        )
        reward = _sigmoid_reward(
            ihsan_score, efficiency_bonus=0.1 if latency_ms <= 1500 else 0.0
        )
        if decision == "APPROVED":
            impt += reward

        now = now + timedelta(milliseconds=latency_ms)
        record = {
            "phase": "action",
            "mode": mode,
            "intent": intent,
            "latency_ms": latency_ms,
            "reward": reward if decision == "APPROVED" else 0.0,
            "decision": decision,
            "ihsan_score": ihsan_score,
            "thermal_energy": thermal_result.profile.total_energy,
            "receipt_seq": receipt.sequence,
            "receipt_hash": receipt.entry_hash,
            "timestamp": now.isoformat(),
        }
        events.append(record)
        return record

    # Phase 1: first interaction (system-2)
    first = execute_action(
        action_seq=1,
        intent="Organize my research papers folder by topic",
        content=(
            "Step 1: scan directory. Step 2: extract topics from filenames. "
            "Step 3: create topic folders and move files. "
            "Step 4: verify state change with deterministic checks."
        ),
        snr_score=0.92,
        latency_ms=3080.0,
        thermal_step=0,
        previous_total_energy=None,
        mode="system2",
    )

    # Phase 2: learning (repeat pattern 4 more times)
    rewards: list[float] = [first["reward"]]
    prev_energy = float(first["thermal_energy"])
    learning_latencies = [2650.0, 2310.0, 1900.0, 1650.0]
    for idx, latency in enumerate(learning_latencies, start=2):
        r = execute_action(
            action_seq=idx,
            intent="Organize folder by topic",
            content=(
                "Step 1: identify files. Step 2: cluster by topic keywords. "
                "Step 3: move files with reversible manifest."
            ),
            snr_score=0.90,
            latency_ms=latency,
            thermal_step=idx - 1,
            previous_total_energy=None,
            mode="system2",
        )
        prev_energy = float(r["thermal_energy"])
        rewards.append(float(r["reward"]))

    successful_learning_actions = sum(
        1
        for e in events
        if e.get("phase") == "action"
        and e.get("mode") == "system2"
        and e.get("decision") == "APPROVED"
    )
    myelinated = successful_learning_actions >= cfg.compile_threshold
    if myelinated:
        impt -= cfg.compile_cost_phase3

    events.append(
        {
            "phase": "myelination",
            "timestamp": now.isoformat(),
            "compiled": myelinated,
            "successful_learning_actions": successful_learning_actions,
            "compile_threshold": cfg.compile_threshold,
            "impt_after_compile": round(impt, 3),
        }
    )

    # Phase 4: system-1 fast path
    fast = execute_action(
        action_seq=6,
        intent="Organize my downloads by topic",
        content=(
            "Reflex path: apply known organization strategy and verify with file count diff."
        ),
        snr_score=0.94,
        latency_ms=375.0,
        thermal_step=6,
        previous_total_energy=None,
        mode="system1" if myelinated else "system2",
    )
    prev_energy = float(fast["thermal_energy"])

    # Phase 5: autopoiesis with consent
    consent = impt >= cfg.compile_cost_phase5
    if consent:
        impt -= cfg.compile_cost_phase5
    events.append(
        {
            "phase": "autopoiesis",
            "timestamp": now.isoformat(),
            "consent": consent,
            "compile_cost": cfg.compile_cost_phase5,
            "impt_after_autopoiesis": round(impt, 3),
        }
    )

    # Phase 6: network contribution reward
    network_reward = 75.0
    impt += network_reward
    events.append(
        {
            "phase": "network",
            "timestamp": now.isoformat(),
            "adopted_users": 47,
            "network_reward": network_reward,
            "impt_after_network": round(impt, 3),
        }
    )

    # Phase 7: convergence summary
    action_events = [e for e in events if e.get("phase") == "action"]
    system1_actions = [e for e in action_events if e.get("mode") == "system1"]
    avg_latency = sum(float(e["latency_ms"]) for e in action_events) / max(
        len(action_events), 1
    )
    avg_ihsan = sum(float(e["ihsan_score"]) for e in action_events) / max(
        len(action_events), 1
    )
    speedup = (
        float(first["latency_ms"]) / float(fast["latency_ms"])
        if float(fast["latency_ms"]) > 0
        else 0.0
    )
    valid_chain, chain_errors = ledger.verify_chain()
    entries = ledger.entries()
    signed_receipts = all(
        isinstance(entry.receipt.get("signature"), dict)
        and len(entry.receipt.get("signature", {}).get("value", "")) > 0
        and len(entry.receipt.get("signature", {}).get("public_key", "")) > 0
        for entry in entries
    )

    summary = {
        "final_state": "FLOURISHING",
        "actions_total": len(action_events),
        "system1_ratio": round(len(system1_actions) / max(len(action_events), 1), 3),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_ihsan": round(avg_ihsan, 3),
        "speedup_system1_vs_system2": round(speedup, 2),
        "impt_balance": round(impt, 3),
        "ledger_height": ledger.count(),
        "ledger_chain_valid": valid_chain,
        "ledger_chain_errors": chain_errors,
        "signed_receipts": signed_receipts,
        "learning_avg_reward": round(sum(rewards) / max(len(rewards), 1), 3),
    }
    events.append({"phase": "convergence", "timestamp": now.isoformat(), **summary})

    return {
        "config": {
            "user_name": cfg.user_name,
            "start_time": cfg.start_time.isoformat(),
            "initial_impt": cfg.initial_impt,
            "strict_signing": strict_signing,
        },
        "events": events,
        "summary": summary,
        "artifacts": {
            "ledger_path": str(ledger_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Node0 lifecycle emulation.")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("sovereign_state") / "lifecycle_emulation",
        help="Directory for generated lifecycle artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <state-dir>/lifecycle_summary.json",
    )
    args = parser.parse_args()

    result = run_lifecycle_emulation(state_dir=args.state_dir)
    output = args.output or (args.state_dir / "lifecycle_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("BIZRA lifecycle emulation complete")
    print(f"state_dir={args.state_dir}")
    print(f"output={output}")
    print(
        "summary: "
        f"state={result['summary']['final_state']} "
        f"actions={result['summary']['actions_total']} "
        f"speedup={result['summary']['speedup_system1_vs_system2']}x "
        f"ledger_valid={result['summary']['ledger_chain_valid']}"
    )


if __name__ == "__main__":
    main()
