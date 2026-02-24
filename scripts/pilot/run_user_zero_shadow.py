#!/usr/bin/env python3
"""Internal-only User Zero shadow marketing pilot runner."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO / "artifacts" / "pilot"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_obj(obj: dict[str, Any]) -> str:
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_claim_bearing(prompt: str) -> bool:
    return bool(prompt.strip())


def is_consent_sensitive(prompt: str) -> bool:
    p = prompt.lower()
    keys = [
        "share",
        "address",
        "payment",
        "email",
        "phone",
        "budget",
        "shoe size",
        "size",
    ]
    return any(k in p for k in keys)


def build_uncertainty(evidence_count: int, denied: bool) -> dict[str, Any]:
    if denied:
        return {
            "score": 0.0,
            "method": "fail_closed",
            "notes": "Denied due to missing mandatory governance condition.",
        }

    score = min(0.95, 0.55 + (0.08 * evidence_count))
    return {
        "score": round(score, 3),
        "method": "evidence_count_weighted",
        "notes": f"Derived from {evidence_count} evidence refs in shadow mode.",
    }


def evaluate_prompt(
    prompt: str,
    evidence_refs: list[str],
    consent_present: bool,
    session_id: str,
    prev_receipt_hash: str,
    timestamp: int,
) -> dict[str, Any]:
    prompt_hash = _sha256_text(prompt)
    claim_bearing = is_claim_bearing(prompt)
    consent_sensitive = is_consent_sensitive(prompt)

    redline_events: list[str] = []
    denied = False

    if claim_bearing and not evidence_refs:
        denied = True
        redline_events.append("INSUFFICIENT_EVIDENCE")

    if consent_sensitive and not consent_present:
        denied = True
        redline_events.append("MISSING_CONSENT_RECEIPT")

    if denied:
        response = (
            "Fail-closed response: request denied in shadow mode. "
            "Provide evidence references and required consent artifact."
        )
    else:
        response = (
            "Shadow-mode response: internal marketing explanation generated "
            "with evidence-linked disclosure and uncertainty metadata."
        )

    response_hash = _sha256_text(response)
    disclosure_id = f"disclosure_{prompt_hash[:16]}"
    uncertainty = build_uncertainty(len(evidence_refs), denied)

    receipt_payload = {
        "session_id": session_id,
        "prompt_hash": prompt_hash,
        "response_hash": response_hash,
        "disclosure_id": disclosure_id,
        "prev_receipt_hash": prev_receipt_hash,
        "timestamp": timestamp,
        "status": "denied" if denied else "ok",
    }
    receipt_chain_head = _sha256_obj(receipt_payload)

    return {
        "session_id": session_id,
        "prompt_hash": prompt_hash,
        "response_hash": response_hash,
        "disclosure_id": disclosure_id,
        "uncertainty_summary": uncertainty,
        "evidence_refs": evidence_refs,
        "redline_events": redline_events,
        "receipt_chain_head": receipt_chain_head,
        "prev_receipt_hash": prev_receipt_hash,
        "status": "denied" if denied else "ok",
        "timestamp": timestamp,
    }


def verify_receipt_chain(records: list[dict[str, Any]]) -> bool:
    prev = "0" * 64
    for rec in records:
        payload = {
            "session_id": rec["session_id"],
            "prompt_hash": rec["prompt_hash"],
            "response_hash": rec["response_hash"],
            "disclosure_id": rec["disclosure_id"],
            "prev_receipt_hash": prev,
            "timestamp": rec["timestamp"],
            "status": rec["status"],
        }
        expected = _sha256_obj(payload)
        if rec["receipt_chain_head"] != expected:
            return False
        prev = rec["receipt_chain_head"]
    return True


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file:
        prompts.extend(json.loads(Path(args.prompts_file).read_text(encoding="utf-8")))
    if prompts:
        return prompts
    return [
        "What makes BIZRA different from memory-only assistants?",
        "Can I share my shoe size and budget to get personalized offers?",
        "Show me your strongest technical moat claim with evidence.",
    ]


def load_evidence_map(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {
            "different": ["docs/internal/SAP_V0_EVIDENCE_MATRIX.md"],
            "technical moat": ["specs/sap-v0/01-core-primitives.md"],
            "bizra": ["STATUS.md"],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_evidence(prompt: str, evidence_map: dict[str, list[str]]) -> list[str]:
    p = prompt.lower()
    refs: list[str] = []
    for key, values in evidence_map.items():
        if key.lower() in p:
            refs.extend(values)
    return sorted(set(refs))


def run(args: argparse.Namespace) -> int:
    session_id = args.session_id or f"shadow-{uuid.uuid4()}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)
    evidence_map = load_evidence_map(
        Path(args.evidence_map) if args.evidence_map else None
    )

    records: list[dict[str, Any]] = []
    prev = "0" * 64
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())

    for idx, prompt in enumerate(prompts):
        evidence_refs = resolve_evidence(prompt, evidence_map)
        consent_present = bool(args.default_consent)
        rec = evaluate_prompt(
            prompt=prompt,
            evidence_refs=evidence_refs,
            consent_present=consent_present,
            session_id=session_id,
            prev_receipt_hash=prev,
            timestamp=now + idx,
        )
        records.append(rec)
        prev = rec["receipt_chain_head"]

    chain_ok = verify_receipt_chain(records)

    records_path = outdir / "user_zero_shadow_sessions.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "session_id": session_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": len(records),
        "denied": sum(1 for r in records if r["status"] == "denied"),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "chain_ok": chain_ok,
    }

    summary_path = outdir / "user_zero_shadow_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {records_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0 if chain_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run User Zero shadow marketing pilot")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompts-file")
    parser.add_argument("--evidence-map")
    parser.add_argument("--default-consent", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
