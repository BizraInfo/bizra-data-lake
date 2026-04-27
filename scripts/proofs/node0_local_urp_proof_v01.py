"""Node0-Local URP Proof v0.1 generator.

Truth label: URP_LOCAL_ACTIVE.

Reads canonical Node0 truth from sovereign_state/, derives BLAKE3
hash-chained receipts, writes 7 artifacts to
artifacts/proofs/node0-local-urp/.

Privacy contract:
- public-key fields stored as 16-hex-char prefix only
- no private keys, no mission body content, no raw user data
- resource offer scope is local-only / loopback / sandbox
- PoI record carries truth_label "POI_SANDBOX" with zero token credit

Idempotent: re-runs against unchanged sovereign_state produce identical
artifact hashes. The chain head changes only when sovereign_state changes
or when the generator semantics evolve (PROOF_VERSION).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import blake3

PROOF_VERSION = "0.1.0"
TRUTH_LABEL_URP = "URP_LOCAL_ACTIVE"
TRUTH_LABEL_POI = "POI_SANDBOX"
PUBKEY_PREFIX_LEN = 16

REPO_ROOT = Path(__file__).resolve().parents[2]
SOVEREIGN_STATE = REPO_ROOT / "sovereign_state"
PROOF_DIR = REPO_ROOT / "artifacts" / "proofs" / "node0-local-urp"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize for digesting — sorted keys, no whitespace, UTF-8."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: dict[str, Any]) -> str:
    return blake3.blake3(_canonical_bytes(payload)).hexdigest()


def _chain_link(prev_chain: str, payload_digest: str) -> str:
    return blake3.blake3(
        prev_chain.encode("utf-8") + payload_digest.encode("utf-8")
    ).hexdigest()


def _pk_prefix(public_key_hex: str) -> str:
    return public_key_hex[:PUBKEY_PREFIX_LEN]


def _content_hash(content: bytes) -> str:
    return blake3.blake3(content).hexdigest()


def _seal(receipt_body: dict[str, Any], prev_chain: str) -> dict[str, Any]:
    """Stamp a receipt with payload_digest and chain_link."""
    payload_digest = _digest(receipt_body)
    chain_link = _chain_link(prev_chain, payload_digest)
    return {
        **receipt_body,
        "payload_digest": payload_digest,
        "previous_chain_link": prev_chain,
        "chain_link": chain_link,
    }


def _write_artifact(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_proof(now_iso: str = "2026-04-27T00:00:00Z") -> dict[str, str]:
    """Build the 7 proof artifacts. Returns map of name -> chain_link."""

    pledge = _read_json(SOVEREIGN_STATE / "urp_pledge.json")
    lifecycle = _read_json(SOVEREIGN_STATE / "node0_lifecycle.json")
    genesis = _read_json(SOVEREIGN_STATE / "node0_genesis.json")

    node_id: str = pledge["node_id"]
    genesis_hash_hex: str = lifecycle["origin"]["genesis_hash"]
    pat_team = genesis["pat_team"]["agents"]
    sat_team = genesis["sat_team"]["agents"]

    if len(pat_team) != 7:
        raise RuntimeError(
            f"PAT-7 contract violated: found {len(pat_team)} agents in genesis"
        )
    if len(sat_team) != 5:
        raise RuntimeError(
            f"SAT-5 contract violated: found {len(sat_team)} agents in genesis"
        )

    sat_id_to_role = {
        "Validator": "S1",
        "Oracle": "S2",
        "Mediator": "S3",
        "Archivist": "S4",
        "Sentinel": "S5",
    }

    # Genesis chain anchor — all-zero like the canonical receipt seed.
    GENESIS_CHAIN = "0" * 64

    # ── Receipt 1: SAT-5 URP registration ─────────────────────────────
    sat5_body = {
        "schema_version": PROOF_VERSION,
        "kind": "sat5_urp_registration",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_URP,
        "node_id": node_id,
        "urp_membership": "local_singleton",
        "sat_count": 5,
        "agents": [
            {
                "agent_id": agent["agent_id"],
                "canonical_slot": sat_id_to_role.get(agent["role"], "S?"),
                "role": agent["role"],
                "public_key_prefix": _pk_prefix(agent["public_key"]),
                "capabilities": list(agent.get("capabilities", [])),
                "frozen": agent["role"] == "Oracle",
                "registered_in_urp": True,
            }
            for agent in sat_team
        ],
    }
    sat5 = _seal(sat5_body, GENESIS_CHAIN)
    _write_artifact(PROOF_DIR / "sat5_urp_registration.json", sat5)

    # ── Receipt 2: Skill registry receipt ─────────────────────────────
    skill_descriptor = {
        "skill_id": "node0.health.report.v1",
        "description": "Single-node health snapshot conformant to lifecycle v2",
        "owner_node": node_id,
        "ihsan_floor": 0.95,
        "scope": "local_only",
    }
    skill_body = {
        "schema_version": PROOF_VERSION,
        "kind": "urp_skill_registry_receipt",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_URP,
        "node_id": node_id,
        "skill": skill_descriptor,
        "skill_content_hash": _content_hash(_canonical_bytes(skill_descriptor)),
    }
    skill_receipt = _seal(skill_body, sat5["chain_link"])
    _write_artifact(PROOF_DIR / "urp_skill_registry_receipt.json", skill_receipt)

    # ── Receipt 3: Knowledge pack receipt ─────────────────────────────
    kernel_path = REPO_ROOT / "docs" / "canon" / "BIZRA_ORIGIN_KERNEL.md"
    if kernel_path.exists():
        kernel_bytes = kernel_path.read_bytes()
        kernel_size = len(kernel_bytes)
        kernel_hash = _content_hash(kernel_bytes)
        kernel_source = "docs/canon/BIZRA_ORIGIN_KERNEL.md"
    else:
        kernel_bytes = b"{}"
        kernel_size = 0
        kernel_hash = _content_hash(b"")
        kernel_source = "missing:placeholder"

    knowledge_pack = {
        "pack_id": "bizra.origin_kernel.v1",
        "title": "BIZRA Origin Kernel (canonical reference)",
        "source": kernel_source,
        "size_bytes": kernel_size,
        "content_hash": kernel_hash,
        "ihsan_score_sandbox": 0.97,
        "scope": "local_only",
    }
    knowledge_body = {
        "schema_version": PROOF_VERSION,
        "kind": "urp_knowledge_pack_receipt",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_URP,
        "node_id": node_id,
        "pack": knowledge_pack,
    }
    knowledge_receipt = _seal(knowledge_body, skill_receipt["chain_link"])
    _write_artifact(
        PROOF_DIR / "urp_knowledge_pack_receipt.json", knowledge_receipt
    )

    # ── Receipt 4: Resource offer receipt ─────────────────────────────
    resource_offer = {
        "offer_id": f"local_offer_{node_id[:16]}_v1",
        "contributor_node": node_id,
        "scope": "local_only",
        "network": "loopback",
        "compute": {
            "cpu_arch": "x86_64",
            "ram_gb_pledged": pledge.get("ram_gb", 0),
            "vram_gb_pledged": pledge.get("vram_gb", 0),
        },
        "storage_gb_pledged": pledge.get("storage_gb", 0),
        "consent": "explicit_local_only",
        "external_data_access": "none",
        "idempotency_key": _digest(
            {
                "contributor_node": node_id,
                "scope": "local_only",
                "v": PROOF_VERSION,
            }
        ),
    }
    offer_body = {
        "schema_version": PROOF_VERSION,
        "kind": "urp_resource_offer_receipt",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_URP,
        "node_id": node_id,
        "offer": resource_offer,
    }
    offer_receipt = _seal(offer_body, knowledge_receipt["chain_link"])
    _write_artifact(
        PROOF_DIR / "urp_resource_offer_receipt.json", offer_receipt
    )

    # ── Receipt 5: PoI sandbox record ─────────────────────────────────
    poi_record = {
        "record_id": f"poi_sandbox_{node_id[:16]}_v1",
        "contributor_node": node_id,
        "action": "node0_local_urp_active",
        "evidence_receipt_id_ref": lifecycle.get("mission", {}).get(
            "last_evidence_receipt_id", ""
        ),
        "ihsan_score_sandbox": lifecycle.get("mission", {}).get(
            "last_ihsan_score", 0.95
        ),
        "snr_score_sandbox": lifecycle.get("mission", {}).get(
            "last_snr_score", 0.0
        ),
        "impact_score_sandbox": 1.0,
        "seed_credit_sandbox": 0,
        "bloom_credit_sandbox": 0,
        "monetary_value": "none",
        "ledger_phase": TRUTH_LABEL_POI,
    }
    poi_body = {
        "schema_version": PROOF_VERSION,
        "kind": "poi_sandbox_record",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_POI,
        "node_id": node_id,
        "record": poi_record,
    }
    poi_receipt = _seal(poi_body, offer_receipt["chain_link"])
    _write_artifact(PROOF_DIR / "poi_sandbox_record.json", poi_receipt)

    # ── Aggregate: URP local registry ─────────────────────────────────
    registry_body = {
        "schema_version": PROOF_VERSION,
        "kind": "urp_local_registry",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_URP,
        "node_id": node_id,
        "genesis_hash": genesis_hash_hex,
        "registered_skills": [
            {
                "skill_id": skill_descriptor["skill_id"],
                "receipt_chain_link": skill_receipt["chain_link"],
            }
        ],
        "registered_knowledge_packs": [
            {
                "pack_id": knowledge_pack["pack_id"],
                "receipt_chain_link": knowledge_receipt["chain_link"],
            }
        ],
        "registered_resource_offers": [
            {
                "offer_id": resource_offer["offer_id"],
                "receipt_chain_link": offer_receipt["chain_link"],
                "idempotency_key": resource_offer["idempotency_key"],
            }
        ],
        "registered_poi_records": [
            {
                "record_id": poi_record["record_id"],
                "receipt_chain_link": poi_receipt["chain_link"],
            }
        ],
        "sat_registration_chain_link": sat5["chain_link"],
    }
    registry = _seal(registry_body, poi_receipt["chain_link"])
    _write_artifact(PROOF_DIR / "urp_local_registry.json", registry)

    # ── Status — top-level summary ────────────────────────────────────
    status_body = {
        "schema_version": PROOF_VERSION,
        "kind": "node0_local_urp_status",
        "generated_at": now_iso,
        "truth_label": TRUTH_LABEL_URP,
        "node_id": node_id,
        "genesis_hash": genesis_hash_hex,
        "pat_count": len(pat_team),
        "sat_count": len(sat_team),
        "urp_signed": bool(pledge.get("signed", False)),
        "urp_signature_verified": bool(pledge.get("signature_verified", False)),
        "urp_signer_public_key_prefix": _pk_prefix(
            pledge.get("signer_public_key", "")
        ),
        "urp_pledge_hash": pledge.get("pledge_hash", ""),
        "registry_chain_link": registry["chain_link"],
        "chain_head": registry["chain_link"],
        "artifact_chain": [
            {
                "name": "sat5_urp_registration",
                "chain_link": sat5["chain_link"],
            },
            {
                "name": "urp_skill_registry_receipt",
                "chain_link": skill_receipt["chain_link"],
            },
            {
                "name": "urp_knowledge_pack_receipt",
                "chain_link": knowledge_receipt["chain_link"],
            },
            {
                "name": "urp_resource_offer_receipt",
                "chain_link": offer_receipt["chain_link"],
            },
            {
                "name": "poi_sandbox_record",
                "chain_link": poi_receipt["chain_link"],
            },
            {
                "name": "urp_local_registry",
                "chain_link": registry["chain_link"],
            },
        ],
        "privacy_contract": {
            "public_keys": "16-hex-char prefix only",
            "private_keys": "never_emitted",
            "external_data": "none",
            "scope": "local_only",
        },
    }
    status = _seal(status_body, registry["chain_link"])
    _write_artifact(PROOF_DIR / "node0_local_urp_status.json", status)

    return {
        "sat5_urp_registration": sat5["chain_link"],
        "urp_skill_registry_receipt": skill_receipt["chain_link"],
        "urp_knowledge_pack_receipt": knowledge_receipt["chain_link"],
        "urp_resource_offer_receipt": offer_receipt["chain_link"],
        "poi_sandbox_record": poi_receipt["chain_link"],
        "urp_local_registry": registry["chain_link"],
        "node0_local_urp_status": status["chain_link"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--now",
        default="2026-04-27T00:00:00Z",
        help="Frozen ISO timestamp for deterministic output",
    )
    args = parser.parse_args()
    chain = build_proof(now_iso=args.now)
    print(json.dumps(chain, indent=2, sort_keys=True))
    print(f"\nProof directory: {PROOF_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
