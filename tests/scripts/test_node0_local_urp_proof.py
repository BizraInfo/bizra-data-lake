"""Node0-Local URP Proof v0.1 contract tests.

Validates the eight contract assertions named in the proof spec:

1. URP status is URP_LOCAL_ACTIVE.
2. SAT-5 registered into local URP (exactly 5 agents).
3. One skill registered, hash-verifiable.
4. One knowledge pack registered, hash-verifiable.
5. One resource offer registered, hash-verifiable + idempotent.
6. One PoI sandbox record exists with truth_label POI_SANDBOX.
7. Hash chain integrity across all receipts.
8. No raw private data leaked into proof artifacts.

Also covers:
- generator idempotence: rerunning yields identical hashes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import blake3
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = REPO_ROOT / "artifacts" / "proofs" / "node0-local-urp"
GENERATOR = REPO_ROOT / "scripts" / "proofs" / "node0_local_urp_proof_v01.py"

EXPECTED_FILES = [
    "node0_local_urp_status.json",
    "urp_local_registry.json",
    "sat5_urp_registration.json",
    "urp_skill_registry_receipt.json",
    "urp_knowledge_pack_receipt.json",
    "urp_resource_offer_receipt.json",
    "poi_sandbox_record.json",
]

CHAIN_ORDER = [
    "sat5_urp_registration.json",
    "urp_skill_registry_receipt.json",
    "urp_knowledge_pack_receipt.json",
    "urp_resource_offer_receipt.json",
    "poi_sandbox_record.json",
    "urp_local_registry.json",
    "node0_local_urp_status.json",
]

GENESIS_CHAIN = "0" * 64


def _read(name: str) -> dict:
    return json.loads((PROOF_DIR / name).read_text(encoding="utf-8"))


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: dict) -> str:
    return blake3.blake3(_canonical_bytes(payload)).hexdigest()


def _chain_link(prev_chain: str, payload_digest: str) -> str:
    return blake3.blake3(
        prev_chain.encode("utf-8") + payload_digest.encode("utf-8")
    ).hexdigest()


def _strip_seal(receipt: dict) -> dict:
    return {
        k: v
        for k, v in receipt.items()
        if k not in {"payload_digest", "previous_chain_link", "chain_link"}
    }


@pytest.fixture(scope="module")
def all_artifacts() -> dict[str, dict]:
    missing = [name for name in EXPECTED_FILES if not (PROOF_DIR / name).exists()]
    if missing:
        pytest.fail(
            f"Missing proof artifacts: {missing}. Run "
            f"scripts/proofs/node0_local_urp_proof_v01.py first."
        )
    return {name: _read(name) for name in EXPECTED_FILES}


# ── Contract 1 ───────────────────────────────────────────────────────
def test_urp_status_is_urp_local_active(all_artifacts):
    status = all_artifacts["node0_local_urp_status.json"]
    assert status["truth_label"] == "URP_LOCAL_ACTIVE"
    assert status["pat_count"] == 7
    assert status["sat_count"] == 5
    assert status["urp_signed"] is True
    assert status["urp_signature_verified"] is True
    assert status["node_id"].startswith("node0_")
    # genesis_hash from sealed lifecycle
    assert re.fullmatch(r"[0-9a-f]{64}", status["genesis_hash"])


# ── Contract 2 ───────────────────────────────────────────────────────
def test_sat5_registered_into_local_urp(all_artifacts):
    sat5 = all_artifacts["sat5_urp_registration.json"]
    assert sat5["truth_label"] == "URP_LOCAL_ACTIVE"
    assert sat5["sat_count"] == 5
    assert len(sat5["agents"]) == 5
    canonical_slots = {a["canonical_slot"] for a in sat5["agents"]}
    assert canonical_slots == {"S1", "S2", "S3", "S4", "S5"}
    # S2 (Oracle) is frozen per canon
    oracle = next(a for a in sat5["agents"] if a["canonical_slot"] == "S2")
    assert oracle["frozen"] is True
    assert oracle["role"] == "Oracle"
    # All registered into URP
    assert all(a["registered_in_urp"] for a in sat5["agents"])


# ── Contract 3 ───────────────────────────────────────────────────────
def test_skill_registered_and_hash_verifiable(all_artifacts):
    receipt = all_artifacts["urp_skill_registry_receipt.json"]
    assert receipt["kind"] == "urp_skill_registry_receipt"
    assert receipt["truth_label"] == "URP_LOCAL_ACTIVE"
    skill = receipt["skill"]
    assert skill["scope"] == "local_only"
    expected = blake3.blake3(_canonical_bytes(skill)).hexdigest()
    assert receipt["skill_content_hash"] == expected
    # full receipt body digest must match payload_digest
    assert _digest(_strip_seal(receipt)) == receipt["payload_digest"]


# ── Contract 4 ───────────────────────────────────────────────────────
def test_knowledge_pack_registered_and_hash_verifiable(all_artifacts):
    receipt = all_artifacts["urp_knowledge_pack_receipt.json"]
    assert receipt["kind"] == "urp_knowledge_pack_receipt"
    pack = receipt["pack"]
    assert pack["scope"] == "local_only"
    assert re.fullmatch(r"[0-9a-f]{64}", pack["content_hash"])
    assert _digest(_strip_seal(receipt)) == receipt["payload_digest"]


# ── Contract 5 ───────────────────────────────────────────────────────
def test_resource_offer_registered_and_idempotent(all_artifacts):
    receipt = all_artifacts["urp_resource_offer_receipt.json"]
    offer = receipt["offer"]
    assert offer["scope"] == "local_only"
    assert offer["network"] == "loopback"
    assert offer["external_data_access"] == "none"
    # idempotency: regenerating the canonical key yields the same hash
    expected_key = blake3.blake3(
        _canonical_bytes(
            {
                "contributor_node": offer["contributor_node"],
                "scope": "local_only",
                "v": receipt["schema_version"],
            }
        )
    ).hexdigest()
    assert offer["idempotency_key"] == expected_key
    assert _digest(_strip_seal(receipt)) == receipt["payload_digest"]


# ── Contract 6 ───────────────────────────────────────────────────────
def test_poi_sandbox_record_present_and_truth_labeled(all_artifacts):
    receipt = all_artifacts["poi_sandbox_record.json"]
    assert receipt["truth_label"] == "POI_SANDBOX"
    rec = receipt["record"]
    assert rec["ledger_phase"] == "POI_SANDBOX"
    assert rec["seed_credit_sandbox"] == 0
    assert rec["bloom_credit_sandbox"] == 0
    assert rec["monetary_value"] == "none"
    assert _digest(_strip_seal(receipt)) == receipt["payload_digest"]


# ── Contract 7 ───────────────────────────────────────────────────────
def test_chain_integrity_across_receipts(all_artifacts):
    prev = GENESIS_CHAIN
    for name in CHAIN_ORDER:
        receipt = all_artifacts[name]
        body = _strip_seal(receipt)
        digest = _digest(body)
        assert receipt["payload_digest"] == digest, f"{name}: payload_digest drift"
        assert receipt["previous_chain_link"] == prev, f"{name}: chain not linked"
        expected_link = _chain_link(prev, digest)
        assert receipt["chain_link"] == expected_link, f"{name}: chain_link drift"
        prev = receipt["chain_link"]
    # Status must echo the registry chain head as its previous_chain_link.
    status = all_artifacts["node0_local_urp_status.json"]
    registry = all_artifacts["urp_local_registry.json"]
    assert status["registry_chain_link"] == registry["chain_link"]


# ── Contract 8 ───────────────────────────────────────────────────────
def test_no_raw_private_data_leaked():
    """Public-key fields are 16-hex-prefixed; no full-length keys appear."""
    forbidden_keys = {
        "private_key",
        "private_key_hex",
        "signing_private_key",
        "signing_private_key_hex",
        "secret",
    }

    def scan(node, path="root"):
        if isinstance(node, dict):
            for k, v in node.items():
                lk = k.lower()
                assert lk not in forbidden_keys, f"private field {path}.{k} present"
                # any explicit *public_key* field must be the 16-char prefix form
                if lk == "public_key":
                    pytest.fail(
                        f"raw public_key field at {path}.{k}; expected "
                        "*_public_key_prefix variant"
                    )
                scan(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                scan(item, f"{path}[{i}]")
        elif isinstance(node, str):
            # Disallow any 64-hex-char run that looks like a full ed25519 pubkey,
            # except in well-known hash-bearing fields. Hashes are tracked by
            # name (content_hash, payload_digest, chain_link, genesis_hash, etc.)
            # and exit through the dict-key path above; this check catches stray
            # keys embedded inside descriptive strings.
            stray = re.search(
                r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",
                node,
            )
            if stray is not None:
                # tolerate when the surrounding key is a known hash field
                if not any(
                    tag in path
                    for tag in (
                        "hash",
                        "digest",
                        "chain_link",
                        "previous_chain_link",
                        "registered_skills",
                        "registered_knowledge_packs",
                        "registered_resource_offers",
                        "registered_poi_records",
                        "sat_registration_chain_link",
                        "registry_chain_link",
                        "chain_head",
                        "idempotency_key",
                        "evidence_receipt_id_ref",
                        "skill_content_hash",
                    )
                ):
                    pytest.fail(
                        f"64-hex-char string outside hash field at {path}: "
                        f"{stray.group(0)[:16]}…"
                    )

    for name in EXPECTED_FILES:
        scan(_read(name), path=name)


# ── Idempotence ──────────────────────────────────────────────────────
def test_generator_idempotent_against_unchanged_state(tmp_path):
    """Re-running the generator produces identical chain heads."""
    import subprocess

    # sovereign_state is local-only / gitignored. The 8 contract tests above
    # validate the committed artifacts and run anywhere. This idempotence
    # test re-invokes the generator, which needs sovereign_state present.
    if not (REPO_ROOT / "sovereign_state" / "urp_pledge.json").exists():
        pytest.skip("sovereign_state/urp_pledge.json absent (local-only state)")

    venv_py = REPO_ROOT / ".venv" / "bin" / "python3"
    py = str(venv_py if venv_py.exists() else "python3")
    before = _read("node0_local_urp_status.json")["chain_head"]
    # Re-run with same frozen timestamp
    res = subprocess.run(
        [py, str(GENERATOR), "--now", "2026-04-27T00:00:00Z"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert res.returncode == 0, res.stderr
    after = _read("node0_local_urp_status.json")["chain_head"]
    assert before == after, "Generator is not idempotent"
