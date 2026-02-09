import hashlib
import json
from pathlib import Path

import pytest

from core.pci.crypto import generate_keypair, sign_message
from core.zpk import ZeroPointKernel, ZPKPolicy


def _write_worker_bundle(
    bundle_dir: Path,
    release_private_key_hex: str,
    version: str = "1.0.0",
    policy_version: int = 1,
    ihsan_policy: float = 0.95,
    valid_signature: bool = True,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    worker_path = bundle_dir / "worker.py"
    worker_code = "def main(context):\n    return {'ok': True, 'version': context['version']}\n"
    worker_bytes = worker_code.encode("utf-8")
    worker_path.write_bytes(worker_bytes)  # write_bytes avoids \r\n conversion on Windows

    worker_hash = hashlib.sha256(worker_bytes).hexdigest()
    signature = sign_message(worker_hash, release_private_key_hex)
    if not valid_signature:
        signature = "00" * 64

    manifest = {
        "version": version,
        "worker_uri": "worker.py",
        "worker_hash": worker_hash,
        "worker_signature": signature,
        "policy_version": policy_version,
        "ihsan_policy": ihsan_policy,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _read_receipts(state_dir: Path):
    receipts_path = state_dir / "receipts" / "zpk_receipts.jsonl"
    if not receipts_path.exists():
        return []
    lines = receipts_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class DummyEventBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_signature_fail_must_not_execute(tmp_path: Path):
    release_private_key_hex, release_public_key_hex = generate_keypair()
    manifest_path = _write_worker_bundle(
        tmp_path / "bundle",
        release_private_key_hex=release_private_key_hex,
        valid_signature=False,
    )

    kernel = ZeroPointKernel(
        state_dir=tmp_path / "state",
        release_public_key_hex=release_public_key_hex,
    )

    result = await kernel.bootstrap(str(manifest_path), policy=ZPKPolicy())

    assert result.success is False
    assert result.executed_version is None

    receipts = _read_receipts(tmp_path / "state")
    assert receipts
    assert all(r.get("type") != "execution" for r in receipts)


@pytest.mark.asyncio
async def test_policy_fail_must_not_execute(tmp_path: Path):
    release_private_key_hex, release_public_key_hex = generate_keypair()
    manifest_path = _write_worker_bundle(
        tmp_path / "bundle",
        release_private_key_hex=release_private_key_hex,
        version="2.0.0",
        policy_version=1,
        ihsan_policy=0.99,
    )

    kernel = ZeroPointKernel(
        state_dir=tmp_path / "state",
        release_public_key_hex=release_public_key_hex,
    )

    result = await kernel.bootstrap(
        str(manifest_path),
        policy=ZPKPolicy(
            allowed_versions={"1.0.0"},
            min_policy_version=1,
            min_ihsan_policy=0.95,
        ),
    )

    assert result.success is False
    assert result.reason == "policy_denied"

    receipts = _read_receipts(tmp_path / "state")
    assert any(r.get("type") == "policy" for r in receipts)
    assert all(r.get("type") != "execution" for r in receipts)


@pytest.mark.asyncio
async def test_fetch_fail_rolls_back_to_last_known_good(tmp_path: Path):
    release_private_key_hex, release_public_key_hex = generate_keypair()
    manifest_path = _write_worker_bundle(
        tmp_path / "bundle",
        release_private_key_hex=release_private_key_hex,
        version="1.0.0",
    )

    state_dir = tmp_path / "state"
    kernel = ZeroPointKernel(
        state_dir=state_dir,
        release_public_key_hex=release_public_key_hex,
    )

    initial = await kernel.bootstrap(str(manifest_path), policy=ZPKPolicy())
    assert initial.success is True
    assert initial.rollback_used is False
    assert initial.executed_version == "1.0.0"

    missing_manifest = tmp_path / "bundle" / "missing_manifest.json"
    rolled = await kernel.bootstrap(str(missing_manifest), policy=ZPKPolicy())
    assert rolled.success is True
    assert rolled.rollback_used is True
    assert rolled.executed_version == "1.0.0"


@pytest.mark.asyncio
async def test_every_execution_produces_append_only_receipt(tmp_path: Path):
    release_private_key_hex, release_public_key_hex = generate_keypair()
    manifest_path = _write_worker_bundle(
        tmp_path / "bundle",
        release_private_key_hex=release_private_key_hex,
        version="1.0.0",
    )

    state_dir = tmp_path / "state"
    kernel = ZeroPointKernel(
        state_dir=state_dir,
        release_public_key_hex=release_public_key_hex,
    )

    receipts_before = _read_receipts(state_dir)
    result1 = await kernel.bootstrap(str(manifest_path), policy=ZPKPolicy())
    result2 = await kernel.bootstrap(str(manifest_path), policy=ZPKPolicy())
    receipts_after = _read_receipts(state_dir)

    assert result1.success is True
    assert result2.success is True
    assert len(receipts_after) > len(receipts_before)

    execution_receipts = [r for r in receipts_after if r.get("type") == "execution"]
    assert len(execution_receipts) == 2

    # Append-only chain should link by prev_hash/hash.
    for i in range(1, len(receipts_after)):
        assert receipts_after[i].get("prev_hash") == receipts_after[i - 1].get("hash")


@pytest.mark.asyncio
async def test_attestation_challenge_response_verifies(tmp_path: Path):
    _, release_public_key_hex = generate_keypair()
    kernel = ZeroPointKernel(
        state_dir=tmp_path / "state",
        release_public_key_hex=release_public_key_hex,
    )

    challenge = kernel.issue_attestation_challenge("federation-verifier", ttl_seconds=300)
    response = await kernel.answer_attestation_challenge(challenge)
    ok, reason = ZeroPointKernel.verify_attestation_response(challenge, response)

    assert ok is True
    assert reason == "ok"


@pytest.mark.asyncio
async def test_bootstrap_emits_signed_receipt_events(tmp_path: Path):
    release_private_key_hex, release_public_key_hex = generate_keypair()
    manifest_path = _write_worker_bundle(
        tmp_path / "bundle",
        release_private_key_hex=release_private_key_hex,
        version="1.0.0",
    )

    bus = DummyEventBus()
    kernel = ZeroPointKernel(
        state_dir=tmp_path / "state",
        release_public_key_hex=release_public_key_hex,
        event_bus=bus,
        event_topic="zpk.bootstrap.receipt",
    )

    result = await kernel.bootstrap(str(manifest_path), policy=ZPKPolicy())
    assert result.success is True
    assert len(bus.events) >= 4  # attestation + fetch + policy + execution

    for event in bus.events:
        assert event.topic == "zpk.bootstrap.receipt"
        receipt = event.payload["receipt"]
        assert "signature" in receipt
        assert "hash" in receipt
