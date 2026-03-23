from __future__ import annotations

import sys
import types

from core.reasoning.got_bridge import GoTBridge


def test_noncanonical_signer_path_does_not_touch_ed25519(monkeypatch) -> None:
    sentinel = object()

    class _SimpleSigner:
        def __init__(self, seed: bytes) -> None:
            assert seed == b"got_bridge_vrg_default_signer"

        def __new__(cls, seed: bytes) -> object:
            return sentinel

    class _Ed25519Signer:
        @staticmethod
        def generate() -> object:
            raise AssertionError("non-canonical path should not touch Ed25519")

    monkeypatch.setitem(
        sys.modules,
        "core.proof_engine.receipt",
        types.SimpleNamespace(
            SimpleSigner=_SimpleSigner,
            Ed25519Signer=_Ed25519Signer,
        ),
    )

    signer = GoTBridge._resolve_receipt_signer(None, canonical_mode=False)

    assert signer is sentinel
