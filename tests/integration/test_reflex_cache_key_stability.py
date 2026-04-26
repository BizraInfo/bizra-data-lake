from __future__ import annotations

import json

from core.sovereign.reflex_compiler import ReflexCompiler


def test_reflex_hash_normalization_is_stable() -> None:
    first = ReflexCompiler._hash_input("Canonical   Mission\nPrompt")
    second = ReflexCompiler._hash_input(" canonical mission prompt ")

    assert first == second


def test_persisted_reflex_entry_uses_stable_cache_key(tmp_path) -> None:
    persistence_path = tmp_path / "node0" / "reflexes.json"
    compiler = ReflexCompiler(persistence_path=persistence_path)
    mission = "Canonical mission prompt"
    pattern_hash = compiler._hash_input(mission)

    compiler.compile_from_candidate(
        pattern_id=pattern_hash,
        input_template=mission,
        output_template="Reject execution and emit blocked receipt.",
        ihsan_score=0.95,
        observation_count=3,
    )
    compiler.save_to_disk()

    persisted = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert pattern_hash in persisted["entries"]
