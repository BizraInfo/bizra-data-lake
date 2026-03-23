"""Tests for سلسلة البذرة — The Seed Chain v1.0"""

from core.prompt.seed_chain import (
    Bayyinah,
    EvidenceTag,
    Hadd,
    Iisal,
    IisalVerdict,
    Niyyah,
    SeedChain,
    Thamara,
    full_seed,
    small_seed,
)


class TestEvidenceClassification:
    def test_unknown_capped_at_half(self):
        b = Bayyinah()
        b.add("unverified claim", EvidenceTag.UNKNOWN, confidence=0.9)
        assert b.items[0].confidence == 0.5

    def test_derived_capped_at_ninety(self):
        b = Bayyinah()
        b.add("inferred fact", EvidenceTag.DERIVED, confidence=1.0)
        assert b.items[0].confidence == 0.9

    def test_verified_keeps_full_confidence(self):
        b = Bayyinah()
        b.add("proven fact", EvidenceTag.VERIFIED, confidence=1.0)
        assert b.items[0].confidence == 1.0

    def test_counts(self):
        b = Bayyinah()
        b.add("a", EvidenceTag.VERIFIED)
        b.add("b", EvidenceTag.UNKNOWN)
        b.add("c", EvidenceTag.UNKNOWN)
        assert b.verified_count == 1
        assert b.unknown_count == 2


class TestHadd:
    def test_constitutional_default_has_five_prohibitions(self):
        h = Hadd.constitutional_default()
        assert len(h.prohibitions) == 5
        assert h.zann_zero is True
        assert h.riba_zero is True
        assert h.ihsan_floor == 0.95

    def test_frozen_ethics_enabled_by_default(self):
        h = Hadd.constitutional_default()
        assert h.frozen_ethics is True


class TestThamara:
    def test_max_confidence_from_verified(self):
        t = Thamara(evidence_inherited=[EvidenceTag.VERIFIED])
        assert t.max_confidence == 1.0

    def test_max_confidence_from_unknown(self):
        t = Thamara(evidence_inherited=[EvidenceTag.UNKNOWN])
        assert t.max_confidence == 0.5

    def test_max_confidence_mixed_takes_minimum(self):
        t = Thamara(evidence_inherited=[EvidenceTag.VERIFIED, EvidenceTag.UNKNOWN])
        assert t.max_confidence == 0.5

    def test_empty_evidence_returns_zero(self):
        t = Thamara()
        assert t.max_confidence == 0.0


class TestIisal:
    def test_should_loop_on_loop_verdict(self):
        r = Iisal(verdict=IisalVerdict.LOOP, loop_count=0)
        assert r.should_loop is True

    def test_should_not_loop_at_max(self):
        r = Iisal(verdict=IisalVerdict.LOOP, loop_count=3, max_loops=3)
        assert r.should_loop is False

    def test_should_not_loop_on_pass(self):
        r = Iisal(verdict=IisalVerdict.PASS)
        assert r.should_loop is False


class TestSeedChain:
    def test_validate_empty_purpose_fails(self):
        chain = SeedChain(niyyah=Niyyah(purpose=""))
        errors = chain.validate()
        assert any("empty purpose" in e for e in errors)

    def test_validate_unknown_with_zann_zero_fails(self):
        chain = SeedChain(niyyah=Niyyah(purpose="test"))
        chain.bayyinah.add("unverified", EvidenceTag.UNKNOWN)
        errors = chain.validate()
        assert any("ZANN_ZERO" in e for e in errors)

    def test_validate_clean_chain_passes(self):
        chain = SeedChain(niyyah=Niyyah(purpose="research"))
        chain.bayyinah.add("proven", EvidenceTag.VERIFIED, source="test")
        errors = chain.validate()
        assert errors == []

    def test_to_prompt_contains_all_sections(self):
        chain = full_seed(
            "Analyze consensus algorithms",
            evidence=[{"claim": "BFT exists", "tag": "VERIFIED", "source": "paper"}],
        )
        prompt = chain.to_prompt()
        assert "Niyyah" in prompt
        assert "Bayyinah" in prompt
        assert "Hadd" in prompt
        assert "Amanah" in prompt
        assert "Thamara" in prompt
        assert "ZANN_ZERO" in prompt
        assert "BFT exists" in prompt

    def test_compute_hash_is_deterministic(self):
        c1 = small_seed("test purpose")
        c2 = small_seed("test purpose")
        assert c1.compute_hash() == c2.compute_hash()

    def test_compute_hash_changes_with_purpose(self):
        c1 = small_seed("purpose A")
        c2 = small_seed("purpose B")
        assert c1.compute_hash() != c2.compute_hash()


class TestFactories:
    def test_small_seed_uses_reflex_mode(self):
        chain = small_seed("quick task")
        assert chain.amanah.reasoning_mode == "reflex"
        assert chain.amanah.max_depth == 1

    def test_small_seed_has_constitutional_hadd(self):
        chain = small_seed("quick task")
        assert chain.hadd.zann_zero is True
        assert len(chain.hadd.prohibitions) == 5

    def test_full_seed_loads_evidence(self):
        chain = full_seed(
            "deep analysis",
            evidence=[
                {"claim": "test passed", "tag": "VERIFIED", "source": "pytest"},
                {"claim": "might work", "tag": "DERIVED"},
            ],
        )
        assert len(chain.bayyinah.items) == 2
        assert chain.bayyinah.verified_count == 1

    def test_full_seed_respects_tone_and_audience(self):
        chain = full_seed("explain", tone="warm", audience="daughter_test")
        assert chain.amanah.tone == "warm"
        assert chain.amanah.audience == "daughter_test"

    def test_full_seed_default_is_deliberative(self):
        chain = full_seed("analyze")
        assert chain.amanah.reasoning_mode == "deliberative"
        assert chain.amanah.max_depth == 5
