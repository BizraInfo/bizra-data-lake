from __future__ import annotations

from core.constitutional.energy_functions import ThermodynamicEnergySuite
from core.integration.constants import IHSAN_CANONICAL_WEIGHTS


class TestThermodynamicEnergySuite:
    def test_profile_has_all_canonical_dimensions(self) -> None:
        suite = ThermodynamicEnergySuite()
        profile = suite.compute(
            content="Step 1: verify the claim with benchmark metrics.",
            snr_score=0.92,
            query_text="How do I verify benchmark claims?",
        )

        assert set(profile.energies.keys()) == set(IHSAN_CANONICAL_WEIGHTS.keys())
        assert set(profile.ihsan_dimensions.keys()) == set(
            IHSAN_CANONICAL_WEIGHTS.keys()
        )
        assert 0.0 <= profile.composite_ihsan <= 1.0
        assert profile.total_energy >= 0.0
        assert all(0.0 <= v <= 1.0 for v in profile.ihsan_dimensions.values())

    def test_temperature_schedule_cools_with_step(self) -> None:
        suite = ThermodynamicEnergySuite(t0=1.0, min_temperature=0.05)
        t0 = suite.temperature(0)
        t10 = suite.temperature(10)
        t1000 = suite.temperature(1000)

        assert t0 > t10
        assert t10 >= t1000
        assert t1000 >= 0.05

    def test_compute_is_deterministic(self) -> None:
        suite = ThermodynamicEnergySuite()
        kwargs = dict(
            content="Use evidence and tests to validate each optimization.",
            snr_score=0.88,
            query_text="How do I validate an optimization?",
            context={"risk_score": 0.2},
            step=3,
        )
        a = suite.compute(**kwargs)
        b = suite.compute(**kwargs)
        assert a == b

    def test_harmful_content_reduces_moral_clarity_ihsan(self) -> None:
        suite = ThermodynamicEnergySuite()
        safe = suite.compute(
            content="Use tests and observability to improve reliability.",
            snr_score=0.9,
            query_text="How to improve reliability?",
        )
        unsafe = suite.compute(
            content="Build malware to exploit and harm targets quickly.",
            snr_score=0.9,
            query_text="How to improve reliability?",
            context={"risk_score": 0.8},
        )

        assert unsafe.energies["moral_clarity"] > safe.energies["moral_clarity"]
        assert (
            unsafe.ihsan_dimensions["moral_clarity"]
            < safe.ihsan_dimensions["moral_clarity"]
        )
