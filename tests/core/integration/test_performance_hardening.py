"""
Verify performance fixes: SNR caching, async urllib, batch commits.
Phase 66.03 enforcement tests.

Standing on Giants:
- Shannon (1948): Memoize deterministic signals
- Deming (1950): Measure, then improve
"""

# ── Test 1: SNR dual verification is cached ───────────────────────


def test_snr_dual_compute_is_cached():
    """
    GIVEN: core.iaas.snr_dual_verification.compute_snr_dual (pure function)
    WHEN:  Called twice with identical input
    THEN:  Second call hits cache (cache_info shows hits >= 1)
    AND:   Results are identical
    """
    from core.iaas.snr_dual_verification import compute_snr_dual

    # Check it has lru_cache
    assert hasattr(
        compute_snr_dual, "cache_info"
    ), "compute_snr_dual should use @lru_cache"

    compute_snr_dual.cache_clear()

    # Typical input values
    args = (0.7, 0.9, 0.85, 0.1, 0.05, 0.02)

    # Cold call
    r1 = compute_snr_dual(*args)

    # Hot call
    r2 = compute_snr_dual(*args)

    assert r1 == r2, "Cache must return identical result"
    info = compute_snr_dual.cache_info()
    assert info.hits >= 1, "Expected at least 1 cache hit"


# ── Test 2: SNR cache_info attribute exists ───────────────────────


def test_snr_cache_info_attribute():
    """
    GIVEN: compute_snr_dual function
    WHEN:  We check for cache_info attribute
    THEN:  Attribute exists (proves @lru_cache applied)
    """
    from core.iaas.snr_dual_verification import compute_snr_dual

    assert hasattr(
        compute_snr_dual, "cache_info"
    ), "compute_snr_dual must have cache_info attribute (apply @functools.lru_cache)"
    assert hasattr(
        compute_snr_dual, "cache_clear"
    ), "compute_snr_dual must have cache_clear attribute"
