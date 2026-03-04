try:
    # Constitution package integration (Step 2): canonical source.
    from bizra_constitution.snr import normalize_snr as _normalize_from_constitution
except Exception:  # pragma: no cover - service-local fallback
    _normalize_from_constitution = None


def normalize_snr_linear(snr_linear: float) -> float:
    """Canonical bounded normalization for linear SNR ratios."""
    if _normalize_from_constitution is not None:
        return float(_normalize_from_constitution(float(snr_linear)))
    value = max(float(snr_linear), 0.0)
    return min(value / (1.0 + value), 1.0)


def snr_score(signal: float, noise: float, eps: float = 1e-9) -> float:
    """Compute normalized SNR score in [0,1] from signal/noise."""
    ratio = max(signal, 0.0) / (max(noise, 0.0) + eps)
    return normalize_snr_linear(ratio)
