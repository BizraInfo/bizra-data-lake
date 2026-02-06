# BIZRA Ingestion Gates
# Lightweight SNR + Ihsān gating hooks for new ingestion

import math
from typing import Any, Dict

from bizra_config import IHSAN_CONSTRAINT, SNR_THRESHOLD


class IngestGate:
    """Lightweight ingestion gate.

    - Computes an estimated SNR from text statistics (fast, no embeddings)
    - Applies Ihsān constraint and SNR threshold
    - Can be enforced or used as a soft signal
    """

    def __init__(self, enforce: bool = False):
        self.enforce = enforce

    def evaluate(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        text = (meta.get("text") or "").strip()
        tokens = text.split()
        token_count = len(tokens)
        unique_count = len(set(tokens)) if tokens else 0

        # Lightweight component estimates
        signal_strength = min(token_count / 500.0, 1.0) if token_count else 0.0
        information_density = (unique_count / token_count) if token_count else 0.0
        symbolic_grounding = 0.3 if meta.get("source_type") else 0.1
        coverage_balance = 1.0 if token_count > 0 else 0.0

        # Weighted geometric mean (matches ARTE weights)
        weights = {
            "signal": 0.35,
            "density": 0.25,
            "grounding": 0.25,
            "balance": 0.15,
        }
        eps = 1e-10
        components = [
            (signal_strength + eps, weights["signal"]),
            (information_density + eps, weights["density"]),
            (symbolic_grounding + eps, weights["grounding"]),
            (coverage_balance + eps, weights["balance"]),
        ]
        snr = math.exp(sum(w * math.log(v) for v, w in components))

        ihsan = snr >= IHSAN_CONSTRAINT
        passed = snr >= SNR_THRESHOLD

        return {
            "snr_estimate": round(float(snr), 4),
            "ihsan_estimate": bool(ihsan),
            "passed": bool(passed),
            "token_count": token_count,
            "unique_ratio": round(float(information_density), 4),
            "enforced": bool(self.enforce),
        }
