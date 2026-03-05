"""
UI/UX APEX shared test fixtures.

Standing on Giants: Norman (affordances, 1988) · Gibson (ecological perception, 1979)
"""

from __future__ import annotations

import re

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

DESIGN_TOKENS = {
    "bg": "#050B14",
    "gold": "#C9A962",
    "ihsan_threshold": UNIFIED_IHSAN_THRESHOLD,
    "snr_threshold": UNIFIED_SNR_THRESHOLD,
}


def assert_no_hardcoded_secrets(component_source: str) -> None:
    """No API keys, tokens, or IP addresses in UI source."""
    forbidden = [
        r"[A-Za-z0-9+/]{32,}={0,2}",  # Base64 blobs
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
    ]
    for pattern in forbidden:
        match = re.search(pattern, component_source)
        if match:
            # Whitelist known safe patterns (version numbers, hash constants)
            text = match.group(0)
            if text in ("127.0.0.1", "0.0.0.0"):
                continue  # Known safe IPs (loopback/bind-all)
            raise AssertionError(
                f"Hardcoded secret candidate found: {text!r} (pattern: {pattern})"
            )


def assert_constitutional_gate_called(mock_gate, action_name: str) -> None:
    """Every UI-triggered mutation must pass through ConstitutionalGate."""
    calls = [c for c in mock_gate.call_args_list if action_name in str(c)]
    assert len(calls) >= 1, f"ConstitutionalGate not called for {action_name}"
