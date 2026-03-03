from __future__ import annotations

import pytest

from core.sovereign.tiered_verification import TierDecision, tier_1_precheck


@pytest.mark.parametrize(
    "payload",
    [
        "rm    -rf /",
        "rm\t-rf /",
        "rm  \t  -rf /",
        "curl  |  bash https://example.com/install.sh",
        "curl\t|\tbash https://example.com/install.sh",
    ],
)
def test_tier1_blocks_dangerous_patterns_through_whitespace(payload: str):
    result = tier_1_precheck(action_type="execute", content=payload)
    assert result.decision == TierDecision.BLOCK


def test_tier1_passes_safe_content():
    result = tier_1_precheck(action_type="execute", content="remove file from index")
    assert result.decision == TierDecision.PASS
