#!/usr/bin/env python3
"""Inject network_asabiyyah into core/constitutional/algorithms.py (GAP-1 fix)."""

import sys

TARGET = r"C:\BIZRA-DATA-LAKE\core\constitutional\algorithms.py"

FUNCTION = '''
def network_asabiyyah(wallets: list[WalletState]) -> int:
    """Compute network-wide Asabiyyah (social cohesion) from wallet state.

    GAP-1: Asabiyyah-Gini Coupling - closes the Khaldunian feedback loop.
    This function was imported by ticker.py but never implemented.

    Three dimensions, weighted per ASABIYYAH_WEIGHTS = (0.4, 0.3, 0.3):
      1. Attestation density (0.4): mutual attestations / max possible pairs
      2. Governance participation (0.3): voting activity rate across network
      3. Cooperation rate (0.3): cooperative actions / total actions

    Returns fixed-point value in [0, FP_ONE].
    Higher = more cohesive network -> khaldunian_throttle boosts minting.
    Lower = fragmented network -> khaldunian_throttle reduces minting.

    Standing on Giants:
    - Ibn Khaldun (1377): Asabiyyah as social cohesion driving civilizational rise
    - Putnam (2000): Social capital as measurable network property
    """
    n = len(wallets)
    if n <= 1:
        return ASAB_NEUTRAL  # Single node: neutral cohesion

    # -- Dimension 1: Attestation Density (weight 0.4) --
    max_pairs = n * (n - 1) // 2  # C(n, 2)
    mutual_count = 0
    for i, w in enumerate(wallets):
        for w2 in wallets[i + 1:]:
            if w2.node_id in w.attestations_given and w.node_id in w2.attestations_given:
                mutual_count += 1
    attest_score = fp_div(fp(mutual_count), fp(max_pairs)) if max_pairs > 0 else FP_ZERO
    attest_score = fp_clamp(attest_score, FP_ZERO, FP_ONE)

    # -- Dimension 2: Governance Participation (weight 0.3) --
    vote_scores = []
    for w in wallets:
        if w.total_actions > 0:
            vote_scores.append(
                fp_clamp(fp_div(fp(w.governance_votes), fp(w.total_actions)), FP_ZERO, FP_ONE)
            )
        else:
            vote_scores.append(FP_ZERO)
    gov_score = fp_div(sum(vote_scores), fp(n)) if n > 0 else FP_ZERO

    # -- Dimension 3: Cooperation Rate (weight 0.3) --
    total_coop = sum(w.cooperative_actions for w in wallets)
    total_acts = sum(w.total_actions for w in wallets)
    coop_score = fp_div(fp(total_coop), fp(total_acts)) if total_acts > 0 else FP_ZERO
    coop_score = fp_clamp(coop_score, FP_ZERO, FP_ONE)

    # -- Weighted combination: (0.4, 0.3, 0.3) --
    w_attest = fp(0.4)
    w_gov = fp(0.3)
    w_coop = fp(0.3)

    combined = fp_add(
        fp_add(fp_mul(w_attest, attest_score), fp_mul(w_gov, gov_score)),
        fp_mul(w_coop, coop_score),
    )
    return fp_clamp(combined, FP_ZERO, FP_ONE)

'''

ANCHOR = "def khaldunian_throttle(gini: int, asabiyyah: int = FP_ZERO) -> int:"

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

if "def network_asabiyyah" in content:
    print("SKIP: network_asabiyyah already exists")
    sys.exit(0)

if ANCHOR not in content:
    print(f"ERROR: anchor not found in {TARGET}")
    sys.exit(1)

patched = content.replace(ANCHOR, FUNCTION + "\n" + ANCHOR)

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(patched)

print(f"PATCHED: network_asabiyyah injected ({TARGET})")
print(f"Total lines: {patched.count(chr(10)) + 1}")
