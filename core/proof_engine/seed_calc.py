"""
SEED Reward Calculator — Proof of Impact to SEED conversion.

Standing on:
  - Shannon (1948): SNR as quality signal
  - Al-Ghazali: Ihsan as constitutional floor
  - Satoshi: reward proportional to verified work

Formula:
  impact = ihsan * snr * log2(actions + 2)
  efficiency = impact / max(1, log2(tokens + 2))
  seed_reward = floor(efficiency * SEED_PER_UNIT)

Constitutional gates:
  - Ihsan < 0.95 → reward = 0 (quality floor)
  - Unsigned receipt → reward = 0 (Amanah)
  - All arithmetic via integer (Adl)
"""

import math

# Constitutional constants (from core/integration/constants.py)
IHSAN_FLOOR = 0.95
SEED_PER_IMPACT_UNIT = 10  # Base SEED per unit of proven impact
ZAKAT_RATE = 0.025  # 2.5% deducted at earn time


def calculate_seed_reward(
    ihsan_score: float,
    snr_score: float = 0.95,
    action_count: int = 1,
    tokens_used: int = 100,
    signed: bool = True,
) -> dict:
    """
    Calculate SEED reward for a completed mission.

    Returns dict with: gross, zakat, net, impact, efficiency, reason
    """
    # Constitutional gates
    if ihsan_score < IHSAN_FLOOR:
        return {
            "gross": 0,
            "zakat": 0,
            "net": 0,
            "impact": 0.0,
            "efficiency": 0.0,
            "reason": f"Ihsan {ihsan_score:.3f} < {IHSAN_FLOOR} floor",
        }

    if not signed:
        return {
            "gross": 0,
            "zakat": 0,
            "net": 0,
            "impact": 0.0,
            "efficiency": 0.0,
            "reason": "Unsigned receipt — Amanah violation",
        }

    # Impact = quality × signal × work volume
    impact = ihsan_score * snr_score * math.log2(action_count + 2)

    # Efficiency = impact per token (reward efficiency, not waste)
    efficiency = impact / max(1.0, math.log2(tokens_used + 2))

    # SEED reward (integer — Adl)
    gross = int(efficiency * SEED_PER_IMPACT_UNIT)
    gross = max(1, gross)  # Minimum 1 SEED for any passing mission

    # Zakat deduction (2.5% at earn time — founder pays first)
    zakat = max(1, int(gross * ZAKAT_RATE)) if gross >= 40 else 0
    net = gross - zakat

    return {
        "gross": gross,
        "zakat": zakat,
        "net": net,
        "impact": round(impact, 4),
        "efficiency": round(efficiency, 4),
        "reason": "Mission complete — Proof of Impact verified",
    }


def format_seed_reward(reward: dict) -> str:
    """Format for TUI display."""
    if reward["net"] == 0:
        return f"0 SEED ({reward['reason']})"
    zakat_str = f" - {reward['zakat']} zakat" if reward["zakat"] > 0 else ""
    return f"+{reward['gross']}{zakat_str} = {reward['net']} SEED (impact: {reward['impact']})"


if __name__ == "__main__":
    # Demo
    print("=== SEED Reward Calculator ===")
    cases = [
        ("Perfect mission", 0.97, 0.95, 5, 200, True),
        ("High quality", 0.99, 0.98, 23, 500, True),
        ("Below Ihsan", 0.90, 0.95, 5, 200, True),
        ("Unsigned", 0.97, 0.95, 5, 200, False),
        ("Simple task", 0.95, 0.85, 1, 50, True),
        ("Complex task", 0.98, 0.97, 50, 1000, True),
    ]
    for name, ihsan, snr, actions, tokens, signed in cases:
        r = calculate_seed_reward(ihsan, snr, actions, tokens, signed)
        print(f"  {name:20s} → {format_seed_reward(r)}")
