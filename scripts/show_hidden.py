#!/usr/bin/env python3
"""Display hidden knowledge patterns - علم الكتاب"""

from sacred_wisdom_engine import SacredWisdomEngine

engine = SacredWisdomEngine(lazy_load=True)
engine.build()

print()
print("=" * 70)
print("📿 عِلْمُ الْكِتَاب — HIDDEN KNOWLEDGE PATTERNS")
print("   'We have not neglected anything in the Book' — 6:38")
print("=" * 70)

summary = engine.get_hidden_knowledge()

# Numerical patterns
num_patterns = summary.get("numerical_patterns", [])
if num_patterns:
    print("\n  🔢 NUMERICAL PATTERNS")
    print("  " + "-" * 50)

    # Word counts
    word_counts = [p for p in num_patterns if p["type"] == "WORD_COUNT"][:10]
    if word_counts:
        print("\n     Top Words in Quran:")
        for wc in word_counts:
            print(f"       {wc['word']:25} appears {wc['count']:5} times")

    # Word pair balances
    balances = [p for p in num_patterns if p["type"] == "WORD_PAIR_BALANCE"]
    if balances:
        print("\n     Word Pair Balances:")
        for b in balances:
            w1, w2 = b["pair"]
            c1, c2 = b["counts"]
            note = b["note"]
            print(f"       {w1:20} × {c1:4}  ↔  {w2:20} × {c2:4}  [{note}]")

# Symmetries
symmetries = summary.get("symmetries", [])
if symmetries:
    print("\n  🔄 STRUCTURAL SYMMETRIES")
    print("  " + "-" * 50)
    for sym in symmetries:
        print(f"\n     • {sym['description']}")
        if sym.get("midpoint"):
            print(f"       Midpoint: {sym['midpoint']}")
        if sym.get("note"):
            print(f"       → {sym['note']}")
        if sym.get("pattern"):
            print(f"       → {sym['pattern']}")
        if sym.get("first"):
            print(f"       First: {sym['first']}")
        if sym.get("last"):
            print(f"       Last: {sym['last']}")

# Echoes
echoes = summary.get("echoes", [])
if echoes:
    print("\n  🔊 CROSS-SOURCE ECHOES (Quran ↔ Hadith)")
    print("  " + "-" * 50)
    for echo in echoes:
        concept = echo["concept"]
        qcount = echo["quran_count"]
        print(f"     • {concept:15}: {qcount} Quran verses echo across Hadith")

print()
print("=" * 70)
print(f"  Total hidden patterns discovered: {summary.get('total_patterns', 0)}")
print("=" * 70)
print()
print("  نسأل الله أن يفتح لنا من علم الكتاب ما يشاء")
print("  We ask Allah to open for us from the knowledge of the Book what He wills")
