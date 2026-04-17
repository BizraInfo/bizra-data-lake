#!/usr/bin/env bash
# BIZRA — Cycle-6 G4 polyglot E2E smoke test (SCAFFOLD)
#
# Intentionally failing until Cycle-6 G4 closes.
# Red CI on e2e-polyglot workflow is the visible pressure gauge
# that G4 is open. Do NOT "fix" this by making it pass without
# implementing the real contract.
#
# See README.md for the contract.

set -euo pipefail

echo "e2e-polyglot: SCAFFOLD — not yet implemented."
echo ""
echo "Cycle-6 G4 is open. Pre-conditions:"
echo "  - G1 persistence         : OPEN (sovereign_state/ bridge pending)"
echo "  - G2 gateway authority   : SEALED (cycle-6/g2-authority-adr.md)"
echo "  - G3 frontend authority  : OPEN"
echo ""
echo "Close G1 + G3 first, then implement this harness."
echo "Gate criterion: cycle-6/niyyah.md §G4"
echo ""
exit 1
