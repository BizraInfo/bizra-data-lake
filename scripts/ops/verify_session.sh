#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
echo "=== ENTROPY ROUTER TESTS ==="
python3 -m pytest tests/core/reasoning/test_entropy_router.py -q --timeout=30 2>&1 | tail -5
echo ""
echo "=== SAT MODULE IMPORT ==="
python3 -c "
from core.sat.mint_court import MintCourt, MintPhase, MintVerdict, EvidenceSnapshot, SATScorecard, MintDistribution, MintReceipt
print('MintCourt imports: OK')
print(f'Phases: {[p.value for p in MintPhase]}')
print(f'Verdicts: {[v.value for v in MintVerdict]}')
" 2>&1
echo ""
echo "=== SAT GATE TESTS ==="
python3 -m pytest tests/core/sat/ -q --timeout=30 2>&1 | tail -5
echo ""
echo "=== GIT STATUS ==="
git log --oneline -5
git diff --stat HEAD~1
