#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
echo "=== CODEBASE METRICS ==="
echo "Python LOC:"
find core/ -name '*.py' -not -path '*__pycache__*' | xargs wc -l 2>/dev/null | tail -1
echo "Rust LOC:"
find bizra-omega/ -name '*.rs' | xargs wc -l 2>/dev/null | tail -1
echo "Test files:"
find tests/ -name 'test_*.py' | wc -l
echo "Core modules:"
ls -d core/*/ | wc -l
echo ""
echo "=== GIT STATUS ==="
git log --oneline -8
echo ""
echo "=== KERNEL DAEMON ==="
curl -s http://localhost:9740/api/status 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Not running"
echo ""
echo "=== HEARTBEAT ==="
curl -s http://localhost:9740/api/heartbeat 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
l=d.get('latest',{})
print(f'Health: {d.get(\"health\",\"?\")}')
print(f'Beat: {l.get(\"beat\",\"?\")}')
print(f'RSS: {l.get(\"memory_rss_mb\",\"?\")} MB')
print(f'Uptime: {l.get(\"uptime_s\",0)/3600:.1f}h')
print(f'Anomalies: {d.get(\"anomalies\",[])}')
" 2>/dev/null || echo "No heartbeat data"
echo ""
echo "=== CONSTITUTIONAL CONSTANTS ==="
python3 -c "
from core.integration.constants import IHSAN_THRESHOLD, ADL_GINI_THRESHOLD, SNR_THRESHOLD, KERNEL_INVARIANTS, IHSAN_WEIGHTS
print(f'IHSAN: {IHSAN_THRESHOLD}')
print(f'GINI: {ADL_GINI_THRESHOLD}')
print(f'SNR: {SNR_THRESHOLD}')
print(f'KERNEL: {KERNEL_INVARIANTS}')
dims = list(IHSAN_WEIGHTS.keys())
print(f'IHSAN dims ({len(dims)}): {dims}')
print(f'Weights sum: {sum(IHSAN_WEIGHTS.values()):.2f}')
"
echo ""
echo "=== DISCOVERY MANIFEST ==="
python3 -c "
import json
m=json.load(open('04_GOLD/discovery_manifest.json'))
print(f'Files: {m[\"total_files\"]:,}')
print(f'Size: {m[\"total_gb\"]} GB')
print(f'Dups: {m[\"duplicates\"][\"groups\"]:,} groups, {m[\"duplicates\"][\"redundant_files\"]:,} redundant')
print(f'Recoverable: {m[\"duplicates\"][\"recoverable_bytes\"]/1e9:.1f} GB')
for src,cnt in m['by_source'].items():
    print(f'  {src.split(\"/\")[-1]}: {cnt:,}')
"
echo ""
echo "=== MINT COURT LAST RUN ==="
python3 -c "
from core.sat.mint_court import MintCourt, MintPhase
from pathlib import Path
court = MintCourt('NODE0_FOUNDER', Path('04_GOLD'))
r = court.run()
print(f'Verdict: {r[\"verdict\"]}')
if r['verdict']=='approved':
    print(f'Valuation: {r[\"scorecard\"][\"valuation_seed\"]:,.2f} SEED')
    print(f'Work SNR: {r[\"scorecard\"][\"snr\"]:.4f}')
    print(f'Ihsan: {r[\"scorecard\"][\"ihsan\"]:.4f}')
    print(f'Founder net: {r[\"distribution\"][\"net_founder\"]:,.2f}')
    print(f'Treasury: {r[\"distribution\"][\"treasury_share\"]:,.2f}')
    print(f'Receipts: {r[\"receipts\"]}')
" 2>&1
