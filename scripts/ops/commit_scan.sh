#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
# Stage PAT scan tools
git add core/pat/sovereign_scan.py core/pat/quick_scan.py
# Commit
git commit -m "feat(pat): sovereign discovery scan + quick scan pipeline

sovereign_scan.py: Full discovery with BLAKE2b fingerprinting, classification,
  dedup detection, and manifest generation (290 lines)
quick_scan.py: Fast census mode for all BIZRA locations (180 lines)

Results: 1,711,954 files, 494 GB, 52,092 dup groups, 155,351 redundant files,
28.8 GB recoverable. Scanned 5 sources in 7,584s (2.1 hours).

Phone (Z Fold6) discovered via MTP but not yet imported."
echo "EXIT: $?"
git log --oneline -5
