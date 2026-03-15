#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
echo "=== GIT LOG ==="
git log --oneline -15
echo ""
echo "=== KERNEL DAEMON STATUS ==="
curl -s http://localhost:9740/api/status 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Daemon not running"
echo ""
echo "=== SAT MODULE CHECK ==="
find core/sat -name '*.py' 2>/dev/null | head -20 || echo "core/sat/ not found"
echo ""
echo "=== GOLD DATA ==="
ls -lh 04_GOLD/*.parquet 2>/dev/null | awk '{print $5, $9}'
echo ""
echo "=== B: PARTITION ==="
ls /mnt/b/ 2>/dev/null | head -5 || echo "B: not mounted"
