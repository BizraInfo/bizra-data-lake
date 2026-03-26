#!/bin/bash
# BIZRA Spearpoint — 3 missions + first daily manifest
echo "=== MISSION 1: What is BIZRA? ==="
curl -s -X POST http://127.0.0.1:9740/api/mission \
  -H Content-Type:application/json \
  -d '{"text":"What is BIZRA?"}' | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'Status: {d.get(\"status\")}')
print(f'Mission: {d.get(\"mission_id\")}')
print(f'Ihsan: {d.get(\"ihsan_score\")}')
print(f'Duration: {d.get(\"duration_ms\")}ms')
print(f'Knowledge: {d.get(\"knowledge_enriched\")}')
print(f'Hash: {d.get(\"evidence_hash\")}')
"
echo ""
echo "=== MISSION 2: What is the Seed Chain? ==="
curl -s -X POST http://127.0.0.1:9740/api/mission \
  -H Content-Type:application/json \
  -d '{"text":"What is the Seed Chain architecture?"}' | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'Status: {d.get(\"status\")}')
print(f'Mission: {d.get(\"mission_id\")}')
print(f'Ihsan: {d.get(\"ihsan_score\")}')
print(f'Duration: {d.get(\"duration_ms\")}ms')
print(f'Hash: {d.get(\"evidence_hash\")}')
"
echo ""
echo "=== MISSION 3: Explain constitutional governance ==="
curl -s -X POST http://127.0.0.1:9740/api/mission \
  -H Content-Type:application/json \
  -d '{"text":"Explain how BIZRA enforces constitutional governance above cognition"}' | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'Status: {d.get(\"status\")}')
print(f'Mission: {d.get(\"mission_id\")}')
print(f'Ihsan: {d.get(\"ihsan_score\")}')
print(f'Duration: {d.get(\"duration_ms\")}ms')
print(f'Hash: {d.get(\"evidence_hash\")}')
"
echo ""
echo "=== ALL 3 MISSIONS COMPLETE ==="
