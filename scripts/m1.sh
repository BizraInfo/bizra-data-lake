#!/bin/bash
curl -s -X POST http://127.0.0.1:9740/api/mission \
  -H Content-Type:application/json \
  -d '{"text":"What is BIZRA?"}' 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'status={d.get(\"status\")} mission={d.get(\"mission_id\")} ihsan={d.get(\"ihsan_score\")} ms={d.get(\"duration_ms\")} hash={d.get(\"evidence_hash\",\"\")[:16]}')
" 2>/dev/null || echo "MISSION_FAILED"
