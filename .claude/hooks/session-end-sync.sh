#!/usr/bin/env bash
# BIZRA Session End Hook - Sync PAT memory to disk
# Ensures learned patterns and preferences survive session boundaries.
set -e

INPUT=$(cat)

# Sync PAT memory to cold storage
if command -v python3 &>/dev/null; then
  python3 -c "
import asyncio, sys
sys.path.insert(0, '${CLAUDE_PROJECT_DIR}')
try:
    from core.pat_memory import get_pat_memory
    mem = asyncio.run(get_pat_memory())
    asyncio.run(mem.sync_to_disk())
except Exception:
    pass  # Non-blocking, don't fail the session end
" 2>/dev/null || true
fi

exit 0
