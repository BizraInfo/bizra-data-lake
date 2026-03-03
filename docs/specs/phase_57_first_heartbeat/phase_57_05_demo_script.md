# Phase 57.05: Demo Script — Step-by-Step Execution

## The Demo Scenario

**Task:** "Research the latest AI agent frameworks and create a briefing on my desktop."

**Why this task:**
- It touches EVERY layer (AHK → Python → Browser → Desktop → Memory → Evidence)
- It produces a tangible artifact (a file on the desktop)
- It's something a real user would actually want
- It's the kind of task that no other tool does end-to-end with proof

## Pre-Demo Checklist

```bash
# 1. Start Node0 (if not already running)
source .venv-linux/bin/activate
python scripts/node0_activate.py start

# 2. Verify Python bridge is listening
curl -s http://127.0.0.1:9742 || echo "Bridge listening (TCP, not HTTP)"

# 3. Start AHK HDA server (Windows side)
# Double-click: filedfs/ahk_bridge.ahk
# OR: AutoHotkey64.exe filedfs/ahk_bridge.ahk

# 4. Start AHK client (Windows side)
# Double-click: bin/bizra_bridge.ahk
# OR: AutoHotkey64.exe bin/bizra_bridge.ahk

# 5. Verify HDA connection
# Press Win+B → should show "BIZRA: Connected | Uptime: Xs"

# 6. Check LM Studio (optional — demo works without it)
curl -s -H "Authorization: Bearer $LM_API_TOKEN" \
  http://172.22.48.1:1234/v1/models | python -m json.tool | head -5
```

## Demo Execution (Live)

### Act 1: The Trigger (5 seconds)

1. **User presses `Win+Shift+B`**
2. AHK captures:
   - Active window title (e.g., "Visual Studio Code")
   - Clipboard content (whatever was last copied)
3. InputBox appears: **"What should BIZRA do?"**
4. User types: `Research the latest AI agent frameworks and create a briefing`
5. Tooltip appears: **"BIZRA: Executing mission..."**

### Act 2: The Decomposition (< 1 second)

Behind the scenes:
```
AHK Client → TCP 9742 → DesktopBridge.execute_mission()
  → MissionOrchestrator.execute()
    → ChannelDispatcher.decompose()
      Keywords matched: "research" → BROWSER, "create" → DESKTOP
      Plan: [BROWSER (search), then DESKTOP (create file)]
    → LivingMemory.retrieve("AI agent frameworks", top_k=3)
      Returns: 0-3 relevant past memories
```

Event bus emits: `mission.started`, `mission.decomposed`

### Act 3: The Execution (3-10 seconds)

**BROWSER channel** (parallel):
```
BrowserMCPClient(mode="direct")
  → DuckDuckGo Lite search: "AI agent frameworks 2026"
  → Fetch top 3 result pages
  → Extract titles, URLs, snippets
  → Returns: 5 research results with summaries
```

**DESKTOP channel** (parallel):
```
HDAClient → TCP 9743 → AHK HDA
  → get_context: {active_window: "VS Code", clipboard: "...", screen: {...}}
  → Returns: Desktop state snapshot with SHA-256 perception hash
```

### Act 4: The Synthesis (1-3 seconds)

**With LLM (Level 3+):**
```
InferenceGateway → LM Studio (172.22.48.1:1234)
  Prompt: "Synthesize these research findings into a professional briefing..."
  → Returns: Formatted markdown briefing
```

**Without LLM (Level 0-2):**
```
Template synthesis:
  # BIZRA Mission Briefing
  **Mission:** Research the latest AI agent frameworks...
  ## Research Findings
  ### 1. CrewAI
  **Source:** https://...
  AI agent orchestration framework...
  ### 2. AutoGen
  ...
```

### Act 5: The Constitutional Gate (< 100ms)

```
SNRApexEngine.analyze()
  Signal: relevance=0.88, groundedness=0.92, coherence=0.90
  Noise: hallucination_risk=0.05, repetition=0.03
  → snr_linear = 12.4
  → snr_normalized = 0.925
  → ihsan_score = 0.96
  → ihsan_achieved = True (0.96 >= 0.95)
  GATE: PASSED
```

### Act 6: The Evidence Trail (< 100ms)

```
emit_receipt(ledger, receipt_id="a1b2c3d4e5f60001", ...)
  → BLAKE3 hash chain extended
  → Ed25519 signature attached
  → Written to sovereignty_evidence.jsonl

LivingMemory.encode(
  content="Mission: Research AI agent frameworks... Result: ...",
  memory_type="EPISODIC",
  importance=0.8
)
  → Stored in memory.db for future retrieval
```

### Act 7: The Delivery (< 1 second)

```
File written: /mnt/c/Users/mumo/Desktop/BIZRA_Brief_20260302_0830.md

Response flows back:
  MissionOrchestrator → DesktopBridge → TCP 9742 → AHK Client

AHK shows tooltip (8 seconds):
  Mission: COMPLETE
  Ihsan: 0.960
  Duration: 4823ms
  Briefing: C:\Users\mumo\Desktop\BIZRA_Brief_20260302_0830.md
  Receipt: a1b2c3d4e5f6...

AHK auto-opens the briefing file in the default markdown viewer.
```

## What the Audience Sees

From the user's perspective, the entire flow takes **5-15 seconds**:

1. Press a hotkey
2. Type what they want
3. See "executing..."
4. Briefing file appears on desktop and opens automatically
5. Tooltip shows quality score and proof receipt ID

**No other system does this.** Not OpenClaw, not AutoGen, not CrewAI.
They don't cross the desktop-browser boundary.
They don't have constitutional governance.
They don't generate hash-chained evidence receipts.
They don't capture desktop perception state.

## Fallback Demo (No AHK, No LM Studio)

If AHK isn't available, the demo still works via CLI:

```bash
# Direct Python invocation
python -c "
import asyncio
from core.sovereign.mission import MissionOrchestrator, MissionRequest

async def demo():
    orch = MissionOrchestrator({'memory_path': '/tmp/demo', 'evidence_path': '/tmp/demo/evidence.jsonl'})
    await orch.initialize()

    result = await orch.execute(MissionRequest(
        mission_id='demo00000001',
        description='Research the latest AI agent frameworks',
        context=DesktopContext('CLI', '', {}),
        timestamp=time.time(),
        source='cli',
    ))

    print(f'Status: {result.status}')
    print(f'Ihsan: {result.ihsan_score}')
    print(f'Briefing: {result.briefing_path}')
    print(f'Receipt: {result.evidence_receipt_id}')

asyncio.run(demo())
"
```

## Recording the Demo (for social media)

```
1. Screen capture: OBS Studio (already in stack via obs_trigger.py)
2. Terminal split: Left = WSL logs, Right = Windows desktop
3. Highlight the tooltip notification and the auto-opened file
4. Show the evidence receipt in the terminal for proof
5. Total video: 30-60 seconds
```
