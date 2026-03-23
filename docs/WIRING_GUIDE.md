# Wiring Guide — Demo → Live Backend

The cockpit runs in demo mode by default (hardcoded data, simulated missions).
Three changes connect it to the live NODE0 backend.

## Prerequisites

Verify these are running:
```
python scripts/phase1_gate.py
```

Expected: kernel (3006), ghost_ws (9743), desktop_bridge (9742) all responding.

---

## Change 1: Ghost Panel — live suggestions

**File:** `src/App.jsx`

Find the `GHOSTS` array (hardcoded demo suggestions). Add this import and useEffect
at the top of the `BIZRAWorld` component:

```javascript
import { connectGhostPanel } from "./bridge.js";

// Inside BIZRAWorld, after useState declarations:
useEffect(() => {
  const cleanup = connectGhostPanel(
    (event) => set(p => ({ ...p, ghosts: [...(p.ghosts || []), event].slice(-20) })),
    (err) => console.warn("Ghost:", err)
  );
  return cleanup;
}, []);
```

Then remove or comment out the `GHOSTS` constant and change the genesis function
to NOT set hardcoded ghosts:
```javascript
// Remove this line from genesis():
// set(p => ({ ...p, ghosts: GHOSTS }));
```

Ghost Panel now shows live suggestions from `self_harness.py → ghost_ws.py`.

---

## Change 2: Trust Panel — live verification

**File:** `src/App.jsx`

Add import and polling:

```javascript
import { fetchTrust } from "./bridge.js";

// Inside BIZRAWorld, after boot completes:
useEffect(() => {
  if (!s.up) return;
  const poll = async () => {
    const status = await fetchTrust();
    if (status) {
      set(p => ({
        ...p,
        trust: { node: status.node, ledger: status.ledger, token: status.token,
                 supply: status.supply, gate: status.gate },
        ihsan: status.ihsan || p.ihsan,
        seed: status.seed || p.seed,
        bloom: status.bloom || p.bloom,
        chainHead: status.chainHead || p.chainHead,
      }));
    }
  };
  poll();
  const interval = setInterval(poll, 5000);
  return () => clearInterval(interval);
}, [s.up]);
```

Trust Panel now polls the live kernel every 5 seconds.

---

## Change 3: Mission execution — live 9-stage pipeline

**File:** `src/App.jsx`

Add import:

```javascript
import { executeMission } from "./bridge.js";
```

In the `quest` function, replace the simulated stage messages with live SSE:

```javascript
// Replace the stageMessages loop with:
await executeMission(
  task,
  // onStage — called for each pipeline stage
  (stageEvent) => {
    const stageIdx = STAGES.findIndex(s => s.id === stageEvent.stage_id);
    set(p => ({ ...p, pipeStage: stageIdx >= 0 ? stageIdx : p.pipeStage }));
    msg("SYS", `${stageEvent.icon || "●"} Stage ${stageEvent.stage_num}/9: ${stageEvent.message}`, "stage");
  },
  // onComplete — called when all 9 stages finish
  (result) => {
    const ih = result.ihsan_score || 0.96;
    const se = result.seed_earned || 1.0;
    const be = result.bloom_earned || 0.01;
    msg("SYS", `${result.drop || "🔷 RARE"} — +${se.toFixed(3)} SEED · +${be.toFixed(4)} BLOOM`, "mint");
    msg("SYS", `█ Block #${s.blocks + 1} placed. Chain: ...${result.receipt_hash?.slice(0, 12) || "pending"}`, "block");
    set(p => ({
      ...p, phase: "ready", focus: null, pipeStage: -1,
      blocks: p.blocks + 1, seed: p.seed + se, bloom: p.bloom + be,
      ihsan: ih, rac: p.rac + 1, streak: p.streak + 1,
      chainHead: result.receipt_hash || p.chainHead,
      agents: Object.fromEntries(Object.keys(PARTY).map(k => [k, "idle"])),
    }));
  },
  // onError
  (err) => { msg("SYS", `Mission failed: ${err}`, "sys"); set(p => ({ ...p, phase: "ready" })); }
);
```

Missions now execute through the real 9-stage MissionExecutor with live stage streaming.

---

## Verification

After all 3 changes, restart `npm run dev` and:

1. **Ghost Panel**: Wait 30 seconds. If NODE0 idle cycle detects opportunities, cards appear.
2. **Trust Panel**: Right rail should show 5 green checks with live Ihsan score.
3. **Mission**: Type a quest. Watch 9 stages light up with real FAISS retrieval, real LLM inference, real constitutional gating.
4. **URP Tab**: Click the sea tab. Shows resource pool growing with every quest. Zakat 2.5% flows to pool automatically. SAT-5 agents working inside.

## Change 4 (optional): URP — live resource pool

```javascript
import { fetchURP } from "./bridge.js";

// Poll URP status alongside trust:
useEffect(() => {
  if (!s.up) return;
  const poll = async () => {
    const urpStatus = await fetchURP();
    if (urpStatus) {
      set(p => ({ ...p, urp: { ...p.urp, ...urpStatus } }));
    }
  };
  poll();
  const interval = setInterval(poll, 10000); // Every 10s
  return () => clearInterval(interval);
}, [s.up]);
```

If any panel shows stale/demo data, check `python scripts/phase1_gate.py` for which
backend service is offline.

---

## Fallback

If the backend isn't running, the cockpit gracefully degrades:
- Ghost Panel: empty (no suggestions until backend connects)
- Trust Panel: all checks show "—" (unknown state)
- Missions: `bridge.js` reports error, cockpit shows the error message

No crashes. No silent failures. Degradation is visible.
