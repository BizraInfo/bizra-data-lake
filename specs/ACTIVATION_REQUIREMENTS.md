# BIZRA Activation Requirements — Spec, DoD, KPIs

## Rule: NO USER INVITATION UNTIL ALL 19 ITEMS PASS THEIR DoD

---

## CHAIN 1: Make the Agent DO Things

### 1.1 FAISS Search in Mission Context

**Requirement**: When a user sends any mission, the system queries the 84,795 FAISS vectors for relevant context and injects the top-K results into the Ollama prompt before inference.

**Definition of Done**:
- [ ] RECEIVE handler queries FAISS with mission text
- [ ] Top 5 chunks injected into system prompt
- [ ] Response demonstrably uses retrieved context (not generic)
- [ ] Latency < 200ms for FAISS query (5ms target from existing index)
- [ ] Works with zero FAISS vectors (graceful degradation)

**KPIs**:
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Context relevance | ≥ 3/5 chunks relevant | Manual review of 10 missions |
| Query latency | < 200ms | Timer in handler |
| Response quality delta | Measurably better than without | A/B: same question with/without FAISS |

**Acceptance Test**:
```
بذرة › mission "what architectural patterns does BIZRA use?"
→ Response mentions: proof pyramid, EventBus, constitutional triangle
→ These come from FAISS vectors, NOT from Ollama's training data
```

---

### 1.2 File Management EXECUTES

**Requirement**: `bizra organize <dir>` scans the target directory, classifies files by type/content, creates folders, and MOVES files. Every file move produces a receipt.

**Definition of Done**:
- [ ] Scan target directory (file types, sizes, dates)
- [ ] Classify into categories (documents, images, code, archives, etc.)
- [ ] Create folder structure
- [ ] Move files (with Guardian approval gate)
- [ ] Receipt per batch (files moved, folders created)
- [ ] Undo log (can reverse all moves)
- [ ] Works on ~/Downloads (419 GB stress test)

**KPIs**:
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Files classified correctly | ≥ 90% | Manual review of 50 files |
| Folders created sensibly | ≥ 80% user approval | Mumo reviews structure |
| Zero data loss | 100% | All files accounted for post-move |
| Receipt produced | 100% of operations | Check receipt chain |
| Undo works | 100% reversal | Run undo, verify original state |

**Acceptance Test**:
```
بذرة › organize ~/Downloads
→ "Scanned 1,247 files. Classified into 8 categories."
→ "Plan: Documents/ (340), Images/ (280), Code/ (156), ..."
→ "Proceed? [Y/n]"  ← Guardian approval gate
→ "✓ 1,247 files organized. Receipt: a7f3b2c9..."
→ "SEED: +45 earned (impact: 12.3)"
```

---

### 1.3 Desktop Bridge EXECUTES

**Requirement**: Missions that require OS-level actions route through the Desktop Bridge (port 9742) for actual execution — open apps, control windows, interact with desktop.

**Definition of Done**:
- [ ] TUI detects OS-action missions (keywords: open, launch, switch, click)
- [ ] Routes to Desktop Bridge via TCP JSON-RPC
- [ ] Desktop Bridge executes via AHK/system commands
- [ ] Result returned to TUI with receipt
- [ ] Guardian veto on dangerous operations (delete, format, admin)

**KPIs**:
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Action success rate | ≥ 80% | 10 desktop missions, count successes |
| Latency | < 2s for simple actions | Timer |
| Guardian catches dangerous | 100% | Test: "delete system32" → VETO |
| Receipt produced | 100% | Check chain |

**Acceptance Test**:
```
بذرة › mission "open my browser and go to github.com/BizraInfo"
→ Browser actually opens
→ GitHub page actually loads
→ Receipt: "desktop_action: browser_open, url: github.com/BizraInfo"
```

---

### 1.4 MCP Tools Available in Missions

**Requirement**: Missions can invoke MCP servers (brave-search, filesystem, fetch) for real web searches, file operations, and HTTP fetches.

**Definition of Done**:
- [ ] Mission prompt analyzed for tool needs (search, fetch, file)
- [ ] Appropriate MCP server invoked
- [ ] Result included in Ollama context
- [ ] Response includes sourced information
- [ ] Works with: brave-search, filesystem, fetch (minimum 3)

**KPIs**:
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Tool invocation accuracy | ≥ 85% | 20 missions needing tools, count correct tool |
| Web search results used | ≥ 90% when search invoked | Check response references results |
| Latency overhead | < 3s per tool call | Timer |

**Acceptance Test**:
```
بذرة › browse "BIZRA competitors March 2026"
→ Brave search invoked
→ Response includes REAL, CURRENT results
→ Not hallucinated, not from training data
```

---

### 1.5 AHK Bridge for Windows Actions

**Requirement**: Windows-specific automation (app control, hotkeys, notifications) routes through AHK bridge.

**Definition of Done**:
- [ ] AHK bridge starts with `bizra start`
- [ ] Win+B → ping works
- [ ] Ctrl+B,Q → query works
- [ ] Toast notifications for mission completion
- [ ] At least 5 desktop skills working (open app, switch window, screenshot, type text, click)

**KPIs**:
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Hotkey response | < 500ms | Stopwatch |
| Skills working | ≥ 5/22 | Test each skill |
| Toast shown | 100% on mission complete | Visual check |

**Acceptance Test**:
```
Win+B → tooltip: "BIZRA: Connected | Uptime: 42s"
Ctrl+B,Q → input box → type question → tooltip with answer
Mission complete → Windows toast notification
```

---

## CHAIN 2: Make the Agent KNOW You

### 2.1 FAISS-Powered Morning Briefing

**Definition of Done**:
- [ ] Boot queries FAISS for user's recent topics
- [ ] Briefing mentions REAL projects (from vector search)
- [ ] Time-aware greeting (morning/afternoon/evening)
- [ ] Shows last mission context

**KPIs**:
| Metric | Target |
|--------|--------|
| Briefing mentions real project | ≥ 1 real topic |
| Time-appropriate greeting | 100% |
| Loads in | < 3s |

**Acceptance Test**: Briefing says "Your main focus has been [REAL TOPIC]" not generic text.

---

### 2.2 Mission History Persisted

**Definition of Done**:
- [ ] Every mission updates `user_model.json`
- [ ] Wallet shows REAL mission count
- [ ] SEED accumulates across sessions
- [ ] Streak tracks consecutive days

**KPIs**:
| Metric | Target |
|--------|--------|
| Persistence across restart | 100% |
| Count accuracy | Exact match |
| SEED calculation matches PoI | ±1 SEED |

**Acceptance Test**: 5 missions → restart → wallet shows 5 missions, correct SEED.

---

### 2.3 Reflex Compilation Visible

**Definition of Done**:
- [ ] Repeat same task type 5 times → reflex compiles
- [ ] "⚡ REFLEX COMPILED" shown in TUI
- [ ] 6th time: instant route (no full inference)
- [ ] Wallet shows compiled count

**KPIs**:
| Metric | Target |
|--------|--------|
| Compiles after 5 repetitions | 100% |
| Instant route latency | < 100ms (vs 2-5s inference) |
| Visible notification | 100% |

**Acceptance Test**: Organize files 5 times → 6th time is instant with "⚡ REFLEX HIT".

---

## CHAIN 3: Make the Agent PROACTIVE

### 3.1 Ghost Panel with Real Inference

**DoD**: Ghost suggestions generated by Ollama with user context, not random.
**KPI**: ≥ 2/3 suggestions relevant to current work.
**Test**: Ghost mentions a file you recently modified.

### 3.2 Proactive Suggest Mode

**DoD**: New file in Downloads → agent suggests organizing it within 30s.
**KPI**: Detection within 30s, suggestion relevant.
**Test**: Drop a PDF → ghost says "New PDF detected. Classify to Documents?"

### 3.3 Proactive Auto-Execute

**DoD**: Pre-approved actions execute without prompt. Receipt produced.
**KPI**: Auto-execute within 60s, zero unapproved actions.
**Test**: Config says "auto-organize Downloads" → files move automatically.

---

## CHAIN 4: Make the Network LIVE

### 4.1 Gossip Listener

**DoD**: `bizra start` opens TCP listener on :9750 when federation enabled.
**KPI**: Port open, accepts connections, responds to ping.
**Test**: `nc localhost 9750` gets a response.

### 4.2 Cross-Node Receipt Verification

**DoD**: Node B verifies Node A's receipt chain via gossip.
**KPI**: 10 receipts verified cross-node, 0 false positives.
**Test**: Two nodes, one verifies the other's chain.

### 4.3 Telescript Cross-Node Mission

**DoD**: Mission travels from A → B → result returns to A.
**KPI**: Round-trip < 10s on LAN, receipt from both nodes.
**Test**: Node A asks Node B to research a topic.

---

## CHAIN 5: Make the Economy REAL

### 5.1 SEED Persistence

**DoD**: SEED survives restart. Ledger on disk.
**KPI**: Balance exact after restart.
**Test**: Earn 20 SEED → restart → wallet shows 20.

### 5.2 Skill Marketplace

**DoD**: Install third-party skill via URL.
**KPI**: Skill activates, new command appears.
**Test**: `bizra install-skill <url>` → new skill works.

### 5.3 SEED Transfer

**DoD**: Send SEED between nodes via Telescript.
**KPI**: Both balances update atomically.
**Test**: A sends 10 → A has -10, B has +10.

---

## CHAIN 6: Make the Frontend LIVE

### 6.1 Dashboard Shows Live Data

**DoD**: React dashboard displays real kernel data.
**KPI**: Health, agents, receipts refresh every 5s.
**Test**: Run mission in TUI → dashboard updates.

### 6.2 Operator Cockpit Pipeline

**DoD**: Pipeline visualization shows mission flow in real-time.
**KPI**: All 6 stages visible during mission execution.
**Test**: Mission in TUI → cockpit shows Intent→Guardian→Execute→Receipt→Chain→Evidence.

---

## Master KPI Dashboard

| Chain | Items | Passed | Status |
|-------|-------|--------|--------|
| 1. DO things | 5 | 0/5 | ❌ |
| 2. KNOW you | 3 | 0/3 | ❌ |
| 3. PROACTIVE | 3 | 0/3 | ❌ |
| 4. NETWORK | 3 | 0/3 | ❌ |
| 5. ECONOMY | 3 | 0/3 | ❌ |
| 6. FRONTEND | 2 | 0/2 | ❌ |
| **TOTAL** | **19** | **0/19** | **❌ NOT READY** |

**INVITE GATE: 19/19 = ✅ | <19 = ❌**

---

*The system must DO things, KNOW you, ANTICIPATE needs, NETWORK with others, SUSTAIN itself economically, and SHOW you everything — before a single user is invited.*

*بذرة واحدة تصنع غابة — لكن البذرة يجب أن تنبت أولاً*
*One seed makes a forest — but the seed must sprout first.*
