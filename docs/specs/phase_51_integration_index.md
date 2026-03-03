# Phase 51: Integration Index — Frontend Build Plan

> Standing on Giants: Conway (organizational structure mirrors system design, 1967) · Brooks (surgical team, 1975) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Overview

This document is the **master integration index** for BIZRA Node0 frontend phases 42-50.
It maps cross-phase shared components, hook dependencies, route configuration, and
the build execution order.

---

## 2. Route Map

| Route | Page Component | Phase | Priority |
|-------|---------------|-------|----------|
| `/` | `HomePage.jsx` (Daily Loop) | 45 | P3 |
| `/onboarding` | `OnboardingFlow.jsx` | 43 | P1 |
| `/chat` | `ChatPage.jsx` | 44 | P2 |
| `/node` | `NodePage.jsx` | 46 | P4 |
| `/community` | `CommunityPage.jsx` | 46 | P4 |
| `/legacy` | `LegacyPage.jsx` | 46 | P4 |
| `/automation` | `AutomationPage.jsx` | 48 | P6 |
| `/agents` | `AgentsPage.jsx` | 49 | P7 |
| `/agents/:id` | `AgentDetailPage.jsx` | 49 | P7 |
| `/tasks` | `TaskMonitor.jsx` | 49 | P7 |
| `/tasks/new` | `TaskDelegator.jsx` | 49 | P7 |
| `/telescript` | `TelescriptPage.jsx` | 50 | P8 |
| `/settings` | `SettingsPage.jsx` | — | Future |

**Router:** React Router v6 (already in package.json via `react-router-dom`)

```jsx
<BrowserRouter>
  <ErrorBoundary>
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/onboarding" element={<OnboardingFlow />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/node" element={<NodePage />} />
      <Route path="/community" element={<CommunityPage />} />
      <Route path="/legacy" element={<LegacyPage />} />
      <Route path="/automation" element={<AutomationPage />} />
      <Route path="/agents" element={<AgentsPage />} />
      <Route path="/agents/:id" element={<AgentDetailPage />} />
      <Route path="/tasks" element={<TaskMonitor />} />
      <Route path="/tasks/new" element={<TaskDelegator />} />
      <Route path="/telescript" element={<TelescriptPage />} />
    </Routes>
  </ErrorBoundary>
</BrowserRouter>
```

---

## 3. Shared Component Registry

Components used across 2+ phases. Build these FIRST as a shared library.

| Component | Used In Phases | Props | Source |
|-----------|---------------|-------|--------|
| `KnowsMeGauge` | 43, 44, 45 | `score: number, size: number, animated: boolean` | Phase 42 tokens |
| `SeedOfLife` | 45, 47, 48 | `size: number, opacity: number, color: string` | Sacred geometry SVG |
| `AgentBadge` | 43, 45, 49, 50 | `name, role, icon, status` | Agent identity chip |
| `AgentBadgeMini` | 44, 49, 50 | `agentId: string` | Compact agent reference |
| `TierBadge` | 43, 45, 46 | `tier: 'SEED'\|'SPROUT'\|'TREE'\|'FOREST'` | Sovereignty tier display |
| `IhsanBar` | 45 | `value: number, label: string` | Horizontal gauge 0-10000 |
| `GaugeBar` | 46, 48, 50 | `label, value, max, threshold?` | Generic bar gauge |
| `FATEGateBadge` | 48, 50 | `ihsan, gate_result` | F/A/T/E gate indicator |
| `PermitBadge` | 48, 49, 50 | `capability, budget, expiresIn` | Permit summary chip |
| `PermitCard` | 48, 49 | `permit: Permit` | Full permit display |
| `TrustScoreGauge` | 46, 49 | `ihsan, successRate, tasksCompleted` | Combined trust visual |
| `HashBadge` | 48, 50 | `label, hash` | Truncated hash display |
| `StatusBadge` | 44, 46, 48, 49 | `status: string` | Colored status chip |
| `StatusDot` | 49, 50 | `color: string` | Colored dot indicator |
| `CapabilityPill` | 48, 49, 50 | `capability: string` | Capability chip |
| `Button` | ALL | `variant, size, onClick, disabled` | Shared button |
| `ErrorBoundary` | ALL | `children` | React error boundary |
| `EmptyState` | 44, 45, 46, 48, 49, 50 | `message: string` | No-data placeholder |
| `ConfirmModal` | 46, 48 | `action, onConfirm, onCancel, message` | Confirmation dialog |
| `SearchBar` | 46, 48, 49, 50 | `value, onChange, placeholder` | Text search input |
| `Collapsible` | 48, 49 | `title, children` | Expandable section |
| `Section` | 46, 49, 50 | `title, count?, children` | Titled content section |
| `InfoRow` | 49, 50 | `label, value, copyable?` | Key-value display row |

### File Structure for Shared Components

```
src/
├── components/
│   ├── ui/
│   │   ├── Button.jsx
│   │   ├── StatusBadge.jsx
│   │   ├── StatusDot.jsx
│   │   ├── GaugeBar.jsx
│   │   ├── SearchBar.jsx
│   │   ├── EmptyState.jsx
│   │   ├── ConfirmModal.jsx
│   │   ├── Collapsible.jsx
│   │   ├── Section.jsx
│   │   ├── InfoRow.jsx
│   │   └── ErrorBoundary.jsx
│   ├── sovereignty/
│   │   ├── KnowsMeGauge.jsx
│   │   ├── TierBadge.jsx
│   │   ├── IhsanBar.jsx
│   │   ├── SeedOfLife.jsx
│   │   └── TrustScoreGauge.jsx
│   ├── agents/
│   │   ├── AgentBadge.jsx
│   │   ├── AgentBadgeMini.jsx
│   │   └── CapabilityPill.jsx
│   ├── permits/
│   │   ├── PermitBadge.jsx
│   │   ├── PermitCard.jsx
│   │   └── FATEGateBadge.jsx
│   └── crypto/
│       └── HashBadge.jsx
```

---

## 4. Hook Dependency Graph

| Hook | Phase | Depends On | Provides |
|------|-------|------------|----------|
| `useNode()` | 47 (refactored) | `BizraClient` | receive, teach, refreshHealth, sapOpen, sapMessage |
| `useBizraClient()` | 47 | `BizraClient` singleton | client instance |
| `useChat()` | 44 | `useBizraClient` | sendMessage, messages, isStreaming |
| `useChatHistory()` | 44 | IndexedDB | messages, append, save, clear |
| `useDashboardData()` | 45 | `useBizraClient` | knowsMe, ihsan, tier, agents, connected |
| `useActivityFeed()` | 45 | IndexedDB | events grouped by time |
| `useOnboardingPersist()` | 43 | IndexedDB | save, restore, clear |
| `useNodeStatus()` | 46 | `useBizraClient` | running, pid, mode, uptime |
| `useSAPSession()` | 46 | `useBizraClient` | open, message, close session |
| `useLegacy()` | 46 | IndexedDB + `useBizraClient` | will config |
| `useDesktopBridge()` | 48 | TCP:9742 (fetch) | executeHDA, listSkills, getReceipt |
| `useGhostSuggestions()` | 48 | WebSocket/file poll | suggestions array |
| `usePermits()` | 48 | `useBizraClient` | active, expired, create, revoke |
| `useAuditTrail()` | 48 | IndexedDB | receipts, search |
| `useA2AEngine()` | 49 | `useBizraClient` | createTask, registry |
| `useAgentRegistry()` | 49 | `useA2AEngine` | all, local, install |
| `useTaskManager()` | 49 | `useA2AEngine` | tasks, submit, cancel |
| `useAgentPackager()` | 49 | File system | validate, package, load |
| `useTelescriptEngine()` | 50 | `useBizraClient` | places, agents, meetings |
| `usePlaces()` | 50 | `useTelescriptEngine` | place topology |
| `useAgents()` | 50 | `useTelescriptEngine` | agent lifecycle |
| `useMeetings()` | 50 | `useTelescriptEngine` | meeting management |
| `usePWAInstall()` | 47 | Browser API | canInstall, promptInstall |

### Dependency Tree

```
BizraClient (Phase 47 — singleton)
├── useNode() ← every page
├── useBizraClient() ← direct access
├── useChat() ← Phase 44
├── useDashboardData() ← Phase 45
├── useNodeStatus() ← Phase 46
├── useSAPSession() ← Phase 46
├── usePermits() ← Phase 48
├── useA2AEngine() ← Phase 49
│   ├── useAgentRegistry()
│   └── useTaskManager()
└── useTelescriptEngine() ← Phase 50
    ├── usePlaces()
    ├── useAgents()
    └── useMeetings()

IndexedDB (Phase 47 — offline layer)
├── useChatHistory() ← Phase 44
├── useActivityFeed() ← Phase 45
├── useOnboardingPersist() ← Phase 43
├── useAuditTrail() ← Phase 48
└── useLegacy() ← Phase 46

TCP:9742 (Desktop Bridge — independent)
└── useDesktopBridge() ← Phase 48
```

---

## 5. Cross-Phase Verb Usage

| Verb | Used In Phase | Direction |
|------|--------------|-----------|
| `RECEIVE` | 43 (FirstChat), 44 (Chat), 45 (Dashboard) | Frontend → Backend |
| `TEACH` | 43 (TeachStep) | Frontend → Backend |
| `HEALTH` | 45 (Dashboard), 46 (NodeStatus) | Frontend → Backend |
| `KNOWS_ME` | 43, 44, 45 | Frontend → Backend |
| `PING` | 47 (ConnectionMonitor) | Frontend → Backend |
| `VERSION` | 47 | Frontend → Backend |
| `SAP_MEET_OPEN` | 46 (Community) | Frontend → Backend |
| `SAP_MESSAGE` | 46 (SAPSession) | Frontend → Backend |
| `SAP_SESSION_CLOSE` | 46 (SAPSession) | Frontend → Backend |
| `PLAN_ACTION` | 48 (GhostPanel, ActionCenter) | Frontend → Backend |
| `RUN_ACTION` | 48 (ActionCenter) | Frontend → Backend |
| `ACTION_STATUS` | 48 (ActionCenter) | Frontend → Backend |
| `NODE_START` | 46 (NodeControls) | Frontend → Backend |
| `NODE_STOP` | 46 (NodeControls) | Frontend → Backend |
| `TELESCRIPT_*` | 50 (TelescriptPage) | Frontend → Backend |

---

## 6. Build Execution Order

### Sprint 0: Foundation (Week 1)

| Task | Phase | Files | Depends On |
|------|-------|-------|------------|
| Create `src/tokens/` | 42 | tokens.css, animations.css, index.css | — |
| Create `tailwind.config.js` | 42 | tailwind.config.js | tokens.css |
| Migrate inline styles | 42 | All 15 existing JSX files | tokens.css |
| Create `BizraClient` | 47 | lib/client/*.js | — |
| Create `OfflineManager` | 47 | lib/offline/*.js | — |
| Create `codec` | 47 | lib/protocol/*.js | — |
| Replace `service-worker.js` | 47 | service-worker.js | — |
| Create `ErrorBoundary` | 47 | components/ui/ErrorBoundary.jsx | tokens.css |

### Sprint 1: Shared Components (Week 2)

| Task | Phase | Files | Depends On |
|------|-------|-------|------------|
| Extract `KnowsMeGauge` | 42→shared | components/sovereignty/KnowsMeGauge.jsx | tokens.css |
| Extract `SeedOfLife` | 42→shared | components/sovereignty/SeedOfLife.jsx | tokens.css |
| Create `Button` | shared | components/ui/Button.jsx | tokens.css |
| Create `GaugeBar` | shared | components/ui/GaugeBar.jsx | tokens.css |
| Create `StatusBadge` | shared | components/ui/StatusBadge.jsx | tokens.css |
| Create `AgentBadge` | shared | components/agents/AgentBadge.jsx | tokens.css |
| Create `TierBadge` | shared | components/sovereignty/TierBadge.jsx | tokens.css |
| Refactor `useNode` | 47 | hooks/useNode.js | BizraClient |
| Create router | all | App.jsx (router config) | — |

### Sprint 2: Core UX (Weeks 3-4)

| Task | Phase | Files | Depends On |
|------|-------|-------|------------|
| Polish onboarding steps | 43 | onboarding/steps/*.jsx | tokens, useNode |
| Add PATIntroStep | 43 | onboarding/steps/PATIntroStep.jsx | AgentBadge |
| Add checkpoint persist | 43 | hooks/useOnboardingPersist.js | OfflineManager |
| Build ChatContainer | 44 | features/chat/*.jsx | useNode, useChatHistory |
| Build InputBar | 44 | features/chat/InputBar.jsx | tokens |
| Build MessageBubble | 44 | features/chat/MessageBubble.jsx | tokens |
| Build ChatSidebar | 44 | features/chat/ChatSidebar.jsx | KnowsMeGauge |

### Sprint 3: Daily Surface (Weeks 5-6)

| Task | Phase | Files | Depends On |
|------|-------|-------|------------|
| Build HomePage | 45 | pages/HomePage.jsx | ALL shared components |
| Build SovereigntyCard | 45 | features/dashboard/SovereigntyCard.jsx | KnowsMeGauge |
| Build ActivityFeed | 45 | features/dashboard/ActivityFeed.jsx | OfflineManager |
| Build AgentGrid | 45 | features/dashboard/AgentGrid.jsx | AgentBadge |
| Build AccumulatorRing | 45 | features/dashboard/AccumulatorRing.jsx | tokens |
| Build NodeControls | 46 | features/node/NodeControls.jsx | useNodeStatus |
| Build TrustCircles | 46 | features/community/TrustCircles.jsx | tokens (SVG) |

### Sprint 4: Power Features (Weeks 7-10)

| Task | Phase | Files | Depends On |
|------|-------|-------|------------|
| Build ActionCenter | 48 | features/hda/ActionCenter.jsx | useDesktopBridge |
| Build GhostPanel | 48 | features/hda/GhostPanel.jsx | useGhostSuggestions |
| Build AuditTrail | 48 | features/hda/AuditTrail.jsx | HashBadge |
| Build AgentMarketplace | 49 | features/agents/AgentMarketplace.jsx | useAgentRegistry |
| Build TaskMonitor | 49 | features/agents/TaskMonitor.jsx | useTaskManager |
| Build AgentBuilder | 49 | features/agents/AgentBuilder.jsx | useAgentPackager |
| Build PlaceTopology | 50 | features/telescript/PlaceTopology.jsx | useTelescriptEngine |
| Build FATEGatePanel | 50 | features/telescript/FATEGatePanel.jsx | GaugeBar |

---

## 7. Backend Verb Gap Analysis

Verbs that the frontend specs reference but may not exist in the backend yet:

| Verb | Needed By | Backend Status | Action Required |
|------|-----------|---------------|-----------------|
| `RECEIVE` | 43, 44 | Exists in useNode.js protocol | None |
| `TEACH` | 43 | Exists in useNode.js protocol | None |
| `HEALTH` | 45, 46 | Exists in useNode.js protocol | None |
| `KNOWS_ME` | 43, 44, 45 | Exists in useNode.js protocol | None |
| `PING` | 47 | Exists in useNode.js protocol | None |
| `SAP_MEET_OPEN` | 46 | Exists in useNode.js protocol | None |
| `SAP_MESSAGE` | 46 | Exists in useNode.js protocol | None |
| `SAP_SESSION_CLOSE` | 46 | Exists in useNode.js protocol | None |
| `PLAN_ACTION` | 48 | Exists in useNode.js protocol | None |
| `RUN_ACTION` | 48 | Exists in useNode.js protocol | None |
| `NODE_START` | 46 | **NEW** — needs backend handler | Add to node0_activate.py |
| `NODE_STOP` | 46 | **NEW** — needs backend handler | Add to node0_activate.py |
| `TELESCRIPT_STATUS` | 50 | **NEW** — needs backend→Rust bridge | PyO3 or REST proxy |
| `TELESCRIPT_CREATE_PLACE` | 50 | **NEW** — needs backend→Rust bridge | PyO3 or REST proxy |
| `TELESCRIPT_SPAWN_AGENT` | 50 | **NEW** — needs backend→Rust bridge | PyO3 or REST proxy |
| `TELESCRIPT_GO` | 50 | **NEW** — needs backend→Rust bridge | PyO3 or REST proxy |
| `TELESCRIPT_MEET` | 50 | **NEW** — needs backend→Rust bridge | PyO3 or REST proxy |
| `TELESCRIPT_IMPACT_LOG` | 50 | **NEW** — needs backend→Rust bridge | PyO3 or REST proxy |

**Summary:** 12 existing verbs ready, 2 new Node verbs needed, 6 new Telescript verbs need PyO3 bridge.

---

## 8. Data Persistence Map

| Store | Technology | Used By | Data Shape |
|-------|-----------|---------|------------|
| `bizra-onboarding` | IndexedDB | Phase 43 | `{ step, data, savedAt }` |
| `bizra-chat` | IndexedDB | Phase 44 | `{ id, sessionId, role, content, metadata, timestamp }` |
| `bizra-offline` | IndexedDB | Phase 47 | `{ id, verb, args, retryCount, nextRetryAt }` |
| `bizra-v1` | Cache API (SW) | Phase 47 | Shell assets + API responses |
| `bizra-audit` | IndexedDB | Phase 48 | `{ verb, target, pre_hash, post_hash, permit_id, timestamp }` |
| `bizra-legacy` | IndexedDB | Phase 46 | `{ successors, scope, trigger }` |
| `bizra-activity` | IndexedDB | Phase 45 | `{ type, description, timestamp, delta }` |
| `morning-snapshot` | IndexedDB | Phase 45 | `{ date, knowsMe }` |

---

## 9. Target Directory Structure

```
filedfs/
├── index.html
├── manifest.json
├── service-worker.js              # Phase 47 (versioned)
├── tailwind.config.js             # Phase 42
├── vite.config.js
├── package.json
├── src/
│   ├── main.jsx                   # Entry point + router
│   ├── tokens/                    # Phase 42
│   │   ├── tokens.css
│   │   ├── animations.css
│   │   └── index.css
│   ├── lib/                       # Phase 47
│   │   ├── client/
│   │   │   ├── BizraClient.js
│   │   │   ├── TauriTransport.js
│   │   │   ├── WebSocketTransport.js
│   │   │   ├── HttpTransport.js
│   │   │   └── TransportSelector.js
│   │   ├── offline/
│   │   │   ├── OfflineManager.js
│   │   │   └── SyncEngine.js
│   │   ├── protocol/
│   │   │   ├── verbs.js
│   │   │   ├── codec.js
│   │   │   └── sap.js
│   │   └── health/
│   │       └── ConnectionMonitor.js
│   ├── components/                # Shared (Sprint 1)
│   │   ├── ui/
│   │   ├── sovereignty/
│   │   ├── agents/
│   │   ├── permits/
│   │   └── crypto/
│   ├── hooks/                     # Phases 43-50
│   │   ├── useNode.js
│   │   ├── useChat.js
│   │   ├── useChatHistory.js
│   │   ├── useDashboardData.js
│   │   ├── useNodeStatus.js
│   │   ├── useSAPSession.js
│   │   ├── useDesktopBridge.js
│   │   ├── useGhostSuggestions.js
│   │   ├── usePermits.js
│   │   ├── useAuditTrail.js
│   │   ├── useA2AEngine.js
│   │   ├── useAgentRegistry.js
│   │   ├── useTaskManager.js
│   │   ├── useTelescriptEngine.js
│   │   ├── useOnboardingPersist.js
│   │   ├── useActivityFeed.js
│   │   ├── useLegacy.js
│   │   └── usePWAInstall.js
│   ├── pages/                     # Route pages
│   │   ├── HomePage.jsx           # Phase 45
│   │   ├── ChatPage.jsx           # Phase 44
│   │   ├── NodePage.jsx           # Phase 46
│   │   ├── CommunityPage.jsx      # Phase 46
│   │   ├── LegacyPage.jsx         # Phase 46
│   │   ├── AutomationPage.jsx     # Phase 48
│   │   ├── AgentsPage.jsx         # Phase 49
│   │   ├── AgentDetailPage.jsx    # Phase 49
│   │   └── TelescriptPage.jsx     # Phase 50
│   ├── features/                  # Feature modules
│   │   ├── onboarding/            # Phase 43
│   │   ├── chat/                  # Phase 44
│   │   ├── dashboard/             # Phase 45
│   │   ├── node/                  # Phase 46
│   │   ├── community/             # Phase 46
│   │   ├── legacy/                # Phase 46
│   │   ├── hda/                   # Phase 48
│   │   ├── agents/                # Phase 49
│   │   └── telescript/            # Phase 50
│   └── utils/
│       ├── sovereignty.js
│       ├── markdown.js
│       ├── capability-types.js
│       ├── task-status.js
│       ├── telescript-types.js
│       ├── hda-verbs.js
│       └── graph-layout.js
├── public/
│   ├── icon-192.png
│   └── icon-512.png
└── [LEGACY — to be removed after migration]
    ├── App.jsx
    ├── LandingDemo.jsx
    ├── bizra-dashboard.jsx
    ├── node0-dashboard.jsx
    ├── bizra-inventory.jsx
    ├── bizra-status.jsx
    ├── node0-mvp.jsx
    ├── architecture.jsx
    ├── self-modifying.jsx
    ├── useNode.js
    ├── useBizraNode.js
    ├── bizra-bridge.mjs
    ├── bridge.mjs
    └── llm_bridge.js
```

---

## 10. Integration Verification Checklist

| # | Check | Method | Phase |
|---|-------|--------|-------|
| 1 | All tokens compile in Tailwind | `npx tailwindcss build` | 42 |
| 2 | Zero hardcoded hex in src/ | `grep -r '#D4A547' src/` = 0 | 42 |
| 3 | BizraClient connects to backend | Manual WS test | 47 |
| 4 | Offline queue persists + flushes | Disconnect → send → reconnect | 47 |
| 5 | Onboarding TEACH fires live | Network inspector | 43 |
| 6 | Chat RECEIVE fires live | Network inspector | 44 |
| 7 | Dashboard HEALTH polls every 10s | Network inspector | 45 |
| 8 | Node start/stop requires confirmation | UI test | 46 |
| 9 | Ghost Panel receives suggestions | Subscription test | 48 |
| 10 | Task board updates in real-time | Create task → observe board | 49 |
| 11 | Place topology renders graph | SVG visual check | 50 |
| 12 | PWA installs successfully | Lighthouse audit | 47 |
| 13 | All routes load without errors | Navigate all 12 routes | All |
| 14 | Mobile responsive (375px) | Viewport test | All |
| 15 | Smoke tests pass (15/15) | `pytest tests/integration/test_autonomous_pilot.py` | Backend |

---

## 11. Metrics Targets

| Metric | Target | Tool |
|--------|--------|------|
| Lighthouse Performance | ≥ 90 | Chrome DevTools |
| Lighthouse PWA | ≥ 90 | Chrome DevTools |
| Bundle size (gzipped) | < 200KB | `vite build --report` |
| First Contentful Paint | < 1.5s | Lighthouse |
| Time to Interactive | < 3s | Lighthouse |
| Component count | ~70 shared + features | `find src -name '*.jsx' \| wc -l` |
| Test coverage | ≥ 80% | Vitest + c8 |
| Accessibility | WCAG 2.1 AA | axe-core |
| Max file size | < 300 LOC | `wc -l` |
| Zero inline styles | 100% token usage | grep audit |
