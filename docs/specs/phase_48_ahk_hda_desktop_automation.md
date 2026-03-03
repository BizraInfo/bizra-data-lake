# Phase 48: AHK + HDA Desktop Automation UI

> Standing on Giants: General Magic (Telescript permits, 1994) · Fitts (target acquisition, 1954) · Norman (affordance + feedback, 1988) · Boyd (OODA perception-action, 1976) · Shannon (SNR entropy gate, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

BIZRA has a **production-grade Human-Desktop Automation** stack with zero frontend:

| Component | Backend Status | Frontend Status |
|-----------|---------------|-----------------|
| 8 HDA verbs (open_app, switch_window, type_text, click_element, screenshot, read_clipboard, file_open, browser_navigate) | Productized in `ahk_bridge.ahk` (1,018 LOC) | **None** |
| Desktop Bridge (TCP:9742 JSON-RPC) | `desktop_bridge.py` (76.7K), auth envelope, rate limiting | **None** |
| Permit system (capability-scoped, budget-constrained) | `permit.py` (422 LOC), 6 capabilities, HMAC-SHA256 | **None** |
| Ghost Overlay (proactive suggestions HUD) | `ghost_overlay.ahk` (403 LOC), frosted-glass | **None** — Windows-only AHK |
| Actuator Skills Ledger (extensible skill registry) | `actuator_skills.py` (123 LOC), 3 baseline skills | **None** |
| Perception-Action Loop (screenshot hash verification) | Pre/post SHA-256 hash diff per action | **None** |
| FATE Gate Pipeline (Ihsan + ADL + SNR) | GateChain → InferenceGateway | **None** |

Users cannot see, approve, audit, or manage desktop automation from the app.
The Ghost Overlay is Windows-only AHK — needs a cross-platform web equivalent.

### What Exists (Backend)

```
bin/bizra_bridge.ahk           — AHK hotkey client (Win+B, Ctrl+B+Q/S)
filedfs/ahk_bridge.ahk        — AHK bridge server (TCP:9742, 8 HDA skills)
scripts/ghost_overlay.ahk      — Frosted-glass proactive HUD
core/bridges/desktop_bridge.py — Python JSON-RPC proxy (76.7K)
core/sovereign/permit.py       — Telescript permit system
core/spearpoint/actuator_skills.py — 3 registered skills
```

### What's Missing (Frontend)

1. **Action Center** — See pending/running/completed HDA actions
2. **Approval Gate** — Approve/reject proposed actions before execution
3. **Permit Dashboard** — View/create/revoke permits
4. **Skill Registry** — Browse, search, manage actuator skills
5. **Ghost Panel** — Cross-platform proactive suggestions (replaces AHK overlay)
6. **Audit Trail** — Receipt chain: pre_hash → action → post_hash, searchable
7. **Live Desktop View** — Screenshot stream from HDA_SCREENSHOT verb

---

## 2. Architecture

```
src/
├── pages/
│   └── AutomationPage.jsx            # HDA hub layout
├── features/
│   └── hda/
│       ├── ActionCenter.jsx           # Pending/running/completed actions
│       ├── ActionCard.jsx             # Individual action with approve/reject
│       ├── ApprovalGate.jsx           # Pre-execution approval modal
│       ├── PermitDashboard.jsx        # Active permits + budget gauges
│       ├── PermitCard.jsx             # Single permit display
│       ├── CreatePermitModal.jsx      # New permit wizard
│       ├── SkillRegistry.jsx          # Actuator skill browser
│       ├── SkillCard.jsx              # Single skill with entropy badge
│       ├── GhostPanel.jsx             # Cross-platform proactive suggestions
│       ├── SuggestionCard.jsx         # Single ghost suggestion
│       ├── AuditTrail.jsx             # Receipt chain timeline
│       ├── ReceiptCard.jsx            # Single perception-action receipt
│       ├── DesktopPreview.jsx         # Live screenshot stream
│       └── FATEGateBadge.jsx          # Gate pipeline status indicator
├── hooks/
│   ├── useDesktopBridge.js            # JSON-RPC client for TCP:9742
│   ├── usePermits.js                  # Permit CRUD + validation
│   ├── useGhostSuggestions.js         # Poll/subscribe for proactive suggestions
│   └── useAuditTrail.js              # Receipt history from IndexedDB
└── utils/
    └── hda-verbs.js                   # Verb definitions + capability mapping
```

### Data Flow

```
Node0 Proactive Kernel → detects opportunity
         ↓
Ghost Panel receives suggestion (WebSocket or file poll)
         ↓
User sees suggestion card with Ihsan precheck badge
         ↓
User clicks "Execute" → ApprovalGate modal
         ↓
Approved → PLAN_ACTION verb → permit check → budget check
         ↓
RUN_ACTION verb → Desktop Bridge (TCP:9742) → AHK Bridge
         ↓
Pre-screenshot → Execute → Post-screenshot → Receipt
         ↓
ActionCenter shows result + receipt hash
         ↓
AuditTrail logs immutable receipt chain
```

---

## 3. Pseudocode: Action Center

```
PROCEDURE ActionCenter():
    STATE actions = useActionHistory()
    STATE filter = 'all'  # 'all' | 'pending' | 'running' | 'completed' | 'failed'

    RENDER:
        <div className="action-center">
            <h2>Desktop Actions</h2>

            # Filter tabs
            <TabBar
                tabs={['all', 'pending', 'running', 'completed', 'failed']}
                active={filter}
                onChange={setFilter}
                counts={{
                    pending: actions.filter(a => a.status == 'pending').length,
                    running: actions.filter(a => a.status == 'running').length,
                }}
            />

            # Action list
            <div className="action-list">
                FOR EACH action IN actions.filter(byStatus(filter)):
                    <ActionCard
                        action={action}
                        onApprove={() => approveAction(action.id)}
                        onReject={() => rejectAction(action.id)}
                        onViewReceipt={() => viewReceipt(action.receipt)}
                    />

                IF actions.length == 0:
                    <EmptyState message="No desktop actions yet." />
            </div>
        </div>

PROCEDURE ActionCard({ action, onApprove, onReject, onViewReceipt }):
    verbIcons = {
        open_app: '🖥️', switch_window: '🪟', type_text: '⌨️',
        click_element: '🖱️', screenshot: '📸', read_clipboard: '📋',
        file_open: '📁', browser_navigate: '🌐'
    }

    RENDER:
        <div className="action-card" data-status={action.status}>
            # Header: verb icon + label
            <div className="action-header">
                <span className="verb-icon">{verbIcons[action.verb]}</span>
                <span className="verb-label">{action.verb}</span>
                <StatusBadge status={action.status} />
                <FATEGateBadge
                    ihsan={action.ihsan_precheck}
                    gate={action.gate_result}
                />
            </div>

            # Payload summary
            <div className="action-body">
                <span className="action-target">{action.target || action.params?.app || action.params?.url}</span>
                IF action.params?.text:
                    <code className="action-text">{truncate(action.params.text, 100)}</code>
            </div>

            # Permit badge
            <PermitBadge
                capability={action.capability}
                budgetRemaining={action.budget_remaining}
                expiresIn={action.permit_expires_in}
            />

            # Actions (for pending)
            IF action.status == 'pending':
                <div className="action-buttons">
                    <Button variant="primary" onClick={onApprove}>Approve</Button>
                    <Button variant="ghost" onClick={onReject}>Reject</Button>
                </div>

            # Receipt (for completed)
            IF action.receipt:
                <ReceiptMini
                    preHash={action.receipt.pre_hash}
                    postHash={action.receipt.post_hash}
                    stateChanged={action.receipt.state_changed}
                    outcomeConfirmed={action.receipt.outcome_confirmed}
                    onClick={onViewReceipt}
                />

            # Timestamp
            <time>{formatRelativeTime(action.timestamp)}</time>
        </div>
```

---

## 4. Pseudocode: Approval Gate

```
PROCEDURE ApprovalGate({ action, onApprove, onReject, onCancel }):
    STATE reviewing = true

    # Constitutional check: Daughter Test
    daughterTestResult = evaluateDaughterTest(action)

    RENDER:
        <Modal title="Approve Desktop Action?" onClose={onCancel}>
            # Action summary
            <div className="approval-summary">
                <h3>{action.verb}: {action.target}</h3>
                IF action.params?.text:
                    <code>{action.params.text}</code>
                IF action.params?.url:
                    <a href={action.params.url}>{action.params.url}</a>
            </div>

            # FATE Gate Results
            <div className="gate-results">
                <GateRow label="Ihsan" score={action.ihsan_score} threshold={0.95}
                         pass={action.ihsan_score >= 0.95} />
                <GateRow label="Entropy" score={action.entropy} threshold={3.5}
                         pass={action.entropy >= 3.5} />
                <GateRow label="Daughter Test" pass={daughterTestResult.pass}
                         reason={daughterTestResult.reason} />
            </div>

            # Capability + Budget
            <div className="permit-info">
                <span>Capability: {action.capability}</span>
                <span>Budget: {action.budget_remaining} actions remaining</span>
                <span>Permit expires: {formatDuration(action.permit_expires_in)}</span>
            </div>

            # Blocked patterns warning
            IF action.blocked:
                <div className="blocked-warning">
                    <span>BLOCKED: {action.blocked_reason}</span>
                    <p>This action matches a restricted pattern and cannot proceed.</p>
                </div>

            # Decision buttons
            IF NOT action.blocked:
                <div className="approval-buttons">
                    <Button variant="primary" onClick={onApprove}>
                        Approve & Execute
                    </Button>
                    <Button variant="ghost" onClick={onReject}>Reject</Button>
                </div>
            ELSE:
                <Button variant="ghost" onClick={onCancel}>Dismiss</Button>
        </Modal>
```

---

## 5. Pseudocode: Permit Dashboard

```
PROCEDURE PermitDashboard():
    STATE permits = usePermits()
    STATE showCreate = false

    RENDER:
        <div className="permit-dashboard">
            <div className="permit-header">
                <h2>Active Permits</h2>
                <Button onClick={() => showCreate = true}>New Permit</Button>
            </div>

            <div className="permit-grid">
                FOR EACH permit IN permits.active:
                    <PermitCard permit={permit} onRevoke={() => permits.revoke(permit.id)} />
            </div>

            # Expired permits (collapsed)
            <Collapsible title={`Expired (${permits.expired.length})`}>
                FOR EACH permit IN permits.expired:
                    <PermitCard permit={permit} expired />
            </Collapsible>

            IF showCreate:
                <CreatePermitModal
                    onClose={() => showCreate = false}
                    onCreate={(config) => permits.create(config)}
                />
        </div>

PROCEDURE PermitCard({ permit, expired, onRevoke }):
    timeLeft = permit.expires_at - Date.now()
    budgetPercent = (permit.budget.actions_used / permit.budget.max_actions) * 100

    RENDER:
        <div className={`permit-card ${expired ? 'expired' : ''}`}>
            # Issuer chain
            <div className="issuer-chain">
                <span>{permit.issuer.name}</span>
                IF permit.issuer.delegated_from:
                    <span> ← {permit.issuer.delegated_from}</span>
            </div>

            # Capabilities (pill badges)
            <div className="capability-pills">
                FOR EACH cap IN permit.capabilities:
                    <CapabilityPill capability={cap} />
            </div>

            # Budget gauge
            <GaugeBar
                label="Budget"
                value={budgetPercent}
                max={100}
                sublabel={`${permit.budget.actions_used}/${permit.budget.max_actions} actions`}
            />

            # TTL countdown
            IF NOT expired:
                <Countdown seconds={timeLeft / 1000} onExpire={() => permits.refresh()} />
            ELSE:
                <span className="expired-label">Expired</span>

            # Revoke button
            IF NOT expired:
                <Button variant="danger" size="sm" onClick={onRevoke}>Revoke</Button>
        </div>
```

---

## 6. Pseudocode: Ghost Panel (Cross-Platform)

```
PROCEDURE GhostPanel():
    STATE suggestions = useGhostSuggestions()
    STATE selectedIndex = 0
    STATE dismissed = new Set()

    # Keyboard shortcuts (mirror AHK hotkeys)
    useHotkey('Shift+G', () => executeSuggestion(suggestions[selectedIndex]))
    useHotkey('Escape', () => dismissSuggestion(suggestions[selectedIndex]))
    useHotkey('ArrowDown', () => selectedIndex = min(selectedIndex + 1, suggestions.length - 1))
    useHotkey('ArrowUp', () => selectedIndex = max(selectedIndex - 1, 0))

    FUNCTION executeSuggestion(suggestion):
        # Route through approval gate
        result = await bizraClient.send('PLAN_ACTION', {
            action_label: suggestion.action_label,
            ahk_action_id: suggestion.ahk_action_id,
            params: suggestion.params
        })
        IF result.ok AND result.permit_status == 'APPROVED':
            await bizraClient.send('RUN_ACTION', {
                plan_id: result.plan_id,
                params: suggestion.params
            })

    RENDER:
        <aside className="ghost-panel" role="complementary" aria-label="Proactive suggestions">
            <h3>Suggestions</h3>

            IF suggestions.length == 0:
                <p className="ghost-empty">No suggestions right now.</p>

            FOR EACH [i, suggestion] IN suggestions.entries():
                IF dismissed.has(suggestion.id): CONTINUE
                <SuggestionCard
                    suggestion={suggestion}
                    selected={i == selectedIndex}
                    onExecute={() => executeSuggestion(suggestion)}
                    onDismiss={() => dismissed.add(suggestion.id)}
                />
        </aside>

PROCEDURE SuggestionCard({ suggestion, selected, onExecute, onDismiss }):
    ihsanBadge = {
        pass:    { color: '--bz-success', icon: '✓', label: 'Approved' },
        blocked: { color: '--bz-error',   icon: '✗', label: 'Blocked' },
        pending: { color: '--bz-gold',    icon: '⏳', label: 'Pending' },
    }
    badge = ihsanBadge[suggestion.ihsan_precheck]

    RENDER:
        <div className={`suggestion-card ${selected ? 'selected' : ''}`}
             role="option" aria-selected={selected}>
            # Ihsan precheck badge
            <span className="ihsan-badge" style={{ color: `var(${badge.color})` }}>
                {badge.icon} {badge.label}
                IF suggestion.ihsan_score:
                    <span className="ihsan-score">{(suggestion.ihsan_score * 100).toFixed(0)}%</span>
            </span>

            # Action description
            <div className="suggestion-body">
                <span className="action-label">{suggestion.action_label}</span>
                <span className="intent-summary">{suggestion.intent_summary}</span>
            </div>

            # Execute / Dismiss
            IF suggestion.ihsan_precheck != 'blocked':
                <Button size="sm" onClick={onExecute}>Execute</Button>
            <button className="dismiss-btn" onClick={onDismiss} aria-label="Dismiss">×</button>
        </div>
```

---

## 7. Pseudocode: Audit Trail

```
PROCEDURE AuditTrail():
    STATE receipts = useAuditTrail()
    STATE search = ''
    STATE dateRange = 'today'  # 'today' | 'week' | 'month' | 'all'

    RENDER:
        <div className="audit-trail">
            <h2>Action Audit Trail</h2>

            # Filters
            <div className="audit-filters">
                <SearchBar value={search} onChange={setSearch} placeholder="Search by verb, target, hash..." />
                <DateRangeSelector value={dateRange} onChange={setDateRange} />
            </div>

            # Receipt timeline
            <div className="receipt-timeline">
                FOR EACH receipt IN filtered(receipts, search, dateRange):
                    <ReceiptCard receipt={receipt} />
            </div>

            # Stats summary
            <div className="audit-stats">
                <StatCard label="Total Actions" value={receipts.length} />
                <StatCard label="State Changes" value={receipts.filter(r => r.state_changed).length} />
                <StatCard label="Confirmed" value={receipts.filter(r => r.outcome_confirmed).length} />
                <StatCard label="Budget Consumed" value={totalBudget(receipts)} />
            </div>
        </div>

PROCEDURE ReceiptCard({ receipt }):
    RENDER:
        <div className="receipt-card">
            # Verb + target
            <div className="receipt-header">
                <span className="verb">{receipt.verb}</span>
                <span className="target">{receipt.target}</span>
                <time>{formatTimestamp(receipt.timestamp)}</time>
            </div>

            # Hash chain visualization
            <div className="hash-chain">
                <HashBadge label="Pre" hash={receipt.pre_hash} />
                <span className="arrow">→</span>
                <span className="action-icon">{verbIcon(receipt.verb)}</span>
                <span className="arrow">→</span>
                <HashBadge label="Post" hash={receipt.post_hash} />
            </div>

            # Verification indicators
            <div className="verification">
                <Indicator label="State Changed" value={receipt.state_changed} />
                <Indicator label="Outcome Confirmed" value={receipt.outcome_confirmed} />
            </div>

            # Permit reference
            <span className="permit-ref">Permit: {truncate(receipt.permit_id, 12)}</span>
        </div>
```

---

## 8. Pseudocode: Desktop Bridge Hook

```
PROCEDURE useDesktopBridge():
    CONST BRIDGE_URL = 'http://localhost:9742'
    STATE connected = false
    STATE rateLimit = { remaining: 20, resetAt: 0 }

    FUNCTION send(method, params):
        # Rate limit check
        IF rateLimit.remaining <= 0 AND Date.now() < rateLimit.resetAt:
            THROW RateLimitError('Desktop bridge rate limit reached')

        request = {
            jsonrpc: '2.0',
            id: nanoid(),
            method: method,
            params: params,
            headers: {
                'X-BIZRA-TOKEN': await getToken(),
                'X-BIZRA-TS': Math.floor(Date.now() / 1000),
                'X-BIZRA-NONCE': nanoid(16),
            }
        }

        response = await fetch(BRIDGE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        })

        data = await response.json()
        rateLimit.remaining -= 1

        IF data.error:
            IF data.error.code == -32000:
                rateLimit.remaining = 0
                rateLimit.resetAt = Date.now() + 1000
            THROW BridgeError(data.error.message, data.error.code)

        RETURN data.result

    # Convenience methods
    FUNCTION ping(): RETURN send('ping', {})
    FUNCTION status(): RETURN send('status', {})
    FUNCTION executeHDA(verb, params): RETURN send('actuator_execute', { action: verb, ...params })
    FUNCTION listSkills(): RETURN send('list_skills', {})
    FUNCTION invokeSkill(name, params): RETURN send('invoke_skill', { skill: name, ...params })
    FUNCTION getReceipt(actionId): RETURN send('get_receipt', { action_id: actionId })

    # Health check on mount
    ON_MOUNT:
        TRY:
            await ping()
            connected = true
        CATCH:
            connected = false

    RETURN { connected, send, ping, status, executeHDA, listSkills, invokeSkill, getReceipt }
```

---

## 9. TDD Anchors

```
TEST_SUITE ahk_hda_frontend:

    # --- Action Center ---
    TEST "action center renders pending actions":
        seed actions with 2 pending, 1 completed
        render <ActionCenter />
        cards = queryAll('.action-card')
        ASSERT cards.length == 3
        ASSERT cards.filter(c => c.dataset.status == 'pending').length == 2

    TEST "approve button fires PLAN_ACTION":
        mock bizraClient
        render <ActionCard action={pendingAction} />
        click 'Approve'
        ASSERT ApprovalGate modal visible

    TEST "filter tabs work":
        render <ActionCenter />
        click 'running' tab
        ASSERT only running actions visible

    # --- Approval Gate ---
    TEST "approval gate shows FATE results":
        render <ApprovalGate action={actionWithGates} />
        ASSERT query('[data-gate="ihsan"]') shows score
        ASSERT query('[data-gate="entropy"]') shows score
        ASSERT query('[data-gate="daughter-test"]') shows result

    TEST "blocked action disables approve button":
        render <ApprovalGate action={{ ...action, blocked: true }} />
        ASSERT 'Approve' button NOT present

    # --- Permit Dashboard ---
    TEST "permit dashboard shows active permits":
        mock 3 active permits
        render <PermitDashboard />
        ASSERT queryAll('.permit-card').length == 3

    TEST "budget gauge reflects usage":
        render <PermitCard permit={{ budget: { max_actions: 30, actions_used: 15 } }} />
        ASSERT gauge at 50%

    TEST "revoke removes permit":
        render <PermitDashboard />
        click 'Revoke' on first permit → confirm
        ASSERT permits.active decreased by 1

    # --- Ghost Panel ---
    TEST "ghost panel renders suggestions":
        mock 2 suggestions
        render <GhostPanel />
        ASSERT queryAll('.suggestion-card').length == 2

    TEST "keyboard Shift+G executes selected":
        mock suggestions + bizraClient
        render <GhostPanel />
        press Shift+G
        ASSERT bizraClient.send CALLED_WITH('PLAN_ACTION', ...)

    TEST "Escape dismisses suggestion":
        render <GhostPanel />
        press Escape
        ASSERT first suggestion dismissed

    TEST "ihsan badge shows correct state":
        render <SuggestionCard suggestion={{ ihsan_precheck: 'blocked' }} />
        ASSERT badge shows '✗ Blocked' in red

    # --- Audit Trail ---
    TEST "audit trail shows receipts":
        seed 5 receipts
        render <AuditTrail />
        ASSERT queryAll('.receipt-card').length == 5

    TEST "hash chain displays pre→action→post":
        render <ReceiptCard receipt={sampleReceipt} />
        ASSERT query('[data-label="Pre"]') shows hash
        ASSERT query('[data-label="Post"]') shows hash

    TEST "search filters by verb":
        seed receipts with various verbs
        render <AuditTrail />
        type 'screenshot' in search
        ASSERT all visible receipts have verb 'screenshot'

    # --- Desktop Bridge Hook ---
    TEST "useDesktopBridge connects on mount":
        mock fetch → { result: 'pong' }
        { connected } = useDesktopBridge()
        ASSERT connected == true

    TEST "rate limit blocks after 20 requests":
        send 20 requests
        ASSERT 21st throws RateLimitError
```

---

## 10. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `/automation` route renders Action Center | Navigate, visual check |
| 2 | Pending actions show Approve/Reject buttons | Visual + test |
| 3 | Approval gate shows FATE gate results | Modal renders scores |
| 4 | Blocked actions cannot be approved | Blocked pattern triggers rejection |
| 5 | Permit dashboard shows budget countdown | Visual gauge + timer |
| 6 | Ghost panel receives live suggestions | WebSocket/poll subscription |
| 7 | Shift+G executes, Escape dismisses | Keyboard test |
| 8 | Audit trail shows hash chain per action | Receipt cards with hashes |
| 9 | Rate limit respected (20 req/s) | Rapid-fire test |
| 10 | All components use Phase 42 tokens | `grep` audit |

---

## 11. Dependency Chain

```
Phase 42 (Tokens) ← visual foundation
Phase 47 (Infra) ← BizraClient for PLAN_ACTION/RUN_ACTION verbs
Phase 44 (Chat) ← reusable message patterns
Phase 48 (This) ← builds on all above + desktop_bridge.py backend
```
