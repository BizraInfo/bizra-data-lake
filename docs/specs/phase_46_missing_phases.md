# Phase 46: Missing Phases — Node Activation, Community Layer, Legacy

> Standing on Giants: Dunbar (social network theory, 1992) · Ostrom (commons governance, 1990) · Lamport (distributed systems, 1978) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

Three major user-facing modules are completely absent from the frontend:

1. **Node Activation** — No UI for activating/deactivating the node, viewing runtime
   status, or managing the proactive kernel lifecycle.
2. **Community Layer** — No UI for SAP (Sovereign Agent Protocol) interactions:
   discovering other nodes, establishing trust, exchanging knowledge.
3. **Legacy** — No UI for sovereignty inheritance, digital will configuration, or
   knowledge transfer to successor nodes.

These are foundational to BIZRA's "every human is a node" vision. Without them,
the app is a personal chatbot rather than a sovereign node in a decentralized network.

| Module | Current | Target |
|--------|---------|--------|
| Node Activation | Terminal-only (`scripts/node0_activate.py`) | GUI: start/stop/status/mode selector |
| Node Status | None in frontend | Real-time: PID, uptime, model fleet, equalizer mode |
| Community | None | SAP_MEET_OPEN, node discovery, trust circle management |
| Trust Circles | Concept only | Dunbar-layered circles (5/15/50/150 nodes) |
| Legacy | None | Digital will, knowledge escrow, successor nomination |
| Inter-node messaging | SAP verbs exist in useNode.js | UI for SAP_MESSAGE, disclosure controls |

---

## 2. Architecture

```
src/
├── pages/
│   ├── NodePage.jsx             # Node activation & status
│   ├── CommunityPage.jsx        # Discovery, trust circles, SAP
│   └── LegacyPage.jsx           # Sovereignty inheritance
├── features/
│   ├── node/
│   │   ├── NodeStatus.jsx       # Live status panel
│   │   ├── NodeControls.jsx     # Start/stop/mode buttons
│   │   ├── ModelFleet.jsx       # Loaded models display
│   │   ├── EqualizerView.jsx    # Equalizer mode + debt gauge
│   │   └── RuntimeLogs.jsx      # Tail of sovereign.log
│   ├── community/
│   │   ├── NodeDiscovery.jsx    # Find other nodes
│   │   ├── TrustCircles.jsx     # Dunbar-layered visualization
│   │   ├── SAPSession.jsx       # Active SAP sessions
│   │   ├── DisclosureControl.jsx # What to share per session
│   │   └── NodeCard.jsx         # Other node identity card
│   └── legacy/
│       ├── DigitalWill.jsx      # Will configuration
│       ├── KnowledgeEscrow.jsx  # Escrow settings
│       ├── SuccessorNomination.jsx # Nominee management
│       └── LegacyTimeline.jsx   # Projected knowledge transfer
└── hooks/
    ├── useNodeStatus.js         # HEALTH polling + PID status
    ├── useSAPSession.js         # SAP verb orchestration
    └── useLegacy.js             # Legacy configuration state
```

---

## 3. Module A: Node Activation

### 3.1 Node Status Panel

```
PROCEDURE NodeStatus():
    STATE status = useNodeStatus()

    RENDER:
        <div className="node-status">
            # Health indicator (large)
            <div className="status-hero">
                <StatusOrb
                    state={status.running ? 'active' : 'stopped'}
                    size={80}
                    pulse={status.running}
                />
                <h2>{status.running ? 'Node Active' : 'Node Stopped'}</h2>
            </div>

            # Metrics grid
            IF status.running:
                <div className="metrics-grid">
                    <MetricCard label="Uptime" value={formatDuration(status.uptime)} />
                    <MetricCard label="PID" value={status.pid} />
                    <MetricCard label="Mode" value={status.mode} highlight />
                    <MetricCard label="Messages" value={status.messageCount} />
                    <MetricCard label="Tokens Used" value={formatNumber(status.tokensUsed)} />
                    <MetricCard label="Ihsan" value={(status.ihsan * 100).toFixed(1) + '%'} />
                </div>
        </div>
```

### 3.2 Node Controls

```
PROCEDURE NodeControls():
    STATE status = useNodeStatus()
    STATE confirming = null  # 'start' | 'stop' | null

    MODES = ['reactive', 'proactive_suggest', 'proactive_auto', 'proactive_partner']

    FUNCTION handleStart(mode):
        # Confirmation gate (Constitutional: user must confirm mutating ops)
        confirming = 'start'

    FUNCTION confirmStart(mode):
        await bizraClient.send('NODE_START', { mode })
        confirming = null

    FUNCTION handleStop():
        confirming = 'stop'

    FUNCTION confirmStop():
        await bizraClient.send('NODE_STOP', {})
        confirming = null

    RENDER:
        <div className="node-controls">
            IF NOT status.running:
                # Mode selector + Start button
                <div className="mode-selector">
                    <h4>Select Mode</h4>
                    FOR EACH mode IN MODES:
                        <ModeCard
                            mode={mode}
                            description={MODE_DESCRIPTIONS[mode]}
                            selected={selectedMode == mode}
                            onClick={() => selectedMode = mode}
                        />
                </div>
                <Button variant="primary" onClick={() => handleStart(selectedMode)}>
                    Activate Node
                </Button>
            ELSE:
                # Running state: show mode + stop
                <div className="current-mode">
                    <span>Mode: {status.mode}</span>
                    <Button variant="danger" onClick={handleStop}>
                        Stop Node
                    </Button>
                </div>

            # Confirmation modal
            IF confirming:
                <ConfirmModal
                    action={confirming}
                    onConfirm={confirming == 'start' ? confirmStart : confirmStop}
                    onCancel={() => confirming = null}
                    message={confirming == 'start'
                        ? `Activate node in ${selectedMode} mode?`
                        : 'Stop the running node? Active sessions will be preserved.'
                    }
                />
        </div>
```

### 3.3 Model Fleet Display

```
PROCEDURE ModelFleet():
    STATE fleet = useModelFleet()

    RENDER:
        <div className="model-fleet">
            <h3>Model Fleet</h3>
            <div className="fleet-grid">
                FOR EACH model IN fleet:
                    <ModelCard
                        name={model.name}
                        role={model.role}           # 'reasoner', 'planner', 'general', etc.
                        loaded={model.loaded}       # boolean
                        vramMB={model.vramUsed}     # VRAM usage
                        lastUsed={model.lastUsed}
                    />
            </div>
            <div className="fleet-summary">
                <span>{fleet.filter(m => m.loaded).length}/{fleet.length} loaded</span>
                <span>VRAM: {totalVRAM(fleet)} / 16 GB</span>
            </div>
        </div>
```

### 3.4 Equalizer View

```
PROCEDURE EqualizerView():
    STATE eq = useEqualizerState()

    MODES = {
        STEADY:       { color: '--bz-success',   label: 'Steady', desc: 'Balanced operation' },
        FLOW:         { color: '--bz-facts',      label: 'Flow',   desc: 'High throughput' },
        RECOVERY:     { color: '--bz-warning',    label: 'Recovery', desc: 'Reducing debt' },
        ACCUMULATION: { color: '--bz-goals',      label: 'Accumulation', desc: 'Building capacity' },
        SATURATION:   { color: '--bz-error',      label: 'Saturation', desc: 'Overloaded — throttling' },
    }

    mode = MODES[eq.mode]

    RENDER:
        <div className="equalizer-view">
            <h3>Cognitive Equalizer</h3>
            # Mode indicator
            <div className="eq-mode" style={{ borderColor: `var(${mode.color})` }}>
                <span className="eq-mode-label">{mode.label}</span>
                <span className="eq-mode-desc">{mode.desc}</span>
            </div>
            # Debt gauge (0-1 scale)
            <GaugeBar label="Cognitive Debt" value={eq.debt} max={1.0}
                      thresholds={[0.3, 0.6, 0.8]} />
            # Ihsan gauge
            <GaugeBar label="Ihsan Score" value={eq.ihsan} max={1.0}
                      thresholds={[0.85, 0.95, 0.99]} />
            # Last command
            IF eq.lastCommand:
                <div className="eq-command">
                    Last: {eq.lastCommand.kind} — {eq.lastCommand.reason}
                </div>
        </div>
```

---

## 4. Module B: Community Layer

### 4.1 Node Discovery

```
PROCEDURE NodeDiscovery():
    STATE discoveredNodes = []
    STATE searchQuery = ''

    FUNCTION discover():
        # Broadcast SAP_MEET_OPEN to local network
        result = await bizraClient.send('SAP_MEET_OPEN', {
            profile: getMyProfile(),
            initiator_role: 'discovery',
            timestamp: Date.now()
        })
        IF result.ok:
            discoveredNodes = parseNodeList(result)

    ON_MOUNT:
        discover()

    RENDER:
        <div className="node-discovery">
            <h2>Discover Nodes</h2>
            <SearchBar value={searchQuery} onChange={setSearchQuery} onRefresh={discover} />

            IF discoveredNodes.length == 0:
                <EmptyState message="No nearby nodes found. Are other nodes running?" />
            ELSE:
                <div className="node-list">
                    FOR EACH node IN filteredNodes(discoveredNodes, searchQuery):
                        <NodeCard
                            node={node}
                            onConnect={() => openSAPSession(node)}
                            trustLevel={getTrustLevel(node.id)}
                        />
                </div>
        </div>
```

### 4.2 Trust Circles (Dunbar Layers)

```
PROCEDURE TrustCircles():
    STATE circles = useTrustCircles()

    # Dunbar's number layers
    LAYERS = [
        { name: 'Inner Circle',  max: 5,   radius: 60,  color: '--bz-gold' },
        { name: 'Close',         max: 15,  radius: 120, color: '--bz-gold-light' },
        { name: 'Friends',       max: 50,  radius: 180, color: '--bz-text-secondary' },
        { name: 'Acquaintances', max: 150, radius: 240, color: '--bz-text-muted' },
    ]

    RENDER:
        <div className="trust-circles">
            <h2>Trust Circles</h2>

            # Concentric circle visualization
            <svg viewBox="0 0 500 500" className="circles-viz">
                FOR EACH [i, layer] IN LAYERS.entries():
                    # Ring
                    <circle cx="250" cy="250" r={layer.radius}
                            fill="none" stroke={`var(${layer.color})`}
                            stroke-width="1" opacity="0.3" />
                    # Label
                    <text x="250" y={250 - layer.radius - 8} text-anchor="middle"
                          fill={`var(${layer.color})`} font-size="10">
                        {layer.name} ({circles[i]?.length || 0}/{layer.max})
                    </text>

                    # Node dots positioned on ring
                    FOR EACH [j, node] IN (circles[i] || []).entries():
                        angle = (j / layer.max) * 2 * PI
                        x = 250 + layer.radius * cos(angle)
                        y = 250 + layer.radius * sin(angle)
                        <circle cx={x} cy={y} r="6" fill={`var(${layer.color})`}
                                onClick={() => viewNode(node)} />
            </svg>

            # Circle management
            <div className="circle-list">
                FOR EACH [i, layer] IN LAYERS.entries():
                    <CircleSection
                        name={layer.name}
                        nodes={circles[i]}
                        maxNodes={layer.max}
                        onAdd={addToCircle(i)}
                        onRemove={removeFromCircle(i)}
                    />
            </div>
        </div>
```

### 4.3 SAP Session

```
PROCEDURE SAPSession({ peer }):
    STATE messages = []
    STATE sessionId = null
    STATE disclosureLevel = 'minimal'  # 'minimal' | 'standard' | 'full'

    ON_MOUNT:
        # Open SAP session
        result = await bizraClient.send('SAP_MEET_OPEN', {
            profile: getMyProfile(),
            initiator_role: 'peer',
            timestamp: Date.now()
        })
        sessionId = result.session_id

    FUNCTION sendMessage(text):
        result = await bizraClient.send('SAP_MESSAGE', {
            session_id: sessionId,
            content: text,
            timestamp: Date.now()
        })
        messages.push({
            role: 'self', content: text,
            disclosure: myDisclosure(disclosureLevel)
        })
        messages.push({
            role: 'peer', content: result.content,
            ihsan: result.ihsan_score, receipt: result.receipt_hash
        })

    FUNCTION closeSession():
        await bizraClient.send('SAP_SESSION_CLOSE', { session_id: sessionId })
        navigate('/community')

    RENDER:
        <div className="sap-session">
            # Peer identity bar
            <PeerBar peer={peer} sessionId={sessionId} />

            # Disclosure control
            <DisclosureControl
                level={disclosureLevel}
                onChange={setDisclosureLevel}
                options={['minimal', 'standard', 'full']}
            />

            # Message thread
            <MessageList messages={messages} sapMode={true} />

            # Input + close
            <InputBar onSend={sendMessage} />
            <Button variant="ghost" onClick={closeSession}>End Session</Button>
        </div>
```

---

## 5. Module C: Legacy

### 5.1 Digital Will

```
PROCEDURE DigitalWill():
    STATE will = useLegacy()

    RENDER:
        <div className="digital-will">
            <h2>Digital Will</h2>
            <p className="subtitle">
                Configure what happens to your sovereign knowledge when you're no longer active.
            </p>

            # Successor nomination
            <Section title="Successors">
                <SuccessorList
                    nominees={will.successors}
                    onAdd={(nodeId) => will.addSuccessor(nodeId)}
                    onRemove={(nodeId) => will.removeSuccessor(nodeId)}
                    onReorder={(order) => will.reorderSuccessors(order)}
                />
            </Section>

            # Knowledge scope
            <Section title="Knowledge Scope">
                <p>What knowledge transfers to your successors?</p>
                <CheckboxGroup
                    options={[
                        { id: 'facts',     label: 'Facts & Data',      checked: will.scope.facts },
                        { id: 'preferences', label: 'Preferences',     checked: will.scope.preferences },
                        { id: 'goals',     label: 'Goals & Plans',     checked: will.scope.goals },
                        { id: 'expertise', label: 'Expertise',         checked: will.scope.expertise },
                        { id: 'patterns',  label: 'Behavioral Patterns', checked: will.scope.patterns },
                        { id: 'relations', label: 'Relationships',     checked: will.scope.relations },
                        { id: 'principles', label: 'Principles',       checked: will.scope.principles },
                    ]}
                    onChange={(id, val) => will.updateScope(id, val)}
                />
            </Section>

            # Activation trigger
            <Section title="Activation">
                <RadioGroup
                    value={will.trigger}
                    onChange={(val) => will.setTrigger(val)}
                    options={[
                        { value: 'inactivity_90d',  label: '90 days of inactivity' },
                        { value: 'inactivity_180d', label: '180 days of inactivity' },
                        { value: 'inactivity_365d', label: '1 year of inactivity' },
                        { value: 'manual',          label: 'Manual activation only' },
                    ]}
                />
            </Section>

            # Escrow preview
            <Section title="Knowledge Escrow">
                <KnowledgeEscrow
                    totalFragments={will.totalFragments}
                    scopedFragments={will.scopedFragments}
                    encrypted={true}
                />
            </Section>

            <Button onClick={will.save}>Save Will</Button>
        </div>
```

---

## 6. TDD Anchors

```
TEST_SUITE missing_phases:

    # --- Node Activation ---
    TEST "node status shows running state":
        mock HEALTH → { state: 'Ready', ihsan: 9700 }
        render <NodeStatus />
        ASSERT StatusOrb state == 'active'
        ASSERT uptime displayed

    TEST "node controls require confirmation to start":
        render <NodeControls />
        select 'proactive_suggest' → click 'Activate Node'
        ASSERT confirmation modal visible
        ASSERT button says 'Confirm'

    TEST "mode selector shows 4 modes":
        render <NodeControls /> (node stopped)
        modes = queryAll('.mode-card')
        ASSERT modes.length == 4

    TEST "model fleet shows loaded count":
        mock fleet → [{ name: 'model-a', loaded: true }, { name: 'model-b', loaded: false }]
        render <ModelFleet />
        ASSERT text CONTAINS '1/2 loaded'

    TEST "equalizer view shows current mode":
        mock equalizer → { mode: 'FLOW', debt: 0.3, ihsan: 0.97 }
        render <EqualizerView />
        ASSERT mode label == 'Flow'
        ASSERT debt gauge at 30%

    # --- Community Layer ---
    TEST "node discovery fires SAP_MEET_OPEN":
        mock bizraClient
        render <NodeDiscovery />
        ASSERT bizraClient.send CALLED_WITH('SAP_MEET_OPEN', ...)

    TEST "trust circles render 4 Dunbar layers":
        render <TrustCircles />
        rings = queryAll('circle[stroke]')
        ASSERT rings.length >= 4

    TEST "SAP session sends and receives messages":
        mock SAP_MESSAGE response
        render <SAPSession peer={mockPeer} />
        type message → send
        ASSERT bizraClient.send CALLED_WITH('SAP_MESSAGE', { session_id: ... })

    TEST "disclosure control changes level":
        render <DisclosureControl />
        select 'full'
        ASSERT disclosureLevel == 'full'

    # --- Legacy ---
    TEST "digital will saves successor list":
        render <DigitalWill />
        add successor → save
        ASSERT will.successors contains new node

    TEST "knowledge scope checkboxes toggle correctly":
        render <DigitalWill />
        uncheck 'patterns'
        ASSERT will.scope.patterns == false

    TEST "activation trigger defaults to 90d":
        render <DigitalWill />
        ASSERT selected trigger == 'inactivity_90d'
```

---

## 7. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `/node` route shows live status | Navigate, visual check |
| 2 | Start/stop requires confirmation modal | Click → confirm → verify |
| 3 | 4 proactive modes selectable | Visual + test |
| 4 | Model fleet shows VRAM usage | Compare with LM Studio |
| 5 | `/community` route shows discovery | Navigate, visual check |
| 6 | Trust circles visualization renders | SVG renders 4 rings |
| 7 | SAP session sends/receives messages | Mock or live SAP test |
| 8 | `/legacy` route shows digital will | Navigate, visual check |
| 9 | Successor nomination persists | Add → refresh → still there |
| 10 | All components use Phase 42 tokens | `grep` audit |

---

## 8. Dependency Chain

```
Phase 42 (Brand Tokens) ← required before any component work
Phase 43 (Onboarding)   ← PAT Intro provides agent context used in AgentGrid
Phase 44 (Chat)          ← Chat patterns reused in SAPSession
Phase 45 (Dashboard)     ← SovereigntyCard, AgentGrid shared
Phase 46 (This phase)    ← builds on all above
```
