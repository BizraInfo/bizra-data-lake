# Phase 50: Telescript Mobile Agent UI

> Standing on Giants: White & Daniels (Telescript, General Magic, 1994) · Lamport (distributed consensus, 1982) · Al-Ghazali (Maqasid al-Shariah → FATE gates, 1095) · Shannon (SNR quality, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

BIZRA implements a modern Telescript engine in Rust (`bizra-telescript`, 1,416 LOC)
with 9 canonical primitives: Authority, Permit, Place, Agent, Ticket, Value,
Meeting, Connection, and ImpactAttestation. The engine enforces FATE gates
(Ihsan ≥ 0.95, Adl Gini ≤ 0.35, SNR ≥ 0.85) on every operation.

**The entire system has zero frontend.** Users cannot:

- Visualize the place topology (where agents live and travel)
- Watch agent state transitions (Created → Active → Traveling → Meeting → Frozen → Terminated)
- Inspect authority chains (Genesis → delegated)
- Issue or verify tickets (travel authorization)
- Monitor meetings between agents
- Track connection quality (SNR, latency)
- View the immutable impact attestation log
- Manage the engine (create places, spawn agents, initiate travel)

| Component | Rust Engine Status | Frontend Status |
|-----------|-------------------|-----------------|
| 9 Telescript primitives | Fully implemented + tested | **None** |
| Authority chain (7 levels) | Blake3 verified delegation | **None** |
| Permit system (9 capabilities, resource limits) | Budget + TTL + FATE gates | **None** |
| Place topology | FATE-gated, capacity-managed | **None** |
| Agent lifecycle (6 states) | State machine + hash verification | **None** |
| Ticket system (travel) | 300s TTL, Blake3 signed | **None** |
| Meeting protocol | Initiator/responder + message exchange | **None** |
| Connection quality | SNR ≥ 0.85 gated | **None** |
| Impact attestation log | Immutable, hash-chained | **None** |
| Gini coefficient (Adl) | Computed for fairness enforcement | **None** |

---

## 2. Architecture

```
src/
├── pages/
│   └── TelescriptPage.jsx           # Telescript world overview
├── features/
│   └── telescript/
│       ├── PlaceTopology.jsx         # Network graph of places
│       ├── PlaceNode.jsx             # Single place in topology
│       ├── AgentLifecycle.jsx        # Agent state machine view
│       ├── AgentSpawn.jsx            # Create new agent wizard
│       ├── AuthorityChain.jsx        # Delegation tree visualizer
│       ├── PermitInspector.jsx       # Permit detail view
│       ├── TicketTracker.jsx         # Active travel tickets
│       ├── MeetingRoom.jsx           # Live meeting view
│       ├── ConnectionGraph.jsx       # Connection quality overlay
│       ├── ImpactLog.jsx             # Attestation timeline
│       ├── FATEGatePanel.jsx         # Live gate status
│       ├── GiniMeter.jsx             # Adl fairness gauge
│       ├── EngineStats.jsx           # Places, agents, meetings count
│       └── ValueLedger.jsx           # Economic value tracking
├── hooks/
│   ├── useTelescriptEngine.js        # Engine interface (via BizraClient)
│   ├── usePlaces.js                  # Place CRUD + topology
│   ├── useAgents.js                  # Agent lifecycle + spawning
│   └── useMeetings.js               # Meeting management
└── utils/
    ├── telescript-types.js            # 9 primitives as JS types
    └── graph-layout.js               # Force-directed layout for topology
```

### Engine Interface

The Rust `TelescriptEngine` is accessed via:
1. **PyO3 bindings** (when available) → direct Rust calls
2. **BizraClient verbs** → TELESCRIPT_* verb family:
   - `TELESCRIPT_STATUS` → engine stats
   - `TELESCRIPT_CREATE_PLACE` → new place
   - `TELESCRIPT_SPAWN_AGENT` → new agent
   - `TELESCRIPT_GO` → initiate travel
   - `TELESCRIPT_ARRIVE` → complete travel
   - `TELESCRIPT_MEET` → initiate meeting
   - `TELESCRIPT_IMPACT_LOG` → query attestations

---

## 3. Pseudocode: Place Topology

```
PROCEDURE PlaceTopology():
    STATE places = usePlaces()
    STATE agents = useAgents()
    STATE connections = useConnections()
    STATE selectedPlace = null
    STATE layout = useForceLayout(places, connections)

    RENDER:
        <div className="place-topology">
            <h2>Telescript World</h2>

            # Engine stats bar
            <EngineStats
                places={places.length}
                agents={agents.length}
                meetings={activeMeetings.length}
                connections={connections.length}
            />

            # SVG topology graph
            <svg viewBox="0 0 800 600" className="topology-canvas">
                # Connection lines (with SNR quality coloring)
                FOR EACH conn IN connections:
                    fromPos = layout.get(conn.local_place)
                    toPos = layout.get(conn.remote_place)
                    snrColor = conn.quality.snr >= 0.85 ? '--bz-success'
                             : conn.quality.snr >= 0.70 ? '--bz-warning'
                             : '--bz-error'
                    <line x1={fromPos.x} y1={fromPos.y}
                          x2={toPos.x} y2={toPos.y}
                          stroke={`var(${snrColor})`}
                          stroke-width={conn.quality.snr * 3}
                          opacity={0.6} />

                # Traveling agents (animated along connection paths)
                FOR EACH agent IN agents.filter(a => a.state == 'Traveling'):
                    ticket = getActiveTicket(agent.id)
                    IF ticket:
                        <TravelingAgentDot
                            from={layout.get(ticket.from_place)}
                            to={layout.get(ticket.to_place)}
                            progress={travelProgress(ticket)}
                            agentName={agent.name}
                        />

                # Place nodes
                FOR EACH place IN places:
                    pos = layout.get(place.id)
                    placeAgents = agents.filter(a => a.current_place == place.id)
                    <PlaceNode
                        x={pos.x} y={pos.y}
                        place={place}
                        agentCount={placeAgents.length}
                        selected={selectedPlace == place.id}
                        onClick={() => selectedPlace = place.id}
                        passesFATE={place.passes_fate}
                    />
            </svg>

            # Selected place detail panel
            IF selectedPlace:
                <PlaceDetailPanel
                    place={places.find(p => p.id == selectedPlace)}
                    agents={agents.filter(a => a.current_place == selectedPlace)}
                    onClose={() => selectedPlace = null}
                    onSpawnAgent={() => openSpawnWizard(selectedPlace)}
                />
        </div>

PROCEDURE PlaceNode({ x, y, place, agentCount, selected, onClick, passesFATE }):
    RENDER:
        <g transform={`translate(${x}, ${y})`} onClick={onClick} className="place-node">
            # Outer ring (FATE status)
            <circle r={30} fill="none"
                    stroke={passesFATE ? 'var(--bz-success)' : 'var(--bz-error)'}
                    stroke-width={selected ? 3 : 1}
                    opacity={selected ? 1 : 0.6} />

            # Inner fill
            <circle r={28} fill="var(--bz-bg-card)" />

            # Agent count
            <text y={-4} text-anchor="middle" fill="var(--bz-text-primary)" font-size="14">
                {agentCount}
            </text>

            # Place name
            <text y={12} text-anchor="middle" fill="var(--bz-text-secondary)" font-size="8">
                {truncate(place.telename, 15)}
            </text>

            # Capacity indicator (arc)
            <CapacityArc current={agentCount} max={place.max_agents} r={30} />
        </g>
```

---

## 4. Pseudocode: Agent Lifecycle View

```
PROCEDURE AgentLifecycle({ agentId }):
    STATE agent = useAgent(agentId)
    STATE history = useAgentHistory(agentId)  # state transition log

    STATES = [
        { name: 'Created',    color: '--bz-text-muted',  icon: '🔵' },
        { name: 'Active',     color: '--bz-success',     icon: '🟢' },
        { name: 'Traveling',  color: '--bz-warning',     icon: '🟡' },
        { name: 'Meeting',    color: '--bz-facts',       icon: '🔵' },
        { name: 'Frozen',     color: '--bz-expertise',   icon: '🧊' },
        { name: 'Terminated', color: '--bz-error',       icon: '🔴' },
    ]

    currentStateIdx = STATES.findIndex(s => s.name == agent.state)

    RENDER:
        <div className="agent-lifecycle">
            # Agent identity
            <div className="agent-identity">
                <h2>{agent.name}</h2>
                <span className="agent-id">{truncate(agent.id, 12)}</span>
            </div>

            # State machine visualization (horizontal pipeline)
            <div className="state-pipeline">
                FOR EACH [i, state] IN STATES.entries():
                    <div className={`state-node ${i == currentStateIdx ? 'active' : ''}`}>
                        <span className="state-icon">{state.icon}</span>
                        <span className="state-name">{state.name}</span>
                        IF i == currentStateIdx:
                            <span className="current-marker">← current</span>
                    </div>
                    IF i < STATES.length - 1:
                        <span className="state-arrow">→</span>
            </div>

            # Current state detail
            <div className="state-detail">
                <InfoRow label="State" value={agent.state} />
                <InfoRow label="Place" value={agent.current_place || 'In transit'} />
                <InfoRow label="Ihsan" value={(agent.ihsan_score / 10).toFixed(1) + '%'} />
                <InfoRow label="Last Activity" value={formatRelativeTime(agent.last_activity)} />
            </div>

            # Permit summary
            <PermitInspector permit={agent.permit} compact />

            # Resource usage
            <div className="resource-usage">
                <GaugeBar label="CPU" value={agent.resource_usage.cpu_used}
                          max={agent.permit.limits.cpu_millicores} />
                <GaugeBar label="Memory" value={agent.resource_usage.memory_used}
                          max={agent.permit.limits.memory_bytes} unit="bytes" />
                <GaugeBar label="Tokens" value={agent.resource_usage.tokens_used}
                          max={agent.permit.limits.inference_tokens} />
            </div>

            # Agent data (key-value store)
            <Section title="Agent Data" count={Object.keys(agent.data).length}>
                <DataTable entries={agent.data} />
            </Section>

            # State transition history
            <Section title="History">
                <TransitionTimeline history={history} />
            </Section>
        </div>
```

---

## 5. Pseudocode: Authority Chain Visualizer

```
PROCEDURE AuthorityChain({ authority }):
    # Build chain from leaf to Genesis
    chain = []
    current = authority
    WHILE current:
        chain.unshift(current)
        current = current.delegated_from

    RENDER:
        <div className="authority-chain">
            <h3>Authority Chain (depth {authority.delegation_depth})</h3>

            <div className="chain-viz">
                FOR EACH [i, auth] IN chain.entries():
                    <div className="chain-node" data-depth={auth.delegation_depth}>
                        # Node badge
                        <div className="chain-badge"
                             style={{ borderColor: i == 0 ? 'var(--bz-gold)' : 'var(--bz-border-default)' }}>
                            <span className="chain-name">{auth.name}</span>
                            <span className="chain-depth">Depth {auth.delegation_depth}</span>
                            <span className="chain-hash" title={auth.chain_hash}>
                                {truncate(bytesToHex(auth.chain_hash), 12)}
                            </span>
                        </div>

                        # Delegation arrow (except last)
                        IF i < chain.length - 1:
                            <div className="delegation-arrow">
                                <span>delegates to</span>
                                <span>↓</span>
                            </div>
                    </div>
            </div>

            # Chain verification
            <div className="chain-verify">
                <span className={chain_valid ? 'valid' : 'invalid'}>
                    {chain_valid ? '✓ Chain verified (Blake3)' : '✗ Chain broken'}
                </span>
            </div>

            # Max depth warning
            IF authority.delegation_depth >= 6:
                <div className="depth-warning">
                    Approaching max delegation depth (7). Consider restructuring.
                </div>
        </div>
```

---

## 6. Pseudocode: Meeting Room

```
PROCEDURE MeetingRoom({ meetingId }):
    STATE meeting = useMeeting(meetingId)

    IF NOT meeting: RETURN <Loading />

    initiator = useAgent(meeting.initiator_id)
    responder = useAgent(meeting.responder_id)
    place = usePlace(meeting.place_id)

    RENDER:
        <div className="meeting-room">
            # Meeting header
            <div className="meeting-header">
                <AgentBadgeMini agent={initiator} />
                <span className="meeting-state">{meeting.state}</span>
                <AgentBadgeMini agent={responder} />
            </div>

            # Place context
            <span className="meeting-place">at {place.telename}</span>

            # Message exchange
            <div className="meeting-exchange">
                FOR EACH msg IN meeting.exchange:
                    <MeetingMessage
                        sender={msg.sender_id == meeting.initiator_id ? initiator : responder}
                        content={msg.content}
                        timestamp={msg.timestamp}
                    />
            </div>

            # Meeting controls
            IF meeting.state == 'Active':
                <div className="meeting-controls">
                    <Button variant="ghost" onClick={() => endMeeting(meetingId)}>
                        End Meeting
                    </Button>
                </div>

            # Duration
            <span className="meeting-duration">
                {formatDuration(
                    (meeting.ended_at || Date.now()) - meeting.started_at
                )}
            </span>
        </div>
```

---

## 7. Pseudocode: Impact Attestation Log

```
PROCEDURE ImpactLog():
    STATE attestations = useImpactLog()
    STATE filter = ''

    RENDER:
        <div className="impact-log">
            <h2>Impact Attestations</h2>
            <p className="subtitle">Immutable, hash-chained record of all agent actions.</p>

            <SearchBar value={filter} onChange={setFilter}
                       placeholder="Filter by agent, action, or hash..." />

            # Timeline
            <div className="attestation-timeline">
                FOR EACH att IN filtered(attestations, filter):
                    <AttestationEntry attestation={att} />
            </div>

            # Chain integrity indicator
            <div className="chain-integrity">
                <span className={chainValid ? 'valid' : 'broken'}>
                    {chainValid
                        ? `✓ Chain intact (${attestations.length} entries)`
                        : '✗ Chain integrity violation detected'}
                </span>
            </div>
        </div>

PROCEDURE AttestationEntry({ attestation }):
    RENDER:
        <div className="attestation-entry">
            <div className="att-header">
                <AgentBadgeMini agentId={attestation.agent_id} />
                <span className="att-action">{attestation.action}</span>
                <span className="att-impact">
                    Impact: {(attestation.impact_score / 100).toFixed(2)}
                </span>
            </div>
            <div className="att-hash">
                <HashBadge label="Hash" hash={bytesToHex(attestation.attestation_hash)} />
            </div>
            <time>{formatTimestamp(attestation.timestamp)}</time>
        </div>
```

---

## 8. Pseudocode: FATE Gate Panel

```
PROCEDURE FATEGatePanel():
    STATE gates = useFATEGates()

    RENDER:
        <div className="fate-panel">
            <h3>FATE Gates</h3>

            # Fidelity
            <GateRow
                letter="F"
                name="Fidelity"
                description="Agent stays true to its permit and authority chain"
                status={gates.fidelity}
            />

            # Accountability
            <GateRow
                letter="A"
                name="Accountability"
                description="All actions recorded in impact attestation log"
                status={gates.accountability}
            />

            # Transparency
            <GateRow
                letter="T"
                name="Transparency"
                description="Decision reasoning available for inspection"
                status={gates.transparency}
            />

            # Ethics (Ihsan + Adl)
            <GateRow
                letter="E"
                name="Ethics"
                description="Ihsan ≥ 0.95, Adl Gini ≤ 0.35"
                status={gates.ethics}
            />

            # Ihsan gauge
            <GaugeBar label="Ihsan (Excellence)" value={gates.ihsan} max={1.0}
                      threshold={0.95} thresholdLabel="Required" />

            # Gini gauge (inverted — lower is better)
            <GiniMeter value={gates.gini} maxAllowed={0.35} />

            # SNR gauge
            <GaugeBar label="SNR (Signal Quality)" value={gates.snr} max={1.0}
                      threshold={0.85} thresholdLabel="Minimum" />
        </div>

PROCEDURE GiniMeter({ value, maxAllowed }):
    percent = (value / 1.0) * 100
    color = value <= maxAllowed ? '--bz-success' : '--bz-error'

    RENDER:
        <div className="gini-meter">
            <div className="gini-label">
                <span>Adl (Justice)</span>
                <span>Gini: {value.toFixed(3)}</span>
            </div>
            <div className="gini-bar">
                <div className="gini-fill" style={{
                    width: `${percent}%`,
                    background: `var(${color})`
                }} />
                # Threshold marker
                <div className="gini-threshold" style={{
                    left: `${maxAllowed * 100}%`
                }}>
                    <span>≤ {maxAllowed}</span>
                </div>
            </div>
        </div>
```

---

## 9. Pseudocode: Agent Spawn Wizard

```
PROCEDURE AgentSpawn({ placeId }):
    STATE name = ''
    STATE code = ''               # Agent script/bytecode
    STATE capabilities = []       # Selected capabilities
    STATE resourceLimits = {
        cpu_millicores: 1000,
        memory_bytes: 256 * 1024 * 1024,   # 256MB
        storage_bytes: 1 * 1024 * 1024 * 1024, # 1GB
        network_bps: 10 * 1024 * 1024,     # 10Mbps
        inference_tokens: 10000,
        ttl_seconds: 3600,                  # 1 hour
    }

    CAPABILITIES = [
        'Go', 'Create', 'Meet', 'Compute', 'Store',
        'Network', 'Inference', 'SelfModify', 'Delegate'
    ]

    FUNCTION spawn():
        # FATE gate check
        place = places.get(placeId)
        IF NOT place.passes_fate:
            SHOW error "Place does not pass FATE gates"
            RETURN

        result = await engine.createAgent({
            name: name,
            permit: { capabilities, limits: resourceLimits, ihsan_requirement: 950 },
            code: encodeCode(code),
            place_id: placeId,
        })

        IF result.ok:
            navigate(`/telescript/agents/${result.agent_id}`)

    RENDER:
        <Modal title="Spawn Agent" size="large">
            <FormField label="Agent Name">
                <input value={name} onChange={setName} placeholder="my-agent" />
            </FormField>

            <FormField label="Capabilities">
                <CheckboxGroup
                    options={CAPABILITIES.map(c => ({ id: c, label: c }))}
                    selected={capabilities}
                    onChange={setCapabilities}
                />
            </FormField>

            <FormField label="Resource Limits">
                <Slider label="CPU (millicores)" value={resourceLimits.cpu_millicores}
                        min={100} max={4000} onChange={v => resourceLimits.cpu_millicores = v} />
                <Slider label="Memory (MB)" value={resourceLimits.memory_bytes / (1024*1024)}
                        min={64} max={4096}
                        onChange={v => resourceLimits.memory_bytes = v * 1024 * 1024} />
                <Slider label="Inference Tokens" value={resourceLimits.inference_tokens}
                        min={1000} max={100000} onChange={v => resourceLimits.inference_tokens = v} />
                <Slider label="TTL (minutes)" value={resourceLimits.ttl_seconds / 60}
                        min={5} max={1440}
                        onChange={v => resourceLimits.ttl_seconds = v * 60} />
            </FormField>

            <FormField label="Agent Code">
                <CodeEditor value={code} onChange={setCode} language="rust" />
            </FormField>

            <Button variant="primary" onClick={spawn}
                    disabled={!name || capabilities.length == 0}>
                Spawn at {places.get(placeId)?.telename}
            </Button>
        </Modal>
```

---

## 10. TDD Anchors

```
TEST_SUITE telescript_frontend:

    # --- Place Topology ---
    TEST "topology renders places as nodes":
        mock engine with 3 places
        render <PlaceTopology />
        nodes = queryAll('.place-node')
        ASSERT nodes.length == 3

    TEST "connection lines colored by SNR":
        mock connection with SNR 0.92
        render <PlaceTopology />
        line = query('line')
        ASSERT line.stroke == green (passes 0.85 threshold)

    TEST "traveling agent animated along path":
        mock agent in 'Traveling' state with active ticket
        render <PlaceTopology />
        ASSERT '.traveling-agent' element exists
        ASSERT element animates from source to destination

    TEST "clicking place opens detail panel":
        render <PlaceTopology />
        click first place node
        ASSERT PlaceDetailPanel visible

    # --- Agent Lifecycle ---
    TEST "state pipeline highlights current state":
        render <AgentLifecycle agentId="agent-1" /> with state='Active'
        ASSERT '.state-node.active' has name 'Active'

    TEST "resource gauges show usage vs limits":
        render <AgentLifecycle agentId="agent-1" />
        ASSERT CPU gauge value matches resource_usage.cpu_used

    # --- Authority Chain ---
    TEST "chain renders Genesis to leaf":
        authority with depth 3
        render <AuthorityChain authority={auth} />
        nodes = queryAll('.chain-node')
        ASSERT nodes.length == 4  # Genesis + 3 delegations

    TEST "chain verification badge shows valid":
        render <AuthorityChain authority={validAuth} />
        ASSERT text CONTAINS '✓ Chain verified'

    # --- Meetings ---
    TEST "meeting room shows exchange messages":
        mock meeting with 4 messages
        render <MeetingRoom meetingId="meet-1" />
        msgs = queryAll('.meeting-message')
        ASSERT msgs.length == 4

    # --- Impact Log ---
    TEST "attestation timeline shows entries":
        mock 10 attestations
        render <ImpactLog />
        entries = queryAll('.attestation-entry')
        ASSERT entries.length == 10

    TEST "chain integrity checks pass":
        mock valid chain
        render <ImpactLog />
        ASSERT text CONTAINS '✓ Chain intact'

    # --- FATE Panel ---
    TEST "FATE gates show all 4 letters":
        render <FATEGatePanel />
        ASSERT query('[data-gate="F"]') EXISTS
        ASSERT query('[data-gate="A"]') EXISTS
        ASSERT query('[data-gate="T"]') EXISTS
        ASSERT query('[data-gate="E"]') EXISTS

    TEST "Gini meter shows threshold marker":
        render <GiniMeter value={0.25} maxAllowed={0.35} />
        ASSERT threshold marker at 35%
        ASSERT fill color is green (below threshold)

    # --- Agent Spawn ---
    TEST "spawn wizard creates agent":
        mock engine
        render <AgentSpawn placeId="place-1" />
        fill name, select capabilities → click Spawn
        ASSERT engine.createAgent CALLED_WITH name, capabilities

    TEST "spawn blocked if place fails FATE":
        mock place with passes_fate=false
        render <AgentSpawn placeId="bad-place" />
        click Spawn
        ASSERT error message about FATE gates
```

---

## 11. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `/telescript` route renders place topology | Navigate, visual check |
| 2 | Places show agent count + FATE status | SVG nodes colored correctly |
| 3 | Traveling agents animate between places | Visual animation check |
| 4 | Agent lifecycle shows 6 states | State pipeline renders |
| 5 | Authority chain renders Genesis → leaf | Chain depth matches |
| 6 | Meeting room shows live message exchange | Mock meeting data |
| 7 | Impact log shows hash-chained entries | Hash badges visible |
| 8 | FATE panel shows F/A/T/E + Ihsan + Gini + SNR | All gauges render |
| 9 | Agent spawn respects FATE gates | Blocked for non-FATE places |
| 10 | All components use Phase 42 tokens | `grep` audit |

---

## 12. Full Dependency Chain (All Phases)

```
FOUNDATION LAYER (build first, in parallel):
  Phase 42 (Brand Tokens)     ← every component
  Phase 47 (Infrastructure)   ← BizraClient, offline, PWA

FEATURE LAYER (build next):
  Phase 43 (Onboarding)       ← first user experience
  Phase 44 (Chat)             ← primary interaction surface
  Phase 45 (Dashboard)        ← daily landing screen

SYSTEM LAYER (build last):
  Phase 46 (Node/Community/Legacy)  ← sovereignty + social
  Phase 48 (AHK+HDA)               ← desktop automation
  Phase 49 (Agent as a Service)     ← agent marketplace + tasks
  Phase 50 (Telescript)             ← mobile agent world ← THIS

Shared components across phases:
  - KnowsMeGauge: 42, 43, 44, 45
  - AgentBadge: 43, 45, 46, 49, 50
  - FATEGateBadge: 48, 50
  - PermitCard: 48, 49, 50
  - TrustScoreGauge: 46, 49
  - HashBadge: 48, 50
```
