# Phase 49: Agent as a Service (AaaS)

> Standing on Giants: General Magic (Telescript agents, 1994) · Fielding (REST, 2000) · Lamport (Byzantine consensus, 1982) · Ostrom (commons governance, 1990) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

BIZRA has a complete A2A (Agent-to-Agent) backend with zero frontend:

| Component | Backend Status | Frontend Status |
|-----------|---------------|-----------------|
| A2A Engine (PCI-signed messages, Ed25519) | `engine.py` (506 LOC) | **None** |
| Agent Cards (identity + capability manifest) | `schema.py` (425 LOC), 7 pre-defined PAT agents | **None** |
| Task Manager (priority queue, parallel exec, decomposition) | `tasks.py` (448 LOC) | **None** |
| Agent Packager (.bizra-agent archives) | `agent_packager.py` (428 LOC) | **None** |
| Transport (Local + UDP + Hybrid) | `transport.py` (319 LOC) | **None** |
| Capability Cards (model credentials) | `capability_card.py` | **None** |
| A2A Protocol YAML (7 agents, 6 task templates, FATE gates) | `a2a_protocol.yaml` (397 lines) | **None** |
| Founder Ops Agent manifest | `agents/founder_ops/manifest.yaml` (211 lines) | **None** |

Users cannot browse agents, delegate tasks, inspect capability cards,
package agents, or monitor task execution from the frontend.

### What Exists (Backend)

```
core/a2a/schema.py          — AgentCard, TaskCard, A2AMessage, Capability
core/a2a/engine.py          — PCI-signed engine, registry, task delegation
core/a2a/tasks.py           — TaskManager (priority queue, parallel, retry)
core/a2a/transport.py       — Local/UDP/Hybrid transport
core/a2a/agent_packager.py  — .bizra-agent ZIP packaging
core/sovereign/capability_card.py — Model capability credentials
bizra-omega/bizra-cli/config/a2a_protocol.yaml — 7 PAT agents + protocol
agents/founder_ops/manifest.yaml — Example agent manifest
```

### What's Missing (Frontend)

1. **Agent Marketplace** — Browse, search, install .bizra-agent packages
2. **Agent Card Viewer** — Detailed view of agent identity, capabilities, trust score
3. **Task Delegation UI** — Create tasks, select target agent, set priority
4. **Task Monitor** — Real-time task status: pending → accepted → in_progress → completed
5. **Capability Browser** — Search agents by capability type
6. **Agent Builder** — Create manifest.yaml through guided wizard
7. **Registry Dashboard** — All discovered agents (local + federated)

---

## 2. Architecture

```
src/
├── pages/
│   ├── AgentsPage.jsx               # Agent marketplace/registry hub
│   └── AgentDetailPage.jsx          # Single agent deep view
├── features/
│   └── agents/
│       ├── AgentMarketplace.jsx     # Browse/search/install agents
│       ├── AgentCardFull.jsx        # Detailed agent card view
│       ├── CapabilityBrowser.jsx    # Search by capability
│       ├── CapabilityBadge.jsx      # Single capability chip
│       ├── TaskDelegator.jsx        # Create + delegate tasks
│       ├── TaskMonitor.jsx          # Live task tracking
│       ├── TaskCard.jsx             # Single task status card
│       ├── TaskTree.jsx             # Parent-child task hierarchy
│       ├── AgentBuilder.jsx         # Manifest creation wizard
│       ├── ManifestPreview.jsx      # YAML preview
│       ├── RegistryDashboard.jsx    # All agents (local + remote)
│       ├── TrustScoreGauge.jsx      # Ihsan + success rate visual
│       └── MessageInspector.jsx     # PCI-signed message viewer
├── hooks/
│   ├── useA2AEngine.js              # Engine interface
│   ├── useAgentRegistry.js          # Agent discovery + registration
│   ├── useTaskManager.js            # Task CRUD + status polling
│   └── useAgentPackager.js          # Package/install .bizra-agent
└── utils/
    ├── capability-types.js           # 12 CapabilityType definitions
    └── task-status.js                # TaskStatus enum + colors
```

---

## 3. Pseudocode: Agent Marketplace

```
PROCEDURE AgentMarketplace():
    STATE agents = useAgentRegistry()
    STATE search = ''
    STATE capFilter = null      # CapabilityType filter
    STATE sortBy = 'trust'      # 'trust' | 'tasks' | 'name' | 'recent'
    STATE installing = null     # agent being installed

    CAPABILITY_TYPES = [
        'REASONING', 'CODE_GENERATION', 'CODE_EXECUTION', 'KNOWLEDGE_RETRIEVAL',
        'DATA_ANALYSIS', 'VISION', 'AUDIO', 'ORCHESTRATION', 'SECURITY',
        'DESIGN', 'FORMATTING', 'CUSTOM'
    ]

    FUNCTION filteredAgents():
        result = agents.all
        IF search: result = result.filter(a => a.name.includes(search) || a.description.includes(search))
        IF capFilter: result = result.filter(a => a.capabilities.some(c => c.type == capFilter))
        SORT result BY sortBy
        RETURN result

    RENDER:
        <div className="agent-marketplace">
            <h2>Agent Marketplace</h2>

            # Search + filters
            <div className="marketplace-controls">
                <SearchBar value={search} onChange={setSearch} placeholder="Search agents..." />
                <CapabilityFilter
                    types={CAPABILITY_TYPES}
                    selected={capFilter}
                    onChange={setCapFilter}
                />
                <SortSelect value={sortBy} onChange={setSortBy} />
            </div>

            # Installed agents section
            <Section title="Your Agents" count={agents.local.length}>
                <div className="agent-grid">
                    FOR EACH agent IN agents.local:
                        <AgentCardCompact
                            agent={agent}
                            onClick={() => navigate(`/agents/${agent.agent_id}`)}
                            installed
                        />
                </div>
            </Section>

            # Available agents section
            <Section title="Available" count={filteredAgents().length}>
                <div className="agent-grid">
                    FOR EACH agent IN filteredAgents():
                        <AgentCardCompact
                            agent={agent}
                            onClick={() => navigate(`/agents/${agent.agent_id}`)}
                            onInstall={() => installing = agent}
                        />
                </div>
            </Section>

            # Install modal
            IF installing:
                <InstallAgentModal
                    agent={installing}
                    onConfirm={() => agents.install(installing.agent_id)}
                    onCancel={() => installing = null}
                />
        </div>
```

---

## 4. Pseudocode: Agent Card (Full View)

```
PROCEDURE AgentCardFull({ agentId }):
    STATE agent = useAgent(agentId)
    STATE tasks = useAgentTasks(agentId)

    IF NOT agent: RETURN <Loading />

    RENDER:
        <div className="agent-card-full">
            # Identity header
            <div className="agent-identity">
                <AgentAvatar name={agent.name} size={80} />
                <div className="agent-info">
                    <h1>{agent.name}</h1>
                    <span className="agent-version">v{agent.version}</span>
                    <p>{agent.description}</p>
                </div>
                <TrustScoreGauge
                    ihsan={agent.ihsan_score}
                    successRate={agent.success_rate}
                    tasksCompleted={agent.tasks_completed}
                />
            </div>

            # Capabilities grid
            <Section title="Capabilities">
                <div className="capability-grid">
                    FOR EACH cap IN agent.capabilities:
                        <CapabilityCard
                            name={cap.name}
                            type={cap.type}
                            description={cap.description}
                            ihsanFloor={cap.ihsan_floor}
                            version={cap.version}
                            parameters={cap.parameters}
                        />
                </div>
            </Section>

            # Endpoint + Federation
            <Section title="Network">
                <InfoRow label="Endpoint" value={agent.endpoint || 'Local only'} />
                <InfoRow label="Federation" value={agent.federation_address || 'Not federated'} />
                <InfoRow label="Public Key" value={truncate(agent.public_key, 24)} copyable />
                <InfoRow label="Card Hash" value={truncate(agent.card_hash, 24)} copyable />
            </Section>

            # Task history
            <Section title="Recent Tasks" count={tasks.length}>
                FOR EACH task IN tasks.slice(0, 10):
                    <TaskCard task={task} compact />
            </Section>

            # Quick delegate button
            <Button variant="primary" onClick={() => openDelegator(agent)}>
                Delegate Task
            </Button>
        </div>
```

---

## 5. Pseudocode: Task Delegator

```
PROCEDURE TaskDelegator({ preselectedAgent }):
    STATE capability = ''
    STATE prompt = ''
    STATE priority = 5           # 1-10
    STATE timeout = 300          # seconds
    STATE targetAgent = preselectedAgent || null
    STATE parameters = {}
    STATE submitting = false

    matchingAgents = useAgentsByCapability(capability)

    FUNCTION submit():
        submitting = true
        result = await a2aEngine.createTask({
            capability_required: capability,
            prompt: prompt,
            parameters: parameters,
            target_agent: targetAgent?.agent_id,
            priority: priority,
            timeout: timeout,
        })
        IF result.ok:
            navigate(`/tasks/${result.task_id}`)
        submitting = false

    RENDER:
        <div className="task-delegator">
            <h2>Delegate Task</h2>

            # Step 1: What capability?
            <FormField label="Capability Required">
                <CapabilitySelect
                    value={capability}
                    onChange={setCapability}
                    options={CAPABILITY_TYPES}
                />
            </FormField>

            # Step 2: Who? (auto-selected or manual)
            <FormField label="Target Agent">
                IF targetAgent:
                    <AgentCardCompact agent={targetAgent} removable
                        onRemove={() => targetAgent = null} />
                ELSE:
                    <AgentPicker
                        agents={matchingAgents}
                        onSelect={(a) => targetAgent = a}
                        emptyMessage={capability
                            ? `No agents with ${capability} capability`
                            : 'Select a capability first'}
                    />
                    IF matchingAgents.length > 0:
                        <p className="auto-route-hint">
                            Best match: {matchingAgents[0].name}
                            (Ihsan: {(matchingAgents[0].ihsan_score * 100).toFixed(0)}%,
                             Success: {(matchingAgents[0].success_rate * 100).toFixed(0)}%)
                        </p>
            </FormField>

            # Step 3: What to do?
            <FormField label="Task Prompt">
                <textarea
                    value={prompt}
                    onChange={setPrompt}
                    placeholder="Describe what you need done..."
                    rows={4}
                />
            </FormField>

            # Step 4: Priority + Timeout
            <div className="task-controls">
                <FormField label="Priority (1-10)">
                    <Slider value={priority} min={1} max={10} onChange={setPriority} />
                </FormField>
                <FormField label="Timeout">
                    <Select value={timeout} onChange={setTimeout}
                        options={[
                            { value: 60, label: '1 minute' },
                            { value: 300, label: '5 minutes' },
                            { value: 900, label: '15 minutes' },
                            { value: 3600, label: '1 hour' },
                        ]}
                    />
                </FormField>
            </div>

            # Submit
            <Button variant="primary" onClick={submit} disabled={!capability || !prompt || submitting}>
                {submitting ? 'Delegating...' : 'Delegate Task'}
            </Button>
        </div>
```

---

## 6. Pseudocode: Task Monitor

```
PROCEDURE TaskMonitor():
    STATE tasks = useTaskManager()
    STATE view = 'board'  # 'board' | 'list' | 'tree'

    RENDER:
        <div className="task-monitor">
            <div className="monitor-header">
                <h2>Tasks</h2>
                <ViewToggle value={view} onChange={setView}
                    options={['board', 'list', 'tree']} />
                <Button onClick={() => navigate('/tasks/new')}>New Task</Button>
            </div>

            IF view == 'board':
                <TaskBoard tasks={tasks} />
            ELSE IF view == 'list':
                <TaskList tasks={tasks} />
            ELSE:
                <TaskTree tasks={tasks} />
        </div>

PROCEDURE TaskBoard({ tasks }):
    COLUMNS = [
        { key: 'pending',     label: 'Pending',     color: '--bz-text-muted' },
        { key: 'accepted',    label: 'Accepted',    color: '--bz-info' },
        { key: 'in_progress', label: 'In Progress', color: '--bz-warning' },
        { key: 'completed',   label: 'Completed',   color: '--bz-success' },
        { key: 'failed',      label: 'Failed',      color: '--bz-error' },
    ]

    RENDER:
        <div className="task-board">
            FOR EACH col IN COLUMNS:
                <div className="board-column">
                    <h4 style={{ color: `var(${col.color})` }}>
                        {col.label}
                        <span className="count">
                            ({tasks.filter(t => t.status == col.key).length})
                        </span>
                    </h4>
                    FOR EACH task IN tasks.filter(t => t.status == col.key):
                        <TaskCard task={task} />
                </div>
        </div>

PROCEDURE TaskCard({ task, compact }):
    statusColors = {
        pending: '--bz-text-muted', accepted: '--bz-info',
        in_progress: '--bz-warning', completed: '--bz-success',
        failed: '--bz-error', cancelled: '--bz-text-muted',
    }

    RENDER:
        <div className="task-card" data-status={task.status}>
            <div className="task-header">
                <StatusDot color={statusColors[task.status]} />
                <span className="task-capability">{task.capability_required}</span>
                <PriorityBadge priority={task.priority} />
            </div>

            <p className="task-prompt">{truncate(task.prompt, compact ? 60 : 200)}</p>

            IF task.assignee_id:
                <AgentBadgeMini agentId={task.assignee_id} />

            IF task.child_task_ids?.length > 0:
                <span className="subtask-count">{task.child_task_ids.length} subtasks</span>

            # Duration / timeout
            IF task.started_at:
                <span className="task-duration">{formatDuration(Date.now() - task.started_at)}</span>

            IF task.status == 'completed' AND task.result:
                <div className="task-result-preview">
                    {truncate(JSON.stringify(task.result), 100)}
                </div>

            IF task.status == 'failed' AND task.error:
                <div className="task-error">{task.error}</div>

            <time>{formatRelativeTime(task.created_at)}</time>
        </div>

PROCEDURE TaskTree({ tasks }):
    # Build parent-child tree
    roots = tasks.filter(t => !t.parent_task_id)

    FUNCTION renderNode(task, depth):
        children = tasks.filter(t => t.parent_task_id == task.task_id)
        RETURN:
            <div className="tree-node" style={{ marginLeft: depth * 24 }}>
                <TaskCard task={task} compact />
                FOR EACH child IN children:
                    renderNode(child, depth + 1)
            </div>

    RENDER:
        <div className="task-tree">
            FOR EACH root IN roots:
                renderNode(root, 0)
        </div>
```

---

## 7. Pseudocode: Agent Builder

```
PROCEDURE AgentBuilder():
    STATE step = 0
    STATE manifest = {
        name: '', display_name: '', version: '1.0.0', description: '',
        capabilities_telescript: [],
        hda_skills: [],
        missions: [],
        permit_defaults: { ttl_seconds: 300, max_actions: 30, max_tokens: 4096, auto_renew: true },
        quality: { min_ihsan: 0.95, min_snr: 0.85, daughter_test: true },
        persona: { tone: '', expertise: '', greeting: '' },
        onboarding_questions: [],
    }
    STATE validation = null
    STATE yamlPreview = ''

    STEPS = [
        { title: 'Identity',     component: IdentityStep },
        { title: 'Capabilities', component: CapabilitiesStep },
        { title: 'Skills',       component: SkillsStep },
        { title: 'Missions',     component: MissionsStep },
        { title: 'Permits',      component: PermitStep },
        { title: 'Review',       component: ReviewStep },
    ]

    FUNCTION validate():
        result = await agentPackager.validate(manifest)
        validation = result

    FUNCTION buildPackage():
        await validate()
        IF validation.valid:
            package = await agentPackager.package(manifest)
            download(package.archivePath, `${manifest.name}-${manifest.version}.bizra-agent`)

    RENDER:
        <div className="agent-builder">
            <h2>Build Agent</h2>

            # Step indicator
            <StepIndicator steps={STEPS} current={step} />

            # Current step
            <STEPS[step].component manifest={manifest} onUpdate={setManifest} />

            # Navigation
            <div className="builder-nav">
                IF step > 0:
                    <Button variant="ghost" onClick={() => step -= 1}>Back</Button>
                IF step < STEPS.length - 1:
                    <Button onClick={() => step += 1}>Next</Button>
                IF step == STEPS.length - 1:
                    <Button variant="primary" onClick={buildPackage}>
                        Build .bizra-agent
                    </Button>
            </div>

            # YAML preview (collapsible)
            <Collapsible title="manifest.yaml preview">
                <ManifestPreview manifest={manifest} />
            </Collapsible>
        </div>
```

---

## 8. TDD Anchors

```
TEST_SUITE agent_as_a_service:

    # --- Marketplace ---
    TEST "marketplace renders installed agents":
        mock 3 local + 5 remote agents
        render <AgentMarketplace />
        ASSERT 'Your Agents' section shows 3
        ASSERT 'Available' section shows 5

    TEST "capability filter narrows results":
        render <AgentMarketplace />
        select filter 'CODE_GENERATION'
        ASSERT only agents with CODE_GENERATION capability shown

    TEST "search filters by name/description":
        render <AgentMarketplace />
        type 'Guardian' in search
        ASSERT results contain 'Guardian' agent

    # --- Agent Card ---
    TEST "agent card shows all capabilities":
        render <AgentCardFull agentId="guardian" />
        caps = queryAll('.capability-card')
        ASSERT caps.length == agent.capabilities.length

    TEST "trust score gauge shows ihsan + success rate":
        render <TrustScoreGauge ihsan={0.97} successRate={0.92} tasksCompleted={150} />
        ASSERT ihsan display == '97%'
        ASSERT success display == '92%'

    # --- Task Delegator ---
    TEST "delegator auto-suggests best agent":
        mock agents with CODE_GENERATION capability
        render <TaskDelegator />
        select 'CODE_GENERATION'
        ASSERT auto-route hint shows best agent

    TEST "delegator creates task":
        mock a2aEngine
        render <TaskDelegator />
        fill form → submit
        ASSERT a2aEngine.createTask CALLED_WITH capability, prompt, priority

    # --- Task Monitor ---
    TEST "task board shows columns":
        render <TaskBoard tasks={sampleTasks} />
        columns = queryAll('.board-column')
        ASSERT columns.length == 5

    TEST "task tree renders hierarchy":
        tasks = [parent, child1, child2, grandchild]
        render <TaskTree tasks={tasks} />
        ASSERT tree depth == 2

    TEST "completed task shows result preview":
        render <TaskCard task={{ status: 'completed', result: { answer: 'done' } }} />
        ASSERT '.task-result-preview' visible

    # --- Agent Builder ---
    TEST "builder validates manifest":
        render <AgentBuilder />
        leave name empty → click 'Build'
        ASSERT validation error for name

    TEST "builder generates .bizra-agent":
        fill all fields → click 'Build'
        ASSERT download triggered
        ASSERT file name matches '{name}-{version}.bizra-agent'

    TEST "YAML preview updates live":
        render <AgentBuilder />
        type name 'test-agent'
        ASSERT YAML preview contains 'name: test-agent'

    # --- Registry ---
    TEST "registry shows local + federated agents":
        mock 3 local + 2 federated
        render <RegistryDashboard />
        ASSERT total shown == 5
```

---

## 9. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `/agents` route renders marketplace | Navigate, visual check |
| 2 | Agent cards show capabilities + trust score | Visual + test |
| 3 | Capability filter works across all agents | Select filter → verify |
| 4 | Task delegation creates task via A2A engine | Mock + network check |
| 5 | Task board shows live status updates | Poll or WebSocket |
| 6 | Task tree renders parent-child hierarchy | Decomposed task visual |
| 7 | Agent builder produces valid manifest.yaml | YAML validation |
| 8 | .bizra-agent download works | Click → file downloaded |
| 9 | PCI-signed messages viewable in inspector | Message detail view |
| 10 | All components use Phase 42 tokens | `grep` audit |

---

## 10. Dependency Chain

```
Phase 42 (Tokens) ← visual foundation
Phase 47 (Infra) ← BizraClient transport
Phase 46 (Community) ← SAP session patterns reused
Phase 48 (HDA) ← permits + capability model shared
Phase 49 (This) ← builds on all above + A2A engine backend
```
