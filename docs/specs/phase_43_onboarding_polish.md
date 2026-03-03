# Phase 43: Onboarding Polish

> Standing on Giants: Nielsen (usability heuristics, 1994) · Krug (Don't Make Me Think, 2000) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

The 5-step onboarding wizard (`filedfs/onboarding/`) is functionally complete but
needs brand alignment, UX polish, and integration with the live Node0 backend.
Steps work in isolation but don't connect to actual TEACH/RECEIVE verbs in
production mode — simulated responses only.

| Dimension | Current | Target |
|-----------|---------|--------|
| Brand alignment | Inline styles, no tokens | Phase 42 tokens throughout |
| Seed Test (TeachStep) | 4 questions, text-only | Richer UI: trait chips, progress ring, persona preview |
| PAT Intro | Not present | Agent roster carousel with role explanations |
| First Session (FirstChatStep) | Simulated responses | Live RECEIVE verb through BizraClient |
| Completion flow | Static summary | Animated sovereignty score reveal + next-steps CTA |
| Accessibility | None | Focus management, aria-labels, keyboard nav |
| Validation | None | Per-field validation with inline errors |
| State persistence | In-memory only | IndexedDB checkpoint — resume after browser close |

### Existing Step Inventory

| Step | File | LOC | Status |
|------|------|-----|--------|
| 1. Verify | VerifyStep.jsx | 351 | Functional, needs token migration |
| 2. Provider | ProviderStep.jsx | 405 | Functional, needs token migration + validation |
| 3. Teach | TeachStep.jsx | 1141 | Functional, needs UX polish + live verb wiring |
| 4. Chat | FirstChatStep.jsx | 448 | Simulated, needs live BizraClient integration |
| 5. Dashboard | DashboardStep.jsx | 649 | Functional, needs animated reveal + CTA |

---

## 2. Architecture

```
onboarding/
├── OnboardingFlow.jsx         # State machine (5 steps) — ADD checkpoint save
├── steps/
│   ├── VerifyStep.jsx         # RESTYLE with tokens
│   ├── ProviderStep.jsx       # RESTYLE + ADD validation
│   ├── TeachStep.jsx          # RESTYLE + ADD persona preview card
│   ├── PATIntroStep.jsx       # NEW — agent roster carousel
│   ├── FirstChatStep.jsx      # WIRE to live BizraClient
│   └── DashboardStep.jsx      # ADD animated sovereignty reveal
├── components/
│   ├── StepShell.jsx          # Shared step layout (title, progress, nav)
│   ├── PersonaPreview.jsx     # Live persona card from TEACH answers
│   ├── AgentCarousel.jsx      # PAT agent roster with roles
│   └── SovereigntyReveal.jsx  # Animated score + tier + confetti
└── hooks/
    └── useOnboardingPersist.js # IndexedDB checkpoint save/restore
```

### Step Flow (revised: 6 steps)

```
[Verify] → [Provider] → [Teach/Seed Test] → [PAT Intro] → [First Chat] → [Dashboard]
  (1/6)      (2/6)          (3/6)              (4/6)          (5/6)          (6/6)
```

---

## 3. Pseudocode: Onboarding State Machine

```
PROCEDURE OnboardingFlow():
    STATE step = 0
    STATE data = {
        installVerified: false,
        provider: null,           # 'lm_studio' | 'ollama' | 'cloud'
        model: '',
        apiKey: '',
        policyHash: '',
        teachData: {
            role: '', values: '', goal: '',
            work_schedule: '', primary_tools: '',
            communication_pref: '', priority_domains: '',
            automation_comfort: ''
        },
        traits: [],               # extracted from TEACH responses
        patAgentsSeen: false,
        firstChatComplete: false,
        sovereigntyScore: 0.0,
    }
    STATE checkpoint = useOnboardingPersist()

    ON_MOUNT:
        restored = checkpoint.restore()
        IF restored:
            data = restored.data
            step = restored.step
            SHOW "Welcome back — resuming from step {step+1}"

    FUNCTION goNext():
        IF NOT validate(step, data): RETURN
        checkpoint.save(step + 1, data)
        step += 1

    FUNCTION goBack():
        step = max(0, step - 1)

    FUNCTION updateState(partial):
        data = { ...data, ...partial }

    RENDER:
        <StepShell
            currentStep={step}
            totalSteps={6}
            title={STEP_TITLES[step]}
            onBack={step > 0 ? goBack : null}
        >
            SWITCH step:
                0 → <VerifyStep data={data} onUpdate={updateState} onNext={goNext} />
                1 → <ProviderStep data={data} onUpdate={updateState} onNext={goNext} />
                2 → <TeachStep data={data} onUpdate={updateState} onNext={goNext} />
                3 → <PATIntroStep data={data} onUpdate={updateState} onNext={goNext} />
                4 → <FirstChatStep data={data} onUpdate={updateState} onNext={goNext} />
                5 → <DashboardStep data={data} />
        </StepShell>
```

---

## 4. Pseudocode: Teach Step Polish

```
PROCEDURE TeachStep(data, onUpdate, onNext):
    STATE currentQuestion = 0
    STATE answers = data.teachData
    STATE extractedTraits = []
    STATE isProcessing = false

    QUESTIONS = [
        { id: 'role',    prompt: "What is your primary role or craft?",     kind: 'expertise', maxLen: 200 },
        { id: 'values',  prompt: "What principles guide your work?",        kind: 'principles', maxLen: 300 },
        { id: 'goal',    prompt: "What is your most important goal right now?", kind: 'goals', maxLen: 200 },
        { id: 'ops',     prompt: "Describe your ideal work day",            kind: 'patterns', maxLen: 300 },
    ]

    FUNCTION submitAnswer(questionId, answer):
        isProcessing = true
        # Fire TEACH verb to live backend
        response = await bizraClient.send('TEACH', {
            kind: QUESTIONS[currentQuestion].kind,
            content: answer,
            confidence: 0.8,
            timestamp: Date.now()
        })

        # Extract traits from response
        IF response.ok:
            newTraits = parseTraits(response.taught, response.kind)
            extractedTraits = [...extractedTraits, ...newTraits]
            onUpdate({ teachData: { ...answers, [questionId]: answer }, traits: extractedTraits })

        isProcessing = false
        IF currentQuestion < QUESTIONS.length - 1:
            currentQuestion += 1
        ELSE:
            onNext()

    RENDER:
        <div className="teach-step">
            # Progress ring showing questions completed
            <ProgressRing current={currentQuestion} total={4} />

            # Current question card
            <QuestionCard
                question={QUESTIONS[currentQuestion]}
                value={answers[QUESTIONS[currentQuestion].id]}
                onChange={(val) => answers[QUESTIONS[currentQuestion].id] = val}
                onSubmit={submitAnswer}
                isProcessing={isProcessing}
            />

            # Live persona preview (updates after each answer)
            <PersonaPreview traits={extractedTraits} />

            # Trait chips extracted so far
            <TraitChipList traits={extractedTraits} />
        </div>
```

---

## 5. Pseudocode: PAT Intro Step (NEW)

```
PROCEDURE PATIntroStep(data, onUpdate, onNext):
    STATE activeAgent = 0

    PAT_AGENTS = [
        { name: 'Scribe',    role: 'Memory & Knowledge',     icon: '📝', color: '--bz-facts',
          desc: 'Records everything you teach. Builds your personal knowledge graph.' },
        { name: 'Guardian',  role: 'Quality & Ethics Gate',   icon: '🛡️', color: '--bz-principles',
          desc: 'Ensures all outputs meet Ihsan (excellence). Your constitutional filter.' },
        { name: 'Strategist', role: 'Goal Planning',          icon: '🎯', color: '--bz-goals',
          desc: 'Breaks down your goals into actionable steps. Tracks progress.' },
        { name: 'Analyst',   role: 'Pattern Recognition',     icon: '📊', color: '--bz-patterns',
          desc: 'Detects patterns in your work. Surfaces insights you might miss.' },
        { name: 'Connector', role: 'Relationship Mapping',    icon: '🔗', color: '--bz-relationships',
          desc: 'Maps connections between your ideas, contacts, and resources.' },
        { name: 'Operator',  role: 'Task Execution',          icon: '⚡', color: '--bz-expertise',
          desc: 'Handles routine tasks. Automates what you approve.' },
        { name: 'Sentinel',  role: 'Security & Privacy',      icon: '🔒', color: '--bz-context',
          desc: 'Guards your data sovereignty. Manages consent and access.' },
    ]

    RENDER:
        <div className="pat-intro">
            <h2>Your Personal Agent Team</h2>
            <p>7 agents working together — you are the sovereign.</p>

            # Carousel with agent cards
            <AgentCarousel
                agents={PAT_AGENTS}
                active={activeAgent}
                onSelect={(i) => activeAgent = i}
            />

            # Selected agent detail
            <AgentDetailCard agent={PAT_AGENTS[activeAgent]} />

            # "Meet them all" progress
            <p>{seenCount}/7 agents explored</p>

            <Button onClick={() => { onUpdate({ patAgentsSeen: true }); onNext(); }}>
                Continue to First Chat →
            </Button>
        </div>
```

---

## 6. Pseudocode: First Chat Step (Live Wiring)

```
PROCEDURE FirstChatStep(data, onUpdate, onNext):
    STATE messages = []
    STATE inputText = ''
    STATE knowsMeScore = 0.0
    STATE isTyping = false
    CONST bizraClient = useBizraClient()

    ON_MOUNT:
        # Send system greeting
        messages.push({
            role: 'assistant',
            content: greeting_for_persona(data.teachData),
            timestamp: Date.now()
        })

    FUNCTION sendMessage():
        IF inputText.trim() == '': RETURN
        userMsg = { role: 'user', content: inputText, timestamp: Date.now() }
        messages.push(userMsg)
        inputText = ''
        isTyping = true

        # Fire RECEIVE verb to live backend
        response = await bizraClient.send('RECEIVE', {
            content: userMsg.content,
            timestamp: userMsg.timestamp
        })

        isTyping = false

        IF response.ok:
            assistantMsg = {
                role: 'assistant',
                content: response.content,
                confidence: response.confidence,
                agentsConsulted: response.agents_consulted,
                guardiansApproved: response.guardian_approved,
                timestamp: Date.now()
            }
            messages.push(assistantMsg)

            # Update KnowsMe score
            knowsMeScore = parseFloat(response.knows_me || '0')

            # After 3+ exchanges, enable completion
            IF messages.filter(m => m.role == 'user').length >= 3:
                onUpdate({ firstChatComplete: true, sovereigntyScore: knowsMeScore })

    RENDER:
        <div className="first-chat">
            # Mini KnowsMe gauge (updates live)
            <MiniGauge score={knowsMeScore} label="KnowsMe" />

            # Message list
            <MessageList messages={messages} />

            # Typing indicator
            IF isTyping: <TypingIndicator />

            # Input bar
            <ChatInput
                value={inputText}
                onChange={setInputText}
                onSend={sendMessage}
                placeholder="Say something to your node..."
            />

            # Completion CTA (appears after 3+ messages)
            IF data.firstChatComplete:
                <Button onClick={onNext}>See your sovereignty score →</Button>
        </div>
```

---

## 7. Pseudocode: Sovereignty Reveal

```
PROCEDURE DashboardStep(data):
    STATE revealed = false
    STATE animPhase = 'counting'  # 'counting' | 'tier' | 'agents' | 'done'

    ON_MOUNT:
        # Animated counter: 0 → data.sovereigntyScore over 2s
        animate(0, data.sovereigntyScore, 2000, (val) => {
            currentDisplayScore = val
        }).then(() => {
            animPhase = 'tier'
            WAIT 500ms
            animPhase = 'agents'
            WAIT 500ms
            animPhase = 'done'
            revealed = true
        })

    RENDER:
        <div className="sovereignty-reveal">
            # Large animated KnowsMe gauge
            <KnowsMeGauge score={currentDisplayScore} size={200} animated={true} />

            # Tier badge (slides in)
            IF animPhase >= 'tier':
                <TierBadge
                    tier={computeTier(data.sovereigntyScore)}
                    animate="slide-up"
                />

            # Agent roster summary
            IF animPhase >= 'agents':
                <AgentRosterSummary traits={data.traits} agentCount={7} />

            # Next steps CTA
            IF revealed:
                <div className="next-steps">
                    <h3>Your node is alive.</h3>
                    <CTAButton label="Open Dashboard" route="/dashboard" primary />
                    <CTAButton label="Teach More" route="/teach" />
                    <CTAButton label="Explore Agents" route="/agents" />
                </div>
        </div>
```

---

## 8. Pseudocode: Checkpoint Persistence

```
PROCEDURE useOnboardingPersist():
    CONST DB_NAME = 'bizra-onboarding'
    CONST STORE_NAME = 'checkpoint'

    FUNCTION save(step, data):
        db = await openDB(DB_NAME, 1, { upgrade(db) { db.createObjectStore(STORE_NAME) } })
        await db.put(STORE_NAME, { step, data, savedAt: Date.now() }, 'current')

    FUNCTION restore():
        db = await openDB(DB_NAME, 1)
        record = await db.get(STORE_NAME, 'current')
        IF record AND (Date.now() - record.savedAt) < 7 * 24 * 3600 * 1000:  # 7 day TTL
            RETURN record
        RETURN null

    FUNCTION clear():
        db = await openDB(DB_NAME, 1)
        await db.delete(STORE_NAME, 'current')

    RETURN { save, restore, clear }
```

---

## 9. TDD Anchors

```
TEST_SUITE onboarding_polish:

    TEST "step progression: 1→2→3→4→5→6":
        render <OnboardingFlow />
        ASSERT currentStep == 0
        fill_verify() → click next → ASSERT currentStep == 1
        fill_provider() → click next → ASSERT currentStep == 2
        fill_teach() → click next → ASSERT currentStep == 3
        browse_agents() → click next → ASSERT currentStep == 4
        send_3_messages() → click next → ASSERT currentStep == 5

    TEST "back button returns to previous step":
        render <OnboardingFlow /> at step 3
        click back → ASSERT currentStep == 2

    TEST "teach step fires TEACH verb":
        mock bizraClient.send
        render <TeachStep />
        fill question 1 → submit
        ASSERT bizraClient.send CALLED_WITH('TEACH', { kind: 'expertise', ... })

    TEST "PAT intro shows all 7 agents":
        render <PATIntroStep />
        agents = queryAll('[data-agent]')
        ASSERT agents.length == 7
        ASSERT agents[0].textContent CONTAINS 'Scribe'

    TEST "first chat fires RECEIVE verb":
        mock bizraClient.send → returns { ok: true, content: 'Hello', knows_me: '0.35' }
        render <FirstChatStep />
        type "Hello" → send
        ASSERT bizraClient.send CALLED_WITH('RECEIVE', { content: 'Hello', ... })
        ASSERT KnowsMeGauge.score == 0.35

    TEST "sovereignty reveal animates score":
        render <DashboardStep data={{ sovereigntyScore: 0.45 }} />
        WAIT 2500ms
        ASSERT gauge.displayValue ≈ 0.45
        ASSERT TierBadge visible with text 'SPROUT'

    TEST "checkpoint saves to IndexedDB":
        render <OnboardingFlow />
        advance to step 3
        checkpoint = await getFromIndexedDB('bizra-onboarding', 'current')
        ASSERT checkpoint.step == 3
        ASSERT checkpoint.data.provider != null

    TEST "checkpoint restores on remount":
        save checkpoint { step: 3, data: { provider: 'ollama' } }
        render <OnboardingFlow />
        ASSERT currentStep == 3
        ASSERT toast "Welcome back" visible

    TEST "validation blocks empty provider":
        render <ProviderStep />
        click next without selecting provider
        ASSERT error message visible
        ASSERT step did NOT advance

    TEST "keyboard navigation: Tab through steps":
        render <OnboardingFlow />
        press Tab → ASSERT focus on first interactive element
        press Enter → ASSERT action triggered
```

---

## 10. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | All steps use Phase 42 tokens (zero inline hex) | `grep -r '#D4A547' onboarding/` = 0 |
| 2 | TEACH verb fires for each question in TeachStep | Network tab shows 4 TEACH requests |
| 3 | RECEIVE verb fires in FirstChatStep | Network tab shows RECEIVE requests |
| 4 | PAT Intro carousel shows 7 agents | Visual inspection + test |
| 5 | Sovereignty reveal animates 0 → score over 2s | Visual inspection |
| 6 | Checkpoint persists across browser close | Close tab, reopen → resumes at last step |
| 7 | Keyboard navigation works for all steps | Tab/Enter through entire flow |
| 8 | Each step < 400 LOC | `wc -l onboarding/steps/*.jsx` |
