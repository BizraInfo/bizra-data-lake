# Phase 73.03: Onboarding Wizard — Consumer + Contributor

**Target:** `/onboarding` and `/onboarding/contributor` routes
**Upstream:** Phase 43 (onboarding polish), `FRONTEND_MASTER_SPEC.md` Section 3

## Existing Assets

| File | LOC | Status |
|---|---|---|
| `filedfs/onboarding/OnboardingFlow.jsx` | ~200 | State machine, 5 steps |
| `filedfs/onboarding/steps/VerifyStep.jsx` | 351 | Functional |
| `filedfs/onboarding/steps/ProviderStep.jsx` | 405 | Functional |
| `filedfs/onboarding/steps/TeachStep.jsx` | 1141 | Functional, simulated |
| `filedfs/onboarding/steps/FirstChatStep.jsx` | 448 | Simulated backend |
| `filedfs/onboarding/steps/DashboardStep.jsx` | 649 | Functional |

**Migration plan:** Extract shared logic from existing JSX files. Wire to
live backend contracts. Add PATIntroStep (Phase 43 spec). Migrate inline
styles to design tokens.

## Consumer Onboarding Flow (6 steps)

```pseudocode
# src/features/onboarding/ConsumerOnboarding.tsx

IMPORT { api } FROM "../../shared/api/client"
IMPORT { OnboardingState } FROM "../../shared/types/api"

STEPS = ["verify", "provider", "teach", "pat-intro", "first-chat", "dashboard"]

PROCEDURE ConsumerOnboarding():
    STATE step: number = 0
    STATE data: OnboardingState = {
        step: 0,
        track: "consumer",
        data: {
            install_verified: false,
            provider: null,
            model: "",
            teach_data: {},
            traits: [],
            pat_agents_seen: false,
            first_chat_complete: false,
            sovereignty_score: 0.0,
        },
        last_checkpoint: new Date().toISOString()
    }

    # Checkpoint resume on mount
    ON_MOUNT:
        TRY:
            saved = await api.onboardingState()
            IF saved AND saved.track == "consumer":
                data = saved
                step = saved.step
                SHOW "Welcome back — resuming from step {step + 1}"
        CATCH:
            # No saved state — start fresh (IndexedDB fallback)
            saved = await idb.get("onboarding_checkpoint")
            IF saved:
                data = saved.data
                step = saved.step

    FUNCTION goNext():
        IF NOT validate(step, data): RETURN
        step += 1
        data.step = step
        data.last_checkpoint = new Date().toISOString()
        # Dual persistence: server + local
        api.saveOnboardingState(data).catch(noop)  # best-effort server
        idb.set("onboarding_checkpoint", { step, data })  # guaranteed local

    FUNCTION goBack():
        step = max(0, step - 1)

    FUNCTION updateData(partial):
        data = { ...data, data: { ...data.data, ...partial } }

    RENDER:
        <StepShell
            currentStep={step}
            totalSteps={6}
            title={STEP_TITLES[step]}
            onBack={step > 0 ? goBack : null}
        >
            SWITCH step:
                0 -> <VerifyStep data={data.data} onUpdate={updateData} onNext={goNext} />
                1 -> <ProviderStep data={data.data} onUpdate={updateData} onNext={goNext} />
                2 -> <TeachStep data={data.data} onUpdate={updateData} onNext={goNext} />
                3 -> <PATIntroStep data={data.data} onUpdate={updateData} onNext={goNext} />
                4 -> <FirstChatStep data={data.data} onUpdate={updateData} onNext={goNext} />
                5 -> <DashboardStep data={data.data} />
        </StepShell>
```

## Step 1: Verify

```pseudocode
PROCEDURE VerifyStep({ data, onUpdate, onNext }):
    """Check that Node0 is reachable and healthy."""
    STATE checking = false
    STATE result = null

    FUNCTION runCheck():
        checking = true
        TRY:
            health = await api.health()
            IF health.status == "healthy":
                result = { ok: true, version: health.version }
                onUpdate({ install_verified: true })
            ELSE:
                result = { ok: false, error: "Node is " + health.status }
        CATCH error:
            result = { ok: false, error: "Cannot reach Node0. Is it running?" }
        checking = false

    RENDER:
        <div>
            <p>"Let's make sure your node is running."</p>
            <Button onClick={runCheck} loading={checking}>"Check Connection"</Button>

            IF result?.ok:
                <SuccessCard message={"Connected to Node0 v" + result.version} />
                <Button onClick={onNext}>"Continue"</Button>

            IF result AND NOT result.ok:
                <ErrorCard message={result.error} />
                <HelpLink href="/site/faq#install">"Need help installing?"</HelpLink>
        </div>
```

## Step 2: Provider

```pseudocode
PROCEDURE ProviderStep({ data, onUpdate, onNext }):
    """Choose LLM backend: LM Studio, Ollama, or Cloud."""
    STATE provider = data.provider
    STATE model = data.model
    STATE validating = false

    PROVIDERS = [
        { id: "lm_studio", name: "LM Studio", desc: "Local models, full privacy" },
        { id: "ollama", name: "Ollama", desc: "Local models, easy setup" },
        { id: "cloud", name: "Cloud", desc: "No GPU needed, data leaves device" },
    ]

    FUNCTION selectProvider(id):
        provider = id
        onUpdate({ provider: id })

    FUNCTION validateAndContinue():
        validating = true
        TRY:
            result = await api.teach({
                kind: "provider_verify",
                content: provider,
                confidence: 1.0,
            })
            IF result.ok:
                onNext()
            ELSE:
                SHOW_ERROR("Could not connect to " + provider)
        CATCH:
            SHOW_ERROR("Verification failed")
        validating = false

    RENDER:
        <div>
            <p>"Choose your AI backend."</p>
            <RadioGroup options={PROVIDERS} value={provider} onChange={selectProvider} />

            IF provider == "cloud":
                <Warning>"Cloud mode sends data off-device. Local is recommended."</Warning>

            <Button onClick={validateAndContinue} disabled={!provider} loading={validating}>
                "Verify & Continue"
            </Button>
        </div>
```

## Step 3: Teach (Seed Test)

```pseudocode
PROCEDURE TeachStep({ data, onUpdate, onNext }):
    """4 questions that personalize the node."""
    STATE questionIdx = 0
    STATE answers = data.teach_data
    STATE traits: string[] = data.traits
    STATE submitting = false

    QUESTIONS = [
        { id: "role",   prompt: "What is your primary role or craft?",     kind: "expertise" },
        { id: "values", prompt: "What principles guide your work?",        kind: "principles" },
        { id: "goal",   prompt: "What is your most important goal now?",   kind: "goals" },
        { id: "ops",    prompt: "Describe your ideal work day.",           kind: "patterns" },
    ]

    FUNCTION submitAnswer(questionId, answer):
        submitting = true
        TRY:
            response = await api.teach({
                kind: QUESTIONS[questionIdx].kind,
                content: answer,
                confidence: 0.8,
            })
            IF response.ok:
                newTraits = response.traits || []
                traits = [...traits, ...newTraits]
                answers = { ...answers, [questionId]: answer }
                onUpdate({ teach_data: answers, traits: traits })
        CATCH:
            # Graceful degrade — save answer locally even if API fails
            answers = { ...answers, [questionId]: answer }
            onUpdate({ teach_data: answers })
        submitting = false

        IF questionIdx < QUESTIONS.length - 1:
            questionIdx += 1
        ELSE:
            onNext()

    RENDER:
        <div>
            <ProgressRing current={questionIdx} total={4} />
            <QuestionCard
                question={QUESTIONS[questionIdx]}
                value={answers[QUESTIONS[questionIdx].id] || ""}
                onSubmit={(val) => submitAnswer(QUESTIONS[questionIdx].id, val)}
                submitting={submitting}
            />
            <PersonaPreview traits={traits} />
        </div>
```

## Step 4: PAT Intro

```pseudocode
PROCEDURE PATIntroStep({ data, onUpdate, onNext }):
    """Introduce the 7 personal agents."""
    STATE roster = null
    STATE currentAgent = 0

    ON_MOUNT:
        TRY:
            roster = await api.agentRoster()
        CATCH:
            # Fallback: show static PAT descriptions
            roster = STATIC_PAT_ROSTER

    PAT_DESCRIPTIONS = {
        "WORKER":      "Executes tasks on your behalf",
        "RESEARCHER":  "Gathers information from your files and the web",
        "GUARDIAN":    "Monitors security and protects your data",
        "SYNTHESIZER": "Connects ideas across your knowledge",
        "VALIDATOR":   "Checks quality before anything ships",
        "COORDINATOR": "Orchestrates your agent team",
    }

    RENDER:
        <div>
            <h2>"Meet your agents"</h2>
            <p>"7 personal agents work for you. They learn your patterns and grow with you."</p>

            IF roster:
                <AgentCarousel
                    agents={roster.agents.filter(a => a.type == "PAT")}
                    descriptions={PAT_DESCRIPTIONS}
                    currentIndex={currentAgent}
                    onNavigate={(idx) => currentAgent = idx}
                />

            <Button onClick={() => { onUpdate({ pat_agents_seen: true }); onNext() }}>
                "Continue"
            </Button>
        </div>
```

## Step 5: First Chat

```pseudocode
PROCEDURE FirstChatStep({ data, onUpdate, onNext }):
    """One real interaction with the node. Proves it works."""
    STATE messages: Message[] = []
    STATE input = ""
    STATE sending = false

    FUNCTION sendMessage():
        IF input.trim() == "": RETURN
        sending = true
        userMsg = { role: "user", content: input, timestamp: Date.now() }
        messages = [...messages, userMsg]
        input = ""

        TRY:
            response = await api.teach({
                kind: "first_chat",
                content: userMsg.content,
                confidence: 0.8,
            })
            botMsg = { role: "assistant", content: response.reply, timestamp: Date.now() }
            messages = [...messages, botMsg]
        CATCH:
            botMsg = { role: "assistant", content: "I'm still warming up. Try again.", timestamp: Date.now() }
            messages = [...messages, botMsg]
        sending = false

    FUNCTION markComplete():
        onUpdate({ first_chat_complete: true })
        onNext()

    RENDER:
        <div>
            <ChatWindow messages={messages} />
            <ChatInput
                value={input}
                onChange={(v) => input = v}
                onSubmit={sendMessage}
                disabled={sending}
                placeholder="Ask your node anything..."
            />
            IF messages.length >= 2:
                <Button onClick={markComplete}>"Continue to Dashboard"</Button>
        </div>
```

## Step 6: Dashboard Activation

```pseudocode
PROCEDURE DashboardStep({ data }):
    """Reveal sovereignty score, show what's next."""
    STATE potential = null
    STATE lifecycle = null
    STATE animating = true

    ON_MOUNT:
        TRY:
            [potential, lifecycle] = await Promise.all([
                api.seedPotential(),
                api.lifecycle(),
            ])
        CATCH:
            potential = { sovereignty_score: 0.0, tier: "SEED" }
            lifecycle = { current_stage: "Seed", progress: 0.0 }
        # Reveal animation
        setTimeout(() => animating = false, 2000)

    RENDER:
        <div>
            IF animating:
                <SovereigntyReveal score={potential?.sovereignty_score || 0} />
            ELSE:
                <SovereigntyCard
                    score={potential.sovereignty_score}
                    tier={potential.tier}
                    stage={lifecycle.current_stage}
                    progress={lifecycle.progress}
                />

                <NextStepsCard>
                    <NextStep icon="message" label="Start a mission" href="/" />
                    <NextStep icon="book" label="Learn more" href="/learn" />
                    <NextStep icon="wallet" label="Check your wallet" href="/wallet" />
                </NextStepsCard>

                <Button href="/" primary>"Go to Dashboard"</Button>
        </div>
```

## Contributor Onboarding (`/onboarding/contributor`)

```pseudocode
# Same StepShell, different steps:
CONTRIBUTOR_STEPS = [
    "verify",           # Same as consumer: check Node0 health
    "environment",      # Check GPU, RAM, disk (hardware survey)
    "provider",         # Same as consumer but with more detail
    "identity",         # Ed25519 key generation + recovery phrase display
    "first-proof",      # Run a real Proof of Impact cycle
    "activation",       # Dashboard with node metrics
]

# Key difference from consumer: Step 4 (identity) shows the recovery
# phrase and requires the user to confirm they've saved it.

PROCEDURE IdentityStep({ data, onUpdate, onNext }):
    STATE identity = null
    STATE confirmed = false

    ON_MOUNT:
        # This calls the genesis ceremony if identity doesn't exist
        identity = await api._fetch("/v1/identity/status")
        IF identity.has_recovery_phrase:
            # Show phrase, require confirmation
            pass

    RENDER:
        <div>
            <h2>"Your sovereign identity"</h2>
            <IdentityCard
                nodeId={identity.node_id}
                publicKey={identity.public_key_short}
                tier={identity.tier}
            />
            IF identity.recovery_phrase AND NOT confirmed:
                <RecoveryPhraseDisplay words={identity.recovery_phrase} />
                <Checkbox
                    label="I have saved my recovery phrase securely"
                    checked={confirmed}
                    onChange={(v) => confirmed = v}
                />
            <Button onClick={onNext} disabled={!confirmed}>"Continue"</Button>
        </div>
```

## TDD Anchors

```pseudocode
TEST "consumer onboarding renders 6 steps":
    page = render(<ConsumerOnboarding />)
    ASSERT page.getByText("1 / 6") IS_VISIBLE

TEST "verify step calls /v1/health":
    page = render(<VerifyStep />)
    fireEvent.click(page.getByText("Check Connection"))
    ASSERT fetch.calledWith("/v1/health")

TEST "teach step sends TEACH verb per question":
    page = render(<TeachStep />)
    fireEvent.submit(page.getByRole("form"))
    ASSERT fetch.calledWith("/v1/onboarding/teach")

TEST "checkpoint saves on step advance":
    page = render(<ConsumerOnboarding />)
    # Advance past verify
    await advanceToStep(page, 1)
    ASSERT idb.get("onboarding_checkpoint").step == 1

TEST "checkpoint restores on remount":
    idb.set("onboarding_checkpoint", { step: 3, data: mockData })
    page = render(<ConsumerOnboarding />)
    ASSERT page.getByText("4 / 6") IS_VISIBLE  # resumed at step 3

TEST "first chat requires 2+ messages before continue":
    page = render(<FirstChatStep />)
    ASSERT page.queryByText("Continue to Dashboard") IS_NULL
    await sendMessage(page, "Hello")
    ASSERT page.getByText("Continue to Dashboard") IS_VISIBLE

TEST "dashboard step shows sovereignty score":
    page = render(<DashboardStep data={mockData} />)
    await waitFor(() => page.getByText("SEED"))
    ASSERT page.getByTestId("sovereignty-score") IS_VISIBLE

TEST "contributor onboarding shows recovery phrase":
    page = render(<IdentityStep />)
    ASSERT page.getByText("recovery phrase") IS_VISIBLE

TEST "contributor cannot proceed without confirming phrase":
    page = render(<IdentityStep />)
    ASSERT page.getByRole("button", { name: "Continue" }).disabled == true
    fireEvent.click(page.getByLabelText("I have saved"))
    ASSERT page.getByRole("button", { name: "Continue" }).disabled == false

TEST "all onboarding steps pass axe audit":
    FOR Step IN [VerifyStep, ProviderStep, TeachStep, PATIntroStep, FirstChatStep, DashboardStep]:
        page = render(<Step data={mockData} />)
        results = await axe(page.container)
        ASSERT results.violations.length == 0
```
