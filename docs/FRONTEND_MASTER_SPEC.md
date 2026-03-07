# Front-End Master Spec

Status: authoritative product surface contract  
Date: 2026-03-06  
Audience: product, design, frontend, desktop, operator-console, onboarding

This document freezes the front-end spearpoint for BIZRA.

It is not a claim that the full UI is already implemented. The current repo contains working prototypes, specs, and design artifacts that establish the correct direction. This document converts that material into one product contract.

Grounding artifacts:
- [WEBSITE_PLAN.md](WEBSITE_PLAN.md)
- [phase_43_onboarding_polish.md](specs/phase_43_onboarding_polish.md)
- [phase_45_daily_loop_dashboard.md](specs/phase_45_daily_loop_dashboard.md)
- [`filedfs/onboarding/OnboardingFlow.jsx`](../filedfs/onboarding/OnboardingFlow.jsx)
- [`filedfs/bizra-dashboard.jsx`](../filedfs/bizra-dashboard.jsx)
- [`filedfs/node0-dashboard.jsx`](../filedfs/node0-dashboard.jsx)
- [`docs/node0_operations_dashboard.html`](node0_operations_dashboard.html)
- [`filedfs/node0-mvp.jsx`](../filedfs/node0-mvp.jsx)

Frozen surface order:

`website -> onboarding wizard -> daily dashboard -> contributor desktop client -> operator/admin console`

## 1. Information Architecture

The front-end is a layered experience system, not one giant app.

### Surface A: Public Web

Purpose:
- trust building
- differentiation
- safety and sovereignty explanation
- conversion into signup, demo, or contributor entry

Primary pages:
- Home
- How It Works
- Safety / Data / Sovereignty
- Demo
- FAQ / Docs
- Sign Up / Join

Core promises it must answer fast:
- what BIZRA is
- why it is safe
- why it is different
- how value appears quickly

Source grounding:
- [WEBSITE_PLAN.md](WEBSITE_PLAN.md:1) already defines the public site stack, page skeleton, typography, FAQ, and public content priorities

### Surface B: Onboarding App

Purpose:
- deliver first value in less than 15 minutes
- convert curiosity into identity, context, and first action

This surface has two distinct tracks and they must stay separate:
- consumer onboarding
- contributor or node onboarding

Consumer onboarding flow:
- Why BIZRA
- Identity
- Personalization
- First Win
- Dashboard activation

Contributor onboarding flow:
- email or OTP plus device identity
- environment or provider check
- discovery, redaction, and pack flow
- first Proof of Impact workflow
- first assignment or mission claim
- live impact and reward view

Source grounding:
- [phase_43_onboarding_polish.md](specs/phase_43_onboarding_polish.md:1)
- [`filedfs/onboarding/OnboardingFlow.jsx`](../filedfs/onboarding/OnboardingFlow.jsx:1)

### Surface C: Core App Shell

Purpose:
- daily use
- habit loop
- progress visibility
- reward and community reinforcement

Primary navigation:
- Home
- Learn
- Earn
- Community
- Wallet / Profile

### Surface D: Contributor Desktop Client

Purpose:
- local node workflows
- proof operations
- expert tooling
- diagnostics and approvals

Modes:
- guided GUI
- TUI/CLI expert mode
- diagnostics and logs

### Surface E: Operator / Admin Console

Purpose:
- runtime visibility
- governance and audit review
- proof and deployment operations
- system atlas and incident response

This is not the main user app. It is the control plane.

### Explicit Non-Goals For The First Tranche

Do not front-load:
- BIZRAverse
- trading UI
- full governance suite
- every speculative application surface

The first tranche is:

`trust -> onboarding -> dashboard -> daily loop`

## 2. Design System

The product system should reuse the existing brand evidence instead of inventing a new aesthetic.

### Core Tokens

Primary direction:
- navy and deep-space backgrounds
- gold accents
- white and light-gray text
- restrained green, red, blue for state and telemetry

Typography:
- Inter for UI
- Noto Sans Arabic for Arabic support
- monospace only for telemetry, proofs, and operator data

Source grounding:
- [WEBSITE_PLAN.md](WEBSITE_PLAN.md:70)
- [`docs/node0_operations_dashboard.html`](node0_operations_dashboard.html:1)
- [`filedfs/bizra-dashboard.jsx`](../filedfs/bizra-dashboard.jsx:1)
- [`filedfs/node0-dashboard.jsx`](../filedfs/node0-dashboard.jsx:1)

### Product Tone Split

This is the main product decision:

- user-facing surfaces are human-first, calm, trustworthy, and fast to understand
- operator and governance surfaces are premium, dense, and control-room oriented

Keep:
- premium polish
- clarity
- confidence

Do not keep for the main user product:
- luxury framing
- exclusivity posture
- “elite users only” cues

That tone belongs only on:
- operator console
- governance surfaces
- enterprise or investor control-room views

### Interaction Rules

Mandatory:
- mobile-first layouts
- visible progress states
- inline validation
- keyboard navigation
- focus management
- tooltip and guided-help support
- consistent loading, error, and empty states

Accessibility baseline:
- WCAG 2.1 AA minimum
- 44px touch targets
- contrast-safe text and control states

Performance baseline:
- public site load target under 3 seconds
- onboarding screens must feel instant on ordinary laptops
- dashboard first meaningful render should favor cached state and then hydrate live data

### Release Gates

Design system freeze requires:
- accessibility review complete
- responsive review complete
- performance review complete
- Arabic branding consistency review complete before logo or identity freeze

## 3. Onboarding Flow

Onboarding is the highest-SNR front-end flow and ships before any broader application shell expansion.

### Consumer Flow

Frozen 4-step compact version:

1. Why BIZRA  
One screen. Clear value. No jargon.

2. Identity  
Email or OTP, account, optional wallet connect, or custodial path.

3. Personalization  
Goals, skill interest, current level, weekly time, preferred guidance style.

4. First Win  
One small task, one small reward, one visible dashboard update.

Success criteria:
- onboarding completion > 85%
- time to first value < 15 minutes
- first reward or proof event within 24 hours

### Contributor / Node Flow

Frozen flow:
- Verify environment
- Choose provider or runtime
- Teach or seed identity
- PAT introduction
- First live chat or first proof flow
- Dashboard / client activation

Source grounding:
- [phase_43_onboarding_polish.md](specs/phase_43_onboarding_polish.md:1)
- [`filedfs/onboarding/OnboardingFlow.jsx`](../filedfs/onboarding/OnboardingFlow.jsx:1)

### What This Flow Must Demonstrate

The user must feel, not infer:
- BIZRA understands context
- BIZRA protects sovereignty
- BIZRA produces a first useful outcome quickly

### Mandatory Onboarding Behaviors

- state persistence or checkpoint resume
- inline validation
- animated but restrained progress cues
- live or realistic first-value reveal
- explicit next step into the dashboard

## 4. App Shell

The daily dashboard is the main product shell.

The current repo already converges on this through multiple prototypes and the Daily Loop spec.

Source grounding:
- [phase_45_daily_loop_dashboard.md](specs/phase_45_daily_loop_dashboard.md:1)
- [`filedfs/bizra-dashboard.jsx`](../filedfs/bizra-dashboard.jsx:1)
- [`filedfs/node0-dashboard.jsx`](../filedfs/node0-dashboard.jsx:1)

### Home Screen Structure

Do not launch with 12 competing modules.

Freeze the shell around four blocks:
- Today
- Progress
- Rewards
- Community

Recommended home layout:
- sovereignty summary or health card
- today plan or next mission
- progress delta or learning state
- rewards and wallet snapshot
- community or accountability activity

For the node-oriented dashboard, the same shell can extend with:
- agent grid
- activity feed
- mission approvals
- accumulator or reward-cycle visualization

### Shell Rules

- `/` is the daily home surface
- every module must answer “what should I do next?”
- card density is allowed, clutter is not
- live telemetry is secondary to user clarity

### App Navigation Freeze

Primary user nav:
- Home
- Learn
- Earn
- Community
- Wallet / Profile

Secondary utilities:
- notifications
- help
- settings

### KPI Gates

First shell release should be measured against:
- task completion >= 95% in guided usability tests
- onboarding to dashboard handoff without drop-off cliff
- daily-return loop visible on first login
- mobile and desktop parity for core flows

## 5. TUI / Operator Split

This split is non-negotiable.

### GUI Is For Mainstream Use

GUI owns:
- discovery
- onboarding
- dashboard
- rewards visibility
- learning and community surfaces

### TUI / CLI Is For Power Users

TUI owns:
- contributor workflows
- node operations
- proof generation
- verification
- diagnostics
- approvals
- logs

The TUI is not the consumer front door.
It is the expert plane.

### Operator Console Owns The Control Plane

The operator console should evolve from the existing atlas and operations dashboard patterns.

It owns:
- runtime status
- deploy and rollout views
- governance and proposal review
- proof and audit views
- system graph and architecture atlas
- incident and SRE actions

Source grounding:
- [`docs/node0_operations_dashboard.html`](node0_operations_dashboard.html:1)
- [`filedfs/node0-mvp.jsx`](../filedfs/node0-mvp.jsx:1)

### Shared Contracts Across GUI, TUI, and Operator

Shared backend contracts:
- onboarding state and identity
- dashboard status models
- mission and proof models
- health and telemetry endpoints
- approval events

Shared experience rule:
- the same truth model can be rendered differently for user, contributor, and operator surfaces
- the user surface must optimize comprehension
- the operator surface can optimize density

### Freeze Decision

Front-end delivery order is now:

1. public trust site
2. onboarding wizard
3. daily dashboard shell
4. contributor desktop client with GUI plus TUI
5. operator/admin console

If a new front-end proposal does not strengthen one of those five in order, it is not first-priority work.
