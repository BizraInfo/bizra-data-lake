# Alpha-100 Rollout Plan — First 100 Users

---

## Objective

Deploy BIZRA Node0 to 100 users across four phases, validating product-market fit, infrastructure stability, and federation readiness. Each phase has hard entry and exit criteria. No phase proceeds until the previous phase's exit criteria are met.

**Reference:** Geoffrey Moore, *Crossing the Chasm* (1991) — focus on a beachhead segment, deliver the whole product, and expand concentrically. The Alpha-100 is not a beta test. It is a beachhead operation.

---

## Phase Overview

| Phase | Timeline | Users | Channel | Method |
|-------|----------|-------|---------|--------|
| 0: Hardening | Week 1-2 | Internal | NODE0 | 100 demo runs, record video |
| 1: Inner Circle | Week 3-4 | 1-10 | Dubai tech | USB deploy, 1:1 onboarding |
| 2: Trusted Network | Week 5-6 | 11-30 | Referred | Docker, group workshops |
| 3: Public Alpha | Week 7-10 | 31-100 | Open | Unified Installer, federation |

---

## Phase 0: Hardening (Week 1-2)

**Users:** Internal team only (NODE0)

**Method:** Run the MoneyShot demo 100 times across different hardware configurations, mission types, and failure scenarios. Record at least 3 polished video variants (investor cut, technical cut, conference cut).

### Entry Criteria

- All 7,907+ tests passing in CI (Python + Rust)
- Ihsan gate operational with threshold >= 0.95
- FATE binding active (Z3 + Dilithium)
- MoneyShot demo script (`scripts/moneyshot_demo.py`) functional in both `--mock` and `--live` modes
- Evidence Ledger appending correctly with BLAKE3 hashes

### Success Metrics

| Metric | Target |
|--------|--------|
| Demo completion rate | >= 98/100 runs |
| Average mission duration | < 120 seconds |
| FATE violations | 0 across all runs |
| Ihsan average | >= 0.95 |
| Crash/hang rate | 0 |
| Video recordings | >= 3 polished variants |

### Exit Criteria

- 98+ successful demo runs out of 100
- Zero crashes or hangs under normal operation
- At least one polished demo video ready for external sharing
- All known P0 and P1 bugs resolved
- Hardware compatibility matrix documented (see requirements below)

### Risk Mitigation

- **Risk:** Demo fails on edge-case hardware configurations.
  **Mitigation:** Test on at least 3 distinct GPU/CPU combinations. Document minimum viable hardware.
- **Risk:** LM Studio or Ollama version incompatibility.
  **Mitigation:** Pin compatible versions in requirements. Test against latest and latest-minus-one.
- **Risk:** Demo fatigue — 100 runs reveals subtle state leaks.
  **Mitigation:** Full system reset between runs. Monitor memory and disk usage trends across all 100 runs.

---

## Phase 1: Inner Circle (Week 3-4)

**Users:** 1-10 (hand-selected from Dubai tech community)

**Method:** USB deployment with pre-configured environment. 1:1 onboarding sessions (60-90 minutes each). Direct communication channel with development team.

### Entry Criteria

- Phase 0 exit criteria met
- USB deployment image built and tested on target hardware
- Onboarding script written and rehearsed
- Support channel established (private Discord or Signal group)
- Telemetry system operational (with explicit opt-in consent)
- Feedback form designed and tested

### Success Metrics

| Metric | Target |
|--------|--------|
| Successful installations | 10/10 |
| Time to first mission | < 15 minutes |
| User-reported bugs (P0/P1) | 0 / <= 3 |
| Daily active usage (days 3-14) | >= 60% of users |
| Net Promoter Score | >= 40 |
| Qualitative: "Would you pay for this?" | >= 7/10 yes |

### Exit Criteria

- All 10 users successfully installed and completed at least one mission
- No P0 bugs open
- All P1 bugs triaged with fix timeline
- At least 6/10 users active on day 14
- Onboarding process refined based on observed friction points
- At least 2 users willing to refer others (entry to Phase 2)

### Risk Mitigation

- **Risk:** USB deployment fails on user's specific Windows/WSL configuration.
  **Mitigation:** Pre-screen user hardware. Prepare a Docker fallback image. Schedule onboarding with buffer time for troubleshooting.
- **Risk:** Users do not understand the value proposition within the onboarding session.
  **Mitigation:** Lead with the MoneyShot demo (show, then explain). Prepare a 1-page "What You Just Saw" summary.
- **Risk:** Privacy concerns about telemetry.
  **Mitigation:** Telemetry is strictly opt-in. Show users exactly what is collected before they consent. Provide a single command to disable.

---

## Phase 2: Trusted Network (Week 5-6)

**Users:** 11-30 (referred by Phase 1 users, plus targeted outreach)

**Method:** Docker-based deployment. Group onboarding workshops (5-8 users per session, 90 minutes). Asynchronous support via Discord.

### Entry Criteria

- Phase 1 exit criteria met
- Docker image published to private registry
- Docker Compose file tested on Linux, macOS, and Windows (WSL2)
- Group workshop agenda and materials prepared
- Known issues from Phase 1 documented in a public-facing FAQ
- At least 2 Phase 1 users willing to co-facilitate workshops

### Success Metrics

| Metric | Target |
|--------|--------|
| Successful installations | >= 18/20 |
| Time to first mission | < 30 minutes (self-serve) |
| User-reported bugs (P0/P1) | 0 / <= 5 |
| Weekly active usage (week 2+) | >= 50% of users |
| Community contributions (bug reports, suggestions) | >= 10 |
| Federation test (node-to-node) | >= 1 successful pair |

### Exit Criteria

- At least 25 users installed and active
- No P0 bugs open
- Docker deployment success rate >= 90% without manual intervention
- At least one successful node-to-node federation test
- Workshop materials refined and documented for reuse
- Community Discord active with organic discussion (not just support requests)

### Risk Mitigation

- **Risk:** Docker deployment fails on specific OS/hardware combinations.
  **Mitigation:** Publish a compatibility matrix. Provide a troubleshooting guide. Maintain a "known issues" document updated within 24 hours of new reports.
- **Risk:** Group workshops are less effective than 1:1 onboarding.
  **Mitigation:** Cap workshops at 8 users. Pair each new user with a Phase 1 "buddy." Record workshops for async review.
- **Risk:** Federation protocol not stable enough for multi-node testing.
  **Mitigation:** Federation is a stretch goal for Phase 2, not a gate. If federation is unstable, document the issues and defer to Phase 3. Do not block the rollout on federation readiness.

---

## Phase 3: Public Alpha (Week 7-10)

**Users:** 31-100 (open registration with lightweight screening)

**Method:** Unified Installer (cross-platform). Self-serve onboarding with documentation. Federation enabled. Community-driven support with team escalation.

### Entry Criteria

- Phase 2 exit criteria met
- Unified Installer tested on Windows 10/11, macOS 13+, Ubuntu 22.04+
- Self-serve onboarding documentation published (installation guide, first mission tutorial, FAQ)
- Federation protocol stable for at least 5 concurrent nodes
- Monitoring and alerting operational (uptime, error rates, user activity)
- Landing page with registration form live

### Success Metrics

| Metric | Target |
|--------|--------|
| Successful installations | >= 60/70 |
| Time to first mission (self-serve) | < 45 minutes |
| User-reported bugs (P0/P1) | 0 / <= 10 |
| Weekly active usage (week 2+) | >= 40% of users |
| Federation nodes active | >= 10 |
| SEED tokens minted (total) | >= 500 |
| Community contributions | >= 50 (issues, PRs, docs) |
| Organic referrals | >= 20% of new users |

### Exit Criteria

- 100 users registered and installed
- At least 70 users completed at least one mission
- Federation network stable with >= 10 active nodes
- No P0 bugs open, P1 backlog manageable (< 10 open)
- Product-market fit signal: >= 40% of users active in week 4
- Decision made on next phase (Beta, fundraise, or pivot)

### Risk Mitigation

- **Risk:** Unified Installer fails on edge-case OS configurations.
  **Mitigation:** Provide Docker as fallback for all platforms. Maintain a real-time compatibility matrix. Staff a dedicated installation support channel during the first 48 hours of public launch.
- **Risk:** Federation creates cascading failures across nodes.
  **Mitigation:** Federation is opt-in, not default. Implement circuit breakers and graceful degradation. Each node must function fully in standalone mode regardless of federation state.
- **Risk:** Support volume overwhelms the team.
  **Mitigation:** Invest in documentation and FAQ before launch. Recruit Phase 1/2 users as community moderators. Set expectations: response time is 24 hours, not instant.
- **Risk:** Low activation rate (install but never use).
  **Mitigation:** Send a "first mission" onboarding email 24 hours after installation. Provide three pre-built mission templates that require zero configuration.

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 3060 (8GB VRAM) or Apple M1 |
| RAM | 16 GB |
| Storage | 20 GB free (SSD recommended) |
| OS | Windows 10/11 (WSL2), macOS 13+, Ubuntu 22.04+ |
| Network | Broadband for initial setup; offline operation supported after |

### Recommended

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4070+ (12GB+ VRAM) or Apple M2 Pro+ |
| RAM | 32 GB |
| Storage | 50 GB free (NVMe SSD) |
| OS | Windows 11 (WSL2), macOS 14+, Ubuntu 24.04 |
| Network | Broadband; static IP preferred for federation |

### Notes

- Apple Silicon users run inference via Ollama with Metal acceleration. Performance is comparable to RTX 3060 for 7B parameter models.
- Systems without a discrete GPU can run in CPU-only mode with reduced inference speed. This is supported but not recommended for production missions.
- Docker deployment requires Docker Desktop 4.25+ (Windows/macOS) or Docker Engine 24+ (Linux).

---

## Software Requirements

| Software | Version | Required | Notes |
|----------|---------|----------|-------|
| Python | 3.11+ | Yes | 3.12 recommended |
| Rust | 1.88+ stable | For building from source | Pre-built binaries provided |
| Docker | 24+ | Optional (Phase 2+) | Required for Docker deployment path |
| LM Studio | Latest | Recommended | Primary local inference backend |
| Ollama | 0.3+ | Alternative | Fallback inference backend |
| Node.js | 18+ | For dashboard | Included in Unified Installer |
| Git | 2.40+ | Yes | For updates and version control |

---

## Support Channels

| Channel | Purpose | Response Time |
|---------|---------|---------------|
| GitHub Issues | Bug reports, feature requests | 48 hours (P0: 4 hours) |
| Discord (private, Phase 1-2) | Real-time support, onboarding | Same day |
| Discord (public, Phase 3) | Community support, discussion | Best effort (community + team) |
| Email (support@bizra.dev) | Sensitive issues, account problems | 24 hours |
| Weekly Office Hours | Live Q&A, demo sessions | Scheduled (Thursdays, 1 hour) |

---

## Feedback Loop

### Weekly Retrospectives

Every Friday, the team conducts a 30-minute retrospective covering:

1. **New users onboarded** — count, success rate, friction points
2. **Bugs reported** — P0/P1/P2 breakdown, resolution rate
3. **Feature requests** — categorized and prioritized
4. **Usage patterns** — mission types, agent utilization, session duration
5. **Federation health** — node count, connection stability, sync latency
6. **Decision log** — what changed this week and why

### Usage Telemetry

All telemetry is opt-in with explicit consent. Users can disable telemetry with a single command:

```bash
bizra config set telemetry.enabled false
```

**What is collected (when opted in):**

- Mission count and completion rate (no mission content)
- Agent utilization (which agents are used, not what they produce)
- Session duration and frequency
- Error codes and stack traces (no user data)
- Hardware profile (GPU, RAM, OS — for compatibility analysis)

**What is never collected:**

- Mission content, prompts, or outputs
- File contents or directory structures
- Personal information beyond what the user explicitly provides
- Network traffic content
- Keystrokes or screen recordings

### Feedback Forms

Structured feedback collected at three points:

1. **Post-onboarding** (within 24 hours of first mission) — installation experience, time to value, initial impressions
2. **Weekly pulse** (automated, optional) — satisfaction score (1-10), one thing to improve, one thing that worked
3. **Phase exit** (at each phase boundary) — comprehensive survey covering product-market fit, willingness to pay, referral likelihood

---

## Success Definition

The Alpha-100 succeeds if, at the end of Week 10:

1. At least 70 of 100 users have completed at least one mission
2. At least 40 of 100 users are active in the final week
3. Federation network has at least 10 stable nodes
4. Net Promoter Score across all phases is >= 30
5. At least 5 users have made community contributions (issues, PRs, docs, or mentorship)
6. The team has a clear, data-backed decision on the next phase

The Alpha-100 fails if:

1. Fewer than 50 users successfully install
2. P0 bug rate exceeds 1 per week sustained over 3+ weeks
3. Fewer than 20 users are active in the final week
4. The team cannot articulate product-market fit with data by Week 10

---

## Standing on Giants

> "The key to crossing the chasm is to target a specific niche, dominate it, and use it as a springboard." — Geoffrey Moore, *Crossing the Chasm* (1991)

The Alpha-100 is not about reaching 100 users. It is about finding the 10 users who cannot live without BIZRA, then using their conviction to reach the next 90. Phase 1 (Inner Circle) is the beachhead. Everything else follows from whether those 10 users stay.

Moore's framework demands a "whole product" — not a minimum viable product, but the minimum product that solves the complete problem for the beachhead segment. For BIZRA's beachhead (Dubai-based technical founders running sovereign AI workloads), the whole product is: local inference, constitutional governance, audit trail, and a working demo they can show to their own investors.

The Alpha-100 is designed to deliver exactly that, and nothing more, until the beachhead is held.
