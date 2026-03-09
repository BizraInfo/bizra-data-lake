# BIZRA Genesis Sprint Blueprint
## From v0.80.0 to v1.0.0-GENESIS

**Authority:** Enforceable Spine v1.0, §10 Roadmap
**Framework:** PMBOK 7th Edition × DevOps × Constitutional Gates
**Duration:** 8 weeks (March 10 — May 4, 2026)
**Team:** 1 engineer (Mumo) + 1 NODE0 (i9-14900HX, 128GB, RTX 4090)
**Budget:** $2,000 (GPU compute for constitutional fine-tuning)
**Gate:** Ramadan 2026 launch window (~March 28)

---

## §1 SCOPE (PMBOK Knowledge Area 1)

### What Ships at Genesis

| Deliverable | Version | Acceptance Criteria | Sprint |
|------------|---------|-------------------|--------|
| ReflexCompiler | v1.0 | 46 tests pass, 50ms cache hit, wired to /v1/plan | 1 |
| MOE Engine | v1.0 | 5 experts loaded, HHMM routing, mission execution | 2 |
| HDA Controller | v1.0 | AHK + MCP tools, tier-gated permissions | 3 |
| Constitutional Fine-Tuning | v1.0 | DEMA persona, Ihsān scorer, LoRA checkpoints | 4 |
| SDPO Closed-Loop | v1.0 | Expert improvement from use, feature flag on | 5 |
| AaaS Protocol | v1.0 | ServiceContract, discovery, settlement | 6 |
| Alpha-10 Package | v1.0 | Installer, docs, support channel, 10 invitations | 7 |
| Genesis Gate | PASS | 68/68 checks, 5 SAT agents approve | 8 |

### What Does NOT Ship

| Excluded | Reason | When |
|----------|--------|------|
| Federation (Layer 2) | Requires 100+ nodes to test properly | Phase 2 |
| Custom model training | Requires forest data (none yet) | Phase 3 |
| Mobile installer | Desktop-first, mobile after stability | Phase 2 |
| Browser extension | MVP is terminal/desktop | Phase 2 |

---

## §2 SCHEDULE (PMBOK Knowledge Area 2)

### Week-by-Week Execution

```
WEEK 1 (Mar 10-14): THE SKELETON
  ┌─────────────────────────────────────────────────────┐
  │ Monday                                              │
  │   AM: git push origin main (10 commits)             │
  │   AM: python bizra_test.py --lock (verify v0.80.0)  │
  │   PM: Copy reflex_compiler.py → core/sovereign/     │
  │   PM: Copy test_reflex_compiler.py → tests/         │
  │                                                     │
  │ Tuesday                                             │
  │   AM: Wire ReflexCompiler into /v1/plan endpoint    │
  │   PM: Integration test: mission → cache → hit       │
  │                                                     │
  │ Wednesday                                           │
  │   AM: Run H-Neuron proof on NODE0 (Phi-2, ~25 min)  │
  │   PM: Film 3-minute demo (4 missions, precipitation)│
  │                                                     │
  │ Thursday                                            │
  │   AM: Delta tests, commit, push                     │
  │   PM: Verify CI pipeline (T0 + T1 gates)            │
  │                                                     │
  │ Friday                                              │
  │   AM: Version lock v0.85.0                          │
  │   PM: Review, retrospective, plan Week 2            │
  └─────────────────────────────────────────────────────┘
  
  Gate: v0.85.0 locked, 50ms demo filmed, CI pipeline green
  Risk: WSL2 filesystem slowness → mitigate: work on Linux filesystem

WEEK 2 (Mar 17-21): THE BRAIN
  ┌─────────────────────────────────────────────────────┐
  │ Monday                                              │
  │   AM: Download base models to NODE0                 │
  │       SmolLM2-135M, Qwen2.5-0.5B, SigLIP-SO400M   │
  │       Whisper-tiny, custom Expert-S scaffold        │
  │   PM: Build core/living_model/moe_engine.py         │
  │                                                     │
  │ Tuesday                                             │
  │   AM: HHMM→expert routing integration               │
  │   PM: Multi-expert composition (vision+reasoning)   │
  │                                                     │
  │ Wednesday                                           │
  │   AM: Integration: 100 missions through MOE         │
  │   PM: Benchmark: latency, memory, Ihsān scores      │
  │                                                     │
  │ Thursday                                            │
  │   AM: Wire MOE into ReflexCompiler pipeline         │
  │   PM: End-to-end: mission → MOE → reflex → cache    │
  │                                                     │
  │ Friday                                              │
  │   AM: Version lock v0.87.0                          │
  │   PM: Ramadan preparation begins (~March 28)        │
  └─────────────────────────────────────────────────────┘
  
  Gate: v0.87.0 locked, MOE engine operational, 5 experts loaded
  Risk: Model download size (~3GB total) → mitigate: quantized GGUF

WEEK 3 (Mar 24-28): THE HANDS — Ramadan begins ~Mar 28
  ┌─────────────────────────────────────────────────────┐
  │ Focus: HDA desktop controller + MCP integration     │
  │ Reduced hours (Ramadan schedule)                    │
  │                                                     │
  │ Mon-Tue: core/hda/desktop_controller.py             │
  │          AHK 2.0 bridge on NODE0                    │
  │          Permission tiers (Novice→Master)            │
  │                                                     │
  │ Wed-Thu: MCP tool registration                      │
  │          filesystem, browser, editor bindings        │
  │          Agent→tool mapping (P3→editor, P2→browser)  │
  │                                                     │
  │ Fri: Version lock v0.89.0                           │
  │      🌙 Ramadan begins — adjusted schedule           │
  └─────────────────────────────────────────────────────┘
  
  Gate: v0.89.0 locked, desktop automation works on NODE0
  Risk: Ramadan schedule → mitigate: reduced scope, core only

WEEK 4 (Mar 31 - Apr 4): THE DNA (Ramadan)
  ┌─────────────────────────────────────────────────────┐
  │ Focus: Constitutional fine-tuning on NODE0 (RTX 4090)│
  │ Reduced hours (Ramadan schedule)                    │
  │                                                     │
  │ Mon: DEMA persona dataset (1,000 conversations)     │
  │ Tue: LoRA fine-tune Expert-T (Qwen2.5-0.5B)        │
  │      Cost: ~$500 equivalent compute on NODE0         │
  │ Wed: Ihsān scorer dataset (10K evidence receipts)    │
  │ Thu: Train Expert-S (Crown) from Evidence Ledger     │
  │ Fri: Version lock v0.91.0                           │
  └─────────────────────────────────────────────────────┘
  
  Gate: v0.91.0 locked, DEMA speaks Arabic-first, Ihsān scorer ±0.05
  Risk: LoRA quality insufficient → mitigate: multiple checkpoints

WEEK 5-6 (Apr 7-18): THE GROWTH (Ramadan)
  ┌─────────────────────────────────────────────────────┐
  │ Focus: SDPO closed-loop → expert improvement        │
  │                                                     │
  │ Week 5:                                             │
  │   SDPO training loop for Expert-R (reasoning)       │
  │   Wire Phase 80 LearningLoopOrchestrator to MOE     │
  │   Version lock v0.93.0                              │
  │                                                     │
  │ Week 6:                                             │
  │   Multi-expert SDPO (Expert-T, Expert-V)            │
  │   Cross-expert routing optimization                 │
  │   A2A intra-node protocol (PAT↔SAT messaging)      │
  │   Version lock v0.95.0                              │
  └─────────────────────────────────────────────────────┘
  
  Gate: v0.95.0 locked, SDPO active (flag on), model improves from use
  Risk: SDPO divergence → mitigate: Ihsān gate prevents bad updates

WEEK 7 (Apr 21-25): THE ECOSYSTEM (Ramadan)
  ┌─────────────────────────────────────────────────────┐
  │ Focus: AaaS protocol + Alpha-10 preparation         │
  │                                                     │
  │ Mon-Tue: ServiceContract implementation             │
  │          Discovery + negotiation + settlement        │
  │ Wed: Alpha-10 installer package                     │
  │      Offline USB bundle (2.5GB)                     │
  │      Installation guide (Arabic + English)           │
  │ Thu: Select Alpha-10 users                          │
  │      Prepare support channel                        │
  │ Fri: Version lock v0.98.0                           │
  └─────────────────────────────────────────────────────┘
  
  Gate: v0.98.0 locked, installer tested on 3 devices, Alpha-10 ready

WEEK 8 (Apr 28 - May 4): GENESIS 🌙 (End of Ramadan)
  ┌─────────────────────────────────────────────────────┐
  │ Focus: Genesis Gate + Launch                        │
  │                                                     │
  │ Mon: Genesis Gate — run all 68 checks               │
  │      L1 Sentinel (12) + L2 Oracle (14) + L3 (10)   │
  │      L4 Conductor (13) + L5 Ambassador (19)         │
  │                                                     │
  │ Tue: Fix any gate failures                          │
  │      Re-run Genesis Gate if needed                  │
  │                                                     │
  │ Wed: v1.0.0-GENESIS locked                          │
  │      Upload demo to YouTube                         │
  │      Post on Twitter/LinkedIn/HackerNews            │
  │                                                     │
  │ Thu: Send Alpha-10 invitations                      │
  │      🌙 Eid al-Fitr — celebration                   │
  │                                                     │
  │ Fri: First Alpha-10 nodes come online               │
  │      Monitor, support, celebrate                    │
  └─────────────────────────────────────────────────────┘
  
  Gate: v1.0.0-GENESIS LOCKED. 68/68 checks. Alpha-10 LIVE.
```

---

## §3 QUALITY (PMBOK Knowledge Area 3)

### Quality Gates (Constitutional — No Bypass)

| Gate | When | Criteria | Who Enforces |
|------|------|----------|-------------|
| **Commit Gate** | Every commit | T0 smoke pass + T1 delta pass | CI pipeline |
| **PR Gate** | Every PR to main | T2 contract pass + coverage ≥ floor | CI pipeline |
| **Version Lock** | End of each week | T3 full pass + coverage ratchet + constants hash | bizra_test.py |
| **Genesis Gate** | Week 8 | 68/68 checks by 5 SAT agents | Manual + automated |
| **Daughter Test** | Every release | "Would you deploy for ديما?" | Mumo (oath) |

### Coverage Ratchet

```
v0.80.0:  64.57% (LOCKED — current floor)
v0.85.0:  target 68% (+3.4%)
v0.90.0:  target 72% (+4%)
v0.95.0:  target 76% (+4%)
v1.0.0:   target 80% (+4%)
```

### Defect Classification (Spine §13)

| Severity | Response Time | Example |
|----------|-------------|---------|
| **CONSTITUTIONAL** | Immediate halt | Ihsān threshold changed, Gini violated |
| **CRITICAL** | Same day | Atomic write bug, identity corruption |
| **HIGH** | Within sprint | Rate limiting missing, auth bypass |
| **MEDIUM** | Within 2 sprints | Coverage drop, documentation gap |
| **LOW** | Backlog | UI polish, logging format |

---

## §4 RISK REGISTER (PMBOK Knowledge Area 4)

| ID | Risk | Probability | Impact | Mitigation |
|----|------|------------|--------|-----------|
| R1 | WSL2 filesystem slowness | HIGH | MEDIUM | Work on native Linux filesystem |
| R2 | LoRA fine-tuning quality insufficient | MEDIUM | HIGH | Multiple checkpoints, evaluate each |
| R3 | Ramadan reduced productivity | CERTAIN | MEDIUM | Reduced scope per week, core features only |
| R4 | Model download blocked/slow | LOW | MEDIUM | Pre-download, USB transfer, mirror |
| R5 | Alpha-10 user hardware incompatible | MEDIUM | LOW | Test on 3+ device types before invite |
| R6 | SDPO divergence (model gets worse) | LOW | HIGH | Ihsān gate blocks bad updates, feature flag |
| R7 | Constitutional constant accidentally changed | LOW | CRITICAL | CI gate + hash comparison every commit |
| R8 | Single point of failure (Mumo alone) | HIGH | HIGH | Document everything, spine is canonical reference |

---

## §5 DEVOPS INTEGRATION

### Pipeline Architecture

```
Developer (Mumo on NODE0)
    │
    ├── git push ─────────────► GitHub
    │                              │
    │                              ├── T0 Smoke (30s)
    │                              ├── T1 Delta (2min)
    │                              ├── Lint + Security
    │                              │
    │                              ▼
    │                         PR to main
    │                              │
    │                              ├── T2 Contract (5min)
    │                              ├── Coverage check (≥ floor)
    │                              ├── Constants hash verify
    │                              │
    │                              ▼
    │                         Merge to main
    │                              │
    ├── bizra_test.py --lock ────► Version Lock Receipt
    │                              │
    │                              ├── T3 Full (20-30min)
    │                              ├── Coverage ratchet
    │                              ├── Lock hash chained
    │                              │
    │                              ▼
    │                         Release Candidate
    │                              │
    │                              ├── T4 Genesis Gate (60min)
    │                              ├── 68 checks, 5 SAT agents
    │                              ├── Empirical validation (10/10)
    │                              ├── Lifecycle proof (19/19)
    │                              │
    │                              ▼
    │                         v1.0.0-GENESIS
    │                              │
    └── Alpha-10 invitations ────► LIVE
```

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Source control | GitHub (BizraInfo org) | Code, specs, CI config |
| CI/CD | GitHub Actions | Automated quality gates |
| Test runner | bizra_test.py | Delta testing, version locks |
| Coverage | pytest-cov | Ratcheted floor |
| Security scan | Bandit + Safety | Vulnerability detection |
| Linting | Ruff + MyPy | Code quality + type safety |
| Deployment | Docker + Vercel | Backend + Frontend |
| Monitoring | Evidence Ledger (local) | Constitutional compliance |

---

## §6 METRICS (KPIs)

### Sprint Health

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tests passing | 100% (excluding environmental) | bizra_test.py --status |
| Coverage | ≥ floor (ratchet) | pytest-cov |
| Build time (T1) | < 2 minutes | CI pipeline |
| Deploy frequency | Weekly version locks | Lock receipts |
| Lead time | < 1 week from idea to locked version | Git history |
| MTTR | < 4 hours for CRITICAL | Anomaly log timestamps |

### Constitutional Health

| Metric | Target | Source |
|--------|--------|--------|
| Ihsān composite | ≥ 0.95 | Empirical validation V1-V10 |
| Gini coefficient | ≤ 0.35 | Empirical validation V4 |
| P5 drift attempts blocked | 100% | Empirical validation V10 |
| Evidence chain integrity | 100% | Empirical validation V7 |
| Self-critique detection | ≥ 95% within 3 ticks | Empirical validation V8 |

---

## §7 COMMUNICATION (PMBOK Knowledge Area 7)

### Stakeholder Map

| Stakeholder | Interest | Communication |
|-------------|---------|---------------|
| Mumo (founder) | Everything | Daily: commit logs. Weekly: lock receipts |
| Alpha-10 users | Stability, features | Weekly: changelog + support channel |
| Investors (future) | Progress, metrics | Monthly: proof summary with evidence |
| Community (future) | Vision, contribution | Quarterly: blog posts + demo videos |

### Documentation Deliverables per Sprint

| Week | Document Update |
|------|----------------|
| 1 | Spine v1.1 (add Gini buffer zone from V4) |
| 2 | Living Organism spec (MOE implementation details) |
| 4 | Constitutional Fine-Tuning report (LoRA results) |
| 7 | Alpha-10 Installation Guide (Arabic + English) |
| 8 | Genesis Proof Summary (investor-grade) |

---

## §8 THE FIRST COMMAND

Everything in this blueprint starts with one terminal command on NODE0:

```bash
# Monday, March 10, 2026, 09:00 GST
cd ~/BIZRA-DATA-LAKE  # Linux filesystem, NOT /mnt/c
git push origin main  # 10 commits, 10,105 tests, 64.57% coverage

# Then:
cp /path/to/reflex_compiler.py core/sovereign/reflex_compiler.py
cp /path/to/test_reflex_compiler.py tests/core/sovereign/test_reflex_compiler.py
python bizra_test.py --delta  # Only test changed modules
git add . && git commit -m "feat(sovereign): O(1) reflex compiler — 46 tests, constitutional gates"
git push origin main

# The demo is now filmable:
# Mission 1: 1800ms (System-2)
# Mission 2: 900ms (learning)
# Mission 3: PRECIPITATION
# Mission 4: 50ms (System-1) — THE WOOW MOMENT
```

---

> **"إن الله يحب إذا عمل أحدكم عملاً أن يتقنه"**
> *"God loves that when one does a work, he perfects it."*
>
> The work is defined. The quality gates are automated.
> The constitutional constants are immutable. The proof is executable.
> The sprint starts Monday. Genesis is in 8 weeks.
>
> **v0.80.0 LOCKED → v1.0.0-GENESIS in 8 sprints.**
