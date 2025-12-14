# BIZRA Context Kernel - Quick Reference Card

## 🚀 Session Initialization (Start Every Chat With This)

### Option 1: Single Command (Fastest)
```
@load-kernel
```

### Option 2: Explicit File Load (Most Reliable)
```
Please read and activate the BIZRA Context Kernel:

Priority 1 (MUST LOAD):
C:\BIZRA-PROJECTS\.claude-kernel\core-context.json

Supporting Files (Load as needed):
C:\BIZRA-PROJECTS\.claude-kernel\hooks.yaml
C:\BIZRA-PROJECTS\.claude-kernel\memory.json
C:\BIZRA-PROJECTS\.claude-kernel\tools.yaml
C:\BIZRA-PROJECTS\.claude-kernel\standards.yaml
```

### Option 3: Python Helper Script
```bash
python C:\BIZRA-PROJECTS\.claude-kernel\load-kernel.py
# (Then copy-paste the generated prompt)
```

---

## 📋 Pre-Configured Workflow Hooks

Use these commands to trigger common workflows:

### Development
- `@kernel-fix` - Backend kernel hardening (1,550 LOC Rust)
- `@frontend-mvp` - Frontend MVP development (2,280 LOC React)
- `@integration-test` - End-to-end testing
- `@deploy-staging` - Deploy to staging environment

### Analysis
- `@arch-review` - Architecture deep dive
- `@quality-audit` - Comprehensive Ihsan quality assessment

### Documentation
- `@api-docs` - Generate OpenAPI documentation
- `@readme` - Create comprehensive README

### Research
- `@archaeology` - Mine 15,000 hours of accumulated work
- `@competitive` - Competitive landscape analysis

### Utilities
- `@sprint-report` - Generate sprint progress report
- `@checkpoint` - Create system backup checkpoint
- `@kernel-update` - Update context kernel with latest state

### Composite Workflows
- `@sprint-complete` - Execute full development sprint
- `@release` - Production release pipeline

---

## 🎯 Current Project Status (Updated: 2025-11-18)

**Backend:**
- Status: ✅ Production-ready
- Quality: 97/100
- LOC: 3,500 Rust
- Tests: 156/156 passing
- Unsafe Code: 0

**Frontend:**
- Status: ⚠️ Critical gap (44% health)
- Target: 85%+ in 15 days
- Framework: React + TypeScript + Vite
- Components: 3 → 25 needed

**Active Sprint:**
- Name: Frontend Excellence Sprint
- Duration: 15 days (2025-11-18 to 2025-12-02)
- Goal: Production-ready unified platform

**Velocity Discovery:**
- Assumed: 450 LOC/day
- Actual: ~4,000 LOC/day (8-10x with AI assistance!)

---

## 💡 Operating Modes

- `/A` - Autonomous mode (full tool usage, multi-step)
- `/S` - System check (time, commands, context)
- `/#` - Max resources (all tools, autonomous agents)
- `/R` - Reasoning together (collaborative analysis)
- `/@` - Continue (seamless progression)

---

## 📊 Success Metrics

**Technical:**
- Backend: 97% → 99%+
- Frontend: 44% → 85%+
- Tests: 156 → 200+
- P99 Latency: <250ms

**Product:**
- Onboarding: 80%+ completion
- First Command: 85%+ success
- Day-7 Retention: 65%+
- Satisfaction: 4.5+ stars

---

## 🔧 Key Paths

**Repositories:**
- Synthesis Orchestrator: `C:\BIZRA-PROJECTS\synthesis_orchestrator`
- Genesis Node: `C:\bizra-genesis-node`
- Frontend MVP: `C:\BIZRA-PROJECTS\bizra-frontend-mvp`
- Kernel Config: `C:\BIZRA-PROJECTS\.claude-kernel`

**Environment:**
- OS: Windows 11 Pro
- Hardware: MSI Titan (i9-14900HX, 128GB, RTX 4090)
- Rust: 1.90.0
- Node: 24.5.0
- Python: 3.13.5

---

## 🎖️ Ihsan Standard (95+ Threshold)

**Formula:**
```
ihsan = 0.3*correctness + 0.3*safety + 0.2*efficiency + 0.2*user_benefit
```

**Enforcement:**
- Zero unsafe code (mandatory)
- 80%+ backend test coverage
- 70%+ frontend test coverage
- <250ms P99 latency
- WCAG 2.2 AA accessibility

---

## 🚨 Remember

1. **Always load kernel at session start** (saves 10-15 minutes of context rebuilding)
2. **Use project_knowledge_search first** (before web search)
3. **Chunk large files** (≤30 lines per write for optimal performance)
4. **Quality over velocity** (scoped excellence beats rushed completion)
5. **Update memory.json** after significant milestones

---

## 📞 Quick Commands for Common Tasks

**Check Status:**
```
/S (Show system status)
What's our current sprint progress?
```

**Start Development:**
```
/@kernel-fix (Start backend work)
/@frontend-mvp (Start frontend work)
```

**Quality Check:**
```
/@quality-audit (Run Ihsan assessment)
Run tests and show coverage
```

**Deploy:**
```
/@deploy-staging (Deploy to staging)
/@release (Production release)
```

---

## 🔄 Kernel Update Protocol

**When to Update:**
- Major milestone completion
- Architecture changes
- New tool integrations
- Significant insights or learnings

**How to Update:**
```
/@kernel-update

Or manually edit:
C:\BIZRA-PROJECTS\.claude-kernel\memory.json
C:\BIZRA-PROJECTS\.claude-kernel\core-context.json
```

**Backup:**
- Kernel is Git-tracked
- Auto-saved after each update
- Version history maintained

---

**إن شاء الله - Excellence through systematic preparation!**

---

*Last Updated: 2025-11-18*
*Version: 1.0.0*
