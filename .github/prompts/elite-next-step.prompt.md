---
name: "Elite Next Step"
description: "Choose and execute the single highest-leverage next implementation slice for BIZRA using production-grade full-stack, DevOps, CI/CD, performance, and QA standards."
argument-hint: "Optional target area or constraint, for example: frontend route wiring, CI/CD hardening, API integration, performance budget, testing gates"
agent: "agent"
---

Advance BIZRA by delivering exactly one concrete, highest-leverage next implementation slice.

Use these repository anchors as primary context before choosing the slice:
- [Phase 72 constitutional kernel](../../docs/specs/phase_72_constitutional_kernel)
- [Phase 73 frontend routes](../../docs/specs/phase_73_frontend_routes)
- [Phase 74 frontend final selection](../../docs/specs/phase_74_frontend_final_selection.md)
- [Phase 75 live domains consolidation](../../docs/specs/phase_75_live_domains_consolidation.md)
- [Frontend workspace](../../frontend)
- [Project guidance](../../CLAUDE.md)

Interpret any user argument as a priority hint, not a license to broaden scope. If no argument is given, determine the next slice yourself.

Task:
1. Inspect the current codebase, active specs, and existing changes.
2. Select exactly one next implementation slice that most improves delivery readiness, product coherence, or production reliability.
3. Prefer work that closes a real gap between specification and implementation, especially in one of these areas:
   - typed API wiring between backend and frontend
   - CI/CD and pipeline automation
   - deploy or environment hardening
   - performance budgets, observability, or regression gates
   - test coverage and quality enforcement
   - route-level implementation needed to unblock a user-facing surface
4. Explain the chosen slice in 2-4 sentences before editing anything.
5. Implement the slice end-to-end with minimal, production-quality changes.
6. Validate the result with the strongest relevant checks you can run.

Required standards:
- Deliver one slice, not a roadmap dump.
- Fix root causes, not cosmetic symptoms.
- Preserve the repository's existing architecture and constitutional constraints.
- Treat DevOps, CI/CD, testing, and performance as first-class engineering requirements, not optional polish.
- Do not invent backend contracts or UI data shapes when repository specs already define them.
- Keep secrets out of code and use environment-driven configuration.
- Prefer incremental, mergeable work with clear acceptance criteria.

Required output format:
- Chosen slice: one short paragraph explaining why this was selected now.
- Implementation: concise summary of what changed.
- Validation: commands run, checks passed, and checks not run.
- Remaining risk: the main residual gap after this slice.

If the requested area is too large, narrow it to the smallest high-impact mergeable unit and state that choice explicitly.