# BIZRA 7B Reasoner + 7B Planner — Expected Outcomes vs Similar‑Size Market Models (v1.0)

This is a **forecast** (not a claim) for what you should expect after completing a high‑quality data refinery + instruction synthesis + LoRA/QLoRA fine‑tuning cycle on your 3‑year BIZRA corpus.

## Baseline comparators (same weight class)
- General 7B–8B instruct models (e.g., modern Llama‑8B/Qwen‑7B/Mistral‑7B class)
- Planning‑oriented 7B variants (AgentFlow‑style, tool‑planner fine‑tunes)

## What you can realistically outperform
### 1) BIZRA‑specific correctness
- **Fewer invented module names / fake endpoints** (because your corpus contains the real ones).
- **Better adherence to your invariants** (SAT veto, URP leases, receipts, sealing).

### 2) Tool‑calling reliability
- Higher rate of producing valid JSON plans and respecting allowlists/timeouts.
- Lower “tool thrash” (calling tools that are blocked or irrelevant).

### 3) Planning throughput (human time)
- Fewer iterations to get from intent → plan → execution steps.
- Stronger separation of *plan* vs *execution* vs *audit*.

## What you will NOT magically outperform
- **World knowledge** breadth versus larger models.
- **Long‑horizon reasoning** compared to much larger reasoning‑tuned models.
- **Vision / multimodal** abilities unless you train/attach them explicitly.

## Concrete KPI targets (measure, don’t guess)
Use your own eval harness; these are good targets:
- **Plan validity:** +15–30% higher than base model (strict schema pass rate).
- **Tool success rate:** +10–25% improvement (successful calls / attempted).
- **Hallucinated interface rate:** 2–4× reduction (made‑up file/function/API claims).
- **Refusal quality:** +20–40% improvement (correctly refusing unsafe/unknown).
- **Latency:** unchanged or slightly higher (LoRA often similar); optimize via quant + residency.

## Specialization expectations
### bizra-planner-7b
Best‑in‑class (for its size) on:
- JSON step plans, dependency ordering, acceptance criteria, and “stop conditions”.
- Breaking big goals into sealable commits/ADRs/tests.

Risk:
- Over‑planning. Mitigate with a “minimum viable plan” style dataset and penalties.

### bizra-reasoner-7b
Best‑in‑class (for its size) on:
- BIZRA architecture reasoning, threat modeling, tradeoff handling, and invariant checks.

Risk:
- Becoming too “inside baseball” and less helpful on generic tasks. Mitigate with a mixed diet.

## The real moat
Your moat is not the 7B weights.
Your moat is:
1) **The refinery** (dedup, provenance, quality)
2) **The eval arena** (negative tests + regressions)
3) **The governance loop** (SAT/FATE/receipts)

If you nail those, a 7B can feel “bigger” than many 13B models *inside your domain*.
