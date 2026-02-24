# Phase 50.1 — RLM Core Algorithm: REPL, Recursion, State

> Standing on Giants: Zhang, Kraska & Khattab (RLM, 2026) · Turing (universal computation, 1936) · McCarthy (LISP/REPL, 1960) · Knuth (structured programming, 1974)

## 1. Algorithm 1 — Formal Definition

From the paper (arXiv:2512.24601v2, §2):

```
Algorithm 1: A recursive language model, around LLM M
Input:  prompt P
Output: response Y

state  ← InitREPL(prompt=P)
state  ← AddFunction(state, sub_RLM_M)
hist   ← [Metadata(state)]

while True do
    code          ← LLM_M(hist)
    (state, stdout) ← REPL(state, code)
    hist          ← hist ∥ code ∥ Metadata(stdout)
    if state[Final] is set then
        return state[Final]
```

### Key Properties

| Property | Description | Complexity |
|----------|-------------|------------|
| **Unbounded input** | P stored as REPL variable, never loaded into M's context | O(1) root context for P |
| **Unbounded output** | Final answer written to `state[Final]` variable | No autoregressive length limit |
| **Symbolic recursion** | `sub_RLM_M` callable from within REPL code | Ω(|P|) semantic work possible |
| **Constant metadata** | Only `len(P)`, short prefix, type metadata in hist | O(K) root context usage |

### Three Design Choices Missing from Prior Scaffolds

1. **Symbolic handle to prompt** — The model receives a variable name, not the text. It writes `prompt[:100]` to peek, `prompt.split("Chapter")` to decompose.

2. **Symbolic recursion** — Code running inside the REPL can invoke `sub_RLM(sub_prompt)`, launching a new LLM call on a slice of the prompt. This is NOT pre-planned delegation — the model decides what to recurse on at runtime.

3. **Unbounded output via state variables** — `state[Final] = accumulated_result` bypasses the autoregressive output cap entirely.

## 2. BIZRA RLM Pseudocode

Adapted for BIZRA's PAT agent pipeline with sovereign constraints:

```python
class BizraRLM:
    """
    Recursive Language Model adapted for BIZRA sovereign cognition.

    Standing on Giants:
    - Zhang et al. (2026): RLM algorithm
    - McCarthy (1960): REPL as computation model
    - Al-Ghazali (1095): Ihsan gate on recursive depth
    """

    def __init__(self, llm_backend, memory_store, ihsan_threshold=0.95):
        self.llm = llm_backend          # LM Studio endpoint
        self.memory = memory_store       # LivingMemory 5-layer store
        self.ihsan_threshold = ihsan_threshold
        self.max_recursion_depth = 3     # Constitutional bound
        self.max_iterations = 20         # Prevent runaway loops

    async def execute(self, prompt: str, task_context: dict) -> RLMResult:
        """
        Execute RLM loop on a prompt.

        The prompt is loaded as an external variable in a sandboxed
        REPL environment. The LLM generates code to probe, decompose,
        and recursively process the prompt.
        """
        # Phase 1: Initialize REPL with prompt as variable
        state = self._init_repl(prompt, task_context)

        # Phase 2: Add sub-RLM function for recursive calls
        state = self._add_sub_rlm(state)

        # Phase 3: Seed history with metadata only (NOT the prompt text)
        hist = [self._extract_metadata(state)]

        # Phase 4: Iterative REPL loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # LLM generates code to probe/process the prompt
            code = await self.llm.generate(
                history=hist,
                system="You are operating inside a Python REPL. "
                       "The user's prompt is loaded as variable `prompt`. "
                       "Write code to examine, decompose, and process it. "
                       "Store your final answer in FINAL_ANSWER. "
                       "You can call lm_query(sub_prompt) for sub-tasks.",
                max_tokens=1200,
            )

            # Execute code in sandboxed REPL
            state, stdout = self._execute_in_repl(state, code)

            # Append code + metadata to history (NOT raw stdout)
            hist.append(code)
            hist.append(self._metadata_of(stdout))

            # Check termination
            if "FINAL_ANSWER" in state.variables:
                answer = state.variables["FINAL_ANSWER"]

                # Ihsan gate on final answer
                if self._passes_ihsan(answer, task_context):
                    return RLMResult(
                        answer=answer,
                        iterations=iteration,
                        state=state,
                        trace=hist,
                    )
                else:
                    # Answer failed Ihsan — allow one more iteration
                    del state.variables["FINAL_ANSWER"]
                    hist.append("[IHSAN_GATE: Answer below threshold, refine.]")

        # Max iterations reached — return best partial
        return RLMResult(
            answer=state.variables.get("FINAL_ANSWER", ""),
            iterations=iteration,
            state=state,
            trace=hist,
            partial=True,
        )

    def _init_repl(self, prompt: str, context: dict) -> REPLState:
        """
        Initialize sandboxed REPL with prompt as variable.

        The prompt is NOT copied into the LLM context.
        Only metadata (length, type, short prefix) is visible.
        """
        state = REPLState()
        state.variables["prompt"] = prompt
        state.variables["prompt_length"] = len(prompt)
        state.variables["prompt_preview"] = prompt[:200]
        state.variables["context"] = context
        return state

    def _add_sub_rlm(self, state: REPLState) -> REPLState:
        """
        Register sub-RLM callable for recursive invocation.

        The sub-RLM function allows code inside the REPL to
        launch new LLM calls on slices of the prompt.
        """
        async def lm_query(sub_prompt: str, depth: int = 0) -> str:
            if depth >= self.max_recursion_depth:
                return "[MAX_RECURSION_DEPTH_REACHED]"

            response = await self.llm.generate(
                history=[{"role": "user", "content": sub_prompt}],
                max_tokens=600,
            )
            return response

        state.functions["lm_query"] = lm_query
        return state

    def _extract_metadata(self, state: REPLState) -> str:
        """
        Extract constant-size metadata about the prompt.

        This is what the root LLM sees — NOT the raw text.
        Analogous to the 'emissions' in an HHMM.
        """
        return (
            f"[REPL Environment]\n"
            f"Variable `prompt`: {state.variables['prompt_length']} chars\n"
            f"Preview: {state.variables['prompt_preview']}\n"
            f"Available functions: print(), lm_query(), len(), "
            f"prompt.split(), prompt.find(), re.search()\n"
            f"Store final answer in: FINAL_ANSWER"
        )
```

## 3. REPL Sandbox Specification

The REPL must be **sandboxed** — it runs code generated by the LLM.

### Allowed Operations
- String operations: `prompt[:n]`, `prompt.split()`, `prompt.find()`, `prompt.count()`
- Regex: `re.search()`, `re.findall()`, `re.sub()`
- Math: arithmetic, `len()`, `min()`, `max()`, `sum()`
- Collections: list/dict creation, iteration, comprehensions
- Sub-RLM calls: `lm_query(sub_prompt)`
- Variable assignment: `state["key"] = value`
- Print for debugging: `print(expr)`

### Blocked Operations (Constitutional Constraint)
- File I/O: `open()`, `os.`, `pathlib.`
- Network: `requests.`, `urllib.`, `socket.`
- System: `subprocess.`, `os.system()`, `exec()`, `eval()` on external input
- Import of non-whitelisted modules

### Execution Timeout
- Per-step: 10 seconds
- Total REPL session: 120 seconds
- Sub-RLM call: 30 seconds per call, 120 seconds total

## 4. Memory Integration

The RLM REPL state integrates with Living Memory:

```
REPL State Variable          →  Living Memory Layer
─────────────────────────────────────────────────────
Intermediate results         →  WORKING (fast, 5-access promotion)
Verified sub-answers         →  EPISODIC (medium, 10-access promotion)
Learned probe strategies     →  PROCEDURAL (slow, glacial promotion)
Final answers                →  SEMANTIC (permanent)
Planned future probes        →  PROSPECTIVE (goal-directed)
```

Each REPL iteration that produces a useful intermediate result triggers a memory ENCODE operation. The HHMM promotion chain then naturally consolidates frequently-used probe strategies into procedural memory, making the agent faster on similar tasks.

## 5. Cost Model

From the paper's Observation 4:

| Metric | Base LLM | RLM | Ratio |
|--------|----------|-----|-------|
| Median cost | $0.13/query | $0.11/query | 0.85x (cheaper) |
| 95th percentile | $0.16/query | $0.90/query | 5.6x (variance) |
| Context scaling | Degrades linearly | Maintains performance | N/A |

For BIZRA local inference (LM Studio), the cost model translates to:
- **Token budget per RLM session**: 5,000 tokens root + N × 600 tokens sub-calls
- **Typical session**: 3-5 iterations, 1-2 sub-calls each = ~8,000 tokens total
- **Maximum session**: 20 iterations, 60 sub-calls = ~41,000 tokens
