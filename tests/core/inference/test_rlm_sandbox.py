from __future__ import annotations

from core.inference.rlm_bridge import REPLState, RLMSandbox, should_use_rlm


def test_repl_state_defaults() -> None:
    state = REPLState()
    assert state.variables == {}
    assert state.stdout == []
    assert state.iteration == 0
    assert state.sub_calls == 0


def test_validate_code_accepts_safe_assignment() -> None:
    sandbox = RLMSandbox()
    allowed, reason = sandbox.validate_code("x = 1 + 2")
    assert allowed is True
    assert reason == "ok"


def test_validate_code_blocks_imports() -> None:
    sandbox = RLMSandbox()
    allowed, reason = sandbox.validate_code("import os")
    assert allowed is False
    assert "blocked" in reason.lower()


def test_validate_code_blocks_eval_and_open() -> None:
    sandbox = RLMSandbox()
    allowed_eval, reason_eval = sandbox.validate_code('eval("1+1")')
    allowed_open, reason_open = sandbox.validate_code('f = open("x")')
    assert allowed_eval is False
    assert allowed_open is False
    assert "blocked" in reason_eval.lower()
    assert "blocked" in reason_open.lower()


def test_execute_persists_state() -> None:
    state = REPLState(variables={"seed": 3})
    sandbox = RLMSandbox(state)

    state, _ = sandbox.execute("x = seed * 2")
    state, _ = sandbox.execute("y = x + 1")

    assert state.variables["x"] == 6
    assert state.variables["y"] == 7
    assert state.iteration == 2


def test_execute_captures_stdout() -> None:
    sandbox = RLMSandbox(REPLState())
    state, output = sandbox.execute('print("hello")')
    assert "hello" in output
    assert state.stdout[-1] == output


def test_execute_blocks_dangerous_code() -> None:
    sandbox = RLMSandbox(REPLState())
    state, output = sandbox.execute("import pathlib")
    assert "SANDBOX_BLOCKED" in output
    assert state.iteration == 1


def test_execute_supports_regex_and_math() -> None:
    state = REPLState(variables={"text": "a1 b2"})
    sandbox = RLMSandbox(state)
    state, _ = sandbox.execute('nums = re.findall(r"\\d", text); root = math.sqrt(16)')
    assert state.variables["nums"] == ["1", "2"]
    assert state.variables["root"] == 4.0


def test_execute_lm_query_with_limit() -> None:
    calls: list[str] = []

    def lm_query(prompt: str) -> str:
        calls.append(prompt)
        return prompt.upper()

    sandbox = RLMSandbox(REPLState(), lm_query_fn=lm_query, max_sub_calls=2)
    state, _ = sandbox.execute(
        "a = lm_query('one')\n"
        "b = lm_query('two')\n"
        "c = lm_query('three')"
    )

    assert state.sub_calls == 2
    assert state.variables["a"] == "ONE"
    assert state.variables["b"] == "TWO"
    assert state.variables["c"] == "[MAX_SUB_CALLS_REACHED]"
    assert calls == ["one", "two"]


def test_should_use_rlm_false_for_coordinator_executor() -> None:
    assert should_use_rlm("coordinator", 99999, 1.0) is False
    assert should_use_rlm("executor", 99999, 1.0) is False


def test_should_use_rlm_true_for_large_prompt() -> None:
    assert should_use_rlm("researcher", 32001, 0.2) is True


def test_should_use_rlm_true_for_complex_mid_prompt() -> None:
    assert should_use_rlm("analyst", 9000, 0.8) is True


def test_should_use_rlm_false_for_small_simple_prompt() -> None:
    assert should_use_rlm("creator", 2000, 0.3) is False
