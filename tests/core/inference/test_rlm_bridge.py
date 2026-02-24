from __future__ import annotations

import pytest

from core.inference.rlm_bridge import BizraRLMBridge, RLMResult


@pytest.mark.asyncio
async def test_execute_rlm_returns_final_answer() -> None:
    async def model(_: str) -> str:
        return 'FINAL_ANSWER = "done"'

    bridge = BizraRLMBridge(max_iterations=5)
    result = await bridge.execute_rlm(
        prompt="summarize",
        task="Return one word",
        agent_model=model,
    )

    assert isinstance(result, RLMResult)
    assert result.success is True
    assert result.final_answer == "done"
    assert result.halted_reason == "final_answer"


@pytest.mark.asyncio
async def test_execute_rlm_stops_on_iteration_limit() -> None:
    async def model(_: str) -> str:
        return "x = 1"

    bridge = BizraRLMBridge(max_iterations=3)
    result = await bridge.execute_rlm(
        prompt="never finishes",
        task="loop",
        agent_model=model,
    )

    assert result.success is False
    assert result.iterations == 3
    assert result.halted_reason == "max_iterations"


@pytest.mark.asyncio
async def test_execute_rlm_handles_markdown_code_fences() -> None:
    async def model(_: str) -> str:
        return "```python\nFINAL_ANSWER = 'ok'\n```"

    bridge = BizraRLMBridge(max_iterations=2)
    result = await bridge.execute_rlm(
        prompt="fenced",
        task="extract",
        agent_model=model,
    )

    assert result.final_answer == "ok"
    assert result.success is True


@pytest.mark.asyncio
async def test_execute_rlm_sub_call_budget_halts() -> None:
    async def model(_: str) -> str:
        return "a = lm_query('x')\nb = lm_query('y')\nFINAL_ANSWER = b"

    bridge = BizraRLMBridge(max_iterations=2, max_sub_calls=1)
    result = await bridge.execute_rlm(
        prompt="budget",
        task="use sub-calls",
        agent_model=model,
        sub_model=lambda q: q + "!",
    )

    assert result.sub_calls == 1
    assert result.final_answer in {"[MAX_SUB_CALLS_REACHED]", "[SUB_MODEL_UNSET]", "[ASYNC_SUB_MODEL_UNSUPPORTED]", "y!"}


@pytest.mark.asyncio
async def test_execute_rlm_empty_prompt_fails_closed() -> None:
    bridge = BizraRLMBridge()
    result = await bridge.execute_rlm(
        prompt="   ",
        task="anything",
        agent_model=lambda _: "FINAL_ANSWER = 'x'",
    )

    assert result.success is False
    assert result.halted_reason == "empty_prompt"


@pytest.mark.asyncio
async def test_execute_rlm_uses_raw_llm_call_for_string_model() -> None:
    calls = []

    async def raw(model: str, prompt: str) -> str:
        calls.append((model, prompt))
        return 'FINAL_ANSWER = "via_raw"'

    bridge = BizraRLMBridge(raw_llm_call=raw, max_iterations=2)
    result = await bridge.execute_rlm(
        prompt="prompt",
        task="task",
        agent_model="test-model",
    )

    assert result.final_answer == "via_raw"
    assert calls


@pytest.mark.asyncio
async def test_execute_rlm_empty_model_response_halts() -> None:
    bridge = BizraRLMBridge(max_iterations=2)
    result = await bridge.execute_rlm(
        prompt="prompt",
        task="task",
        agent_model=lambda _: "",
    )

    assert result.success is False
    assert result.halted_reason == "empty_model_response"
