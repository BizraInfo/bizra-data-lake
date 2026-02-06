"""
Tests for DeepSeek R1 think token stripping and response utilities.
===============================================================================

Tests the strip_think_tokens function and related utilities for handling
DeepSeek R1's chain-of-thought tokens.

Created: 2026-02-05 | BIZRA Node0 Genesis
"""

import pytest
from core.inference.response_utils import (
    strip_think_tokens,
    strip_reasoning_tokens,
    strip_all_reasoning_tokens,
    normalize_response,
    extract_think_content,
    has_think_tokens,
)


class TestStripThinkTokens:
    """Tests for strip_think_tokens function."""

    def test_basic_think_block(self):
        """Test stripping a single <think> block."""
        content = "<think>Let me analyze this problem step by step...</think>The answer is 42."
        result = strip_think_tokens(content)
        assert result == "The answer is 42."

    def test_multiple_think_blocks(self):
        """Test stripping multiple <think> blocks."""
        content = (
            "<think>First, let me consider option A...</think>"
            "Option A is valid. "
            "<think>Now let me verify option B...</think>"
            "Option B is also valid. The best choice is A."
        )
        result = strip_think_tokens(content)
        assert result == "Option A is valid. Option B is also valid. The best choice is A."

    def test_multiline_think_block(self):
        """Test stripping <think> block with multiple lines."""
        content = """<think>
Step 1: Parse the input
Step 2: Analyze the data
Step 3: Compute the result
</think>
The computed result is 123."""
        result = strip_think_tokens(content)
        assert result == "The computed result is 123."

    def test_unclosed_think_tag(self):
        """Test handling unclosed <think> tag (truncated output)."""
        content = "Starting analysis... <think>Let me think about this"
        result = strip_think_tokens(content)
        assert result == "Starting analysis..."

    def test_orphaned_closing_tag(self):
        """Test handling orphaned </think> tag."""
        content = "Some text</think> more text"
        result = strip_think_tokens(content)
        assert result == "Some text more text"

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        content = "<THINK>Uppercase thinking</THINK>Result"
        result = strip_think_tokens(content)
        assert result == "Result"

        content2 = "<Think>Mixed case</Think>Answer"
        result2 = strip_think_tokens(content2)
        assert result2 == "Answer"

    def test_think_with_attributes(self):
        """Test <think> tags with attributes."""
        content = '<think reasoning="step1">Analysis here</think>Final answer.'
        result = strip_think_tokens(content)
        assert result == "Final answer."

    def test_empty_content(self):
        """Test empty string input."""
        assert strip_think_tokens("") == ""
        assert strip_think_tokens(None) == ""

    def test_no_think_tokens(self):
        """Test content without think tokens (passthrough)."""
        content = "This is a normal response without any special tokens."
        result = strip_think_tokens(content)
        assert result == content

    def test_only_think_content(self):
        """Test content that is only think tokens."""
        content = "<think>All thinking, no output</think>"
        result = strip_think_tokens(content)
        assert result == ""

    def test_whitespace_cleanup(self):
        """Test that extra whitespace is cleaned up."""
        content = "<think>Thinking...</think>\n\n\n\nResult"
        result = strip_think_tokens(content)
        assert result == "Result"

    def test_nested_angle_brackets(self):
        """Test content with angle brackets inside think block."""
        content = "<think>if x < 5 && y > 3 then compute</think>Value: 10"
        result = strip_think_tokens(content)
        assert result == "Value: 10"

    def test_code_blocks_preserved(self):
        """Test that code blocks outside think are preserved."""
        content = """<think>Analyzing code...</think>
Here is the solution:
```python
def hello():
    print("Hello, World!")
```"""
        result = strip_think_tokens(content)
        assert "def hello():" in result
        assert 'print("Hello, World!")' in result


class TestStripReasoningTokens:
    """Tests for generic strip_reasoning_tokens function."""

    def test_custom_tag_name(self):
        """Test stripping custom tag names."""
        content = "<thinking>Deep thought...</thinking>Conclusion"
        result = strip_reasoning_tokens(content, "thinking")
        assert result == "Conclusion"

    def test_reasoning_tag(self):
        """Test stripping <reasoning> tags."""
        content = "<reasoning>Step by step analysis</reasoning>Final answer: 42"
        result = strip_reasoning_tokens(content, "reasoning")
        assert result == "Final answer: 42"


class TestStripAllReasoningTokens:
    """Tests for strip_all_reasoning_tokens function."""

    def test_multiple_tag_types(self):
        """Test stripping multiple different reasoning tag types."""
        content = (
            "<think>First thought</think>"
            "<thinking>Second thought</thinking>"
            "<reasoning>Third analysis</reasoning>"
            "Final answer."
        )
        result = strip_all_reasoning_tokens(content)
        assert result == "Final answer."


class TestHasThinkTokens:
    """Tests for has_think_tokens function."""

    def test_detects_think_tokens(self):
        """Test detection of think tokens."""
        assert has_think_tokens("<think>content</think>result")
        assert has_think_tokens("<THINK>content</THINK>result")
        assert has_think_tokens("prefix<think>content</think>suffix")

    def test_no_think_tokens(self):
        """Test content without think tokens."""
        assert not has_think_tokens("normal content")
        assert not has_think_tokens("")
        assert not has_think_tokens(None)


class TestExtractThinkContent:
    """Tests for extract_think_content function."""

    def test_extract_single_block(self):
        """Test extracting single think block."""
        content = "<think>My internal reasoning</think>Output"
        result = extract_think_content(content)
        assert result == "My internal reasoning"

    def test_extract_multiple_blocks(self):
        """Test extracting multiple think blocks."""
        content = "<think>First thought</think>Text<think>Second thought</think>"
        result = extract_think_content(content)
        assert "First thought" in result
        assert "Second thought" in result
        assert "---" in result

    def test_no_think_blocks(self):
        """Test content without think blocks."""
        content = "No thinking here"
        result = extract_think_content(content)
        assert result is None


class TestNormalizeResponse:
    """Tests for normalize_response function."""

    def test_strips_think_tokens(self):
        """Test that normalize also strips think tokens."""
        content = "<think>Thinking...</think>Answer"
        result = normalize_response(content)
        assert result == "Answer"

    def test_preserves_formatting(self):
        """Test that formatting is preserved by default."""
        content = "Line 1\n\nLine 2\n\nLine 3"
        result = normalize_response(content, preserve_formatting=True)
        assert "\n\n" in result


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_deepseek_r1_style_output(self):
        """Test handling typical DeepSeek R1 output format."""
        content = """<think>
Let me analyze this mathematical problem step by step.

1. First, I need to identify the equation type
2. This appears to be a quadratic equation: x^2 + 5x + 6 = 0
3. I can factor this as (x + 2)(x + 3) = 0
4. Therefore x = -2 or x = -3

Let me verify:
- For x = -2: (-2)^2 + 5(-2) + 6 = 4 - 10 + 6 = 0 (correct)
- For x = -3: (-3)^2 + 5(-3) + 6 = 9 - 15 + 6 = 0 (correct)
</think>

The solutions to the equation x^2 + 5x + 6 = 0 are:

**x = -2** and **x = -3**

Both solutions can be verified by substitution into the original equation."""

        result = strip_think_tokens(content)

        # Think content should be gone
        assert "<think>" not in result
        assert "Let me analyze" not in result
        assert "step by step" not in result.lower() or "step" not in result[:50]

        # Final answer should remain
        assert "x = -2" in result
        assert "x = -3" in result
        assert "Both solutions can be verified" in result

    def test_preserves_user_facing_content(self):
        """Test that user-facing content is fully preserved."""
        final_answer = """Here is your Python code:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

This recursive implementation has O(2^n) time complexity."""

        content = f"<think>Analyzing the request for Fibonacci...</think>{final_answer}"
        result = strip_think_tokens(content)

        assert result == final_answer.strip()
        assert "def fibonacci(n):" in result
        assert "O(2^n)" in result
