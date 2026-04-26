"""Tests for the pattern registry (YAML loader + query helpers)."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.execution_flywheel.pattern_registry import (
    get_pattern,
    list_patterns,
    load_patterns,
    parse_minimal_yaml,
    query_by_trigger,
)
from tools.execution_flywheel.schemas import Pattern


VALID_YAML = textwrap.dedent(
    """
    version: "0.1"
    patterns:
      - pattern_id: TEST_PATTERN
        name: Test pattern
        severity: minor
        triggers:
          - keyword: some_trigger
            description: A test trigger
        risks:
          - a risk
        guard_actions:
          - do nothing
        source:
          - unit test
    """
).strip()


class MinimalYamlTests(unittest.TestCase):
    def test_parses_simple_mapping(self) -> None:
        parsed = parse_minimal_yaml('version: "0.1"\nfoo: bar\n')
        self.assertEqual(parsed, {"version": "0.1", "foo": "bar"})

    def test_parses_block_list(self) -> None:
        parsed = parse_minimal_yaml(
            textwrap.dedent(
                """
                items:
                  - one
                  - two
                """
            ).strip()
        )
        self.assertEqual(parsed, {"items": ["one", "two"]})

    def test_parses_list_of_mappings(self) -> None:
        parsed = parse_minimal_yaml(
            textwrap.dedent(
                """
                triggers:
                  - keyword: a
                    description: first
                  - keyword: b
                    description: second
                """
            ).strip()
        )
        self.assertEqual(
            parsed,
            {
                "triggers": [
                    {"keyword": "a", "description": "first"},
                    {"keyword": "b", "description": "second"},
                ]
            },
        )

    def test_quoted_values_preserve_specials(self) -> None:
        parsed = parse_minimal_yaml(
            textwrap.dedent(
                """
                items:
                  - "PR #49"
                  - "inspect origin/<branch>:<file>"
                """
            ).strip()
        )
        self.assertEqual(parsed["items"], ["PR #49", "inspect origin/<branch>:<file>"])


class PatternLoaderTests(unittest.TestCase):
    def _write(self, text: str) -> str:
        fd = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        fd.write(text)
        fd.close()
        return fd.name

    def test_loads_valid_pattern(self) -> None:
        path = self._write(VALID_YAML)
        patterns = load_patterns(path)
        self.assertEqual(len(patterns), 1)
        p = patterns[0]
        self.assertEqual(p.pattern_id, "TEST_PATTERN")
        self.assertEqual(p.severity, "minor")
        self.assertEqual(p.default_decision, "")
        self.assertEqual(len(p.triggers), 1)
        self.assertEqual(p.triggers[0].keyword, "some_trigger")

    def test_missing_required_field_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Pattern.from_dict({"name": "x", "severity": "minor"})
        with self.assertRaises(ValueError):
            Pattern.from_dict({"pattern_id": "x", "severity": "minor"})
        with self.assertRaises(ValueError):
            Pattern.from_dict({"pattern_id": "x", "name": "y"})

    def test_invalid_severity_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Pattern.from_dict({"pattern_id": "x", "name": "y", "severity": "catastrophic"})

    def test_invalid_default_decision_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Pattern.from_dict(
                {
                    "pattern_id": "x",
                    "name": "y",
                    "severity": "minor",
                    "default_decision": "NUKE_FROM_ORBIT",
                }
            )

    def test_default_decision_accepted(self) -> None:
        p = Pattern.from_dict(
            {
                "pattern_id": "x",
                "name": "y",
                "severity": "critical",
                "default_decision": "ABORT",
            }
        )
        self.assertEqual(p.default_decision, "ABORT")

    def test_list_and_query_helpers(self) -> None:
        patterns = [
            Pattern.from_dict(
                {
                    "pattern_id": "A",
                    "name": "a",
                    "severity": "minor",
                    "triggers": [{"keyword": "foo"}],
                }
            ),
            Pattern.from_dict(
                {
                    "pattern_id": "B",
                    "name": "b",
                    "severity": "critical",
                    "triggers": [{"keyword": "bar"}, {"keyword": "foo"}],
                }
            ),
        ]
        self.assertEqual(list_patterns(patterns), ["A", "B"])
        self.assertEqual(get_pattern(patterns, "B").pattern_id, "B")
        self.assertIsNone(get_pattern(patterns, "C"))
        hits = query_by_trigger(patterns, "foo")
        self.assertEqual([p.pattern_id for p in hits], ["A", "B"])
        hits = query_by_trigger(patterns, "BAR")
        self.assertEqual([p.pattern_id for p in hits], ["B"])
        self.assertEqual(query_by_trigger(patterns, "nope"), [])

    def test_packaged_patterns_registry_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        patterns = load_patterns(root / "patterns.yaml")
        self.assertGreaterEqual(len(patterns), 5)
        ids = list_patterns(patterns)
        for required in (
            "PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT",
            "AUDIT_YAML_INLINE_COMMENT_PARSE_FAILURE",
            "SECRET_SCANNER_SNR_NOISE_COLLAPSE",
            "DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT",
            "BOTTLENECK_SHIFT_AFTER_SECRET_GATE_CLEARS",
        ):
            self.assertIn(required, ids)
        cred = get_pattern(patterns, "DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT")
        self.assertIsNotNone(cred)
        self.assertEqual(cred.default_decision, "ABORT")


if __name__ == "__main__":
    unittest.main()
