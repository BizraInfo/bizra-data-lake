"""Contract tests for ``tools/audit/claim_drift_probe.py``.

The probe is wired as Gate 6 (canonical-validation-gate.yml). Its behaviour is
load-bearing for claim discipline in CI, so regressions must be caught here
before they reach main.

Coverage:
    * ``_unsuppressed_line_matches`` — per-line semantics of the allow marker.
    * ``scan_file`` — H1 per-line suppression, H4 double-canonical firing rules,
      H0 missing-file handling.
    * ``LogSink`` — no-op when path is None; NDJSON emission otherwise.
    * ``main`` — CI-mode exit codes (pass and fail paths).
    * ``_resolve_log_path`` — resolution policy for CI vs local.

Every test is hermetic: a fresh tmpdir is wired into the probe via the module's
``REPO`` constant so the real repo is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pytest

PROBE_PATH = Path(__file__).resolve().parents[3] / "tools" / "audit" / "claim_drift_probe.py"


def _load_probe():
    """Load the probe module fresh each test so the ``REPO`` patch sticks."""
    spec = importlib.util.spec_from_file_location(
        "_claim_drift_probe_under_test", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yield a probe module whose ``REPO`` points at ``tmp_path``."""
    module = _load_probe()
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "DEFAULT_LOG_PATH", tmp_path / ".cursor" / "log.ndjson")
    return module


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def _findings_for(probe, rel: str) -> list[dict]:
    sink = probe.LogSink(path=None, run_id="test")
    return probe.scan_file(rel, sink)


def _by_id(findings: Iterable[dict], hid: str) -> list[dict]:
    return [f for f in findings if f["hypothesisId"] == hid]


# ---------------------------------------------------------------------------
# _unsuppressed_line_matches — the H4 fix contract
# ---------------------------------------------------------------------------


def test_unsuppressed_line_matches_returns_hit_line_numbers(probe):
    lines = ["no hit", "Node0 proves the seed can live alone", "also no hit"]
    assert probe._unsuppressed_line_matches(probe.H4_A, lines) == [2]


def test_unsuppressed_line_matches_skips_lines_with_allow_marker(probe):
    lines = [
        "no hit",
        "Node0 proves the seed can live alone <!-- claim-probe: allow -->",
        "no hit",
    ]
    assert probe._unsuppressed_line_matches(probe.H4_A, lines) == []


def test_unsuppressed_line_matches_returns_empty_on_no_hits(probe):
    assert probe._unsuppressed_line_matches(probe.H4_A, ["nothing", "matches"]) == []


# ---------------------------------------------------------------------------
# scan_file — H0 / H1 / H4 behaviour
# ---------------------------------------------------------------------------


def test_scan_file_missing_emits_h0(probe, tmp_path: Path):
    findings = _findings_for(probe, "does/not/exist.md")
    assert len(findings) == 1
    assert findings[0]["hypothesisId"] == "H0"
    assert findings[0]["missing"] is True


def test_scan_file_h1_ready_for_production_fires(probe, tmp_path: Path):
    _write(tmp_path, "brief.md", "# Status\n\nWe are READY FOR PRODUCTION today.\n")
    findings = _findings_for(probe, "brief.md")
    h1 = _by_id(findings, "H1")
    assert len(h1) == 1
    assert "READY FOR PRODUCTION" in h1[0]["matched"].upper()
    assert h1[0]["line"] == 3


def test_scan_file_h1_suppressed_by_per_line_allow_marker(probe, tmp_path: Path):
    _write(
        tmp_path,
        "brief.md",
        "# Register\n\n"
        "The phrase 'READY FOR PRODUCTION' is prohibited "
        "<!-- claim-probe: allow -->\n",
    )
    findings = _findings_for(probe, "brief.md")
    assert _by_id(findings, "H1") == []


def test_scan_file_h1_suppression_does_not_leak_across_lines(probe, tmp_path: Path):
    # Marker on line 3 must NOT suppress a hit on line 5 (per-line semantics).
    body = (
        "# Doc\n"
        "\n"
        "Prior register entry <!-- claim-probe: allow -->\n"
        "\n"
        "Today we are READY FOR PRODUCTION.\n"
    )
    _write(tmp_path, "brief.md", body)
    findings = _findings_for(probe, "brief.md")
    h1 = _by_id(findings, "H1")
    assert len(h1) == 1
    assert h1[0]["line"] == 5


def test_scan_file_h4_fires_when_both_canonical_sentences_present(
    probe, tmp_path: Path
):
    body = (
        "# Transition\n"
        "Node0 proves the seed can live alone.\n"
        "Each human node mints PAT-7 inside the URP.\n"
    )
    _write(tmp_path, "doc.md", body)
    findings = _findings_for(probe, "doc.md")
    h4 = _by_id(findings, "H4")
    assert len(h4) == 1
    assert h4[0]["line"] == 2  # first unsuppressed legacy hit


def test_scan_file_h4_fires_single_finding_even_with_multiple_repeats(
    probe, tmp_path: Path
):
    body = (
        "Node0 proves the seed can live alone.\n"
        "Node0 proves the seed can live alone again.\n"
        "Each human node mints PAT-7.\n"
        "Each human node mints PAT-7 twice.\n"
    )
    _write(tmp_path, "doc.md", body)
    findings = _findings_for(probe, "doc.md")
    assert len(_by_id(findings, "H4")) == 1


def test_scan_file_h4_suppressed_when_marker_on_legacy_line(probe, tmp_path: Path):
    body = (
        "Node0 proves the seed can live alone <!-- claim-probe: allow -->\n"
        "Each human node mints PAT-7.\n"
    )
    _write(tmp_path, "doc.md", body)
    assert _by_id(_findings_for(probe, "doc.md"), "H4") == []


def test_scan_file_h4_suppressed_when_marker_on_canonical_line(
    probe, tmp_path: Path
):
    body = (
        "Node0 proves the seed can live alone.\n"
        "Each human node mints PAT-7 <!-- claim-probe: allow -->\n"
    )
    _write(tmp_path, "doc.md", body)
    assert _by_id(_findings_for(probe, "doc.md"), "H4") == []


def test_scan_file_h4_fires_when_marker_is_on_unrelated_line(
    probe, tmp_path: Path
):
    """Regression test for the H4 file-scope suppression bug (commit d93a6525).

    Pre-fix: a single allow marker anywhere in the file suppressed H4.
    Post-fix: suppression is strictly per-line, so this must fire.
    """
    body = (
        "# Header\n"
        "Unrelated paragraph <!-- claim-probe: allow --> discussing registers.\n"
        "\n"
        "Node0 proves the seed can live alone.\n"
        "Each human node mints PAT-7 in the URP.\n"
    )
    _write(tmp_path, "doc.md", body)
    h4 = _by_id(_findings_for(probe, "doc.md"), "H4")
    assert len(h4) == 1, "H4 must fire — marker was on an unrelated line"
    assert h4[0]["line"] == 4


def test_scan_file_h4_requires_both_sentences(probe, tmp_path: Path):
    _write(tmp_path, "doc.md", "Node0 proves the seed can live alone.\n")
    assert _by_id(_findings_for(probe, "doc.md"), "H4") == []


def test_scan_file_h1_allcaps_guarantee_fires(probe, tmp_path: Path):
    _write(tmp_path, "pitch.md", "We offer guaranteed uptime.\n")
    h1 = _by_id(_findings_for(probe, "pitch.md"), "H1")
    assert any("guaranteed" in f["matched"].lower() for f in h1)


def test_scan_file_h1_production_ready_accepts_truth_label(probe, tmp_path: Path):
    # Truth-labelled variants must not trip the H1 heuristic.
    body = "Node0 is production-ready PARTIAL as of 2026-04-26.\n"
    _write(tmp_path, "doc.md", body)
    assert _by_id(_findings_for(probe, "doc.md"), "H1") == []


# ---------------------------------------------------------------------------
# LogSink
# ---------------------------------------------------------------------------


def test_logsink_with_none_path_is_noop(probe):
    sink = probe.LogSink(path=None, run_id="x")
    sink.emit(hypothesisId="H0", location="x", message="y", data={})


def test_logsink_writes_ndjson(probe, tmp_path: Path):
    log = tmp_path / "log.ndjson"
    sink = probe.LogSink(path=log, run_id="abc")
    sink.emit(hypothesisId="H1", location="a:1", message="hit", data={"k": "v"})
    sink.emit(hypothesisId="H2", location="a:2", message="hit2", data={})
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["runId"] == "abc"
    assert first["sessionId"] == probe.SESSION_ID
    assert first["hypothesisId"] == "H1"
    assert first["location"] == "a:1"


# ---------------------------------------------------------------------------
# _resolve_log_path
# ---------------------------------------------------------------------------


def test_resolve_log_path_returns_explicit_arg(probe, tmp_path: Path):
    explicit = tmp_path / "custom.log"
    assert probe._resolve_log_path(explicit, ci_mode=True) == explicit
    assert probe._resolve_log_path(explicit, ci_mode=False) == explicit


def test_resolve_log_path_ci_default_is_none(probe):
    assert probe._resolve_log_path(None, ci_mode=True) is None


def test_resolve_log_path_local_default_is_debug_log(probe):
    assert probe._resolve_log_path(None, ci_mode=False) == probe.DEFAULT_LOG_PATH


# ---------------------------------------------------------------------------
# main — CI exit codes
# ---------------------------------------------------------------------------


def _seed_empty_clean_and_watch_sets(
    probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace CLEAN_SET / WATCH_SET with a single empty file so main() runs clean."""
    clean_rel = "clean.md"
    watch_rel = "watch.md"
    _write(tmp_path, clean_rel, "# Clean\nNothing prohibited here.\n")
    _write(tmp_path, watch_rel, "# Watch\nNo hits either.\n")
    monkeypatch.setattr(probe, "CLEAN_SET", (clean_rel,))
    monkeypatch.setattr(probe, "WATCH_SET", (watch_rel,))


def test_main_ci_mode_passes_when_clean_set_has_no_gating_findings(
    probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    _seed_empty_clean_and_watch_sets(probe, tmp_path, monkeypatch)
    exit_code = probe.main(["--ci", "--run-id", "test-pass"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "verdict=PASS" in captured.out


def test_main_ci_mode_fails_when_clean_set_has_h1_hit(
    probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    clean_rel = "bad.md"
    _write(tmp_path, clean_rel, "# Doc\nWe are READY FOR PRODUCTION.\n")
    monkeypatch.setattr(probe, "CLEAN_SET", (clean_rel,))
    monkeypatch.setattr(probe, "WATCH_SET", ())
    exit_code = probe.main(["--ci", "--run-id", "test-fail"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "verdict=FAIL" in captured.out
    assert "CI gate FAILED" in captured.err


def test_main_non_ci_mode_never_fails_even_with_hits(
    probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clean_rel = "bad.md"
    _write(tmp_path, clean_rel, "READY FOR PRODUCTION.\n")
    monkeypatch.setattr(probe, "CLEAN_SET", (clean_rel,))
    monkeypatch.setattr(probe, "WATCH_SET", ())
    assert probe.main(["--run-id", "test-local"]) == 0


def test_main_writes_summary_json(
    probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_empty_clean_and_watch_sets(probe, tmp_path, monkeypatch)
    summary_path = tmp_path / "summary.json"
    probe.main(
        [
            "--ci",
            "--run-id",
            "test-summary",
            "--summary-json",
            str(summary_path),
        ]
    )
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["ci_mode"] is True
    assert data["clean_set"]["files"] == 1
    assert data["watch_set"]["files"] == 1


# ---------------------------------------------------------------------------
# Sanity: the real CLEAN_SET / WATCH_SET declared in the module
# ---------------------------------------------------------------------------


def test_real_clean_set_is_not_empty_and_disjoint_from_watch_set():
    module = _load_probe()
    assert len(module.CLEAN_SET) >= 3
    assert set(module.CLEAN_SET).isdisjoint(set(module.WATCH_SET))


def test_real_pattern_tables_are_well_formed():
    module = _load_probe()
    for table in (module.H1_PATTERNS, module.H2_PATTERNS, module.H3_PATTERNS, module.H5_PATTERNS):
        assert len(table) >= 1
        for pattern, why in table:
            re.compile(pattern, re.IGNORECASE)  # must compile
            assert isinstance(why, str) and why
