# Phase 5 — Report Generation and Pytest Integration

> Standing on Giants: Tufte (visual information display, 1983) ·
> Shannon (information theory, 1948)

## Overview

Two output paths:

1. **Report generation** — JSON (always) + HTML (optional) from HarnessResult
2. **Pytest integration** — `@pytest.mark.harness` marker + `conftest.py`
   fixtures for running scenarios as tests

## Part A: Report Generation

### File: `core/harness/report.py`

```pseudocode
IMPORTS:
    import json
    from pathlib import Path
    from core.harness.types import HarnessResult, Verdict, PillarName

# ── JSON Report ────────────────────────────────────────────────────

FUNCTION generate_json_report(result: HarnessResult, output_dir: Path) -> Path:
    """Write a JSON report to disk.

    File: {output_dir}/harness_{run_id}.json
    Size target: < 100 KB for typical runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"harness_{result.run_id}.json"

    report = {
        "meta": {
            "generator": "bizra-harness",
            "version": "1.0.0",
        },
        "summary": result.to_dict(),
        "pillars": {},
        "guardrails": [],
        "regression": None,
        "bench": None,
    }

    # Pillar detail
    FOR pillar_name, pillar_result IN result.pillars.items():
        report["pillars"][pillar_name.value] = {
            "passed":      pillar_result.passed,
            "duration_ms": round(pillar_result.duration_ms, 1),
            "score":       pillar_result.score,
            "error":       pillar_result.error,
            "details":     _safe_serialize(pillar_result.details),
        }

    # Guardrail detail
    FOR gr IN result.guardrail_results:
        report["guardrails"].append({
            "name":    gr.name if hasattr(gr, 'name') else str(gr),
            "passed":  gr.passed if hasattr(gr, 'passed') else True,
        })

    # Regression detail
    IF result.regression:
        rr = result.regression
        report["regression"] = {
            "baseline_run_id":  rr.baseline_run_id,
            "snr_delta":        round(rr.snr_delta, 4),
            "ihsan_delta":      round(rr.ihsan_delta, 4),
            "is_regression":    rr.is_regression,
            "regressed_pillars": [p.value for p in rr.regressed_pillars],
        }

    # Bench detail
    IF result.bench_result:
        report["bench"] = result.bench_result.to_dict()

    path.write_text(json.dumps(report, indent=2, default=str))
    RETURN path


FUNCTION _safe_serialize(obj: Any) -> Any:
    """Recursively sanitize for JSON (no secrets, no huge blobs)."""
    IF isinstance(obj, dict):
        RETURN {
            k: _safe_serialize(v) for k, v in obj.items()
            IF NOT k.startswith("_")        # skip private
            AND NOT any(s in k.lower() for s in ("token", "key", "secret", "password"))
        }
    IF isinstance(obj, (list, tuple)):
        RETURN [_safe_serialize(v) for v in obj[:100]]   # cap list length
    IF isinstance(obj, (str, int, float, bool, type(None))):
        IF isinstance(obj, str) AND len(obj) > 2000:
            RETURN obj[:2000] + "...[truncated]"
        RETURN obj
    RETURN str(obj)    # fallback: stringify unknown types


# ── HTML Report (Optional) ─────────────────────────────────────────

FUNCTION generate_html_report(result: HarnessResult, output_dir: Path) -> Path:
    """Generate a self-contained HTML report.

    No external dependencies — inline CSS, no JS frameworks.
    Uses semantic HTML5 with a pillar status grid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"harness_{result.run_id}.html"

    verdict_color = {
        Verdict.PASS: "#22c55e",          # green
        Verdict.FAIL: "#ef4444",          # red
        Verdict.INCONCLUSIVE: "#f59e0b",  # amber
    }

    pillar_rows = ""
    FOR pname, presult IN sorted(result.pillars.items(), key=lambda x: x[0].value):
        status = "PASS" if presult.passed else "FAIL"
        color = "#22c55e" if presult.passed else "#ef4444"
        score_str = f"{presult.score:.3f}" if presult.score is not None else "N/A"
        error_str = presult.error or ""
        pillar_rows += f"""
        <tr>
            <td>{pname.value}</td>
            <td style="color:{color};font-weight:bold">{status}</td>
            <td>{score_str}</td>
            <td>{presult.duration_ms:.0f}ms</td>
            <td class="error">{error_str}</td>
        </tr>"""

    regression_section = ""
    IF result.regression:
        rr = result.regression
        reg_color = "#ef4444" if rr.is_regression else "#22c55e"
        regression_section = f"""
        <section>
            <h2>Regression Analysis</h2>
            <p>Baseline: <code>{rr.baseline_run_id}</code></p>
            <p>SNR delta: <span style="color:{reg_color}">
                {rr.snr_delta:+.4f}</span></p>
            <p>Ihsan delta: <span style="color:{reg_color}">
                {rr.ihsan_delta:+.4f}</span></p>
            <p>Regression: <strong style="color:{reg_color}">
                {"YES" if rr.is_regression else "NO"}</strong></p>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BIZRA Harness Report — {result.run_id[:8]}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a2e; }}
  h1 {{ border-bottom: 3px solid {verdict_color[result.verdict]}; }}
  .verdict {{ font-size: 2rem; font-weight: bold;
              color: {verdict_color[result.verdict]}; }}
  .meta {{ display: grid; grid-template-columns: repeat(3, 1fr);
           gap: 1rem; margin: 1rem 0; }}
  .meta-card {{ background: #f8f9fa; border-radius: 8px; padding: 1rem;
                text-align: center; }}
  .meta-card .value {{ font-size: 1.5rem; font-weight: bold; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; }}
  .error {{ color: #ef4444; font-size: 0.85rem; max-width: 300px;
            overflow: hidden; text-overflow: ellipsis; }}
  footer {{ margin-top: 2rem; color: #94a3b8; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>BIZRA Harness Report</h1>
<p class="verdict">{result.verdict.value.upper()}</p>
<p>Run: <code>{result.run_id}</code> |
   {result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")} |
   Mode: {result.config.mode.value} |
   Duration: {result.total_duration_ms:.0f}ms</p>

<div class="meta">
  <div class="meta-card">
    <div>SNR Score</div>
    <div class="value">{result.snr_score:.4f}</div>
    <div>Floor: {result.config.snr_floor}</div>
  </div>
  <div class="meta-card">
    <div>Ihsan Score</div>
    <div class="value">{result.ihsan_score:.4f}</div>
    <div>Floor: {result.config.ihsan_floor}</div>
  </div>
  <div class="meta-card">
    <div>Tier</div>
    <div class="value">{result.tier}</div>
    <div>{result.passed_count}/{len(result.pillars)} pillars passed</div>
  </div>
</div>

<section>
<h2>Pillar Results</h2>
<table>
  <thead><tr>
    <th>Pillar</th><th>Status</th><th>Score</th><th>Duration</th><th>Error</th>
  </tr></thead>
  <tbody>{pillar_rows}</tbody>
</table>
</section>

{regression_section}

<section>
<h2>Claim</h2>
<blockquote>{result.config.claim}</blockquote>
</section>

<footer>
Generated by BIZRA Harness v1.0 |
Standing on Giants: Shannon, Deming, Lamport, Al-Ghazali
</footer>
</body>
</html>"""

    path.write_text(html)
    RETURN path
```

## Part B: Pytest Integration

### Marker and Conftest

```pseudocode
# tests/conftest.py — add harness marker

# In existing pytest marker registration:
pytest.ini_options.markers.append(
    "harness: BIZRA unified harness tests (deselect with '-m \"not harness\"')"
)

# tests/core/harness/conftest.py

IMPORTS:
    import pytest
    from core.harness.types import (
        HarnessConfig, HarnessResult, Verdict, PillarName, PillarResult, RunMode,
    )
    from core.harness.runner import HarnessRunner
    from core.harness.scenarios import ScenarioLibrary, BUILTIN_SCENARIOS
    from core.harness.persistence import BaselineStore

@pytest.fixture
FUNCTION harness_runner():
    """Provide a HarnessRunner with default settings."""
    RETURN HarnessRunner()

@pytest.fixture
FUNCTION scenario_library():
    """Provide the default scenario library."""
    RETURN ScenarioLibrary.default()

@pytest.fixture
FUNCTION baseline_store(tmp_path):
    """Provide a temp-dir BaselineStore for test isolation."""
    RETURN BaselineStore(base_dir=tmp_path / ".spearpoint")

@pytest.fixture
FUNCTION harness_config():
    """Default harness config factory."""
    FUNCTION _factory(**overrides) -> HarnessConfig:
        defaults = {
            "mode": RunMode.SMOKE,       # fast for unit tests
            "emit_receipt": False,        # skip evidence in unit tests
            "compare_baseline": False,    # skip regression in unit tests
        }
        defaults.update(overrides)
        RETURN HarnessConfig(**defaults)
    RETURN _factory

# Convenience fixture for making minimal results in tests
@pytest.fixture
FUNCTION make_result():
    FUNCTION _factory(
        run_id="test-run",
        verdict=Verdict.PASS,
        snr=0.92,
        ihsan=0.96,
        **kwargs
    ) -> HarnessResult:
        pillars = {
            PillarName.RUNTIME_BOOT: PillarResult(
                pillar=PillarName.RUNTIME_BOOT,
                passed=True, duration_ms=10.0,
            ),
            PillarName.SNR_CHECK: PillarResult(
                pillar=PillarName.SNR_CHECK,
                passed=True, duration_ms=5.0, score=snr,
            ),
        }
        RETURN HarnessResult(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            config=HarnessConfig(),
            verdict=verdict,
            snr_score=snr,
            ihsan_score=ihsan,
            tier="OPERATIONAL",
            pillars=pillars,
            total_duration_ms=15.0,
            guardrail_results=[],
            all_gates_passed=True,
            **kwargs,
        )
    RETURN _factory
```

### Parametrized Scenario Tests

```pseudocode
# tests/core/harness/test_scenarios_parametric.py

IMPORTS:
    import pytest
    from core.harness.scenarios import BUILTIN_SCENARIOS
    from core.harness.runner import HarnessRunner
    from core.harness.types import HarnessConfig, RunMode, Verdict

# Generate test IDs from scenario IDs
SCENARIO_IDS = list(BUILTIN_SCENARIOS.keys())
SCENARIO_LIST = list(BUILTIN_SCENARIOS.values())

@pytest.mark.harness
@pytest.mark.parametrize("scenario", SCENARIO_LIST, ids=SCENARIO_IDS)
@pytest.mark.asyncio
async def test_scenario_executes_without_crash(scenario, harness_config):
    """Every built-in scenario should run without raising."""
    config = harness_config(
        scenario_id=scenario.id,
        mode=RunMode.SMOKE,         # force smoke for speed
        compare_baseline=False,
        emit_receipt=False,
    )
    runner = HarnessRunner()
    result = await runner.run(config)
    # We don't assert PASS — some scenarios might legitimately FAIL
    # We assert the harness itself doesn't crash
    assert result.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.INCONCLUSIVE)
    assert result.total_duration_ms > 0

@pytest.mark.harness
@pytest.mark.asyncio
async def test_basic_claim_passes_in_healthy_system(harness_config):
    """The basic_claim scenario should PASS in a healthy environment."""
    config = harness_config(
        scenario_id="basic_claim",
        compare_baseline=False,
        emit_receipt=False,
    )
    runner = HarnessRunner()
    result = await runner.run(config)
    assert result.verdict == Verdict.PASS
```

### CLI Integration

```pseudocode
# scripts/run_harness.py (or as part of bizra CLI)

"""
Usage:
    python -m core.harness                       # default: standard mode
    python -m core.harness --mode smoke          # fast check
    python -m core.harness --scenario basic_claim
    python -m core.harness --update-baseline     # seal this run
    python -m core.harness --report html         # generate HTML report
    pytest -m harness                            # run as pytest suite
"""

IMPORTS:
    import argparse
    from core.harness.runner import run_harness
    from core.harness.types import HarnessConfig, RunMode
    from core.harness.report import generate_json_report, generate_html_report

FUNCTION main():
    parser = argparse.ArgumentParser(description="BIZRA Unified Harness")
    parser.add_argument("--mode", choices=["smoke", "standard", "full", "benchmark"],
                        default="standard")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--claim", type=str, default=None)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--report", choices=["json", "html", "both"], default="json")
    parser.add_argument("--output-dir", type=str, default=".spearpoint/reports")
    args = parser.parse_args()

    config = HarnessConfig(
        mode=RunMode(args.mode),
        scenario_id=args.scenario,
        claim=args.claim or "System meets constitutional thresholds",
        update_baseline=args.update_baseline,
        output_dir=Path(args.output_dir),
    )

    result = run_harness(config)

    # Print summary
    print(f"\nVerdict: {result.verdict.value.upper()}")
    print(f"SNR: {result.snr_score:.4f}  Ihsan: {result.ihsan_score:.4f}")
    print(f"Pillars: {result.passed_count}/{len(result.pillars)} passed")
    print(f"Duration: {result.total_duration_ms:.0f}ms")

    # Generate reports
    output_dir = Path(args.output_dir)
    IF args.report in ("json", "both"):
        path = generate_json_report(result, output_dir)
        print(f"JSON report: {path}")
    IF args.report in ("html", "both"):
        path = generate_html_report(result, output_dir)
        print(f"HTML report: {path}")

    # Exit code: 0 for PASS, 1 for FAIL, 2 for INCONCLUSIVE
    MATCH result.verdict:
        CASE Verdict.PASS:         sys.exit(0)
        CASE Verdict.FAIL:         sys.exit(1)
        CASE Verdict.INCONCLUSIVE: sys.exit(2)

IF __name__ == "__main__":
    main()
```

## TDD Anchors

```python
# test_report.py — Phase 5 validation

def test_json_report_creates_file(tmp_path, make_result):
    result = make_result()
    path = generate_json_report(result, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["summary"]["verdict"] == "pass"
    assert "pillars" in data

def test_json_report_under_100kb(tmp_path, make_result):
    result = make_result()
    path = generate_json_report(result, tmp_path)
    assert path.stat().st_size < 100_000

def test_html_report_creates_file(tmp_path, make_result):
    result = make_result()
    path = generate_html_report(result, tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "<html" in content
    assert result.verdict.value.upper() in content

def test_html_report_contains_pillar_table(tmp_path, make_result):
    result = make_result()
    path = generate_html_report(result, tmp_path)
    content = path.read_text()
    for pillar_name in result.pillars:
        assert pillar_name.value in content

def test_safe_serialize_strips_secrets():
    d = {"api_key": "secret123", "name": "test", "token": "abc"}
    safe = _safe_serialize(d)
    assert "api_key" not in safe
    assert "token" not in safe
    assert safe["name"] == "test"

def test_safe_serialize_truncates_long_strings():
    d = {"content": "x" * 5000}
    safe = _safe_serialize(d)
    assert len(safe["content"]) <= 2020   # 2000 + "...[truncated]"

def test_safe_serialize_caps_list_length():
    d = {"items": list(range(200))}
    safe = _safe_serialize(d)
    assert len(safe["items"]) == 100
```
