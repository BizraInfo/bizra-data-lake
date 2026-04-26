"""Omnidirectional Hyper-dimensional Audit Engine — orchestrator.

Usage:
  python -m tools.audit.omni_audit.run_audit --repo-root . --out-dir docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts

Read-only. Never writes to source files. Writes only under --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

# ensure package-style import works when invoked via `-m` or as a file
_PKG_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from tools.audit.omni_audit.schemas import (
    EvidenceItem, Claim, Finding, Risk, Mitigation, Kpi, Gate
)
from tools.audit.omni_audit import (
    evidence_index, claim_scanner, dependency_inventory,
    secret_pattern_scanner, code_risk_scanner, website_claim_capture,
    snr_classifier, hhmm_taxonomy, graph_export, urp_canonicality,
)


def _load_yaml_config(path: Path) -> dict:
    """Tiny YAML subset parser — stdlib only, handles nested dict/list + strings."""
    text = path.read_text(encoding="utf-8")
    def _strip_inline_comment(val: str) -> str:
        """Remove YAML comments outside simple quoted strings."""
        in_quote = False
        quote = ""
        for i, ch in enumerate(val):
            if ch in ("'", '"'):
                if not in_quote:
                    in_quote = True
                    quote = ch
                elif quote == ch:
                    in_quote = False
                    quote = ""
            elif ch == "#" and not in_quote:
                return val[:i].rstrip()
        return val

    def _coerce(val: str):
        v = _strip_inline_comment(val).strip()
        if v.startswith('"') and v.endswith('"'):
            return v.strip('"')
        if v.startswith("'") and v.endswith("'"):
            return v.strip("'")
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), stripped))

    root: dict = {}
    stack: list[tuple[int, object]] = [(-1, root)]

    for idx, (indent, line) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        # list item
        if line.startswith("- "):
            if isinstance(parent, list):
                parent.append(_coerce(line[2:]))
            continue

        k, _, v = line.partition(":")
        key = k.strip()
        value = v.strip()
        if not isinstance(parent, dict):
            continue

        if value == "":
            next_container: object = {}
            if idx + 1 < len(lines):
                next_indent, next_line = lines[idx + 1]
                if next_indent > indent and next_line.startswith("- "):
                    next_container = []
            parent[key] = next_container
            stack.append((indent, next_container))
        else:
            parent[key] = _coerce(value)

    return root


def _derive_findings(evidence: List[EvidenceItem], claims: List[Claim],
                     secrets: List[dict], code_risks: List[dict],
                     deps: dict, captures: List[dict],
                     urp_observations: List[dict] | None = None) -> List[Finding]:
    """Cross-cut the raw artifacts into structured Finding objects."""
    findings: List[Finding] = []
    seq = 1

    def _add(domain, subsystem, summary, evidence_paths, severity="MEDIUM",
             confidence=0.75, signal_score=0.7, noise_score=0.3,
             actionable=True, next_action=""):
        nonlocal seq
        findings.append(Finding(
            finding_id=f"F{seq:04d}",
            domain=domain, subsystem=subsystem, summary=summary,
            evidence_paths=evidence_paths, severity=severity,
            confidence=confidence, signal_score=signal_score,
            noise_score=noise_score, actionable=actionable,
            next_action=next_action,
        ))
        seq += 1

    # Secrets
    if secrets:
        _add("SECURITY", "secrets",
             f"Secret-pattern scanner surfaced {len(secrets)} candidate match(es). "
             "Review each for false positives; any real key must be rotated.",
             evidence_paths=[], severity="HIGH" if len(secrets) else "LOW",
             signal_score=0.85 if secrets else 0.15,
             next_action="Triage each match; rotate any real credential; add allow-listed paths.",
             confidence=0.7)
    else:
        _add("SECURITY", "secrets",
             "No matches from secret-pattern scanner in configured scan roots.",
             evidence_paths=[], severity="LOW",
             signal_score=0.2, noise_score=0.4,
             next_action="Expand scan roots / add CI gate for ongoing coverage.")

    # Dep gaps
    for gap in deps.get("gaps", []):
        _add("DEPENDENCY", "lockfiles", gap,
             evidence_paths=[], severity="MEDIUM",
             signal_score=0.75,
             next_action="Add/pin lockfile; establish SBOM generation in CI.")

    # Code risk aggregations
    rule_counts: dict = {}
    for r in code_risks:
        rule_counts[r["rule"]] = rule_counts.get(r["rule"], 0) + 1
    # Surface high-signal rules.
    highlights = [
        ("RS_UNWRAP", "Rust .unwrap() usage — failure modes become panics. Review hot paths."),
        ("RS_UNSAFE_BLOCK", "Rust unsafe { } blocks — document invariants + add tests."),
        ("PY_SHELL_TRUE", "Python subprocess shell=True — command-injection surface."),
        ("PY_BROAD_EXCEPT", "Python broad `except Exception` — may mask errors."),
        ("RS_TODO", "Rust TODO/FIXME markers — tech-debt signal."),
        ("PY_TODO", "Python TODO/FIXME markers — tech-debt signal."),
    ]
    for rule, summary in highlights:
        n = rule_counts.get(rule, 0)
        if n == 0:
            continue
        sev = "MEDIUM" if rule in ("RS_UNSAFE_BLOCK", "PY_SHELL_TRUE") else "LOW"
        sig = 0.7 if rule in ("RS_UNSAFE_BLOCK", "PY_SHELL_TRUE") else 0.45
        _add("CODE_QUALITY", rule, f"{summary} ({n} occurrences)",
             evidence_paths=[], severity=sev, signal_score=sig,
             next_action="Triage; establish rule-level budgets; raise severity on hot paths.")

    # Claims (prohibited + needs_rewrite are signal-high)
    cls_counts: dict = {}
    for c in claims:
        cls_counts[c.classification] = cls_counts.get(c.classification, 0) + 1
    if cls_counts.get("PROHIBITED", 0):
        _add("PUBLIC_CLAIMS", "prohibited",
             f"{cls_counts['PROHIBITED']} PROHIBITED-class claim patterns matched across scanned docs.",
             evidence_paths=[c.source for c in claims if c.classification == "PROHIBITED"][:5],
             severity="HIGH", signal_score=0.9,
             next_action="Rewrite or remove each PROHIBITED match before any public reuse.")
    if cls_counts.get("NEEDS_REWRITE", 0):
        _add("PUBLIC_CLAIMS", "needs_rewrite",
             f"{cls_counts['NEEDS_REWRITE']} NEEDS_REWRITE claim patterns matched (exact numbers / brittle / cost).",
             evidence_paths=[c.source for c in claims if c.classification == "NEEDS_REWRITE"][:5],
             severity="HIGH", signal_score=0.88,
             next_action="Remove from hero copy; move to under-the-hood page with receipts.")
    if cls_counts.get("PROOF_REQUIRED", 0):
        _add("PUBLIC_CLAIMS", "proof_required",
             f"{cls_counts['PROOF_REQUIRED']} PROOF_REQUIRED claim patterns — need public receipt chain.",
             evidence_paths=[c.source for c in claims if c.classification == "PROOF_REQUIRED"][:5],
             severity="MEDIUM", signal_score=0.75,
             next_action="Publish a receipt per claim OR soften to directional wording.")

    # Website captures
    for cap in captures:
        if not cap.get("fetch_ok") and "pre-check" in (cap.get("source") or ""):
            _add("PUBLIC_CLAIMS", "website_rendering",
                 f"Website {cap['url']} is client-side-rendered; non-JS fetchers see "
                 f"only shell. Social/link previews may be degraded.",
                 evidence_paths=[], severity="MEDIUM", signal_score=0.8,
                 next_action="Add OG meta tags in shell HTML; consider SSR/prerender "
                             "for link-preview surfaces.")
        if cap.get("url") == "https://bizra.info" and cap.get("redirected"):
            _add("PUBLIC_CLAIMS", "redirects",
                 "bizra.info 302 → bizra.ai confirmed; no split claim surface.",
                 evidence_paths=[], severity="LOW", signal_score=0.6,
                 noise_score=0.3, actionable=False,
                 next_action="None — keep as brand-defense redirect.")

    # Architecture / doctrine signals from evidence
    doc_count = sum(1 for e in evidence if e.evidence_class == "DOCTRINE")
    if doc_count > 0:
        _add("DOCUMENTATION", "doctrine_surface_area",
             f"{doc_count} doctrine-class documents present (manifestos, canonical docs, READMEs).",
             evidence_paths=[], severity="LOW", signal_score=0.55,
             next_action="Index and deduplicate; canon-store ingestion gate is the "
                         "single forward path for doctrine → runtime.")
    if any(e.path.endswith("CLAUDE.md") for e in evidence):
        _add("DOCUMENTATION", "agent_instructions",
             "Top-level CLAUDE.md present — stable high-level agent contract.",
             evidence_paths=["CLAUDE.md"], severity="LOW", signal_score=0.6,
             next_action="Review quarterly; keep in sync with module decomposition.")
    if any("canon_packs/README.md" in e.path for e in evidence):
        _add("ARCHITECTURE", "canon_separation",
             "Cognitive Foundry canon-packs sit outside runtime canon; "
             "Canon Store Ingestion Gate is explicitly required before runtime contact.",
             evidence_paths=["tools/cognitive_foundry/claude_lane/canon_packs/README.md"],
             severity="LOW", signal_score=0.85,
             next_action="Design Canon Store Ingestion Gate spec before any ingest.")

    # Canonical acronym drift
    urp_observations = urp_observations or []
    urp_alternates = [
        item for item in urp_observations
        if item.get("classification") == "ALTERNATE"
    ]
    if urp_alternates:
        expansions = sorted({item["expansion"] for item in urp_alternates})
        _add("DOCUMENTATION", "urp_canonicality",
             "URP expands to alternate meanings across documentation: "
             + ", ".join(expansions) + ".",
             evidence_paths=[item["path"] for item in urp_alternates[:5]],
             severity="MEDIUM", confidence=0.86, signal_score=0.82,
             next_action="Lock a canonical URP definition and rewrite or alias historical expansions.")

    return findings


def _derive_kpis(claims: List[Claim]) -> List[Kpi]:
    kpis = [
        Kpi(kpi_id="K01", label="Ihsan threshold", target=">=0.95",
            classification="TARGET", source="core/integration/constants.py"),
        Kpi(kpi_id="K02", label="ADL Gini hard gate", target="<=0.35",
            classification="TARGET", source="core/integration/constants.py"),
        Kpi(kpi_id="K03", label="CI coverage floor", target=">=65%",
            classification="TARGET", source="pyproject.toml"),
        Kpi(kpi_id="K04", label="Public cost-per-action", target="directional only",
            classification="UNVERIFIED", source="bizra.ai (pre-check)"),
        Kpi(kpi_id="K05", label="Public SNR", target="directional only",
            classification="UNVERIFIED", source="bizra.ai (pre-check)"),
        Kpi(kpi_id="K06", label="Test-count claim", target="receipt-backed link",
            classification="UNVERIFIED", source="bizra.ai (pre-check)"),
    ]
    return kpis


def _derive_gates() -> List[Gate]:
    return [
        Gate(gate_id="G_A1", tier="A", label="Node0 identity sealed (genesis)",
             status="PASS",
             evidence_path="docs/canon/NODE0_DEFINITION_OF_DONE.md (if present) "
                           "or memory anchor.",
             next_action="Keep sealed; do not alter genesis."),
        Gate(gate_id="G_A2", tier="A", label="Canonical receipt emits on every effect",
             status="PASS",
             evidence_path="bizra-omega/bizra-core/src/canonical_receipt.rs",
             next_action="Periodic contract tests."),
        Gate(gate_id="G_B1", tier="B", label="Reflex persistence across restart",
             status="PASS",
             evidence_path="bizra-omega/bizra-agent/src/persistence.rs",
             next_action="None."),
        Gate(gate_id="G_B2", tier="B", label="FATE gate chain wired end-to-end",
             status="PASS",
             evidence_path="Node0 closure row 4 replay test",
             next_action="None."),
        Gate(gate_id="G_C1", tier="C", label="Dema face reads authoritative chain head",
             status="PASS",
             evidence_path="services/node_gateway/ + trust-surface",
             next_action="None."),
        Gate(gate_id="G_C2", tier="C", label="Canon-store ingestion gate spec exists",
             status="FAIL",
             evidence_path="(not created yet)",
             next_action="Spec-first design; no code until authorized."),
        Gate(gate_id="G_D1", tier="D", label="Public site claim discipline in place",
             status="FAIL",
             evidence_path="docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md",
             next_action="Remove or receipt-ify C4/C5/C7/C9 on bizra.ai."),
        Gate(gate_id="G_D2", tier="D", label="Privacy policy published",
             status="NOT_TESTED",
             evidence_path="(unknown — check bizra.ai sub-pages)",
             next_action="Publish or disable 'no telemetry' / 'no cloud' claims."),
        Gate(gate_id="G_D3", tier="D", label="Headless-Chromium DOM capture of live site",
             status="NOT_TESTED",
             evidence_path="(not performed)",
             next_action="Capture + version for future audits."),
        Gate(gate_id="G_E1", tier="E", label="Genesis-100 cohort scaling path defined",
             status="BLOCKED",
             evidence_path="(no GTM doc set on disk)",
             next_action="Author Node0→100 activation plan."),
    ]


def _derive_risks(findings: List[Finding]) -> tuple:
    """Derive a small canonical risk set linked to findings."""
    risks: List[Risk] = []
    mitigations: List[Mitigation] = []
    rseq = 1
    mseq = 1

    def _add_risk(finding_id, desc, impact, likelihood, mitigation_desc, effort):
        nonlocal rseq, mseq
        mid = f"M{mseq:03d}"; mseq += 1
        mitigations.append(Mitigation(mitigation_id=mid, description=mitigation_desc, effort=effort))
        risks.append(Risk(risk_id=f"X{rseq:03d}", finding_id=finding_id,
                          description=desc, impact=impact, likelihood=likelihood,
                          mitigation_ids=[mid]))
        rseq += 1

    for f in findings:
        if f.subsystem == "needs_rewrite" or f.subsystem == "prohibited":
            _add_risk(f.finding_id, "Public overclaim → ad-platform rejection / regulator risk",
                      "HIGH", "MEDIUM",
                      "Remove or receipt-ify C-class claims before paid launch.", "S")
        if f.subsystem == "secrets" and f.severity == "HIGH":
            _add_risk(f.finding_id, "Leaked credential in repo → account compromise",
                      "CRITICAL", "MEDIUM",
                      "Rotate, remove, add pre-commit scan.", "M")
        if f.subsystem == "lockfiles":
            _add_risk(f.finding_id, "Non-reproducible build / supply-chain drift",
                      "MEDIUM", "HIGH",
                      "Pin lockfiles + publish SBOM in CI.", "M")
        if f.subsystem == "website_rendering":
            _add_risk(f.finding_id, "SPA shell returned by default — weak SEO / degraded link previews",
                      "LOW", "HIGH",
                      "Add OG tags in shell / SSR hero.", "S")
        if f.subsystem == "canon_separation":
            _add_risk(f.finding_id, "Accidental canon-pack ingestion without gate",
                      "HIGH", "LOW",
                      "Lock ingestion behind human-gated tool; no auto-ingest.", "M")
        if f.subsystem == "urp_canonicality":
            _add_risk(f.finding_id, "URP acronym drift weakens architecture and public-claim truth",
                      "MEDIUM", "MEDIUM",
                      "Publish canonical URP definition and redirect legacy aliases.", "S")

    return risks, mitigations


def main(argv=None):
    parser = argparse.ArgumentParser(prog="omni_audit")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--website", action="append", default=[])
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--config",
                        default=str(Path(__file__).parent / "audit_config.yaml"))
    args = parser.parse_args(argv)

    t0 = time.time()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    if cfg_path.suffix == ".json":
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = _load_yaml_config(cfg_path)
    scope = cfg.get("scope", {})
    runtime_cfg = cfg.get("runtime", {})
    output_cfg = cfg.get("output", {})

    no_network = args.no_network or runtime_cfg.get("default_no_network", True)
    max_bytes = runtime_cfg.get("max_file_bytes_scanned", 524288)

    # 1. Evidence index
    evidence = evidence_index.build_evidence_index(
        repo_root=repo_root,
        include_suffixes=scope.get("evidence_include_suffixes", [".md"]),
        include_basenames=scope.get("evidence_include_basenames", []),
        exclude_dirs=scope.get("exclude_dirs", []),
        limit=output_cfg.get("evidence_index_limit", 2000),
    )
    evidence_index.write_outputs(evidence, out_dir)

    # 2. Website capture (before claim scan so claims can see captures)
    urls = args.website if args.website else [
        scope.get("website", {}).get("bizra_ai", "https://bizra.ai"),
        scope.get("website", {}).get("bizra_info", "https://bizra.info"),
    ]
    captures = website_claim_capture.capture(urls, no_network=no_network, out_dir=out_dir)

    # 3. Claim scan (over docs + website captures)
    claims = claim_scanner.scan_claims(
        repo_root=repo_root,
        claim_scan_roots=scope.get("claim_scan_roots", []),
        exclude_dirs=scope.get("exclude_dirs", []),
        website_captures=captures,
        limit=output_cfg.get("claims_register_limit", 500),
    )
    claim_scanner.write_outputs(claims, out_dir)

    # 4. Dependency inventory
    deps = dependency_inventory.inventory(repo_root)
    dependency_inventory.write_outputs(deps, out_dir)

    # 5. Secret-pattern scanner
    secrets = secret_pattern_scanner.scan(
        repo_root=repo_root,
        roots=scope.get("secret_scan_roots", []),
        top_level_globs=scope.get("secret_scan_top_level_globs", []),
        max_bytes=max_bytes,
        limit=output_cfg.get("secret_findings_limit", 200),
    )
    secret_pattern_scanner.write_outputs(secrets, out_dir)

    # 6. Code-risk scanner
    code_risks = code_risk_scanner.scan(
        repo_root=repo_root,
        python_roots=scope.get("python_risk_roots", []),
        rust_roots=scope.get("rust_risk_roots", []),
        max_bytes=max_bytes,
        limit=output_cfg.get("code_risk_limit", 1000),
    )
    code_risk_scanner.write_outputs(code_risks, out_dir)

    # 7. Canonical acronym drift
    urp_observations = urp_canonicality.scan(
        repo_root=repo_root,
        roots=scope.get("urp_canonicality_roots", scope.get("claim_scan_roots", [])),
        exclude_dirs=scope.get("exclude_dirs", []),
        max_bytes=max_bytes,
        limit=output_cfg.get("urp_canonicality_limit", 200),
    )
    urp_canonicality.write_outputs(urp_observations, out_dir)

    # 8. Derive structured findings
    findings = _derive_findings(
        evidence, claims, secrets, code_risks, deps, captures, urp_observations
    )

    # 9. SNR classification
    snr = snr_classifier.classify(findings)
    snr_classifier.write_outputs(snr, out_dir)

    # 10. HHMM taxonomy
    tree = hhmm_taxonomy.build(findings)
    hhmm_taxonomy.write_outputs(tree, out_dir)

    # 11. KPIs, gates, risks, mitigations + graph
    kpis = _derive_kpis(claims)
    gates = _derive_gates()
    risks, mitigations = _derive_risks(findings)
    graph = graph_export.build(evidence, claims, findings, risks, mitigations, kpis, gates)
    graph_export.write_outputs(graph, out_dir)

    # 12. Summary
    summary = {
        "duration_seconds": round(time.time() - t0, 2),
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "no_network": no_network,
        "counts": {
            "evidence": len(evidence),
            "claims": len(claims),
            "findings": len(findings),
            "secrets": len(secrets),
            "code_risks": len(code_risks),
            "urp_canonicality": len(urp_observations),
            "kpis": len(kpis),
            "gates": len(gates),
            "risks": len(risks),
            "mitigations": len(mitigations),
            "graph_nodes": len(graph["nodes"]),
            "graph_edges": len(graph["edges"]),
        },
        "snr_counts": snr["counts"],
        "hhmm_counts_by_domain": tree["counts_by_domain"],
        "dep_gaps": deps.get("gaps", []),
        "urp_alternate_expansions": sorted({
            item["expansion"] for item in urp_observations
            if item.get("classification") == "ALTERNATE"
        }),
        "website_captures": [
            {"url": c.get("url"), "final_url": c.get("final_url"),
             "status": c.get("status"), "fetch_ok": c.get("fetch_ok"),
             "redirected": c.get("redirected")}
            for c in captures
        ],
    }
    summary_path = out_dir / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also emit a small findings register for report-writing convenience.
    findings_path = out_dir / "findings.json"
    with findings_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(fd) for fd in findings], f, indent=2, ensure_ascii=False)

    # Gates + KPIs + Risks registers.
    with (out_dir / "gates.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(g) for g in gates], f, indent=2, ensure_ascii=False)
    with (out_dir / "kpis.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(k) for k in kpis], f, indent=2, ensure_ascii=False)
    with (out_dir / "risks.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in risks], f, indent=2, ensure_ascii=False)
    with (out_dir / "mitigations.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in mitigations], f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
