# Phase 3: Elevation — Days 11-14

**Ihsan Gate:** Benevolence (Maslahah/Transparency)
**Objective:** Publish the public proof artifact. Declare Node0 Mission OS v1.
Only capabilities that survived Phases 0-2 may be elevated.

## Task 3.1: Evidence Bundle Generation

**Purpose:** Produce a single, downloadable artifact that contains everything
a reviewer needs to independently verify BIZRA's claims.

### Bundle Structure

```
bizra-evidence-bundle-v1.0.0/
├── README.md                           # What this is, how to verify
├── VERIFICATION.md                     # Step-by-step replay instructions
├── TRUTH_LABELS.md                     # Every claim with status tag
├── receipts/
│   ├── receipt_chain.jsonl             # Full immutable receipt chain
│   └── chain_integrity.json            # Hash verification report
├── manifests/
│   ├── daily_manifest_day01.json       # Daily heartbeat manifests
│   ├── daily_manifest_day02.json
│   └── ...
├── probes/
│   ├── probe_negative_path.json        # Individual probe results
│   ├── probe_timeout.json
│   ├── probe_dependency_failure.json
│   ├── probe_replay_divergence.json
│   ├── probe_reflex_provenance.json
│   ├── probe_policy_mismatch.json
│   └── probe_fallback_removal.json
├── benchmarks/
│   ├── membrane_tax.json               # Constitutional overhead measurements
│   ├── canonical_e2e.json              # End-to-end benchmark
│   └── golden_vector.json             # Cross-language sealing proof
├── ci/
│   ├── canonical_validation_gate.log   # Last passing CI run
│   └── golden_vector_ci.log            # Cross-language sealing CI
├── identity/
│   ├── node0_public_key.hex            # Ed25519 public key (NOT private)
│   └── genesis_receipt.json            # First receipt in chain
└── bundle_integrity.json               # BLAKE3 hash of entire bundle
```

### Pseudocode — Bundle Generator

```
FUNCTION generate_evidence_bundle(output_dir):
    bundle = EvidenceBundle(version="1.0.0")

    # 1. Collect receipts
    receipt_chain = read_receipt_chain()
    bundle.add("receipts/receipt_chain.jsonl", receipt_chain.export())
    bundle.add("receipts/chain_integrity.json", receipt_chain.verify_all())

    # 2. Collect manifests
    FOR manifest IN daily_manifests():
        bundle.add(f"manifests/{manifest.filename}", manifest.export())

    # 3. Collect probe results
    FOR probe IN SAPE_PROBES:
        result = probe.last_result()
        bundle.add(f"probes/probe_{probe.name}.json", result.export())

    # 4. Collect benchmarks
    bundle.add("benchmarks/membrane_tax.json", last_benchmark("membrane"))
    bundle.add("benchmarks/canonical_e2e.json", last_benchmark("canonical"))
    bundle.add("benchmarks/golden_vector.json", golden_vector_result())

    # 5. Collect CI logs
    bundle.add("ci/canonical_validation_gate.log", last_ci_log("canonical"))

    # 6. Identity (public only — NEVER include private key)
    bundle.add("identity/node0_public_key.hex", node_public_key())
    bundle.add("identity/genesis_receipt.json", receipt_chain.first())

    # 7. Generate truth label matrix
    truth_matrix = generate_truth_labels()
    bundle.add("TRUTH_LABELS.md", truth_matrix.render_markdown())

    # 8. Generate verification instructions
    bundle.add("VERIFICATION.md", render_verification_guide())
    bundle.add("README.md", render_bundle_readme())

    # 9. Seal the bundle
    bundle_hash = bundle.compute_integrity_hash()
    bundle.add("bundle_integrity.json", {
        "algorithm": "BLAKE3",
        "hash": bundle_hash,
        "files": bundle.file_count(),
        "generated_at": now_iso(),
        "node_id": "node0-genesis",
    })

    bundle.write_to(output_dir)
    RETURN bundle_hash
```

### TDD Anchors

```
TEST bundle_contains_all_required_sections
TEST bundle_integrity_hash_verifiable
TEST bundle_does_not_contain_private_keys
TEST bundle_receipt_chain_is_valid
TEST bundle_truth_labels_match_current_state
```

---

## Task 3.2: Truth Label Matrix

**Purpose:** Every claim in public documentation must carry a truth label.
This is the Ghazali principle: honest labeling is not optional.

### Label Taxonomy

```
[ENFORCEMENT: PROVEN]     — Verified by test, benchmark, or formal proof
[ENFORCEMENT: VALIDATED]  — Verified in controlled EV&V environment
[ENFORCEMENT: WIRED]      — Code exists and compiles, not yet production-proven
[OPTIMIZATION: PARTIAL]   — Partially implemented, known gaps documented
[OPTIMIZATION: PLANNED]   — Designed but not yet coded
```

### Pseudocode — Truth Label Audit

```
FUNCTION audit_truth_labels(repo_root):
    claims = extract_all_claims(repo_root)  # Parse README, docs, etc.
    results = []

    FOR claim IN claims:
        evidence = find_evidence(claim)

        IF evidence.test_passing AND evidence.benchmark_exists:
            label = "PROVEN"
        ELIF evidence.test_passing:
            label = "VALIDATED"
        ELIF evidence.code_exists:
            label = "WIRED"
        ELIF evidence.spec_exists:
            label = "PLANNED"
        ELSE:
            label = "UNSUBSTANTIATED"  # Must be removed or demoted

        results.append(TruthLabel(claim, label, evidence))

    # Generate matrix
    matrix = TruthLabelMatrix(results)
    overclaims = [r for r in results if r.label == "UNSUBSTANTIATED"]

    IF overclaims:
        PRINT f"WARNING: {len(overclaims)} overclaims found — must be corrected"
        FOR oc IN overclaims:
            PRINT f"  - {oc.claim}: no evidence found"

    RETURN matrix
```

---

## Task 3.3: Release Process

### Pseudocode — v1.0.0 Release

```
FUNCTION release_v1():
    # 1. Pre-flight
    ASSERT heartbeat_proof_exists(), "24h heartbeat must complete first"
    ASSERT golden_vector_ci_passes(), "Cross-language sealing must pass"
    ASSERT all_probes_pass(), "All 7 SAPE probes must pass"
    ASSERT truth_labels_clean(), "No unsubstantiated claims"

    # 2. Generate evidence bundle
    bundle_hash = generate_evidence_bundle("./evidence-bundle-v1.0.0/")

    # 3. Update public documentation
    update_readme_with_truth_labels()
    update_metrics_canonical()
    update_giants_md()

    # 4. Create Git tag
    tag_message = f"""
    Node0 Mission OS v1.0.0 — Canonical Proof Artifact

    Evidence bundle hash: {bundle_hash}
    Heartbeat proof: 288/288 ticks, 0 failures
    Constitutional coherence: 1.00
    Canon readiness: 100%

    The evidence era begins.
    """
    git_tag("v1.0.0", tag_message, signed=true)

    # 5. Create GitHub Release
    github_release("v1.0.0", {
        title: "Node0 Mission OS v1.0.0 — Canonical Proof Artifact",
        body: release_notes(),
        assets: [
            "evidence-bundle-v1.0.0.tar.gz",
            "heartbeat_proof.json",
            "bundle_integrity.json",
        ],
    })

    # 6. Deploy proof-of-life dashboard
    deploy_sovereign_dashboard(port=9742)

    PRINT "v1.0.0 released. The world can now verify."
```

---

## Phase 3 Exit Criteria

- [ ] Evidence bundle generated with all required sections
- [ ] Truth label matrix clean (zero unsubstantiated claims)
- [ ] Bundle integrity hash verifiable
- [ ] Public documentation updated with truth labels
- [ ] Git tag v1.0.0 created and pushed
- [ ] GitHub Release created with evidence bundle
- [ ] Sovereign dashboard deployed with proof-of-life
- [ ] Release announcement prepared

## Final Acceptance

The sprint is COMPLETE when a reviewer can:
1. Download the evidence bundle
2. Follow VERIFICATION.md step by step
3. Independently verify every receipt in the chain
4. Confirm all truth labels match the evidence
5. Reproduce the golden vector digest
6. Validate the heartbeat proof

**If any step fails, the release is not ready.**
