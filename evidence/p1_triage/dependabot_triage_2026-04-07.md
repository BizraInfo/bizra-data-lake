# Dependabot Triage — 2026-04-09 (Day 3)

## Summary

9 open alerts triaged → 8 dismissed (phantom), 1 remaining (known P1).

## Methodology

1. `cargo audit` on active workspace (`bizra-omega/`): **0 vulnerabilities**
2. Cross-referenced each alert's `manifest_path` against repo
3. Discovered 8/9 alerts target `runtime/Cargo.lock` or `filedfs/package.json` — neither path exists in repo
4. Active workspace has all patched versions

## Dismissed (Phantom Manifests)

| Alert | Package | Severity | Manifest | Our Version | Fix Version | Verdict |
|-------|---------|----------|----------|-------------|-------------|---------|
| #21 | pyo3 | low | runtime/Cargo.lock | 0.24.2 (bizra-omega) | 0.24.1 | PATCHED — phantom manifest |
| #22 | bytes | medium | runtime/Cargo.lock | 1.11.1 (bizra-omega) | 1.11.1 | PATCHED — phantom manifest |
| #23 | time | medium | runtime/Cargo.lock | 0.3.47 (bizra-omega) | 0.3.47 | PATCHED — phantom manifest |
| #24 | tar | medium | runtime/Cargo.lock | NOT IN TREE | 0.4.45 | NOT A DEPENDENCY |
| #25 | tar | medium | runtime/Cargo.lock | NOT IN TREE | 0.4.45 | NOT A DEPENDENCY |
| #26 | rustls-webpki | medium | runtime/Cargo.lock | 0.103.10 (bizra-omega) | 0.103.10 | PATCHED — phantom manifest |
| #27 | pyo3 | low | runtime/crates/bizra_bridge/Cargo.toml | 0.24.2 (bizra-omega) | 0.24.1 | PATCHED — phantom manifest |
| #28 | vite | medium | filedfs/package.json | N/A | 6.4.2 | PHANTOM — filedfs/ does not exist |

All dismissed via GitHub API with reason `inaccurate` and triage comment.

## Remaining Open

| Alert | Package | Severity | Manifest | Issue |
|-------|---------|----------|----------|-------|
| #30 | vite | medium | frontend/package-lock.json | vitest 2.x bundles vite@5.4.21 (transitively) — below CVE range but Dependabot flags it. Requires vitest 2→4 upgrade (breaking). |

**Classification:** P1-FRONTEND-TOOLCHAIN-UPGRADE (same as Day 2 assessment)

## cargo audit Output

```
Scanning Cargo.lock for vulnerabilities (594 crate dependencies)
warning: 1 allowed warning found
  - paste 1.0.15 (unmaintained, via pqcrypto-mldsa — no alternative for post-quantum)
```

**0 vulnerabilities. 0 high. 0 medium. 0 low.**

## Spearpoint Reference
- BIZRA-STS-001 Day 3
- Date: 2026-04-09
