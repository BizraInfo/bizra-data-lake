# P0-DEPBOT TRIAGE — Day 2, 2026-04-08

## 1. PACKAGE
- **Name:** vite
- **Ecosystem:** npm
- **Current version:** ^6.0.0 (devDependency in frontend/package.json)
- **Patched version:** 6.4.2
- **Vulnerable range:** >= 6.0.0, <= 6.4.1
- **Dependabot alert:** #29

## 2. EXPOSURE
- **Runtime-facing or dev-only?** Dev-only. Vite is a build tool and dev server. Not shipped to production.
- **Used in production code paths or only in tests/tooling?** Dev server + build tooling only. The vulnerability is in the HMR WebSocket of the dev server.
- **Reachable from external input or internal-only?** Only exploitable if dev server is exposed to network via `--host` or `server.host` config. Default: localhost only. BIZRA frontend uses `npm run dev` (localhost).

## 3. EXPLOITABILITY
- **CVE ID:** CVE-2026-39363
- **CVSS score:** Not yet scored (0 in API, pending NVD assessment)
- **Vector:** Network — requires WebSocket connection without Origin header to dev server
- **Public exploit available?** Yes — PoC in advisory. Connect to HMR WebSocket, send `vite:invoke` with `fetchModule` + `file://...?raw` to read arbitrary files.
- **Preconditions:** Dev server must be bound to 0.0.0.0 (not default), WebSocket must not be disabled.

## 4. UPSTREAM STATUS
- **Patch already released?** Yes — vite 6.4.2
- **Breaking changes in patch?** No — patch release (6.4.1 → 6.4.2), security fix only
- **Estimated upgrade effort:** 5 minutes. Update `^6.0.0` to `^6.4.2` in frontend/package.json, run `npm install`.

## 5. CONSTITUTIONAL VERDICT
- **Blocks public trust surface?** No — dev-only dependency, not reachable in production, default config is safe (localhost-only).
- **Action:** FIX_TODAY
- **Triage decision rationale:** Patch is trivial (semver-compatible, no breaking changes). Even though exposure is dev-only, a public repo claiming constitutional sovereignty should not carry named high-severity CVEs when the fix is 5 minutes of work. Fix today, close the alert, remove the asterisk.

## ADDITIONAL OPEN ALERTS (10 moderate/low)

| # | Package | Ecosystem | Severity | Summary |
|---|---------|-----------|----------|---------|
| 30 | vite | npm | medium | Path Traversal in Optimized Deps .map Handling |
| 28 | vite | npm | medium | Path Traversal in Optimized Deps .map Handling |
| 27 | pyo3 | rust | low | Buffer overflow in PyString::from_object |
| 26 | rustls-webpki | rust | medium | CRL matching logic flaw |
| 25 | tar | rust | medium | chmod arbitrary dirs via symlinks |
| 24 | tar | rust | medium | PAX size headers ignored |
| 23 | time | rust | medium | Stack exhaustion DoS |
| 22 | bytes | rust | medium | Integer overflow in BytesMut::reserve |
| 21 | pyo3 | rust | low | Buffer overflow in PyString::from_object (dup of #27) |
| 15 | picomatch | npm | medium | Method Injection in POSIX Character Classes |

**Note:** Alerts #30 and #28 (vite path traversal, moderate) are separate from CVE-2026-39363. They affect vitest's transitive vite@5.4.21, which requires vitest 4.x (breaking change). Reclassified as P1-FRONTEND-TOOLCHAIN-UPGRADE.

## RESOLUTION — 2026-04-08

**Dependency tree proof:**
```
@bizra/ddagi-os@0.3.0
├─┬ @vitejs/plugin-react@4.7.0
│ └── vite@6.4.2 deduped         ← PATCHED
├── vite@6.4.2                    ← PATCHED
└─┬ vitest@2.1.9
  ├─┬ @vitest/mocker@2.1.9
  │ └── vite@5.4.21 deduped      ← NOT AFFECTED (< 6.0.0, outside CVE range)
  ├─┬ vite-node@2.1.9
  │ └── vite@5.4.21              ← NOT AFFECTED (< 6.0.0, outside CVE range)
  └── vite@5.4.21                ← NOT AFFECTED (< 6.0.0, outside CVE range)
```

**CVE-2026-39363 vulnerable range:** >= 6.0.0, <= 6.4.1
**Result:** Zero vulnerable vite paths remain. All resolved versions are either patched (6.4.2) or outside the affected range (5.4.21).

**P0-DEPBOT status: CLOSED**
**Closure commit:** fix(security): patch P0-DEPBOT — vite ^6.4.2 (CVE-2026-39363)
**Remaining:** P1-FRONTEND-TOOLCHAIN-UPGRADE for vitest 2.x → 4.x (separate, non-urgent, breaking change)
