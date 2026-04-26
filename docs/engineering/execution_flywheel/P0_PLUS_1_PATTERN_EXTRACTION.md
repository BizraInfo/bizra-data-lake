# P0+1 Pattern Extraction

## Source

The P0+1 hardening session closed a cluster of audit + secret-hygiene
failures. The session addendum is archived at:

```
docs/audits/omnidirectional_hyperdimensional_audit_v0_1/
  P0_PLUS_1_HARDENING_ADDENDUM_2026_04_24.md
```

Observable outcomes (summarised from the addendum; none reproduced here):

- Audit found 35 secret-pattern matches on the initial run.
- Runtime dev-default credential fallbacks were removed.
- Scanner false-positive noise (self-scan, placeholder and env substitution
  matches) was reduced.
- Regression tests were added against the YAML loader and scanner.
- Audit rerun produced a final secret findings count of zero.

## Patterns extracted

Four new patterns land in `patterns.yaml`, plus one previously-encoded
pattern (`PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT`) from PR #49.

### `AUDIT_YAML_INLINE_COMMENT_PARSE_FAILURE`

- **severity** `high`, **default_decision** `REVALIDATE`
- **Triggers:** audit engine crash during YAML load, `TypeError int vs str`,
  inline YAML comments near numeric config values.
- **Guard actions:** sanitize inline comments outside quoted strings; add a
  regression test for the YAML loader with inline comments.

### `SECRET_SCANNER_SNR_NOISE_COLLAPSE`

- **severity** `high`, **default_decision** `REVALIDATE`
- **Triggers:** high secret finding count, self-scan matches,
  placeholder/env-substitution matches.
- **Guard actions:** dedupe overlapping scanner roots; exclude logs and
  scanner self-reference; suppress safe placeholders/env substitutions;
  rerun audit into `/tmp` for diff comparison.

### `DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT`

- **severity** `critical`, **default_decision** `ABORT`
- **Triggers:** committed default DSN/Redis/Neo4j fallback credential,
  credential-bearing URL printed in logs.
- **Guard actions:** require operator-supplied env var; fail closed for
  strict backend modes; degrade only to local non-network persistence when
  explicitly safe; mask connection URLs in logs.

### `BOTTLENECK_SHIFT_AFTER_SECRET_GATE_CLEARS`

- **severity** `high`, **default_decision** `REVALIDATE`
- **Triggers:** secret findings zero, rotation not required, public claims
  risky.
- **Guard actions:** shift priority from secret triage to public claim
  discipline; recommend P0.2 website claim cleanup.

## What the extraction deliberately excludes

- **No secret values.** No raw credentials, dev-defaults, or redacted
  previews appear in the registry. Only trigger *keywords*.
- **No scanner output or counts.** The "35 matches" figure is context in
  this extraction doc; it is not encoded into any pattern. Patterns stay
  signal-preserving, noise-resistant.
- **No session-specific paths.** The addendum path is quoted once here for
  provenance; it is not baked into any pattern's guard-action list.

## Law of Assumption mapping

Each pattern turns an assumption the system was making into an observable
check:

- "audit will run" → REVALIDATE before trusting audit output when YAML is
  ambiguous.
- "scanner findings are signal" → REVALIDATE when self-scan noise is
  detectable.
- "dev defaults are safe" → ABORT when a credential fallback is committed.
- "secret zero means we are done" → REVALIDATE into PUBLIC_CLAIMS.
