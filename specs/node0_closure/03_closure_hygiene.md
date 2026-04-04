# Phase 3: Closure Hygiene
## Goal: CI all-green, secrets configured, deps pinned
### References: 00_master_spec.md §4 Phase 3

---

## 1. Problem Statement

Five operational hygiene items block closure certification:

| Item | Severity | Current State |
|------|----------|---------------|
| BIZRA_ADMIN_TOKEN | MEDIUM | GitHub secret not configured |
| cargo fmt | LOW | Pre-existing formatting drift on CI |
| ChromaDB pin | MEDIUM | Using `:latest` tag (6 months stale) |
| pip-audit gate | LOW | Missing from CI matrix |
| Redis TLS in dev | LOW | Open in dev compose |

None are architectural. All are closure-blocking gates.

## 2. Pseudocode: BIZRA_ADMIN_TOKEN Setup

```
PROCEDURE configure_admin_token():
    # Step 1: Generate PAT with required scopes
    # Human action — cannot be automated
    PROMPT user:
        "Go to GitHub → Settings → Developer Settings → PATs
         Create token with scopes: admin:repo_hook, repo
         Name: BIZRA_ADMIN_TOKEN
         Expiry: 90 days"

    # Step 2: Store as repository secret
    PROMPT user:
        "Go to BizraInfo/bizra-data-lake → Settings → Secrets → Actions
         New repository secret:
           Name: BIZRA_ADMIN_TOKEN
           Value: <paste PAT>"

    # Step 3: Validate CI picks it up
    RUN: gh workflow run branch-protection-audit.yml
    ASSERT: workflow completes without 'BIZRA_ADMIN_TOKEN not set' warning

    # Step 4: Verify graceful degradation still works
    # The workflow already handles missing token for scheduled runs
    # (commit 69edcc31, lines 47-59)
```

## 3. Pseudocode: cargo fmt Fix

```
PROCEDURE fix_cargo_fmt():
    cd bizra-omega/

    # Step 1: Identify all formatting issues
    cargo fmt --all -- --check 2>&1 | tee /tmp/fmt_diff.txt

    # Step 2: Auto-fix
    cargo fmt --all

    # Step 3: Verify clippy still passes (fmt can shift lint boundaries)
    cargo clippy --workspace --all-targets -- -D warnings

    # Step 4: Verify tests still pass
    cargo test --workspace --release

    # Step 5: Commit
    git add -A bizra-omega/
    git commit -m "style(omega): cargo fmt alignment for CI gate"
```

## 4. Pseudocode: ChromaDB Version Pin

```
PROCEDURE pin_chromadb():
    # Step 1: Identify current running version
    CURRENT_VERSION = docker inspect chromadb/chroma:latest | jq image.tag
    CURRENT_DIGEST = docker inspect --format='{{.RepoDigests}}' <container>

    # Step 2: Pin in both compose files
    FOR compose IN [
        "docker-compose.unified.yml",
        "/mnt/c/BIZRA-Dual-Agentic-system--main/docker-compose.yml"
    ]:
        REPLACE "chromadb/chroma:latest"
        WITH    "chromadb/chroma:0.5.23@sha256:<digest>"
        # Use exact version + digest for supply-chain integrity

    # Step 3: Verify containers start with pinned version
    docker compose -f docker-compose.unified.yml pull vectors
    docker compose -f docker-compose.unified.yml up -d vectors
    ASSERT: docker inspect vectors | version == "0.5.23"
```

## 5. Pseudocode: pip-audit CI Gate

```
# .github/workflows/ci.yml addition

  security-python:
    name: "Python Security Audit"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<sha>
      - uses: actions/setup-python@<sha>
        with: { python-version: "3.12" }
      - run: pip install pip-audit
      - run: pip-audit -r requirements.txt --desc
        # Fail on HIGH+ severity, warn on MEDIUM
      - run: pip-audit -r requirements.txt --fix --dry-run
        # Show what would be fixed
```

## 6. Pseudocode: Redis TLS in Dev Compose

```
PROCEDURE secure_redis_dev():
    # NOT a hard blocker — dev environment
    # Document the risk and plan for production

    # Option A: Self-signed cert for dev
    IN docker-compose.unified.yml, synapse service:
        command: [
            redis-server,
            --appendonly, "yes",
            --requirepass, "${REDIS_PASSWORD}",
            --tls-port, "6379",
            --tls-cert-file, "/tls/redis.crt",
            --tls-key-file, "/tls/redis.key",
            --tls-ca-cert-file, "/tls/ca.crt",
        ]

    # Option B (recommended for dev): Document risk, defer to production
    ADD comment in compose:
        "# SECURITY: Redis TLS disabled in dev. Enable for production.
         # See: specs/node0_closure/03_closure_hygiene.md §6"

    # Production will use managed Redis (AWS ElastiCache / Azure Cache)
    # with TLS enabled by default
```

## 7. Implementation Checklist

```
[ ] BIZRA_ADMIN_TOKEN — user configures GitHub secret
[ ] cargo fmt — run formatter, verify clippy + tests
[ ] ChromaDB — identify version, pin with digest in both composes
[ ] pip-audit — add CI gate step
[ ] Redis TLS — document risk, add TODO comment
[ ] Dependabot #15 — address moderate vulnerability flagged by GitHub
```

## 8. TDD Anchors

```python
# tests/integration/test_closure_hygiene.py

def test_chromadb_version_pinned():
    """Compose files do not use :latest for ChromaDB."""
    for compose_file in COMPOSE_FILES:
        content = Path(compose_file).read_text()
        assert "chromadb/chroma:latest" not in content

def test_no_hardcoded_passwords_in_compose():
    """Default passwords are only in ${VAR:-default} form, not bare."""
    for compose_file in COMPOSE_FILES:
        content = Path(compose_file).read_text()
        # Defaults in env var syntax are acceptable for dev
        # Bare passwords are not
        assert "password:" not in content.lower() or "${" in content

def test_redis_auth_required():
    """Redis synapse requires authentication."""
    import redis
    r = redis.Redis(host="localhost", port=6380)
    with pytest.raises(redis.AuthenticationError):
        r.ping()

def test_cargo_fmt_clean():
    """Rust workspace passes cargo fmt check."""
    result = subprocess.run(
        ["cargo", "fmt", "--all", "--", "--check"],
        cwd="bizra-omega", capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()
```

## 9. Validation Gate

```
ALL of:
  [ ] CI workflow: branch-protection-audit passes (or gracefully degrades)
  [ ] cargo fmt --check returns 0
  [ ] cargo clippy returns 0 warnings
  [ ] ChromaDB version pinned in both compose files
  [ ] No new pip-audit HIGH findings
  [ ] Dependabot #15 addressed
```

---

*Hygiene is not glamorous. It is the difference between*
*"credible proof-bearing life" and "certified operational."*
