# Phase 37 — DDAGI OS v4.0-GENESIS: Production Deployment Contract

> Formalizes the container build, K8s security context, and runtime directory requirements. Derived from live production debugging (SAPE session, 2026-02-18).

Standing on Giants: Burns (K8s Patterns, 2019) + Bernstein (Zero Trust, 2014) + Docker (BuildKit, 2018)

---

## 1. Container Build Invariants

### 1.1 UID/GID Contract

```
INVARIANT: Container UID must match K8s securityContext.runAsUser

  Dockerfile:  useradd -u 1000 -g 1000 bizra
  K8s spec:    securityContext.runAsUser: 1000
                securityContext.runAsGroup: 1000
                securityContext.fsGroup: 1000

ROOT CAUSE (resolved 2026-02-18):
  useradd -r (system user) assigned UID 999, but K8s securityContext
  overrode to UID 1000. All chown/COPY --chown operations used UID 999.
  Result: sovereign_state/ owned by 999, process running as 1000.
  Fix: Pin UID/GID to 1000 explicitly in Dockerfile.
```

### 1.2 Multi-Stage Build Contract

```
MODULE DockerBuildContract:

  STAGE builder:
    BASE: python:3.12-slim-bookworm
    PURPOSE: Compile dependencies into isolated venv
    OUTPUTS: /opt/venv (complete virtual environment)
    INVARIANT: No source code or secrets in this stage

  STAGE runtime:
    BASE: python:3.12-slim-bookworm
    USER: bizra (UID 1000, GID 1000)
    WORKDIR: /app
    INPUTS:
      /opt/venv          FROM builder   # Dependencies
      /app/core/         COPY           # Application code
      /app/tests/        COPY           # Test suite
      /app/pyproject.toml COPY          # Package metadata
      /app/data/         COPY           # Static data assets

    INVARIANT: No build tools in runtime stage
    INVARIANT: No .git, .env, or credential files in image
    INVARIANT: Non-root user (bizra:1000) for all operations
```

### 1.3 Runtime Directory Tree

```
MODULE RuntimeDirectoryContract:

  REQUIRED_DIRS = {
    "/app/sovereign_state":              "Auth DB, runtime state",
    "/app/sovereign_state/checkpoints":  "State checkpoints (StateCheckpointer)",
    "/app/sovereign_state/users":        "User profile storage",
    "/app/sovereign_state/agent_db":     "AgentDB HNSW index + SQLite",
    "/app/sovereign_state/living_memory":"LivingMemoryCore persistence",
    "/app/.spearpoint":                  "Spearpoint orchestrator state",
    "/app/.spearpoint/hypothesis_memory":"Hypothesis persistence",
    "/app/logs":                         "Application logs",
  }

  FOR dir IN REQUIRED_DIRS:
    REQUIRE dir EXISTS
    REQUIRE dir OWNED_BY uid=1000, gid=1000
    REQUIRE dir WRITABLE_BY uid=1000

  # These directories are created in Dockerfile BEFORE USER switch:
  # RUN mkdir -p <all dirs> && chown -R bizra:bizra <parent dirs>

  # TDD ANCHOR: test_all_runtime_dirs_exist_in_container
  # TDD ANCHOR: test_all_runtime_dirs_writable_by_uid_1000
  # TDD ANCHOR: test_sqlite_creates_db_in_sovereign_state
```

---

## 2. Kubernetes Security Context

### 2.1 Pod Security

```
MODULE K8sSecurityContract:

  STRUCT PodSecurityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault

  STRUCT ContainerSecurityContext:
    allowPrivilegeEscalation: false
    capabilities:
      drop: ["ALL"]
    readOnlyRootFilesystem: false  # Required for SQLite WAL

  INVARIANT: runAsUser == Dockerfile UID (1000)
  INVARIANT: runAsNonRoot == true (defense in depth)
  INVARIANT: No privileged containers
  INVARIANT: All capabilities dropped

  # readOnlyRootFilesystem is false because:
  # 1. SQLite needs WAL/journal files adjacent to DB
  # 2. Python __pycache__ writes (PYTHONDONTWRITEBYTECODE mitigates)
  # 3. /tmp usage by libraries
  # Future: Mount emptyDir volumes for state dirs to enable readOnlyRoot

  # TDD ANCHOR: test_pod_runs_as_uid_1000
  # TDD ANCHOR: test_pod_runs_as_non_root
  # TDD ANCHOR: test_no_privilege_escalation
```

### 2.2 NetworkPolicy (Zero-Trust)

```
MODULE NetworkPolicyContract:

  # Applied to both bizra and bizra-staging namespaces
  POLICY = DefaultDenyAll + ExplicitAllow

  INGRESS_ALLOW:
    - FROM: ingress-nginx namespace  TO: ports [3001, 8000]  # Traefik -> app
    - FROM: bizra pods               TO: ports [3001, 8000, 7946/UDP, 7654]  # Intra-ns
    - FROM: monitoring namespace     TO: port 9090  # Prometheus scrape

  EGRESS_ALLOW:
    - TO: kube-dns pods              PORT: 53/UDP   # DNS resolution
    - TO: bizra pods                 ALL ports      # Intra-namespace
    - TO: ollama pods                PORT: 11434    # LLM inference
    - TO: external (non-RFC1918)     PORT: 443      # HTTPS only

  EGRESS_DENY:
    - TO: 10.0.0.0/8                # No internal east-west escape
    - TO: 172.16.0.0/12             # No Docker network escape
    - TO: 192.168.0.0/16            # No host network escape

  # TDD ANCHOR: test_default_deny_blocks_unallowed_traffic
  # TDD ANCHOR: test_ingress_allows_traefik
  # TDD ANCHOR: test_egress_blocks_rfc1918
  # TDD ANCHOR: test_egress_allows_dns
```

### 2.3 HPA Configuration

```
MODULE HPAContract:

  STRUCT HPASpec:
    minReplicas: 1
    maxReplicas: 3                   # Production
    # maxReplicas: 5                 # Staging (overlay)

    metrics:
      - resource: cpu
        target: averageUtilization: 70%
      - resource: memory
        target: averageUtilization: 80%

    behavior:
      scaleUp:
        stabilizationWindowSeconds: 0   # Immediate scale-up
        policies:
          - type: Percent
            value: 100
            periodSeconds: 15
      scaleDown:
        stabilizationWindowSeconds: 300  # 5-min cooldown
        policies:
          - type: Percent
            value: 10
            periodSeconds: 60

  INVARIANT: minReplicas >= 1 (no scale-to-zero)
  INVARIANT: scaleDown.stabilization >= 300s (prevent flapping)

  # TDD ANCHOR: test_hpa_exists_for_elite_deployment
  # TDD ANCHOR: test_hpa_min_replicas_at_least_1
  # TDD ANCHOR: test_hpa_scale_down_stabilization_gte_300
```

---

## 3. Health Check Contract

### 3.1 Liveness Probe

```
MODULE HealthCheckContract:

  ENDPOINT: GET /v1/health
  PORT: 8000
  EXPECTED_RESPONSE:
    status: "healthy" | "degraded" | "unknown"
    version: str
    health_score: float [0.0, 1.0]
    subsystems: Dict[str, str]      # name -> "active" | "unavailable" | "stub"

  PROBE_CONFIG:
    initialDelaySeconds: 15          # App takes ~12s to boot
    periodSeconds: 30
    timeoutSeconds: 10
    failureThreshold: 3
    successThreshold: 1

  HEALTH_SCORING:
    total = len(subsystems)
    active = count(s for s in subsystems.values() if s == "active")
    health_score = active / total

    IF health_score >= 0.8:  status = "healthy"
    ELIF health_score >= 0.5: status = "degraded"
    ELSE: status = "unknown"

  # Subsystems expected in production:
  EXPECTED_ACTIVE = [
    "cognitive_fusion",
    "embedding_service",
    "memory_coordinator",
    "evidence_ledger",
  ]

  EXPECTED_OPTIONAL = [
    "graph_of_thoughts",      # Requires LLM backend
    "snr_maximizer",          # Requires LLM backend
    "guardian_council",       # Requires LLM backend
    "autonomous_loop",        # Stub until Phase 38
  ]

  # TDD ANCHOR: test_health_endpoint_returns_200
  # TDD ANCHOR: test_health_includes_all_subsystems
  # TDD ANCHOR: test_health_score_matches_active_ratio
  # TDD ANCHOR: test_core_subsystems_active_on_boot
```

### 3.2 Auth Layer Boot Contract

```
MODULE AuthBootContract:

  DEPENDENCIES:
    sovereign_state/users.db        # SQLite user store (auto-created)
    BIZRA_JWT_SECRET env var        # JWT signing key (optional, auto-generated)

  BOOT_SEQUENCE:
    1. UserStore.__init__(db_path="sovereign_state/users.db")
       -> Creates SQLite DB if not exists
       -> Runs schema migrations
    2. JWTAuth.__init__(secret=env.BIZRA_JWT_SECRET or auto_generate())
       -> WARNING if auto-generated (not persistent across restarts)
    3. Middleware.__init__(auth=jwt_auth, store=user_store)
    4. Log "Phase 21: Auth layer initialized"

  FAIL_CLOSED_POLICY:
    IF any step raises exception:
      Log "SECURITY: Auth layer failed to initialize: {error}"
      Protected endpoints DENY all requests
      Health endpoint remains accessible (unauthenticated)

  REQUIRED_ENV_VARS:
    BIZRA_JWT_SECRET:    Optional (auto-generated if absent, WARNING logged)
    BIZRA_API_TOKEN:     Optional (for API token auth mode)

  # TDD ANCHOR: test_auth_initializes_with_writable_dir
  # TDD ANCHOR: test_auth_fails_closed_on_permission_error
  # TDD ANCHOR: test_auth_warns_on_auto_generated_jwt_secret
  # TDD ANCHOR: test_auth_creates_sqlite_db_on_first_boot
```

---

## 4. Deployment Checklist (Runbook)

```
MODULE DeploymentRunbook:

  PRE_DEPLOY:
    [ ] Docker image built: docker build -f deploy/Dockerfile.elite -t bizra-elite:$TAG .
    [ ] Image imported to cluster: k3d image import bizra-elite:$TAG -c bizra-prod
    [ ] No hardcoded secrets in image: docker history --no-trunc | grep -i secret
    [ ] UID matches K8s: kubectl get deploy -o jsonpath='{.spec.template.spec.securityContext}'

  DEPLOY:
    [ ] Rolling update: kubectl set image deployment/bizra-elite bizra-elite=bizra-elite:$TAG -n bizra
    [ ] Rollout status: kubectl rollout status deployment/bizra-elite -n bizra --timeout=120s
    [ ] No permission errors: kubectl logs deployment/bizra-elite -n bizra | grep "Permission denied"
    [ ] Auth initialized: kubectl logs deployment/bizra-elite -n bizra | grep "Auth layer initialized"

  POST_DEPLOY:
    [ ] Health check passes: kubectl exec ... -- curl -s localhost:8000/v1/health
    [ ] Core subsystems active: cognitive_fusion, embedding_service, memory_coordinator, evidence_ledger
    [ ] sovereign_state writable: kubectl exec ... -- touch sovereign_state/test && rm sovereign_state/test
    [ ] HPA operational: kubectl get hpa -n bizra
    [ ] NetworkPolicy applied: kubectl get networkpolicy -n bizra

  ROLLBACK:
    kubectl rollout undo deployment/bizra-elite -n bizra
```

---

## 5. Image Tagging Convention

```
MODULE ImageTagging:

  FORMAT: bizra-elite:v{MAJOR}.{MINOR}.{PATCH}-{QUALIFIER}

  QUALIFIERS:
    sape      = Security Audit + Performance Enhancement release
    rc{N}     = Release candidate
    staging   = Staging-only build
    omega     = Rust-native build (bizra-omega crate)

  EXAMPLES:
    bizra-elite:v1.2.2-sape     # Current production (SAPE fixes + Dockerfile UID fix)
    bizra-elite:v1.2.1-sape     # Previous (auth PermissionError)
    bizra-elite:v3.1-omega      # Rust build in staging

  IMMUTABILITY:
    NEVER use :latest in production
    NEVER reuse a tag for different content
    Tags are immutable once deployed to a namespace

  # TDD ANCHOR: test_image_tag_matches_semver_format
  # TDD ANCHOR: test_no_latest_tag_in_production_deployments
```

---

## 6. Environment Variable Contract

```
MODULE EnvVarContract:

  # Injected at runtime via K8s ConfigMap/Secret — NEVER baked into image

  REQUIRED (fail-fast on absence):
    None — all have safe defaults for container boot

  RECOMMENDED (WARNING if absent):
    BIZRA_JWT_SECRET:        str    # JWT signing key (persistent across restarts)
    BIZRA_API_TOKEN:         str    # API authentication token

  OPTIONAL (defaults in code):
    BIZRA_ENV:               str    # "production" | "staging" | "development"
    IHSAN_THRESHOLD:         float  # Default: 0.95 (from constants.py)
    SNR_THRESHOLD:           float  # Default: 0.85 (from constants.py)
    LOG_LEVEL:               str    # Default: "INFO"
    PYTHONUNBUFFERED:        str    # Default: "1" (set in Dockerfile)
    PYTHONDONTWRITEBYTECODE: str    # Default: "1" (set in Dockerfile)

  FORBIDDEN in container image:
    - Private keys (Ed25519, Fernet, etc.)
    - Database passwords
    - API tokens for external services
    - .env files

  # TDD ANCHOR: test_no_secrets_in_docker_image_layers
  # TDD ANCHOR: test_env_defaults_produce_bootable_container
```

---

## 7. TDD Anchor Summary

| Module | Test Count | Key Assertion |
|--------|-----------|---------------|
| RuntimeDirectoryContract | 3 | All dirs exist + writable + SQLite works |
| K8sSecurityContract | 3 | UID 1000, non-root, no privesc |
| NetworkPolicyContract | 4 | Default deny, allow traefik/DNS, block RFC1918 |
| HPAContract | 3 | Exists, min>=1, stabilization>=300s |
| HealthCheckContract | 4 | 200 OK, all subsystems, score formula |
| AuthBootContract | 4 | Writable dir, fail-closed, JWT warning, DB creation |
| ImageTagging | 2 | Semver format, no :latest |
| EnvVarContract | 2 | No secrets in layers, bootable defaults |
| **Total** | **25** | |
