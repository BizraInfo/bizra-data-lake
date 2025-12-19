# HG-RTP Telemetry Activation Backlog (Sprint 0 Grooming)

## Priority 1 – Instrumentation Enablement
1. **Story**: Compile eBPF telemetry probes for target kernels
   - *Description*: Build `bizra_telemetry_probe.c` into bytecode for kernels 5.15 & 6.2, publish artifacts.
   - *Acceptance*: CI job emits signed `.o` files; unit tests verify helper usage; checksum logged.
   - *Dependencies*: Kernel headers package cache.
2. **Story**: Automate probe deployment via Ansible + systemd
   - *Description*: Idempotent role to install probes, create `bizra-telemetry.service` with health check.
   - *Acceptance*: Molecule tests simulate install/rollback; service restarts on boot.
3. **Task**: Add feature flags for MODEL_WARM_SURFACE / MODEL_COLD_CORE tracking
   - *Acceptance*: Config map exposes toggles; integration test verifies selective event capture.

## Priority 2 – Telemetry Daemon Deployment
4. **Story**: Containerize `bizra_telemetry_daemon.rs`
   - *Description*: Create Dockerfile (musl static), GitHub Actions build & push to registry.
   - *Acceptance*: Image includes ringbuffer consumer, Merkle batcher; vulnerability scan clean.
5. **Story**: Implement ringbuffer batch ingest & HotStuff client stub
   - *Acceptance*: Load test shows ≥1M events/sec throughput; Merkle root stored; tx submitted to mock BlockGraph.
6. **Task**: Configure batching parameters & alert thresholds
   - *Acceptance*: Helm values expose `BATCH_SIZE`, `MAX_LATENCY_MS`; alerts fire when exceeded.

## Priority 3 – Ledger & HyperGraph Integration
7. **Story**: Extend HyperGraph schema for TELEMETRY_EVENT nodes
   - *Acceptance*: Migration adds `timestamp_ns`, `BlockGraph_Tx`, `weight`; regression tests pass.
8. **Story**: Wire Ghost Entity pre-check post-ingest
   - *Acceptance*: GED job flags unmeasured schema entries; pipeline fails if count > threshold.
9. **Task**: Mock BlockGraph testnet deployment scripts
   - *Acceptance*: Terraform applies local HotStuff cluster; smoke test passes.

## Priority 4 – Validation & Observability
10. **Story**: Create kernel microbench for BPF overhead metrics
    - *Acceptance*: Benchmark report shows ≤5 µs overhead; stored in Lexicon Ledger.
11. **Story**: Build Grafana dashboards for telemetry health
    - *Acceptance*: Panels for throughput, loss, latency; alert hook to Ops channel.
12. **Task**: Draft telemetry rollback runbook with Ihsān checklist
    - *Acceptance*: Runbook published; tabletop exercise signed off.

## Grooming Notes
- All stories tagged `HG-RTP-PhaseA` and linked to blueprint.
- Definition of Ready: requirements clarified, dependencies available, tests outlined.
- Definition of Done: code merged, CI green, documentation updated, Ihsān review logged.
