# HG-RTP Telemetry Activation Sprint Plan

## Objective
Deploy Layers 0-1 of the HyperGraphRAG Runtime Telemetry Pipeline on staging, proving sub-5 µs syscall overhead, lossless event capture (≥1 M events/sec), and ledger commitment integrity.

## Sprint Scope (2 Weeks)
1. **Instrumentation Enablement**
   - Compile `bizra_telemetry_probe.c` eBPF artifacts for target kernels (Ubuntu 22.04 LTS, kernel 5.15 and 6.2 variants).
   - Automate probe loading via Ansible role + systemd unit; ensure idempotent startup.
   - Implement feature flags to toggle MODEL_WARM_SURFACE / MODEL_COLD_CORE tracking.
2. **Telemetry Daemon Deployment**
   - Build `bizra_telemetry_daemon.rs` container with musl static linking.
   - Provision ringbuffer consumer service (systemd) with configurable batch size (default 10,000 events).
   - Integrate Merkle batcher + threshold signature stub (use dev key shares on staging).
3. **Ledger & HyperGraph Integration**
   - Mock BlockGraph endpoint (local HotStuff testnet) capturing PoI attestion txs.
   - Extend HyperGraph ingestion with TELEMETRY_EVENT nodes and `timestamp_ns`, `BlockGraph_Tx` fields.
   - Wire GED pre-check: mark unmeasured schema entities post-ingest.
4. **Testing & Validation**
   - Regression: kernel microbench (BPF) verifying overhead ≤5 µs per probe.
   - Load: synthetic token generation 1.2 M events/sec for 10 minutes, zero loss.
   - Integrity: verify Merkle root stored on BlockGraph equals local batch digest.
5. **Observability & Docs**
   - Grafana dashboards: ringbuffer depth, event throughput, overhead metrics.
   - Update runbook: probe troubleshooting, rollback procedures, Ihsān checklists.

## Deliverables
- Signed staging deployment manifests (GitOps).
- Telemetry dashboard URLs and alerting rules (latency, loss, ledger mismatch).
- Validation report with metrics, evidence artifacts, Ihsān compliance review.
- Decision log entry authorizing production rollout (pending Phase B readiness).

## Definition of Done
- eBPF probes auto-load on staging boot with health checks.
- Telemetry daemon commits batches to BlockGraph mock and HyperGraph within <3 s end-to-end.
- Lossless event run + audit logs attached to Lexicon Ledger.
- Stakeholders sign Ihsān review (excellence: met KPIs, benevolence: user impact assessed, Adl: fairness controls documented).
