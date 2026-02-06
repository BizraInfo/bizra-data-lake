# BIZRA Node0 Deployment Manifest

Genesis Block - Flagship Development Environment

## Overview

This directory contains the complete deployment manifest for BIZRA Node0, the Genesis Block of the BIZRA ecosystem. It defines hardware requirements, service configurations, startup procedures, and health monitoring for the flagship development environment.

**Constitutional Constraint:** Ihsan >= 0.95

## Directory Structure

```
deploy/node0/
├── node0-manifest.yaml          # Hardware & service specifications
├── systemd-services/            # Linux systemd service units
│   ├── bizra-api.service        # Rust API server
│   ├── bizra-dashboard.service  # React dashboard
│   ├── bizra-inference.service  # LM Studio/Ollama manager
│   └── bizra-sovereign.service  # Apex Sovereign Engine
├── startup.sh                   # Unified startup script
├── health-check.py              # Python health monitoring
├── secrets.env.template         # Secrets template
└── README.md                    # This file
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| CPU | 8 cores | 16 cores |
| RAM | 64 GB | 128 GB |
| Disk | 500 GB NVMe | 1 TB NVMe |

## Quick Start

### 1. Check Hardware Requirements

```bash
./startup.sh --check-only
```

### 2. Configure Secrets

```bash
cp secrets.env.template /etc/bizra/secrets.env
# Edit /etc/bizra/secrets.env with your values
chmod 600 /etc/bizra/secrets.env
```

### 3. Install systemd Services

```bash
sudo cp systemd-services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bizra-{api,dashboard,inference,sovereign}
```

### 4. Start All Services

```bash
./startup.sh
```

### 5. Verify Health

```bash
python health-check.py
```

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TIER 1: Infrastructure                          │
│                     PostgreSQL (5432) | Redis (6379)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         TIER 2: Inference Layer                         │
│            LM Studio (192.168.56.1:1234) | Ollama (11434)               │
│                         [RTX 4090 GPU-bound]                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                        TIER 3: Application Layer                        │
│                API Server (3001) | Dashboard (5173)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         TIER 4: Sovereign Layer                         │
│                        Sovereign Engine (8080)                          │
│      [GraphOfThoughts | SNRMaximizer | GuardianCouncil | OmegaEngine]   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quality Constraints

### Ihsan Threshold

The system enforces Ihsan (excellence) >= 0.95 as a hard constitutional constraint.

| Level | Threshold | Action |
|-------|-----------|--------|
| HEALTHY | >= 0.95 | Normal operation |
| DEGRADED | >= 0.80 | Alerts, investigation |
| UNHEALTHY | < 0.80 | Service degradation |

### SNR Thresholds

Signal-to-Noise Ratio enforces data quality:

| Layer | Threshold |
|-------|-----------|
| Data | >= 0.90 |
| Information | >= 0.95 |
| Knowledge | >= 0.99 |
| Wisdom | >= 0.999 |

## Monitoring

### Continuous Health Check

```bash
python health-check.py --continuous --interval 30
```

### JSON Output (for integration)

```bash
python health-check.py --json
```

### Prometheus Metrics

- API Server: http://localhost:3001/metrics
- Sovereign Engine: http://localhost:8080/metrics
- Prometheus UI: http://localhost:9090
- Grafana UI: http://localhost:3000

## Troubleshooting

### Service Won't Start

1. Check hardware requirements: `./startup.sh --check-only`
2. Check systemd status: `systemctl status bizra-<service>`
3. Check logs: `journalctl -u bizra-<service> -f`

### GPU Not Available

1. Verify NVIDIA driver: `nvidia-smi`
2. Check CUDA: `nvcc --version`
3. Use `--skip-gpu` for testing: `./startup.sh --skip-gpu`

### Inference Backend Down

Priority order (fail-closed):
1. LM Studio (192.168.56.1:1234) - PRIMARY
2. Ollama (localhost:11434) - FALLBACK
3. DENY - complete failure

Check LM Studio on the Windows host, then check Ollama:
```bash
curl http://192.168.56.1:1234/v1/models
curl http://localhost:11434/api/tags
```

### Low Ihsan Score

1. Check Sovereign Engine health: `curl localhost:8080/health`
2. Review recent queries: Check logs for validation failures
3. Investigate SNR scores: May indicate data quality issues

## Standing on Giants

- Shannon (information theory)
- Lamport (distributed systems)
- Nygard (resilience patterns)
- Besta (Graph-of-Thoughts)
- Anthropic (Constitutional AI)

## License

BIZRA Sovereignty License - All rights reserved.
