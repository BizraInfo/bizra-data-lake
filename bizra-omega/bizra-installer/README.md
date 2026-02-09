# bizra-installer

**BIZRA Universal CLI** — Complete command-line interface for Sovereign Node operations.

## Overview

One-shot installer and management CLI for BIZRA sovereign nodes:

- **Hardware Detection** — Auto-detect GPU, RAM, CPU for optimal model recommendations
- **Model Cache** — Download, verify, and manage LLM model files
- **Node Setup** — Initialize data directory, keys, and configuration
- **Service Management** — Start/stop API gateway, gossip, and inference

## Usage

```bash
# Install and initialize a new node
bizra-install init

# Detect hardware and recommend models
bizra-install detect

# Start all services
bizra-install start
```

## Configuration

| Constant | Default |
|----------|---------|
| Data directory | `~/.bizra` |
| API port | 3001 |
| Gossip port | 7946 |

## Standing on Giants

Torvalds (Unix philosophy), Pike (Go CLI patterns), Stallman (GNU)

## License

MIT
