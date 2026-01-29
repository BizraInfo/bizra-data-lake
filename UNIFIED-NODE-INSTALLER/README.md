# BIZRA Unified Node Installer

**One installer. Zero friction. Complete freedom.**

## What This Gives You

By running ONE command, anyone gets:

1. **Personal AI Think Tank (PAT)** - Your own team of AI agents, customized to YOUR goals
2. **Resource Sharing** - Contribute idle compute, earn tokens
3. **Gateway to Free Virtual World** - Decentralized social space with no algorithm manipulation
4. **Network Benefits** - More users = faster, smarter, safer, more private

## Quick Start

### Windows
```
Double-click: installers/windows/install.bat
```

### Mac/Linux
```bash
./installers/linux/install.sh
```

### One-liner (Linux/Mac)
```bash
curl -sSL https://bizra.ai/install | bash
```

## What Happens

1. **Hardware Detection** - Automatically determines your system tier
2. **Profile Creation** - 5-minute wizard to customize your PAT
3. **Installation** - Downloads and configures everything
4. **Launch** - Your node starts at `http://localhost:8888`

## System Requirements

| Tier | RAM | GPU | Experience |
|------|-----|-----|------------|
| Potato | <8GB | No | Text-only PAT |
| Normal | 8-16GB | Optional | Full PAT |
| Gaming | 16-32GB | 6GB+ | Multi-agent teams |
| Server | 32GB+ | 12GB+ | Full validator |

**Minimum:** 4GB RAM, 10GB disk space
**Recommended:** 8GB RAM, 50GB disk space

## Project Structure

```
UNIFIED-NODE-INSTALLER/
├── ARCHITECTURE.md          # Detailed system design
├── README.md               # This file
│
├── bootstrap/
│   └── install.py          # Main installer script (Python)
│
├── core/
│   ├── main.py             # Node entry point
│   ├── pat_engine.py       # Personal Agentic Team
│   └── network_node.py     # P2P network + tokens
│
└── installers/
    ├── windows/
    │   └── install.bat     # Windows entry point
    └── linux/
        └── install.sh      # Linux/Mac entry point
```

## The Vision

This installer is the bridge between:
- **15,000 hours** of BIZRA development
- **8 billion potential nodes** worldwide

Every person who runs this installer becomes part of a decentralized AI civilization where:
- Your data stays YOURS
- No algorithms manipulate what you see
- You choose your own feed
- Contributing resources earns you real value
- The network gets better as it grows

## Development Status

- [x] Architecture design
- [x] Cross-platform bootstrap (install.py)
- [x] PAT Engine (personalized agent team)
- [x] Network Node (resources, tokens)
- [x] Windows installer (install.bat)
- [x] Linux/Mac installer (install.sh)
- [ ] Docker deployment option
- [ ] Native binary builds
- [ ] Auto-update system
- [ ] Full LLM integration
- [ ] Production P2P network

## Contributing

This is the foundation. The pieces are here. Now we build.

---

**Built with 15,000 hours of vision.**
**Ready for 8 billion nodes.**

*"No assumptions - only verified excellence."*
