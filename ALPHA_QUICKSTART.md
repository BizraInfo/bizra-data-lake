# BIZRA Alpha — Quick Start Guide

Welcome to BIZRA. You're one of the first 100 sovereign nodes.

## What You Need

- A computer (Windows, Mac, or Linux)
- 4 GB free disk space
- Your invite code (you received this from Mumo)

## Setup (5 minutes)

### Step 1: Install Ollama (your local AI runtime)

Go to [ollama.ai/download](https://ollama.ai/download) and install it.

Then open a terminal and run:
```bash
ollama pull qwen2.5:3b
```
This downloads a small AI model (2 GB). It runs entirely on YOUR device.

### Step 2: Clone BIZRA

```bash
git clone https://github.com/BizraInfo/bizra-data-lake.git
cd bizra-data-lake
```

### Step 3: Set up Python

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install -e .
```

### Step 4: Build the Rust binary (optional, for full performance)

```bash
cd bizra-omega
cargo build --release -p bizra-node
cd ..
```

### Step 5: Run your first mission

```bash
./scripts/bizra
```

You'll see:
```
  ╔═══════════════════════════════════════════════════════╗
  ║           بِذْرَة  —  BIZRA SOVEREIGN AI              ║
  ╚═══════════════════════════════════════════════════════╝

  بذرة › mission "Hello, I'm a new sovereign node"
```

That's it. You're sovereign. Your AI runs on YOUR hardware. You own every thought.

## What Happens Next

- **12 agents** are minted for you (7 personal, 5 for the network)
- **Every mission** earns you SEED tokens (proof-of-impact)
- **Your data** stays on YOUR device (constitutional guarantee)
- **The sea grows** as you contribute knowledge through the membrane

## Commands

| Command | What it does |
|---------|-------------|
| `bizra mission "your task"` | Run a mission through your PAT team |
| `bizra agents` | See your 12 sovereign agents |
| `bizra wallet` | Check your SEED balance |
| `bizra briefing` | Morning briefing from your AI |
| `bizra status` | System health check |
| `bizra organize ~/Downloads` | Organize files with Guardian approval |
| `bizra ghost` | See what your agents are thinking |

## Your Invite Code

Use your code when connecting to the network:
```bash
./scripts/bizra activate --invite YOUR_CODE_HERE
```

## Help

- Documentation: [github.com/BizraInfo/bizra-data-lake](https://github.com/BizraInfo/bizra-data-lake)
- Start Here: [START_HERE.md](START_HERE.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Philosophy

> Every human is a node. Every node is a seed.
> بذرة واحدة تصنع غابة — One seed makes a forest.

Your AI runs on YOUR hardware. You own every thought. Welcome to the forest.
