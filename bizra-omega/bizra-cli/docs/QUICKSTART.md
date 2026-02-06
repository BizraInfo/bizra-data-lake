# BIZRA CLI Quickstart

Get up and running in 5 minutes.

## Prerequisites

- Windows 10/11 or Linux (WSL supported)
- Rust toolchain (1.75+)
- LM Studio running at `192.168.56.1:1234` (optional but recommended)
  - If auth is enabled, set `LMSTUDIO_API_KEY`

## Step 1: Build

```bash
cd bizra-omega
cargo build -p bizra-cli --release
```

## Step 2: Run

**Windows:**
```batch
bizra.bat
```

**Linux/WSL:**
```bash
./target/release/bizra
```

## Step 3: Try Commands

```bash
# Show your node status
bizra status

# List your PAT agents
bizra agent list

# Start the TUI
bizra
```

## Step 4: Use the TUI

Once in the TUI:

1. Press `Tab` to switch between views
2. Press `j`/`k` to navigate agents
3. Press `i` to enter chat mode
4. Press `/` for command mode
5. Press `q` to quit

## Step 5: First Command

Type `/help` to see all available commands:

```
/agent list          - List PAT agents
/task add "title"    - Add a task
/research "topic"    - Quick research
/morning             - Morning briefing
```

## What's Next?

- [Personalize your profile](guides/03-PERSONALIZATION.md)
- [Learn about PAT agents](guides/04-PAT-AGENTS.md)
- [Set up voice interface](guides/07-VOICE-INTERFACE.md)

---

**Tip:** Run `/morning` each day for your personalized briefing!
