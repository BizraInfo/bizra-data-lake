# Voice Interface Guide

Interact with your PAT team through voice.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Voice Commands](#voice-commands)
4. [Agent Voices](#agent-voices)
5. [Voice Modes](#voice-modes)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The voice interface enables natural speech interaction with your PAT team using PersonaPlex technology.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VOICE ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────┐     ┌───────────────┐     ┌───────────────┐               │
│   │   User    │     │    Speech     │     │   Agent       │               │
│   │   Voice   │ ──> │  Recognition  │ ──> │   Router      │               │
│   └───────────┘     └───────────────┘     └───────┬───────┘               │
│                                                   │                        │
│                                                   ▼                        │
│                                           ┌───────────────┐               │
│                                           │  PAT Agent    │               │
│                                           │  Processing   │               │
│                                           └───────┬───────┘               │
│                                                   │                        │
│                                                   ▼                        │
│   ┌───────────┐     ┌───────────────┐     ┌───────────────┐               │
│   │   User    │ <── │  PersonaPlex  │ <── │   Response    │               │
│   │   Hear    │     │    TTS        │     │   Format      │               │
│   └───────────┘     └───────────────┘     └───────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Voice Features

| Feature | Description |
|---------|-------------|
| **Wake Word** | Activate with "BIZRA" |
| **Agent Voices** | Unique voice per agent |
| **Continuous Mode** | Hands-free conversation |
| **Push-to-Talk** | Manual activation |
| **Voice Commands** | Special voice shortcuts |

---

## Setup

### Requirements

- **Hardware**: Microphone, speakers/headphones
- **Software**: PersonaPlex model or compatible TTS
- **GPU**: RTX 4090 recommended (or similar)

### Installation

#### Option 1: Docker (Recommended)

```bash
# Start voice server with Docker
cd ~/.bizra
docker compose up -d voice-server

# Verify it's running
docker ps | grep personaplex
```

#### Option 2: WSL

```bash
# In WSL
cd ~/.bizra/voice
./start-voice-server.sh

# Connect from Windows CLI
bizra voice connect --host localhost:8765
```

#### Option 3: Native (Advanced)

```bash
# Install dependencies
pip install nvidia-personaplex torch torchaudio

# Download voice models
bizra voice download-models

# Start server
bizra voice start-server
```

### Configuration

```yaml
# ~/.bizra/config/sovereign_profile.yaml
voice:
  enabled: true
  provider: "personaplex"

  # Model settings
  model:
    path: "~/.bizra/voice/models"
    device: "cuda"
    precision: "float16"

  # Audio settings
  audio:
    sample_rate: 24000
    input_device: "default"
    output_device: "default"
    vad_threshold: 0.5

  # Recognition settings
  recognition:
    language: "en-US"
    wake_word: "bizra"
    continuous: false
```

### First Time Setup

```bash
# Test microphone
bizra voice test-mic

# Test speakers
bizra voice test-audio

# Calibrate wake word
bizra voice calibrate

# Test full pipeline
bizra voice test
```

---

## Voice Commands

### Basic Commands

| Say | Action |
|-----|--------|
| "BIZRA" | Wake up (activate) |
| "Stop" / "Quiet" | Stop speaking |
| "Switch to developer" | Change agent |
| "What's my schedule?" | Get info |
| "Research [topic]" | Start research |

### Agent Commands

```
"BIZRA, switch to Guardian"
"BIZRA, ask the Researcher about quantum computing"
"BIZRA, have the Developer review my code"
```

### Task Commands

```
"BIZRA, add a task: Review the authentication module"
"BIZRA, what are my pending tasks?"
"BIZRA, mark task 42 complete"
```

### Quick Commands

```
"BIZRA, morning briefing"
"BIZRA, standup report"
"BIZRA, summarize this page"
```

### Control Commands

```
"Stop" - Stop current speech
"Pause" - Pause voice mode
"Continue" - Resume
"Repeat" - Repeat last response
"Slower" - Speak more slowly
"Faster" - Speak more quickly
```

---

## Agent Voices

Each PAT agent has a distinct voice personality.

### Voice Profiles

| Agent | Voice Character | Traits |
|-------|-----------------|--------|
| **Guardian** | Warm, authoritative | Calm, protective, thoughtful |
| **Strategist** | Clear, confident | Analytical, measured |
| **Researcher** | Engaged, curious | Enthusiastic about knowledge |
| **Developer** | Direct, precise | Technical, efficient |
| **Analyst** | Measured, analytical | Data-focused, clear |
| **Reviewer** | Careful, constructive | Thorough, helpful |
| **Executor** | Action-oriented | Efficient, decisive |

### Voice Configuration

```yaml
# Per-agent voice settings
pat_team:
  agents:
    guardian:
      voice:
        model: "NATF3.pt"          # Voice model file
        speed: 1.0                  # Speaking speed
        pitch: 1.0                  # Voice pitch
        style: "warm"               # Speaking style
        language: "en-US"

    researcher:
      voice:
        model: "researcher.pt"
        speed: 1.1                  # Slightly faster
        pitch: 1.0
        style: "engaging"

    developer:
      voice:
        model: "developer.pt"
        speed: 1.0
        pitch: 0.95                 # Slightly lower
        style: "technical"
```

### Switching Voices

```bash
# Via CLI
bizra voice agent guardian
/voice agent researcher

# Via voice
"BIZRA, let me talk to the Developer"
```

---

## Voice Modes

### Push-to-Talk Mode

Manual activation for each command.

```yaml
voice:
  mode: "push-to-talk"
  activation_key: "ctrl+space"
```

**Usage:**
1. Hold `Ctrl+Space`
2. Speak command
3. Release key
4. Agent responds

### Continuous Mode

Hands-free conversation.

```yaml
voice:
  mode: "continuous"
  wake_word: "bizra"
  end_phrase: "thank you"
```

**Usage:**
1. Say "BIZRA"
2. Speak naturally
3. Say "Thank you" to end

### Dictation Mode

Long-form dictation without commands.

```yaml
voice:
  mode: "dictation"
```

**Usage:**
1. `/voice dictation start`
2. Speak freely
3. `/voice dictation stop`
4. Text appears in input

### Command Mode

Only recognize commands, no conversation.

```yaml
voice:
  mode: "command"
  commands_only: true
```

---

## Configuration

### Full Voice Configuration

```yaml
voice:
  # Enable/disable
  enabled: true

  # Mode settings
  mode: "push-to-talk"        # push-to-talk | continuous | dictation | command
  wake_word: "bizra"
  end_phrase: "thank you"

  # Provider settings
  provider: "personaplex"
  model:
    id: "nvidia/personaplex-7b-v1"
    path: "~/.bizra/voice/models"
    device: "cuda:0"
    precision: "float16"
    load_in_8bit: false

  # Audio input
  input:
    device: "default"
    sample_rate: 16000
    channels: 1
    chunk_size: 1024
    vad:
      enabled: true
      threshold: 0.5
      min_speech_duration: 0.25
      min_silence_duration: 0.3

  # Audio output
  output:
    device: "default"
    sample_rate: 24000
    channels: 1
    volume: 0.8

  # Speech recognition
  recognition:
    language: "en-US"
    model: "whisper-large-v3"
    alternatives: 3

  # Text-to-speech
  synthesis:
    default_voice: "guardian"
    speed: 1.0
    pitch: 1.0
    prosody: "natural"

  # Agent voices
  agent_voices:
    guardian: "NATF3.pt"
    strategist: "strategist.pt"
    researcher: "researcher.pt"
    developer: "developer.pt"
    analyst: "analyst.pt"
    reviewer: "reviewer.pt"
    executor: "executor.pt"

  # Performance
  performance:
    streaming: true
    buffer_size: 4096
    timeout: 30
    max_concurrent: 1
```

### Audio Device Configuration

```bash
# List available devices
bizra voice list-devices

# Set input device
bizra voice set-input "Microphone (Realtek)"

# Set output device
bizra voice set-output "Speakers (Realtek)"
```

### Voice Model Management

```bash
# List available voices
bizra voice list-models

# Download voice model
bizra voice download guardian

# Install custom voice
bizra voice install my_voice.pt

# Set default voice
bizra voice set-default guardian
```

---

## Troubleshooting

### Common Issues

#### No Wake Word Detection

**Symptoms:** Saying "BIZRA" doesn't activate

**Solutions:**
```bash
# Recalibrate
bizra voice calibrate

# Check microphone
bizra voice test-mic

# Adjust sensitivity
bizra voice config --vad-threshold 0.3
```

#### Poor Recognition

**Symptoms:** Commands not understood correctly

**Solutions:**
```bash
# Check noise levels
bizra voice test-environment

# Use push-to-talk mode
bizra voice config --mode push-to-talk

# Train on your voice
bizra voice train
```

#### Audio Output Issues

**Symptoms:** Can't hear agent responses

**Solutions:**
```bash
# Test audio output
bizra voice test-audio

# Check volume
bizra voice config --volume 0.8

# List and select output device
bizra voice list-devices --output
bizra voice set-output "Device Name"
```

#### High Latency

**Symptoms:** Long delay between speaking and response

**Solutions:**
```yaml
# Reduce model precision
voice:
  model:
    precision: "float16"
    load_in_8bit: true

# Enable streaming
voice:
  performance:
    streaming: true
```

#### GPU Memory Issues

**Symptoms:** Out of memory errors

**Solutions:**
```yaml
# Reduce model size
voice:
  model:
    load_in_8bit: true

# Offload to CPU
voice:
  model:
    device: "cpu"
```

### Diagnostic Commands

```bash
# Full diagnostic
bizra voice diagnose

# Test pipeline
bizra voice test --full

# Check GPU
bizra voice gpu-status

# View logs
bizra voice logs --tail 50
```

### Status Check

```bash
bizra voice status
```

**Output:**
```
Voice Interface Status
──────────────────────

Provider: PersonaPlex
Status: Running

Audio:
  Input: Microphone (Realtek) ✓
  Output: Speakers (Realtek) ✓
  VAD: Active

Model:
  TTS: NATF3.pt (loaded)
  STT: whisper-large-v3 (loaded)
  Device: cuda:0
  Memory: 4.2GB / 16GB

Active:
  Mode: push-to-talk
  Agent: Guardian
  Speaking: No
```

---

## Voice Interface Commands

| Command | Description |
|---------|-------------|
| `/voice on` | Enable voice |
| `/voice off` | Disable voice |
| `/voice status` | Show status |
| `/voice agent <name>` | Switch agent voice |
| `/voice test` | Test pipeline |
| `/voice config` | Configure settings |
| `/voice calibrate` | Calibrate wake word |
| `/voice train` | Train on your voice |

---

## Tips

1. **Quiet environment** — Reduce background noise for better recognition
2. **Clear speech** — Speak clearly, not too fast
3. **Use wake word** — "BIZRA" at the start helps recognition
4. **Push-to-talk** — More reliable than continuous mode
5. **Agent context** — Specify agent for better routing: "BIZRA, ask Developer..."

---

## Next Steps

- [MCP Integration](08-MCP-INTEGRATION.md) — External tool integration
- [Hooks Automation](09-HOOKS-AUTOMATION.md) — Voice-triggered automation
- [Config Reference](../reference/CONFIG-REFERENCE.md) — Full configuration

---

**Your voice, your command.** 🎤
