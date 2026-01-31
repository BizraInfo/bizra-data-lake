# BIZRA VOICE: PERSONAPLEX INTEGRATION PLAN

**Model:** nvidia/personaplex-7b-v1  
**Type:** Full-duplex Speech-to-Speech  
**License:** NVIDIA Open Model License (Commercial OK)  
**Released:** 2026-01-15

---

## Why PersonaPlex?

| Feature | Traditional Stack | PersonaPlex |
|---------|-------------------|-------------|
| Architecture | Whisper → LLM → TTS | Single unified model |
| Latency | ~2-3s total | ~170ms response |
| Interruptions | Manual VAD | Native support |
| Voice cloning | Separate model | Built-in |
| Persona control | Prompt engineering | Native feature |

**PersonaPlex replaces 3 separate models with 1.**

---

## Hardware Requirements

| Component | Requirement | RTX 4090 Laptop |
|-----------|-------------|-----------------|
| VRAM | 16GB+ | 16GB ✅ |
| GPU | Ampere/Hopper | Ada Lovelace ✅ |
| Audio | 24kHz | Standard ✅ |

With `--cpu-offload`, can run on less VRAM but slower.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BIZRA VOICE STACK                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐       │
│  │   Browser    │◄──►│   PersonaPlex    │◄──►│   BIZRA      │       │
│  │   WebAudio   │    │   Server (8998)  │    │   Inference  │       │
│  └──────────────┘    └──────────────────┘    └──────────────┘       │
│                              │                       │               │
│                              ▼                       ▼               │
│                      ┌──────────────┐        ┌──────────────┐       │
│                      │ Voice Prompt │        │ Qwen 1.5B    │       │
│                      │ (NATM1.pt)   │        │ (knowledge)  │       │
│                      └──────────────┘        └──────────────┘       │
│                                                                      │
│  Mode 1: PersonaPlex standalone (voice + persona)                   │
│  Mode 2: PersonaPlex + Qwen hybrid (voice + deep reasoning)         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## BIZRA Persona Prompt

```text
You are BIZRA, a sovereign AI assistant. You help humans with الإحسان 
(excellence). You are wise, precise, and helpful. You never assume — 
you ask when uncertain. You are built on local compute, respect privacy, 
and operate with transparency. Your voice is calm and confident.
```

---

## Setup Commands

```bash
# Step 1: Run setup script
chmod +x scripts/setup_personaplex.sh
./scripts/setup_personaplex.sh

# Step 2: Accept license (browser)
# https://huggingface.co/nvidia/personaplex-7b-v1

# Step 3: Set token
export HF_TOKEN=your_token_here

# Step 4: Launch server
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"

# Step 5: Open browser
# https://localhost:8998
```

---

## Voice Options

| Voice | Type | Character |
|-------|------|-----------|
| NATM1 | Natural Male | Calm, professional |
| NATF2 | Natural Female | Warm, engaging |
| VARM0 | Variety Male | Expressive |

For BIZRA, recommend **NATM1** or **NATF2**.

---

## Sprint Update

This accelerates the timeline significantly:

| Original Plan | With PersonaPlex |
|---------------|------------------|
| Day 12-14: Voice Interface | Day 3: Voice Interface |
| Whisper + TTS + routing | Single model |
| ~2s latency | ~170ms latency |

**Voice moves from Day 12 to Day 3.**

---

## Next Steps

1. Run setup script
2. Accept HuggingFace license
3. Launch PersonaPlex server
4. Test with default assistant prompt
5. Create BIZRA persona prompt
6. Integrate with inference gateway

---

*"The voice is the bridge between machine and human. PersonaPlex makes it instant."*
