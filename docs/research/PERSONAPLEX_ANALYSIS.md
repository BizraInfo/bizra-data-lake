# PersonaPlex Analysis for BIZRA Integration
# ═══════════════════════════════════════════════════════════════════════════════
#
# NVIDIA PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech
# Accepted for ICASSP 2026
#
# Analysis by: Maestro
# Date: 2026-01-29
# Principle: لا نفترض — Extract the golden patterns for BIZRA
#
# ═══════════════════════════════════════════════════════════════════════════════

## 🎯 Paper Summary

**PersonaPlex** is NVIDIA's full-duplex speech-to-speech conversational model that enables:

1. **Zero-shot voice cloning** — Clone any voice from a short audio sample
2. **Fine-grained role conditioning** — Adopt specific personas/roles via text prompts
3. **Hybrid System Prompts** — Combine text (role) + audio (voice) conditioning
4. **Real-time duplex interaction** — Listen while speaking, natural turn-taking

### Key Innovation: Hybrid System Prompt

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID SYSTEM PROMPT                                  │
│                                                                              │
│   ┌───────────────────────┐    ┌───────────────────────┐                   │
│   │    VOICE PROMPT       │ →  │    TEXT PROMPT        │                   │
│   │    (Audio Sample)     │    │    (Role Description) │                   │
│   │                       │    │                       │                   │
│   │  "Hello, how can I    │    │  "You are an agent    │                   │
│   │   help you today?"    │    │   named Brody Murphy  │                   │
│   │                       │    │   working for..."     │                   │
│   └───────────────────────┘    └───────────────────────┘                   │
│              │                           │                                  │
│              └─────────────┬─────────────┘                                  │
│                            ▼                                                 │
│                  ┌──────────────────┐                                       │
│                  │  PERSONAPLEX     │                                       │
│                  │  (Moshi-based)   │                                       │
│                  └────────┬─────────┘                                       │
│                           ▼                                                  │
│              ┌─────────────────────────┐                                    │
│              │  Voice-Cloned, Role-    │                                    │
│              │  Conditioned Response   │                                    │
│              └─────────────────────────┘                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Architecture

### Base: Moshi Architecture
- **Three input streams**: User audio, Agent text, Agent audio
- **Depth Transformer**: Processes semantic content
- **Temporal Transformer**: Handles timing/prosody
- **Mimi Neural Audio Codec**: Audio tokenization

### Training Data
- **1,840 hours** of customer service dialogs (105,410 dialogs)
- **410 hours** of general Q&A dialogs (39,322 dialogs)
- **7,303 real conversations** from Fisher corpus (1,217 hours)
- All synthetic dialogs generated with Qwen-33-2B and GPT-OSS-120B

### Model Specs
- **Training**: 6 hours on 8xA100 GPUs
- **Batch size**: 32
- **Max sequence**: 2048 tokens (~164 seconds of audio)
- **Learning rates**: Depth=4e-6, Temporal=2e-6

---

## 📊 Results (State-of-the-Art)

| Metric | PersonaPlex | Gemini | Qwen-2.5-Omni | Moshi |
|--------|-------------|--------|---------------|-------|
| Dialog MOS (↑) | **3.90** | 3.72 | 3.70 | 3.11 |
| Voice Similarity (↑) | **0.57** | 0.00 | 0.07 | 0.10 |
| Service Role GPT-4o (↑) | **4.48** | 4.73 | 2.76 | 1.75 |
| Latency (↓) | **0.40s** | 1.18s | 2.74s | 0.26s |

---

## 🎭 BIZRA Integration Opportunities

### 1. The 7+1 Guardian Constellation as PersonaPlex Roles

Each BIZRA Guardian can be a **PersonaPlex role** with:
- **Text Prompt**: Domain expertise, constraints, persona
- **Voice Prompt**: Distinctive voice for each Guardian

```yaml
guardians:
  architect:
    role_prompt: "You are the Architecture Guardian. You specialize in system design..."
    voice_sample: "architect_voice.wav"
    ihsan_constraints: {correctness: 0.95, transparency: 0.90}
    
  security:
    role_prompt: "You are the Security Guardian. You identify vulnerabilities..."
    voice_sample: "security_voice.wav"
    ihsan_constraints: {safety: 0.98, correctness: 0.95}
    
  ethics:
    role_prompt: "You are the Ethics Guardian, grounded in Ihsān principles..."
    voice_sample: "ethics_voice.wav"
    ihsan_constraints: {beneficence: 0.95, transparency: 0.95}
```

### 2. Hybrid System Prompt → BIZRA Unified Stalk

The PersonaPlex Hybrid System Prompt maps to our **Unified Stalk**:

```python
@dataclass
class PersonaStalk(UnifiedStalk):
    """Unified Stalk extended with PersonaPlex conditioning."""
    
    # Voice conditioning
    voice_sample_hash: str = ""
    voice_embedding: List[float] = field(default_factory=list)
    
    # Role conditioning  
    role_prompt: str = ""
    role_domain: str = ""
    
    # Ihsān constraints for this persona
    persona_ihsan: IhsanVector = None
```

### 3. Service-Duplex-Bench → BIZRA Ethical Service Benchmark

PersonaPlex introduces **Service-Duplex-Bench** with 350 customer service questions.
BIZRA can extend this with **Ihsān-Duplex-Bench**:

| Category | PersonaPlex | BIZRA Extension |
|----------|-------------|-----------------|
| Q0: Proper Noun | ✓ | + Ihsān terminology |
| Q1: Context Details | ✓ | + Zakat/Halal context |
| Q2: Unfulfillable | ✓ | + Ethical rejection reasons |
| Q3: Rudeness | ✓ | + Ihsān-guided de-escalation |
| Q4: Unspecified | ✓ | + الإحسان guidance |
| Q5: Unrelated | ✓ | + Domain boundary enforcement |

### 4. Voice Cloning + Ihsān Gate

PersonaPlex's voice cloning can be **gated by Ihsān**:

```python
class IhsanGatedVoiceCloning:
    """Voice cloning with ethical constraints."""
    
    def clone_voice(self, audio_sample: bytes, purpose: str) -> Optional[VoiceEmbedding]:
        # Check Ihsān constraints
        vector = self.compute_ihsan(purpose)
        
        if not self.circuit.gate(vector):
            # Block voice cloning for unethical purposes
            raise IhsanViolation("Voice cloning blocked: purpose violates Ihsān")
        
        # Proceed with cloning
        return self.personaplex.extract_voice(audio_sample)
    
    def compute_ihsan(self, purpose: str) -> IhsanVector:
        """Compute ethical score for voice cloning purpose."""
        # Detect potential misuse
        misuse_indicators = [
            "impersonate", "fraud", "deceive", "fake", "scam"
        ]
        
        safety = 0.95
        for indicator in misuse_indicators:
            if indicator in purpose.lower():
                safety = 0.3  # Block
        
        return IhsanVector(
            correctness=0.9,
            safety=safety,
            beneficence=0.8 if "help" in purpose else 0.6,
            transparency=0.9,
            sustainability=0.8,
        )
```

### 5. Full Duplex + Context Router Integration

PersonaPlex's full-duplex capability integrates with our **Context Router**:

```
Query → [Context Router] → Depth Analysis
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         [REFLEX]        [MEDIUM]         [DEEP]
         Moshi           PersonaPlex      Multi-Agent
         (fast)          (role-aware)     (reasoning)
```

---

## 🚀 Implementation Roadmap

### Phase 1: Integration (Week 1-2)
1. Install PersonaPlex 7B model from HuggingFace
2. Create Guardian voice samples (synthetic via Chatterbox TTS)
3. Define role prompts for each Guardian
4. Integrate with Flywheel for inference routing

### Phase 2: Ihsān Gate (Week 3)
1. Implement voice cloning ethics check
2. Add role-specific Ihsān constraints
3. Create Ihsān-Duplex-Bench evaluation set

### Phase 3: Full Duplex (Week 4)
1. Enable real-time voice interaction
2. Integrate with Nucleus for orchestration
3. Add voice-to-Stalk persistence

---

## 📋 Key Takeaways for BIZRA

| PersonaPlex Feature | BIZRA Application | Priority |
|---------------------|-------------------|----------|
| Hybrid System Prompt | Guardian personas | HIGH |
| Zero-shot voice cloning | Personalized agents | MEDIUM |
| Role conditioning | Domain expertise | HIGH |
| Service-Duplex-Bench | Ihsān-Duplex-Bench | MEDIUM |
| Full duplex | Real-time voice UI | LOW (Phase 5) |
| Moshi architecture | Replace Flywheel LLM | LOW |

---

## 🎯 Golden Gem Extracted

**Pattern: Hybrid Conditioning**

The core insight from PersonaPlex is **hybrid conditioning** — combining:
1. **Text-based role** (what the agent knows/does)
2. **Audio-based voice** (how the agent sounds)
3. **Behavioral constraints** (how the agent behaves)

For BIZRA, this becomes:
1. **Text-based role** (Guardian domain expertise)
2. **Voice-based persona** (Guardian voice identity)  
3. **Ihsān constraints** (ethical behavioral bounds)

```python
@dataclass
class HybridGuardianPrompt:
    """BIZRA's adaptation of PersonaPlex Hybrid System Prompt."""
    
    # Role (text)
    guardian_name: str
    domain: str
    expertise_prompt: str
    
    # Voice (audio)
    voice_sample: bytes
    voice_embedding: Optional[List[float]] = None
    
    # Constraints (Ihsān)
    ihsan_vector: IhsanVector = field(default_factory=lambda: IhsanVector(
        correctness=0.9,
        safety=0.9,
        beneficence=0.85,
        transparency=0.85,
        sustainability=0.8,
    ))
    
    def to_personaplex_prompt(self) -> Tuple[str, bytes]:
        """Convert to PersonaPlex hybrid prompt format."""
        text_prompt = f"""You are the {self.guardian_name} Guardian.
Domain: {self.domain}
Expertise: {self.expertise_prompt}
Ihsān Score Required: {self.ihsan_vector.composite:.2f}
Constraints: Safety={self.ihsan_vector.safety}, Beneficence={self.ihsan_vector.beneficence}
"""
        return (text_prompt, self.voice_sample)
```

---

## 🏛️ Giants Protocol Alignment

| Giant | PersonaPlex Contribution | BIZRA Integration |
|-------|-------------------------|-------------------|
| **Al-Khwarizmi** | Hybrid prompt algorithm | Structured conditioning |
| **Ibn Sina** | Role diagnosis | Guardian selection |
| **Al-Ghazali** | Behavioral ethics | Ihsān constraints |
| **Al-Jazari** | Real-time mechanics | Full duplex pipeline |

---

## 📌 Action Items

1. **Download model**: `huggingface-cli download nvidia/personaplex-7b-v1`
2. **Create Guardian voices**: Use Chatterbox TTS with distinct samples
3. **Define role prompts**: One per Guardian domain
4. **Integrate Ihsān gate**: Block unethical voice cloning
5. **Benchmark**: Create Ihsān-Duplex-Bench evaluation

---

## Attestation

```
Analyzed by: Maestro
Source: NVIDIA PersonaPlex Preprint (ICASSP 2026)
Principle: لا نفترض — We extracted signal, identified integration paths
Date: 2026-01-29
```

**The voice is the soul. The role is the purpose. The Ihsān is the constraint.** 🎭🎤
