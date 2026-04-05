# 03 — Context-Aware Gate Activation (alpha_t)

**Status:** SPEC  
**New module:** `bizra-ttrl/src/context_gate.rs`  
**Depends on:** 01_engram_tiered (tier lookup), existing OmniKernel

---

## 1. Motivation

v0.90.0's EngramCache uses a simple confidence threshold to decide Hit vs Miss:

```rust
// v0.90.0: binary gate — confidence above floor → hit
Some(entry) if entry.confidence >= min_confidence => Hit
```

This ignores **context relevance**. A factual entry might have high confidence but be
irrelevant to the current reasoning context. Conversely, a lower-confidence entry might
be exactly what the current token needs.

v0.91.0 introduces alpha_t — a context-aware gating signal that measures the semantic
relevance of the retrieved Engram embedding to the current hidden state:

```
alpha_t = sigmoid( (h · e) / sqrt(d) )
```

Where:
- `h` = current hidden state vector (from the transformer)
- `e` = retrieved Engram embedding
- `d` = hidden dimension (for numerical stability)
- `sigmoid` = standard logistic function

## 2. Gate Activation Semantics

```
alpha_t ∈ [0, 1]

alpha_t > 0.7:  Strong Engram injection (entity retrieval, pattern matching)
                 Mean for entity tasks: 0.72
                 → Route to GPU-light path

alpha_t ∈ [0.3, 0.7]:  Mixed mode (composition tasks)
                        Mean for mixed tasks: 0.48
                        → Engram provides substrate, MoE reasons over it

alpha_t < 0.3:  Weak/no injection (novel reasoning, deduction)
                 Mean for reasoning tasks: 0.24
                 → Full MoE path, Engram contribution minimal
```

## 3. Data Structures

```pseudocode
struct ContextGate:
    # Learned scaling parameters (from Engram compiler / training)
    temperature: f64       # scales the dot product (default: 1.0)
    bias: f64              # additive bias before sigmoid (default: 0.0)
    
    # Telemetry
    activations: Vec<f64>  # rolling window of recent alpha_t values
    window_size: usize     # default: 1000
    
    # Task-category statistics
    category_means: HashMap<TaskCategory, RunningMean>

enum TaskCategory:
    EntityRetrieval    # expected alpha_t ~ 0.72
    PatternMatching    # expected alpha_t ~ 0.68
    Reasoning          # expected alpha_t ~ 0.24
    Deduction          # expected alpha_t ~ 0.22
    Composition        # expected alpha_t ~ 0.48
    Mixed              # catch-all

struct GateActivation:
    alpha_t: f64
    hidden_norm: f64       # ||h|| for diagnostics
    engram_norm: f64       # ||e|| for diagnostics
    dot_product: f64       # h · e before scaling
    inferred_category: TaskCategory
```

## 4. Computation

```pseudocode
fn compute_alpha_t(gate, hidden_state, engram_embedding, hidden_dim) -> GateActivation:
    """
    Compute the context-aware gating signal.
    
    hidden_state: [f64; hidden_dim] — current transformer hidden state
    engram_embedding: [f64; hidden_dim] — retrieved Engram entry's embedding
    hidden_dim: usize — dimensionality
    """
    
    # Dot product
    dot = dot_product(hidden_state, engram_embedding)
    
    # Scale by sqrt(d) for numerical stability (Vaswani attention scaling)
    scaled = (dot / sqrt(hidden_dim as f64)) * gate.temperature + gate.bias
    
    # Sigmoid activation
    alpha_t = sigmoid(scaled)
    
    # Telemetry
    gate.activations.push_rolling(alpha_t)
    
    # Infer task category from alpha_t distribution
    category = infer_category(alpha_t)
    gate.category_means.get_or_default(category).update(alpha_t)
    
    return GateActivation {
        alpha_t,
        hidden_norm: l2_norm(hidden_state),
        engram_norm: l2_norm(engram_embedding),
        dot_product: dot,
        inferred_category: category,
    }

fn sigmoid(x: f64) -> f64:
    1.0 / (1.0 + exp(-x))

fn infer_category(alpha_t: f64) -> TaskCategory:
    if alpha_t > 0.65:
        return EntityRetrieval   # high static knowledge density
    elif alpha_t > 0.55:
        return PatternMatching   # structural pattern retrieval
    elif alpha_t > 0.35:
        return Composition       # mixed static + novel
    elif alpha_t > 0.20:
        return Reasoning         # primarily novel
    else:
        return Deduction         # almost entirely novel
```

## 5. Injection Strength

The gate activation modulates *how much* of the Engram embedding is injected into
the transformer processing stream:

```pseudocode
fn apply_engram_injection(hidden_state, engram_embedding, alpha_t) -> Vec<f64>:
    """
    Weighted blend of transformer hidden state and Engram embedding.
    
    At alpha_t = 1.0: output is entirely Engram (pure retrieval)
    At alpha_t = 0.0: output is entirely hidden_state (pure reasoning)
    """
    result = []
    for i in 0..hidden_state.len():
        result.push(
            alpha_t * engram_embedding[i] + (1.0 - alpha_t) * hidden_state[i]
        )
    return result
```

### Injection Layer Selection

From the evaluation data: Engram injection is most effective at layers 2-3 of the
transformer (87% of entity pattern resolved by layer 3). Injection at later layers
wastes the attention already computed.

```pseudocode
fn should_inject_at_layer(layer_index: usize, total_layers: usize) -> bool:
    """Only inject at early layers (2-3) where static patterns are formed."""
    layer_index >= 2 and layer_index <= 3
```

## 6. Integration with OmniKernel

The context gate replaces the simple confidence check in the OmniKernel's Tier-2 path:

```pseudocode
# v0.90.0 (current):
match engram_cache.lookup(intent_bytes, min_confidence):
    Hit { value, .. } => return EngramHit(value)
    Miss => proceed to full inference

# v0.91.0 (new):
match engram_cache.lookup(intent_bytes, min_confidence):
    Hit { value, confidence, embedding } =>
        activation = context_gate.compute_alpha_t(
            current_hidden_state,
            embedding,
            hidden_dim
        )
        if activation.alpha_t >= GATE_INJECTION_THRESHOLD:
            # Strong injection — serve from Engram
            injected = apply_engram_injection(
                current_hidden_state, embedding, activation.alpha_t
            )
            return EngramHit(value, activation)
        else:
            # Weak injection — proceed to MoE but provide Engram as context
            provide_engram_context(embedding, activation.alpha_t)
            proceed to full inference with enriched context
    Miss => proceed to full inference
```

## 7. TDD Anchors

```
TEST gate_01: alpha_t = 1.0 for identical hidden_state and embedding
    SET h = [1.0, 0.0, 1.0, 0.0]
    SET e = [1.0, 0.0, 1.0, 0.0]
    COMPUTE alpha_t with temperature=1.0, bias=0.0
    ASSERT alpha_t > 0.95 (near 1.0, sigmoid saturates)

TEST gate_02: alpha_t = 0.5 for orthogonal vectors
    SET h = [1.0, 0.0, 0.0, 0.0]
    SET e = [0.0, 1.0, 0.0, 0.0]
    COMPUTE alpha_t
    ASSERT alpha_t ≈ 0.5 (dot product = 0, sigmoid(0) = 0.5)

TEST gate_03: alpha_t < 0.5 for anti-correlated vectors
    SET h = [1.0, 1.0, 1.0, 1.0]
    SET e = [-1.0, -1.0, -1.0, -1.0]
    COMPUTE alpha_t
    ASSERT alpha_t < 0.1 (negative dot product → low sigmoid)

TEST gate_04: temperature scaling amplifies gate signal
    COMPUTE alpha_t with temperature=1.0, get result_a
    COMPUTE alpha_t with temperature=2.0, same vectors, get result_b
    ASSERT |result_b - 0.5| > |result_a - 0.5| (more extreme)

TEST gate_05: injection blends correctly at alpha_t = 0.5
    SET h = [1.0, 0.0], e = [0.0, 1.0], alpha_t = 0.5
    result = apply_engram_injection(h, e, 0.5)
    ASSERT result == [0.5, 0.5]

TEST gate_06: injection at alpha_t = 0.0 returns pure hidden_state
    result = apply_engram_injection(h, e, 0.0)
    ASSERT result == h

TEST gate_07: injection at alpha_t = 1.0 returns pure engram
    result = apply_engram_injection(h, e, 1.0)
    ASSERT result == e

TEST gate_08: category inference matches expected ranges
    ASSERT infer_category(0.72) == EntityRetrieval
    ASSERT infer_category(0.48) == Composition
    ASSERT infer_category(0.24) == Reasoning

TEST gate_09: telemetry tracks rolling window correctly
    RUN 1500 activations with window_size = 1000
    ASSERT activations.len() == 1000 (oldest 500 evicted)

TEST gate_10: should_inject_at_layer only true for layers 2-3
    ASSERT should_inject_at_layer(0, 32) == false
    ASSERT should_inject_at_layer(1, 32) == false
    ASSERT should_inject_at_layer(2, 32) == true
    ASSERT should_inject_at_layer(3, 32) == true
    ASSERT should_inject_at_layer(4, 32) == false
```

## 8. Edge Cases

- **Zero-norm vectors**: If `||h|| = 0` or `||e|| = 0`, dot product is 0, alpha_t = 0.5.
  Treat as inconclusive — fall through to MoE.
- **NaN/Inf from exp overflow**: For very large dot products, `sigmoid(x)` approaches 1.0
  but `exp(-x)` may underflow. Use numerically stable sigmoid: `if x >= 0: 1/(1+exp(-x))
  else: exp(x)/(1+exp(x))`.
- **Missing embedding**: Some EngramEntries may not have embeddings (legacy entries from
  v0.90.0). Fall back to confidence-only gating for these entries.
- **Dimensionality mismatch**: If hidden_dim changes between model scales, the gate must
  normalize. The `sqrt(d)` scaling handles this naturally.
