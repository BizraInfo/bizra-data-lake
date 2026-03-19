# Wire 3 — Sync→Async Inference Bridge

## Problem

`AgentRuntime::receive()` is synchronous (no tokio runtime).
`InferenceGateway::generate()` is async (requires tokio).
`Node::run()` is a synchronous stdin loop.

The node must call Ollama/LM Studio without converting the entire
Node to async. This is a boundary problem, not an architecture problem.

## Solution: Embedded Tokio Runtime

```rust
// bizra-agent/src/runtime.rs — add to AgentRuntime

use bizra_inference::{InferenceGateway, InferenceRequest, InferenceResponse};
use tokio::runtime::Runtime as TokioRuntime;

/// Lazy-initialized tokio runtime for inference calls.
/// Single-threaded — the Node is already I/O-bound on stdin.
fn inference_runtime() -> &'static TokioRuntime {
    static RT: std::sync::OnceLock<TokioRuntime> = std::sync::OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
    })
}
```

## Pseudocode: receive() with Live Inference

```rust
impl AgentRuntime {
    pub fn receive(&mut self, msg: Message, timestamp: u64) -> RuntimeResponse {
        // ── Step 1: Guard (existing) ──
        if self.state() != RuntimeState::Ready {
            return RuntimeResponse::degraded("system degraded");
        }

        // ── Step 2: Intent classification (existing) ──
        let (intent, confidence) = IntentClassifier::classify(msg.content());

        // ── Step 3: Context assembly (existing) ──
        let context = self.assembler.assemble(&msg, &self.pipeline);

        // ── Step 4: Guardian pre-check (existing) ──
        let guardian_ok = self.orchestrator.guardian_precheck(&msg, self.current_ihsan);
        if !guardian_ok {
            return RuntimeResponse::vetoed("Guardian rejected");
        }

        // ── Step 5: *** NEW — Live inference call *** ──
        let llm_response = self.call_inference(&msg, &context, intent);

        // ── Step 6: Guardian post-check on LLM output ──
        let ihsan_score = self.score_ihsan(&llm_response);
        if ihsan_score < self.config.ihsan_floor {
            // Emit ihsan.breach → subscriber #9 halts
            return RuntimeResponse::rejected(ihsan_score);
        }

        // ── Step 7: Memory extraction (existing) ──
        let extracted = self.extract_memory(&msg, timestamp);

        // ── Step 8: Build response ──
        RuntimeResponse {
            content: llm_response.text,
            is_ok: true,
            vetoed: false,
            ihsan_score,
            fragments_extracted: extracted,
            session_messages: self.session_message_count,
            context_richness: context.richness(),
            ..Default::default()
        }
    }

    /// Bridge: sync caller → async inference → sync result
    fn call_inference(
        &self,
        msg: &Message,
        context: &AgentContext,
        intent: UserIntent,
    ) -> InferenceResponse {
        let request = InferenceRequest {
            prompt: self.build_prompt(msg, context, intent),
            model: self.select_model(intent),
            max_tokens: 2048,
            temperature: 0.7,
            system: Some(self.system_prompt()),
        };

        // Block on async — this is the sync→async bridge
        match inference_runtime().block_on(self.gateway.generate(request)) {
            Ok(response) => response,
            Err(e) => {
                // Graceful degradation: return error as response
                InferenceResponse {
                    text: format!("[inference error: {e}]"),
                    tokens_used: 0,
                    model: "none".into(),
                    latency_ms: 0,
                }
            }
        }
    }
}
```

## Where gateway lives

```rust
pub struct AgentRuntime {
    // ... existing fields ...
    /// Inference gateway (lazy-initialized on first call)
    gateway: Option<InferenceGateway>,
}

impl AgentRuntime {
    fn ensure_gateway(&mut self) -> &InferenceGateway {
        if self.gateway.is_none() {
            self.gateway = Some(InferenceGateway::auto_discover());
        }
        self.gateway.as_ref().unwrap()
    }
}
```

## InferenceGateway::auto_discover()

```rust
// bizra-inference/src/gateway.rs

impl InferenceGateway {
    /// Discover available backends in priority order.
    /// 1. LM Studio (WSL gateway IP, auto-detected)
    /// 2. Ollama (localhost:11434)
    /// 3. None (offline mode — return template responses)
    pub fn auto_discover() -> Self {
        let lmstudio_host = std::env::var("LMSTUDIO_HOST")
            .unwrap_or_else(|_| detect_wsl_gateway());

        let backends = vec![];
        // Try LM Studio first
        // if reachable: push LMStudioBackend
        // Try Ollama
        // if reachable: push OllamaBackend
        // If nothing: offline mode

        InferenceGateway { backends, timeout: 120 }
    }
}
```

## Cargo.toml Change

```toml
# bizra-agent/Cargo.toml — add dependency
[dependencies]
bizra-inference = { path = "../bizra-inference" }
tokio = { version = "1", features = ["rt", "macros"] }
```

## TDD Anchors

```rust
#[cfg(test)]
mod wire3_tests {
    use super::*;

    #[test]
    fn inference_runtime_initializes_once() {
        let rt1 = inference_runtime();
        let rt2 = inference_runtime();
        assert!(std::ptr::eq(rt1, rt2));
    }

    #[test]
    fn receive_with_offline_gateway_returns_graceful_error() {
        let mut rt = AgentRuntime::for_user(0xBEEF);
        // No Ollama/LM Studio running
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "What is BIZRA?",
            1000,
            IhsanScore::from_raw(9900),
        );
        let resp = rt.receive(msg, 1000);
        // Should not panic — graceful degradation
        assert!(resp.content.contains("inference error") || resp.is_ok);
    }

    #[test]
    fn receive_with_low_ihsan_response_is_rejected() {
        // Mock: gateway returns low-quality response
        // score_ihsan() returns 0.80
        // Response should be rejected with ihsan breach
    }

    #[tokio::test]
    async fn ollama_backend_reachable() {
        // Only runs when Ollama is up
        let backend = OllamaBackend::new("http://localhost:11434");
        let req = InferenceRequest {
            prompt: "Say hello".into(),
            model: "qwen2.5:3b".into(),
            max_tokens: 32,
            temperature: 0.1,
            system: None,
        };
        let resp = backend.generate(&req).await;
        assert!(resp.is_ok());
    }
}
```

## Blast Radius

| File | Change | Risk |
|------|--------|------|
| `bizra-agent/Cargo.toml` | Add bizra-inference + tokio deps | Low — workspace already uses both |
| `bizra-agent/src/runtime.rs` | Add `call_inference()`, `ensure_gateway()` | Medium — touches receive() |
| `bizra-inference/src/gateway.rs` | Add `auto_discover()` | Low — new method |
| All existing tests | No change — template responses still work when gateway is None | Zero |
