//! Inference Gateway Fallback and Resilience Tests
//!
//! Tests for backend failover, timeout handling, and tier escalation.
//!
//! Standing on Giants: Shannon (Information Theory), Gerganov (llama.cpp)

use async_trait::async_trait;
use bizra_core::{Constitution, NodeIdentity};
use bizra_inference::{
    backends::{Backend, BackendError},
    gateway::{GatewayError, InferenceGateway, InferenceRequest, InferenceResponse},
    selector::{ModelSelector, ModelTier, TaskComplexity},
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// MOCK BACKENDS
// ============================================================================

/// Mock backend that always succeeds with configurable delay
struct MockSuccessBackend {
    name: String,
    delay_ms: u64,
    call_count: AtomicUsize,
}

impl MockSuccessBackend {
    fn new(name: &str, delay_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            delay_ms,
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Backend for MockSuccessBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;

        Ok(InferenceResponse {
            request_id: request.id.clone(),
            text: format!("Response from {}", self.name),
            model: self.name.clone(),
            tier: ModelTier::Edge,
            completion_tokens: 10,
            duration_ms: self.delay_ms,
            tokens_per_second: 0.0, // Will be calculated by gateway
        })
    }

    async fn health_check(&self) -> bool {
        true
    }
}

/// Mock backend that always fails
struct MockFailingBackend {
    name: String,
    error_message: String,
    call_count: AtomicUsize,
}

impl MockFailingBackend {
    fn new(name: &str, error_message: &str) -> Self {
        Self {
            name: name.to_string(),
            error_message: error_message.to_string(),
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Backend for MockFailingBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate(
        &self,
        _request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(BackendError::Generation(self.error_message.clone()))
    }

    async fn health_check(&self) -> bool {
        false
    }
}

/// Mock backend that times out (sleeps forever)
struct MockTimeoutBackend {
    name: String,
    call_count: AtomicUsize,
}

impl MockTimeoutBackend {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Backend for MockTimeoutBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate(
        &self,
        _request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // Sleep for a very long time to trigger timeout
        tokio::time::sleep(Duration::from_secs(3600)).await;
        unreachable!()
    }

    async fn health_check(&self) -> bool {
        true
    }
}

/// Mock backend that fails N times then succeeds
struct MockRetryBackend {
    name: String,
    fail_count: AtomicUsize,
    max_failures: usize,
    call_count: AtomicUsize,
}

impl MockRetryBackend {
    fn new(name: &str, max_failures: usize) -> Self {
        Self {
            name: name.to_string(),
            fail_count: AtomicUsize::new(0),
            max_failures,
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Backend for MockRetryBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let failures = self.fail_count.fetch_add(1, Ordering::SeqCst);

        if failures < self.max_failures {
            Err(BackendError::Connection("Transient failure".into()))
        } else {
            Ok(InferenceResponse {
                request_id: request.id.clone(),
                text: "Success after retries".into(),
                model: self.name.clone(),
                tier: ModelTier::Edge,
                completion_tokens: 10,
                duration_ms: 10,
                tokens_per_second: 1000.0,
            })
        }
    }

    async fn health_check(&self) -> bool {
        self.fail_count.load(Ordering::SeqCst) >= self.max_failures
    }
}

// ============================================================================
// BASIC GATEWAY TESTS
// ============================================================================

/// Test: Gateway returns NoBackend error when no backend registered
#[tokio::test]
async fn test_gateway_no_backend_error() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let request = InferenceRequest {
        id: "test_001".into(),
        prompt: "Hello".into(),
        complexity: TaskComplexity::Simple,
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;

    match result {
        Err(GatewayError::NoBackend(tier)) => {
            assert_eq!(tier, ModelTier::Edge);
        }
        other => panic!("Expected NoBackend error, got {:?}", other),
    }
}

/// Test: Gateway properly routes to registered backend
#[tokio::test]
async fn test_gateway_routes_to_backend() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let backend = Arc::new(MockSuccessBackend::new("test_backend", 10));
    gateway
        .register_backend(ModelTier::Edge, backend.clone())
        .await;

    let request = InferenceRequest {
        id: "test_002".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;

    assert!(result.is_ok());
    assert_eq!(backend.calls(), 1);

    let response = result.unwrap();
    assert_eq!(response.model, "test_backend");
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

/// Test: Backend errors are properly propagated
#[tokio::test]
async fn test_gateway_backend_error_propagation() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let failing = Arc::new(MockFailingBackend::new("failing", "Connection refused"));
    gateway.register_backend(ModelTier::Edge, failing).await;

    let request = InferenceRequest {
        id: "test_003".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;

    match result {
        Err(GatewayError::Backend(BackendError::Generation(msg))) => {
            assert!(msg.contains("Connection refused"));
        }
        other => panic!("Expected Backend error, got {:?}", other),
    }
}

/// Test: Timeout errors are properly handled
#[tokio::test]
async fn test_gateway_timeout_handling() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);
    // Note: Default timeout is used from crate::DEFAULT_TIMEOUT_SECS

    let timeout_backend = Arc::new(MockTimeoutBackend::new("timeout"));
    gateway
        .register_backend(ModelTier::Edge, timeout_backend.clone())
        .await;

    let request = InferenceRequest {
        id: "test_004".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    // Use a shorter timeout for testing
    let result = tokio::time::timeout(Duration::from_millis(100), gateway.infer(request)).await;

    // Either the gateway times out internally, or our test timeout triggers
    assert!(result.is_err() || matches!(result.unwrap(), Err(GatewayError::Timeout)));
}

// ============================================================================
// TIER SELECTION TESTS
// ============================================================================

/// Test: Gateway uses selector for tier determination
#[tokio::test]
async fn test_gateway_tier_selection_simple() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let edge_backend = Arc::new(MockSuccessBackend::new("edge", 10));
    let local_backend = Arc::new(MockSuccessBackend::new("local", 10));

    gateway
        .register_backend(ModelTier::Edge, edge_backend.clone())
        .await;
    gateway
        .register_backend(ModelTier::Local, local_backend.clone())
        .await;

    // Simple task should route to Edge
    let simple_request = InferenceRequest {
        id: "simple".into(),
        prompt: "2+2".into(),
        complexity: TaskComplexity::Simple,
        preferred_tier: None, // Let selector decide
        ..Default::default()
    };

    let response = gateway.infer(simple_request).await.unwrap();
    assert_eq!(response.tier, ModelTier::Edge);
    assert_eq!(edge_backend.calls(), 1);
    assert_eq!(local_backend.calls(), 0);
}

/// Test: Complex tasks route to Local tier
#[tokio::test]
async fn test_gateway_tier_selection_complex() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let edge_backend = Arc::new(MockSuccessBackend::new("edge", 10));
    let local_backend = Arc::new(MockSuccessBackend::new("local", 10));

    gateway
        .register_backend(ModelTier::Edge, edge_backend.clone())
        .await;
    gateway
        .register_backend(ModelTier::Local, local_backend.clone())
        .await;

    // Complex task should route to Local
    let complex_request = InferenceRequest {
        id: "complex".into(),
        prompt: "Explain quantum computing in detail".into(),
        complexity: TaskComplexity::Complex,
        preferred_tier: None,
        ..Default::default()
    };

    let response = gateway.infer(complex_request).await.unwrap();
    assert_eq!(response.tier, ModelTier::Local);
    assert_eq!(local_backend.calls(), 1);
}

/// Test: Preferred tier overrides selector
#[tokio::test]
async fn test_gateway_preferred_tier_override() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let edge_backend = Arc::new(MockSuccessBackend::new("edge", 10));
    let local_backend = Arc::new(MockSuccessBackend::new("local", 10));

    gateway
        .register_backend(ModelTier::Edge, edge_backend.clone())
        .await;
    gateway
        .register_backend(ModelTier::Local, local_backend.clone())
        .await;

    // Simple task but force Local tier
    let request = InferenceRequest {
        id: "forced".into(),
        prompt: "2+2".into(),
        complexity: TaskComplexity::Simple,
        preferred_tier: Some(ModelTier::Local), // Force Local
        ..Default::default()
    };

    let response = gateway.infer(request).await.unwrap();
    assert_eq!(response.tier, ModelTier::Local);
    assert_eq!(local_backend.calls(), 1);
    assert_eq!(edge_backend.calls(), 0);
}

// ============================================================================
// METRICS CALCULATION TESTS
// ============================================================================

/// Test: Duration is properly tracked
#[tokio::test]
async fn test_gateway_duration_tracking() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let backend = Arc::new(MockSuccessBackend::new("test", 50)); // 50ms delay
    gateway.register_backend(ModelTier::Edge, backend).await;

    let request = InferenceRequest {
        id: "duration_test".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let response = gateway.infer(request).await.unwrap();

    // Duration should be at least the backend delay
    assert!(
        response.duration_ms >= 50,
        "Duration {} should be >= 50ms",
        response.duration_ms
    );
}

/// Test: Tokens per second calculation
#[tokio::test]
async fn test_gateway_tokens_per_second_calculation() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let backend = Arc::new(MockSuccessBackend::new("test", 100)); // 100ms delay
    gateway.register_backend(ModelTier::Edge, backend).await;

    let request = InferenceRequest {
        id: "tps_test".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let response = gateway.infer(request).await.unwrap();

    // TPS should be calculated: tokens * 1000 / duration_ms
    // Backend returns 10 tokens, ~100ms duration
    // Expected TPS: 10 * 1000 / 100 = 100
    assert!(response.tokens_per_second > 0.0);
    assert!(
        response.tokens_per_second < 200.0,
        "TPS {} seems too high",
        response.tokens_per_second
    );
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

/// Test: Gateway handles concurrent requests
#[tokio::test]
async fn test_gateway_concurrent_requests() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = Arc::new(InferenceGateway::new(identity, constitution));

    let backend = Arc::new(MockSuccessBackend::new("concurrent", 50));
    gateway
        .register_backend(ModelTier::Edge, backend.clone())
        .await;

    // Spawn 10 concurrent requests
    let mut handles = vec![];
    for i in 0..10 {
        let gw = gateway.clone();
        let handle = tokio::spawn(async move {
            let request = InferenceRequest {
                id: format!("concurrent_{}", i),
                prompt: format!("Request {}", i),
                preferred_tier: Some(ModelTier::Edge),
                ..Default::default()
            };
            gw.infer(request).await
        });
        handles.push(handle);
    }

    // All should succeed
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(_)) = handle.await {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 10, "All concurrent requests should succeed");
    assert_eq!(backend.calls(), 10, "Backend should be called 10 times");
}

/// Test: Gateway handles mixed success/failure under concurrency
#[tokio::test]
async fn test_gateway_concurrent_mixed_results() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = Arc::new(InferenceGateway::new(identity, constitution));

    let success_backend = Arc::new(MockSuccessBackend::new("success", 10));
    let failing_backend = Arc::new(MockFailingBackend::new("failing", "Error"));

    gateway
        .register_backend(ModelTier::Edge, success_backend)
        .await;
    gateway
        .register_backend(ModelTier::Local, failing_backend)
        .await;

    let mut handles = vec![];

    // Mix of Edge (success) and Local (failure) requests
    for i in 0..20 {
        let gw = gateway.clone();
        let tier = if i % 2 == 0 {
            ModelTier::Edge
        } else {
            ModelTier::Local
        };

        let handle = tokio::spawn(async move {
            let request = InferenceRequest {
                id: format!("mixed_{}", i),
                prompt: "Test".into(),
                preferred_tier: Some(tier),
                ..Default::default()
            };
            gw.infer(request).await
        });
        handles.push((i, handle));
    }

    let mut successes = 0;
    let mut failures = 0;

    for (i, handle) in handles {
        match handle.await.unwrap() {
            Ok(_) => {
                assert!(i % 2 == 0, "Success should be Edge tier (even index)");
                successes += 1;
            }
            Err(_) => {
                assert!(i % 2 == 1, "Failure should be Local tier (odd index)");
                failures += 1;
            }
        }
    }

    assert_eq!(successes, 10);
    assert_eq!(failures, 10);
}

// ============================================================================
// SELECTOR UNIT TESTS
// ============================================================================

/// Test: Complexity estimation boundaries
#[test]
fn test_complexity_estimation_boundaries() {
    // Very short prompt
    let simple = TaskComplexity::estimate("Hi", 10);
    assert_eq!(simple, TaskComplexity::Simple);

    // Medium prompt with code indicators
    let medium = TaskComplexity::estimate("Write a function: ```python\ndef foo(): pass```", 200);
    assert!(matches!(
        medium,
        TaskComplexity::Medium | TaskComplexity::Complex
    ));
}

/// Test: Tier selection mapping
#[test]
fn test_tier_selection_mapping() {
    let selector = ModelSelector::default();

    assert_eq!(
        selector.select_tier(&TaskComplexity::Simple),
        ModelTier::Edge
    );
    assert_eq!(
        selector.select_tier(&TaskComplexity::Medium),
        ModelTier::Edge
    );
    assert_eq!(
        selector.select_tier(&TaskComplexity::Complex),
        ModelTier::Local
    );
    assert_eq!(
        selector.select_tier(&TaskComplexity::Expert),
        ModelTier::Pool
    );
}

// ============================================================================
// EDGE CASES
// ============================================================================

/// Test: Empty prompt handling
#[tokio::test]
async fn test_gateway_empty_prompt() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let backend = Arc::new(MockSuccessBackend::new("test", 10));
    gateway.register_backend(ModelTier::Edge, backend).await;

    let request = InferenceRequest {
        id: "empty".into(),
        prompt: "".into(), // Empty prompt
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    // Should still process (backend may handle empty prompt)
    let result = gateway.infer(request).await;
    assert!(result.is_ok());
}

/// Test: Very long prompt handling
#[tokio::test]
async fn test_gateway_long_prompt() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let backend = Arc::new(MockSuccessBackend::new("test", 10));
    gateway.register_backend(ModelTier::Edge, backend).await;

    let long_prompt = "word ".repeat(10000); // ~50KB prompt

    let request = InferenceRequest {
        id: "long".into(),
        prompt: long_prompt,
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;
    assert!(result.is_ok());
}

/// Test: Request with all options
#[tokio::test]
async fn test_gateway_full_request() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let backend = Arc::new(MockSuccessBackend::new("test", 10));
    gateway.register_backend(ModelTier::Local, backend).await;

    let request = InferenceRequest {
        id: "full_request".into(),
        prompt: "Explain quantum computing".into(),
        system: Some("You are a physics professor".into()),
        max_tokens: 2048,
        temperature: 0.3,
        complexity: TaskComplexity::Expert,
        preferred_tier: Some(ModelTier::Local),
    };

    let result = gateway.infer(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.request_id, "full_request");
}

/// Test: Default request values
#[test]
fn test_inference_request_defaults() {
    let default = InferenceRequest::default();

    assert!(!default.id.is_empty(), "ID should be auto-generated");
    assert!(default.prompt.is_empty());
    assert!(default.system.is_none());
    assert_eq!(default.max_tokens, 1024);
    assert!((default.temperature - 0.7).abs() < 0.001);
    assert_eq!(default.complexity, TaskComplexity::Medium);
    assert!(default.preferred_tier.is_none());
}
