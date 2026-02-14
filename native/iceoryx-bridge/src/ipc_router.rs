//! IPC Router - Message routing between components

use napi::bindgen_prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use crate::{
    BridgeConfig, BridgeStats, GateResponse, InferenceRequest, InferenceResponse,
    MessageEnvelope, GateResultDetail,
};

/// Statistics tracker
struct StatsTracker {
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    errors: AtomicU64,
    latencies: RwLock<Vec<u64>>,
}

impl StatsTracker {
    fn new() -> Self {
        Self {
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            latencies: RwLock::new(Vec::with_capacity(1000)),
        }
    }

    fn record_send(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    fn record_recv(&self) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)] // Scaffolding: used when error reporting is wired up
    fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    async fn record_latency(&self, latency_ns: u64) {
        let mut latencies = self.latencies.write().await;
        if latencies.len() >= 10000 {
            latencies.remove(0);
        }
        latencies.push(latency_ns);
    }

    async fn get_stats(&self) -> BridgeStats {
        let latencies = self.latencies.read().await;

        let avg_latency_ns = if latencies.is_empty() {
            0
        } else {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        };

        let p99_latency_ns = if latencies.is_empty() {
            0
        } else {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1)]
        };

        BridgeStats {
            messages_sent: self.messages_sent.load(Ordering::Relaxed) as i64,
            messages_received: self.messages_received.load(Ordering::Relaxed) as i64,
            avg_latency_ns: avg_latency_ns as i64,
            p99_latency_ns: p99_latency_ns as i64,
            errors: self.errors.load(Ordering::Relaxed) as i64,
            buffer_utilization: 0.0, // TODO: Calculate from actual buffer
        }
    }
}

/// Pending request tracker
#[allow(dead_code)] // Scaffolding: fields used when response routing is wired up
struct PendingRequest {
    request_id: String,
    sender: tokio::sync::oneshot::Sender<InferenceResponse>,
    start_time: std::time::Instant,
}

/// IPC Router handles message routing between components
#[allow(dead_code)] // Scaffolding: fields used when Iceoryx2 pub/sub is wired up
pub struct IpcRouter {
    config: BridgeConfig,
    stats: Arc<StatsTracker>,
    pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
    running: Arc<std::sync::atomic::AtomicBool>,
    // In a real implementation, these would be Iceoryx2 publishers/subscribers
    // For now, we simulate with channels
    inference_tx: Option<mpsc::Sender<MessageEnvelope>>,
    inference_rx: Option<mpsc::Receiver<MessageEnvelope>>,
}

impl IpcRouter {
    /// Create a new IPC router
    pub fn new(config: &BridgeConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel(config.buffer_slots as usize);

        Ok(Self {
            config: config.clone(),
            stats: Arc::new(StatsTracker::new()),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            inference_tx: Some(tx),
            inference_rx: Some(rx),
        })
    }

    /// Start the router
    pub async fn start(&mut self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);

        // In a real implementation, this would:
        // 1. Create Iceoryx2 publishers/subscribers
        // 2. Map shared memory segments
        // 3. Start background processing tasks

        tracing::info!(
            "IPC Router started with segment: {}",
            self.config.segment_name
        );

        Ok(())
    }

    /// Stop the router
    pub async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);

        // Clean up any pending requests
        let mut pending = self.pending_requests.write().await;
        pending.clear();

        tracing::info!("IPC Router stopped");

        Ok(())
    }

    /// Send an inference request and wait for response
    pub async fn send_inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();
        self.stats.record_send();

        let (tx, _rx) = tokio::sync::oneshot::channel();

        // Register pending request
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(
                request.id.clone(),
                PendingRequest {
                    request_id: request.id.clone(),
                    sender: tx,
                    start_time: start,
                },
            );
        }

        // Create and send envelope
        let envelope = MessageEnvelope::inference_request(&request, "orchestrator");

        if let Some(tx) = &self.inference_tx {
            tx.send(envelope).await
                .map_err(|_| Error::from_reason("Failed to send message"))?;
        }

        // In a real implementation, we'd wait for the actual response
        // For now, simulate a response
        let response = self.simulate_inference(&request).await;

        // Record latency
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_latency(latency_ns).await;
        self.stats.record_recv();

        Ok(response)
    }

    /// Send a gate validation request
    pub async fn send_gate_validation(&self, output_json: &str) -> Result<GateResponse> {
        let start = std::time::Instant::now();
        self.stats.record_send();

        // Parse the output
        let output: InferenceResponse = serde_json::from_str(output_json)
            .map_err(|e| Error::from_reason(format!("Invalid output JSON: {}", e)))?;

        // Simulate gate validation
        let response = self.simulate_gate_validation(&output).await;

        // Record latency
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_latency(latency_ns).await;
        self.stats.record_recv();

        Ok(response)
    }

    /// Echo test for benchmarking
    pub async fn echo_test(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let envelope = MessageEnvelope::inference_request(request, "benchmark");

        // Simulate zero-copy round trip
        let _decoded: InferenceRequest = envelope.decode()?;

        Ok(InferenceResponse {
            id: request.id.clone(),
            content: "echo".to_string(),
            model_id: request.model_id.clone(),
            tokens_generated: 1,
            generation_time_ms: 0,
            ihsan_score: 1.0,
            snr_score: 1.0,
            success: true,
            error: None,
        })
    }

    /// Get current statistics
    pub fn get_stats(&self) -> BridgeStats {
        // Block on async for sync interface
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.stats.get_stats())
        })
    }

    /// Simulate inference (for development/testing)
    async fn simulate_inference(&self, request: &InferenceRequest) -> InferenceResponse {
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        InferenceResponse {
            id: request.id.clone(),
            content: format!(
                "Simulated response to: {}... (max {} tokens)",
                &request.prompt[..request.prompt.len().min(50)],
                request.max_tokens
            ),
            model_id: request.model_id.clone(),
            tokens_generated: request.max_tokens.min(50),
            generation_time_ms: 10,
            ihsan_score: 0.97,
            snr_score: 0.92,
            success: true,
            error: None,
        }
    }

    /// Simulate gate validation (for development/testing)
    async fn simulate_gate_validation(&self, output: &InferenceResponse) -> GateResponse {
        let schema_passed = !output.content.is_empty();
        let snr_passed = output.snr_score >= 0.85;
        let ihsan_passed = output.ihsan_score >= 0.95;
        let license_passed = true; // Assume valid license

        let all_passed = schema_passed && snr_passed && ihsan_passed && license_passed;

        GateResponse {
            id: output.id.clone(),
            passed: all_passed,
            gate_results: vec![
                GateResultDetail {
                    gate: "SCHEMA".to_string(),
                    passed: schema_passed,
                    score: if schema_passed { 1.0 } else { 0.0 },
                    reason: if schema_passed { None } else { Some("Empty content".to_string()) },
                    time_ns: 100,
                },
                GateResultDetail {
                    gate: "SNR".to_string(),
                    passed: snr_passed,
                    score: output.snr_score,
                    reason: if snr_passed {
                        None
                    } else {
                        Some(format!("SNR {} < 0.85", output.snr_score))
                    },
                    time_ns: 200,
                },
                GateResultDetail {
                    gate: "IHSAN".to_string(),
                    passed: ihsan_passed,
                    score: output.ihsan_score,
                    reason: if ihsan_passed {
                        None
                    } else {
                        Some(format!("Ihsān {} < 0.95", output.ihsan_score))
                    },
                    time_ns: 500,
                },
                GateResultDetail {
                    gate: "LICENSE".to_string(),
                    passed: license_passed,
                    score: 1.0,
                    reason: None,
                    time_ns: 50,
                },
            ],
            final_score: output.ihsan_score,
            pci_signature: if all_passed {
                Some("sig:ed25519:simulated".to_string())
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_router_creation() {
        let config = BridgeConfig::default();
        let router = IpcRouter::new(&config).unwrap();
        assert!(!router.running.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_router_start_stop() {
        let config = BridgeConfig::default();
        let mut router = IpcRouter::new(&config).unwrap();

        router.start().await.unwrap();
        assert!(router.running.load(Ordering::SeqCst));

        router.stop().await.unwrap();
        assert!(!router.running.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_inference_request() {
        let config = BridgeConfig::default();
        let router = IpcRouter::new(&config).unwrap();

        let request = InferenceRequest {
            id: "test-1".to_string(),
            prompt: "Hello, world!".to_string(),
            model_id: "test-model".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        };

        let response = router.send_inference(request).await.unwrap();
        assert!(response.success);
        assert_eq!(response.id, "test-1");
    }
}
