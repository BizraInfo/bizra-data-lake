//! Inference Gateway

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::backends::{Backend, BackendError};
use crate::selector::{ModelSelector, ModelTier, TaskComplexity};
use bizra_core::{Constitution, NodeIdentity};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub system: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub complexity: TaskComplexity,
    pub preferred_tier: Option<ModelTier>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: String::new(),
            system: None,
            max_tokens: 1024,
            temperature: 0.7,
            complexity: TaskComplexity::Medium,
            preferred_tier: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub text: String,
    pub model: String,
    pub tier: ModelTier,
    pub completion_tokens: usize,
    pub duration_ms: u64,
    pub tokens_per_second: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("No backend for tier {0:?}")]
    NoBackend(ModelTier),
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("Timeout")]
    Timeout,
}

pub struct InferenceGateway {
    identity: Arc<NodeIdentity>,
    constitution: Arc<Constitution>,
    selector: ModelSelector,
    backends: RwLock<Vec<(ModelTier, Arc<dyn Backend>)>>,
    timeout: Duration,
}

impl InferenceGateway {
    pub fn new(identity: NodeIdentity, constitution: Constitution) -> Self {
        Self {
            identity: Arc::new(identity),
            constitution: Arc::new(constitution),
            selector: ModelSelector::new(),
            backends: RwLock::new(Vec::new()),
            timeout: Duration::from_secs(crate::DEFAULT_TIMEOUT_SECS),
        }
    }

    pub async fn register_backend(&self, tier: ModelTier, backend: Arc<dyn Backend>) {
        self.backends.write().await.push((tier, backend));
    }

    pub async fn infer(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, GatewayError> {
        let start = Instant::now();
        let tier = request
            .preferred_tier
            .unwrap_or_else(|| self.selector.select_tier(&request.complexity));

        let backend = self.get_backend(tier).await?;

        let result = tokio::time::timeout(self.timeout, backend.generate(&request)).await;

        match result {
            Ok(Ok(mut response)) => {
                response.duration_ms = start.elapsed().as_millis() as u64;
                response.tier = tier;
                if response.duration_ms > 0 {
                    response.tokens_per_second =
                        (response.completion_tokens as f32 * 1000.0) / response.duration_ms as f32;
                }
                Ok(response)
            }
            Ok(Err(e)) => Err(GatewayError::Backend(e)),
            Err(_) => Err(GatewayError::Timeout),
        }
    }

    async fn get_backend(&self, tier: ModelTier) -> Result<Arc<dyn Backend>, GatewayError> {
        let backends = self.backends.read().await;
        backends
            .iter()
            .find(|(t, _)| *t == tier)
            .map(|(_, b)| b.clone())
            .ok_or(GatewayError::NoBackend(tier))
    }
}
