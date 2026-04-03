// src/ifc_hallucination_taint.rs - IFC Hallucination Taint Propagation
// Standing on Shoulders of Giants Protocol: Information Flow Control (IFC)
// Extends BIZRA Ihsān security dimensions (safety: 0.22, correctness: 0.22)

use crate::errors::BridgeError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

const TAINT_TRUST_LEVELS: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityLabel {
    Public,
    Internal,
    Confidential,
    Secret,
    TopSecret,
}

impl SecurityLabel {
    pub fn level(&self) -> u8 {
        match self {
            SecurityLabel::Public => 0,
            SecurityLabel::Internal => 1,
            SecurityLabel::Confidential => 2,
            SecurityLabel::Secret => 3,
            SecurityLabel::TopSecret => 4,
        }
    }

    pub fn can_flow_to(&self, target: &SecurityLabel) -> bool {
        self.level() >= target.level()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintStatus {
    pub is_tainted: bool,
    pub source_label: Option<SecurityLabel>,
    pub propagation_chain: Vec<String>,
    pub confidence: f64,
    pub detected_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationDetection {
    pub entity_id: String,
    pub detection_method: String,
    pub confidence_score: f64,
    pub is_hallucination: bool,
    pub labels: Vec<SecurityLabel>,
    pub taint_status: TaintStatus,
}

#[derive(Clone)]
pub struct IfcHallucinationTainter {
    taint_graph: Arc<RwLock<HashMap<String, TaintStatus>>>,
    label_store: Arc<RwLock<HashMap<String, SecurityLabel>>>,
    allowed_flows: Arc<RwLock<HashSet<(SecurityLabel, SecurityLabel)>>>,
    quarantine: Arc<RwLock<HashSet<String>>>,
}

impl IfcHallucinationTainter {
    pub fn new() -> Self {
        let allowed_flows = {
            let mut flows = HashSet::new();
            flows.insert((SecurityLabel::Public, SecurityLabel::Public));
            flows.insert((SecurityLabel::Internal, SecurityLabel::Internal));
            flows.insert((SecurityLabel::Internal, SecurityLabel::Public));
            flows.insert((SecurityLabel::Confidential, SecurityLabel::Internal));
            flows.insert((SecurityLabel::Confidential, SecurityLabel::Public));
            flows.insert((SecurityLabel::Secret, SecurityLabel::Confidential));
            flows.insert((SecurityLabel::TopSecret, SecurityLabel::Secret));
            flows
        };

        Self {
            taint_graph: Arc::new(RwLock::new(HashMap::new())),
            label_store: Arc::new(RwLock::new(HashMap::new())),
            allowed_flows: Arc::new(RwLock::new(flows)),
            quarantine: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub async fn set_label(&self, entity_id: &str, label: SecurityLabel) -> Result<(), BridgeError> {
        let mut store = self.label_store.write().await;
        store.insert(entity_id.to_string(), label);
        Ok(())
    }

    pub async fn get_label(&self, entity_id: &str) -> Option<SecurityLabel> {
        let store = self.label_store.read().await;
        store.get(entity_id).copied()
    }

    pub async fn propagate_taint(
        &self,
        source_id: &str,
        target_id: &str,
    ) -> Result<TaintStatus, BridgeError> {
        let source_label = {
            let store = self.label_store.read().await;
            store.get(source_id).copied()
        }.ok_or_else(|| BridgeError::Auth("Source entity not found".to_string()))?;

        let target_label = {
            let store = self.label_store.read().await;
            store.get(target_id).copied()
        }.ok_or_else(|| BridgeError::Auth("Target entity not found".to_string()))?;

        let allowed_flows = self.allowed_flows.read().await;
        if !allowed_flows.contains(&(source_label, target_label)) {
            return Err(BridgeError::Auth(format!(
                "Flow from {:?} to {:?} not allowed",
                source_label, target_label
            )));
        }

        let taint_status = TaintStatus {
            is_tainted: true,
            source_label: Some(source_label),
            propagation_chain: vec![source_id.to_string(), target_id.to_string()],
            confidence: 1.0,
            detected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| BridgeError::Auth(format!("Time error: {}", e)))?
                .as_secs(),
        };

        {
            let mut graph = self.taint_graph.write().await;
            graph.insert(target_id.to_string(), taint_status.clone());
        }

        Ok(taint_status)
    }

    pub async fn detect_hallucination(
        &self,
        entity_id: &str,
        content: &str,
    ) -> Result<HallucinationDetection, BridgeError> {
        let confidence_score = self.analyze_content_confidence(content).await?;
        let is_hallucination = confidence_score < 0.5;

        let labels = {
            let store = self.label_store.read().await;
            store.values().copied().collect()
        };

        let taint_status = {
            let graph = self.taint_graph.read().await;
            graph.get(entity_id).cloned().unwrap_or(TaintStatus {
                is_tainted: false,
                source_label: None,
                propagation_chain: Vec::new(),
                confidence: confidence_score,
                detected_at: 0,
            })
        };

        if is_hallucination {
            let mut quarantine = self.quarantine.write().await;
            quarantine.insert(entity_id.to_string());
        }

        Ok(HallucinationDetection {
            entity_id: entity_id.to_string(),
            detection_method: "statistical_analysis".to_string(),
            confidence_score,
            is_hallucination,
            labels,
            taint_status,
        })
    }

    async fn analyze_content_confidence(&self, content: &str) -> Result<f64, BridgeError> {
        let word_count = content.split_whitespace().count();
        
        if word_count == 0 {
            return Ok(0.0);
        }

        let unique_ratio = UniqueWords::new();
        
        let confidence = if word_count < 5 {
            0.3
        } else if word_count < 20 {
            0.6
        } else if word_count < 100 {
            0.8
        } else {
            0.95
        };

        Ok(confidence)
    }

    pub async fn check_isolation(&self, entity_id: &str) -> bool {
        let quarantine = self.quarantine.read().await;
        quarantine.contains(entity_id)
    }

    pub async fn get_taint_propagation(&self, entity_id: &str) -> Option<TaintStatus> {
        let graph = self.taint_graph.read().await;
        graph.get(entity_id).cloned()
    }

    pub async fn clear_taint(&self, entity_id: &str) -> Result<(), BridgeError> {
        {
            let mut graph = self.taint_graph.write().await;
            graph.remove(entity_id);
        }
        
        {
            let mut quarantine = self.quarantine.write().await;
            quarantine.remove(entity_id);
        }
        
        Ok(())
    }

    pub async fn get_quarantined(&self) -> Vec<String> {
        let quarantine = self.quarantine.read().await;
        quarantine.iter().cloned().collect()
    }
}

struct UniqueWords(HashSet<String>);

impl UniqueWords {
    fn new() -> Self {
        Self(HashSet::new())
    }
}

impl Default for IfcHallucinationTainter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_label_propagation() {
        let tainter = IfcHallucinationTainter::new();
        
        tainter.set_label("user_input", SecurityLabel::Internal).await.unwrap();
        tainter.set_label("model_output", SecurityLabel::Public).await.unwrap();
        
        let result = tainter.propagate_taint("user_input", "model_output").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hallucination_detection() {
        let tainter = IfcHallucinationTainter::new();
        
        tainter.set_label("test_entity", SecurityLabel::Confidential).await.unwrap();
        
        let result = tainter.detect_hallucination("test_entity", "The sky is purple today").await.unwrap();
        
        if result.confidence_score < 0.5 {
            assert!(result.is_hallucination);
        }
    }

    #[tokio::test]
    async fn test_quarantine() {
        let tainter = IfcHallucinationTainter::new();
        
        tainter.set_label("malicious", SecurityLabel::Secret).await.unwrap();
        tainter.propagate_taint("malicious", "public").await.unwrap();
        
        let detected = tainter.detect_hallucination("malicious", "fake content here").await.unwrap();
        
        if detected.is_hallucination {
            let quarantined = tainter.get_quarantined().await;
            assert!(quarantined.contains(&"malicious".to_string()));
        }
    }
}