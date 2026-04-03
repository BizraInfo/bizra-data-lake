//! src/kernel/contract.rs
//! Formal inter-layer contract trait for BIZRA's 7-layer sovereign kernel.
//!
//! Provides compile-time enforcement of layer interfaces and runtime contract
//! checking at boundaries with taint validation and Ihsan scoring.
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::kernel::contract::{LayerContract, LayerInput, LayerOutput, LayerPipeline};
//! use crate::ifc::{TaintLabel, SecrecyLevel, IntegrityLevel};
//!
//! // Define a custom layer
//! struct ValidationLayer;
//!
//! impl LayerContract for ValidationLayer {
//!     fn layer_id(&self) -> u8 { 1 }
//!     fn layer_name(&self) -> &'static str { "Validation" }
//!
//!     fn verify_input(&self, input: &LayerInput) -> Result<(), LayerError> {
//!         // Check input requirements
//!         Ok(())
//!     }
//!
//!     fn execute(&self, input: LayerInput) -> Result<LayerOutput, LayerError> {
//!         // Process data with taint tracking
//!         let output_taint = TaintLabel::new(
//!             input.taint.secrecy,
//!             IntegrityLevel::Validated,
//!             input.taint.source.clone(),
//!         );
//!
//!         Ok(LayerOutput::new(
//!             serde_json::json!({"validated": true}),
//!             output_taint,
//!             50,
//!         ).with_ihsan_score(0.97))
//!     }
//! }
//!
//! // Build and execute pipeline
//! let mut pipeline = LayerPipeline::new().with_ihsan_threshold(0.95);
//! pipeline.push(Box::new(ValidationLayer))?;
//!
//! let input = LayerInput::new(
//!     serde_json::json!({"task": "process"}),
//!     TaintLabel::new(SecrecyLevel::Internal, IntegrityLevel::Untrusted, "user".into()),
//!     0,
//! );
//!
//! let output = pipeline.execute(input)?;
//! ```

use crate::ifc::{IFCViolation, TaintLabel};
use serde_json::Value as JsonValue;
use std::time::Instant;
use thiserror::Error;

/// Layer execution errors with fail-closed semantics
#[derive(Error, Debug)]
pub enum LayerError {
    #[error("Layer {layer} verification failed: {reason}")]
    VerificationFailed { layer: u8, reason: String },

    #[error("Layer {layer} execution failed: {reason}")]
    ExecutionFailed { layer: u8, reason: String },

    #[error("Taint violation: {source}")]
    TaintViolation { source: IFCViolation },

    #[error("Ihsan score {score:.3} below threshold {threshold:.3}")]
    IhsanBelowThreshold { score: f64, threshold: f64 },
}

/// Input to a layer with taint tracking
#[derive(Debug, Clone)]
pub struct LayerInput {
    pub data: JsonValue,
    pub taint: TaintLabel,
    pub source_layer: u8,
    pub request_id: Option<String>,
}

impl LayerInput {
    pub fn new(data: JsonValue, taint: TaintLabel, source_layer: u8) -> Self {
        Self {
            data,
            taint,
            source_layer,
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
}

/// Output from a layer with taint and metrics
#[derive(Debug, Clone)]
pub struct LayerOutput {
    pub data: JsonValue,
    pub taint: TaintLabel,
    pub ihsan_score: Option<f64>,
    pub processing_time_ms: u128,
}

impl LayerOutput {
    pub fn new(data: JsonValue, taint: TaintLabel, processing_time_ms: u128) -> Self {
        Self {
            data,
            taint,
            ihsan_score: None,
            processing_time_ms,
        }
    }

    pub fn with_ihsan_score(mut self, score: f64) -> Self {
        self.ihsan_score = Some(score);
        self
    }
}

/// Formal contract that all kernel layers must implement
pub trait LayerContract: Send + Sync {
    /// Layer identifier (1-7)
    fn layer_id(&self) -> u8;

    /// Human-readable layer name
    fn layer_name(&self) -> &'static str;

    /// Verify input meets this layer's requirements
    fn verify_input(&self, input: &LayerInput) -> Result<(), LayerError>;

    /// Execute layer logic on verified input
    fn execute(&self, input: LayerInput) -> Result<LayerOutput, LayerError>;

    /// Check that output taint is at least as restrictive as input taint
    fn validate_taint_flow(
        &self,
        input: &LayerInput,
        output: &LayerOutput,
    ) -> Result<(), LayerError> {
        // Default implementation: output secrecy >= input secrecy (no declassification)
        if output.taint.secrecy < input.taint.secrecy {
            return Err(LayerError::TaintViolation {
                source: IFCViolation::SecrecyViolation {
                    from_level: input.taint.secrecy,
                    to_level: output.taint.secrecy,
                    field: format!("layer_{}", self.layer_id()),
                },
            });
        }

        // Output integrity should not exceed input (can't create trust)
        if output.taint.integrity > input.taint.integrity {
            tracing::warn!(
                layer = self.layer_id(),
                name = self.layer_name(),
                input_integrity = ?input.taint.integrity,
                output_integrity = ?output.taint.integrity,
                "Layer promoted integrity level"
            );
        }

        Ok(())
    }
}

/// Pipeline that chains layers together with contract enforcement
pub struct LayerPipeline {
    layers: Vec<Box<dyn LayerContract>>,
    ihsan_threshold: f64,
}

impl LayerPipeline {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            ihsan_threshold: 0.95,
        }
    }

    pub fn with_ihsan_threshold(mut self, threshold: f64) -> Self {
        self.ihsan_threshold = threshold;
        self
    }

    /// Add a layer to the pipeline (must be in order 1-7)
    pub fn push(&mut self, layer: Box<dyn LayerContract>) -> Result<(), LayerError> {
        let new_layer_id = layer.layer_id();

        // Validate layer ordering
        if let Some(last_layer) = self.layers.last() {
            let last_id = last_layer.layer_id();
            if new_layer_id <= last_id {
                return Err(LayerError::VerificationFailed {
                    layer: new_layer_id,
                    reason: format!(
                        "Layers must be added in order: {} after {}",
                        new_layer_id, last_id
                    ),
                });
            }
        }

        self.layers.push(layer);
        Ok(())
    }

    /// Execute input through all layers in the pipeline
    pub fn execute(&self, mut input: LayerInput) -> Result<LayerOutput, LayerError> {
        let mut current_output: Option<LayerOutput> = None;

        for layer in &self.layers {
            tracing::debug!(
                layer_id = layer.layer_id(),
                layer_name = layer.layer_name(),
                source_layer = input.source_layer,
                "Executing layer"
            );

            // Verify input contract
            layer.verify_input(&input)?;

            // Execute layer
            let start = Instant::now();
            let output = layer.execute(input.clone())?;
            let elapsed = start.elapsed().as_millis();

            tracing::info!(
                layer_id = layer.layer_id(),
                processing_time_ms = elapsed,
                ihsan_score = ?output.ihsan_score,
                "Layer execution complete"
            );

            // Validate taint flow
            layer.validate_taint_flow(&input, &output)?;

            // Check Ihsan threshold if score is present
            if let Some(score) = output.ihsan_score {
                if score < self.ihsan_threshold {
                    return Err(LayerError::IhsanBelowThreshold {
                        score,
                        threshold: self.ihsan_threshold,
                    });
                }
            }

            // Prepare input for next layer
            input = LayerInput {
                data: output.data.clone(),
                taint: output.taint.clone(),
                source_layer: layer.layer_id(),
                request_id: input.request_id.clone(),
            };

            current_output = Some(output);
        }

        current_output.ok_or_else(|| LayerError::ExecutionFailed {
            layer: 0,
            reason: "Pipeline is empty".to_string(),
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Default for LayerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifc::{IntegrityLevel, SecrecyLevel};

    /// Mock layer for testing
    struct MockLayer {
        id: u8,
        name: &'static str,
        should_fail: bool,
        output_secrecy: SecrecyLevel,
        ihsan_score: Option<f64>,
    }

    impl MockLayer {
        fn new(id: u8, name: &'static str) -> Self {
            Self {
                id,
                name,
                should_fail: false,
                output_secrecy: SecrecyLevel::Internal,
                ihsan_score: Some(0.97),
            }
        }

        fn with_failure(mut self) -> Self {
            self.should_fail = true;
            self
        }

        fn with_secrecy(mut self, secrecy: SecrecyLevel) -> Self {
            self.output_secrecy = secrecy;
            self
        }

        fn with_ihsan_score(mut self, score: f64) -> Self {
            self.ihsan_score = Some(score);
            self
        }
    }

    impl LayerContract for MockLayer {
        fn layer_id(&self) -> u8 {
            self.id
        }

        fn layer_name(&self) -> &'static str {
            self.name
        }

        fn verify_input(&self, _input: &LayerInput) -> Result<(), LayerError> {
            if self.should_fail {
                return Err(LayerError::VerificationFailed {
                    layer: self.id,
                    reason: "Mock verification failure".to_string(),
                });
            }
            Ok(())
        }

        fn execute(&self, input: LayerInput) -> Result<LayerOutput, LayerError> {
            let output_taint = TaintLabel::new(
                self.output_secrecy,
                input.taint.integrity,
                input.taint.source.clone(),
            );

            let output = LayerOutput::new(
                serde_json::json!({"layer": self.id, "processed": true}),
                output_taint,
                10,
            );

            let output = if let Some(score) = self.ihsan_score {
                output.with_ihsan_score(score)
            } else {
                output
            };

            Ok(output)
        }
    }

    #[test]
    fn test_pipeline_execution_with_three_layers() {
        let mut pipeline = LayerPipeline::new();

        pipeline.push(Box::new(MockLayer::new(1, "Layer1"))).unwrap();
        pipeline.push(Box::new(MockLayer::new(2, "Layer2"))).unwrap();
        pipeline.push(Box::new(MockLayer::new(3, "Layer3"))).unwrap();

        let input = LayerInput::new(
            serde_json::json!({"test": "data"}),
            TaintLabel::new(SecrecyLevel::Internal, IntegrityLevel::Validated, "test".into()),
            0,
        );

        let result = pipeline.execute(input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.data["layer"], 3);
        assert!(output.ihsan_score.is_some());
    }

    #[test]
    fn test_taint_violation_detection() {
        let mut pipeline = LayerPipeline::new();

        // Layer that tries to declassify (Internal -> Public)
        pipeline
            .push(Box::new(MockLayer::new(1, "Declassifier").with_secrecy(SecrecyLevel::Public)))
            .unwrap();

        let input = LayerInput::new(
            serde_json::json!({"secret": "data"}),
            TaintLabel::new(SecrecyLevel::Internal, IntegrityLevel::Validated, "test".into()),
            0,
        );

        let result = pipeline.execute(input);
        assert!(result.is_err());
        assert!(matches!(result, Err(LayerError::TaintViolation { .. })));
    }

    #[test]
    fn test_ihsan_threshold_enforcement() {
        let mut pipeline = LayerPipeline::new().with_ihsan_threshold(0.95);

        // Layer with score below threshold
        pipeline
            .push(Box::new(MockLayer::new(1, "LowQuality").with_ihsan_score(0.90)))
            .unwrap();

        let input = LayerInput::new(
            serde_json::json!({"test": "data"}),
            TaintLabel::new(SecrecyLevel::Internal, IntegrityLevel::Validated, "test".into()),
            0,
        );

        let result = pipeline.execute(input);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(LayerError::IhsanBelowThreshold { .. })
        ));
    }

    #[test]
    fn test_layer_ordering_validation() {
        let mut pipeline = LayerPipeline::new();

        pipeline.push(Box::new(MockLayer::new(1, "Layer1"))).unwrap();

        // Try to add layer 3 without layer 2 (should succeed - just checks monotonic)
        let result = pipeline.push(Box::new(MockLayer::new(3, "Layer3")));
        assert!(result.is_ok());

        // Try to add layer 2 after layer 3 (should fail)
        let result = pipeline.push(Box::new(MockLayer::new(2, "Layer2")));
        assert!(result.is_err());
    }

    #[test]
    fn test_verification_failure_blocks_execution() {
        let mut pipeline = LayerPipeline::new();

        pipeline
            .push(Box::new(MockLayer::new(1, "Failing").with_failure()))
            .unwrap();

        let input = LayerInput::new(
            serde_json::json!({"test": "data"}),
            TaintLabel::new(SecrecyLevel::Internal, IntegrityLevel::Validated, "test".into()),
            0,
        );

        let result = pipeline.execute(input);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(LayerError::VerificationFailed { .. })
        ));
    }
}
