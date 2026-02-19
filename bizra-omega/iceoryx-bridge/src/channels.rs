//! Predefined IPC Channels for BIZRA Sovereign LLM

use napi_derive::napi;
use serde::{Deserialize, Serialize};

/// Channel definitions for the BIZRA IPC topology
#[napi]
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    /// Orchestrator → Sandbox: Inference requests
    InferenceRequest,

    /// Sandbox → Orchestrator: Inference responses
    InferenceResponse,

    /// Orchestrator → FATE Gate: Validation requests
    GateRequest,

    /// FATE Gate → Orchestrator: Validation responses
    GateResponse,

    /// Model registry updates (broadcast)
    ModelRegistry,

    /// Control channel (health, shutdown, config)
    Control,

    /// Metrics and telemetry
    Metrics,
}

impl Channel {
    /// Get the Iceoryx2 service name for this channel
    pub fn service_name(&self) -> &'static str {
        match self {
            Channel::InferenceRequest => "bizra/inference/request",
            Channel::InferenceResponse => "bizra/inference/response",
            Channel::GateRequest => "bizra/gate/request",
            Channel::GateResponse => "bizra/gate/response",
            Channel::ModelRegistry => "bizra/registry",
            Channel::Control => "bizra/control",
            Channel::Metrics => "bizra/metrics",
        }
    }

    /// Get the maximum message size for this channel
    pub fn max_message_size(&self) -> usize {
        match self {
            Channel::InferenceRequest => 64 * 1024,      // 64KB
            Channel::InferenceResponse => 1024 * 1024,   // 1MB
            Channel::GateRequest => 1024 * 1024,         // 1MB
            Channel::GateResponse => 64 * 1024,          // 64KB
            Channel::ModelRegistry => 256 * 1024,        // 256KB
            Channel::Control => 4 * 1024,                // 4KB
            Channel::Metrics => 16 * 1024,               // 16KB
        }
    }

    /// Get the buffer slot count for this channel
    pub fn buffer_slots(&self) -> usize {
        match self {
            Channel::InferenceRequest => 32,
            Channel::InferenceResponse => 32,
            Channel::GateRequest => 64,
            Channel::GateResponse => 64,
            Channel::ModelRegistry => 4,
            Channel::Control => 8,
            Channel::Metrics => 128,
        }
    }
}

/// Channel topology for the sovereign runtime
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelTopology {
    /// All active channels
    pub channels: Vec<ChannelInfo>,

    /// Total shared memory size in bytes
    pub total_memory_bytes: i64,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub name: String,
    pub service_name: String,
    pub max_message_size: i64,
    pub buffer_slots: i32,
    pub memory_bytes: i64,
}

impl ChannelTopology {
    /// Calculate the required topology for sovereign runtime
    pub fn sovereign_runtime() -> Self {
        let channels = [
            Channel::InferenceRequest,
            Channel::InferenceResponse,
            Channel::GateRequest,
            Channel::GateResponse,
            Channel::ModelRegistry,
            Channel::Control,
            Channel::Metrics,
        ];

        let channel_infos: Vec<ChannelInfo> = channels
            .iter()
            .map(|ch| {
                let memory = ch.max_message_size() * ch.buffer_slots();
                ChannelInfo {
                    name: format!("{:?}", ch),
                    service_name: ch.service_name().to_string(),
                    max_message_size: ch.max_message_size() as i64,
                    buffer_slots: ch.buffer_slots() as i32,
                    memory_bytes: memory as i64,
                }
            })
            .collect();

        let total_memory: i64 = channel_infos.iter().map(|c| c.memory_bytes).sum();

        Self {
            channels: channel_infos,
            total_memory_bytes: total_memory,
        }
    }
}

/// Get the channel topology for sovereign runtime
#[napi]
pub fn get_sovereign_topology() -> ChannelTopology {
    ChannelTopology::sovereign_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_service_names() {
        assert_eq!(
            Channel::InferenceRequest.service_name(),
            "bizra/inference/request"
        );
        assert_eq!(
            Channel::GateResponse.service_name(),
            "bizra/gate/response"
        );
    }

    #[test]
    fn test_topology_calculation() {
        let topology = ChannelTopology::sovereign_runtime();
        assert_eq!(topology.channels.len(), 7);
        assert!(topology.total_memory_bytes > 0);
    }
}
