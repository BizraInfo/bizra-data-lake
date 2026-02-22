// bizra-agent/src/ffi.rs
// ============================================================
// FFI Bridge — C-ABI for Desktop & Python Integration
// ============================================================
// Exposes the agent runtime across language boundaries:
//   Desktop (Electron/Tauri) → Rust agent runtime
//   Python engine → Rust agent runtime
//
// All functions are unsafe extern "C" with opaque handles.
// Fixed-size buffers for zero-allocation across FFI boundary.
// ============================================================

use crate::orchestrator::TaskOrchestrator;
use crate::roster::AgentRoster;
use crate::types::*;
use bizra_hooks::IhsanScore;
use bizra_memory::MemoryPipeline;

// ============================================================
// FFI TYPES
// ============================================================

/// FFI result codes
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiResult {
    Ok = 0,
    ErrNull = -1,
    ErrDegraded = -2,
    ErrVetoed = -3,
    ErrBufferTooSmall = -4,
    ErrInvalidUtf8 = -5,
    ErrAgentUnavailable = -6,
    ErrInternal = -7,
}

/// Fixed-size string buffer for FFI exchange
#[repr(C)]
pub struct FfiStringBuffer {
    pub data: [u8; 4096],
    pub len: u32,
}

impl FfiStringBuffer {
    pub fn empty() -> Self {
        Self {
            data: [0u8; 4096],
            len: 0,
        }
    }

    pub fn write(&mut self, text: &str) -> bool {
        let bytes = text.as_bytes();
        if bytes.len() > 4096 {
            let truncated = bytes.len().min(4096);
            self.data[..truncated].copy_from_slice(&bytes[..truncated]);
            self.len = truncated as u32;
            false // Truncated
        } else {
            self.data[..bytes.len()].copy_from_slice(bytes);
            self.len = bytes.len() as u32;
            true
        }
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("")
    }
}

/// FFI-compatible message input
#[repr(C)]
pub struct FfiMessage {
    pub session_id: u32,
    pub sequence: u32,
    pub content: [u8; 4096],
    pub content_len: u32,
    pub timestamp: u64,
    pub ihsan: u16,
}

impl FfiMessage {
    pub fn to_message(&self) -> Option<Message> {
        let content = core::str::from_utf8(&self.content[..self.content_len as usize]).ok()?;

        Some(Message::inbound(
            MessageId::new(self.session_id, self.sequence),
            content,
            self.timestamp,
            IhsanScore::from_raw(self.ihsan),
        ))
    }
}

/// FFI-compatible response output
#[repr(C)]
pub struct FfiResponse {
    pub content: [u8; 8192],
    pub content_len: u32,
    pub confidence: u16,
    pub context_richness: f32,
    pub agents_consulted: u8,
    pub ihsan: u16,
    pub vetoed: u8,
    pub memory_fragments_extracted: u32,
}

impl FfiResponse {
    pub fn empty() -> Self {
        Self {
            content: [0u8; 8192],
            content_len: 0,
            confidence: 0,
            context_richness: 0.0,
            agents_consulted: 0,
            ihsan: 0,
            vetoed: 0,
            memory_fragments_extracted: 0,
        }
    }

    pub fn from_result(result: &crate::orchestrator::OrchestrationResult) -> Self {
        let mut resp = Self::empty();
        let content_bytes = result.response.content.as_str().as_bytes();
        let len = content_bytes.len().min(8192);
        resp.content[..len].copy_from_slice(&content_bytes[..len]);
        resp.content_len = len as u32;
        resp.confidence = (result.response.confidence.base * 10000.0) as u16;
        resp.context_richness = result.response.context_richness;
        resp.agents_consulted = result.agents_consulted;
        resp.ihsan = result.response.ihsan_at_generation.raw();
        resp.vetoed = if result.response.vetoed { 1 } else { 0 };
        resp.memory_fragments_extracted = result.memory_fragments_extracted as u32;
        resp
    }
}

/// FFI-compatible health snapshot
#[repr(C)]
pub struct FfiHealth {
    pub messages_processed: u64,
    pub total_tasks: u64,
    pub total_vetoes: u64,
    pub agents_available: u8,
    pub agents_degraded: u8,
    pub team_health: f32,
    pub knows_me_score: f32,
    pub memory_fragments: u32,
    pub memory_insights: u32,
    pub ihsan: u16,
}

// ============================================================
// RUNTIME HANDLE — opaque pointer for FFI consumers
// ============================================================

pub struct AgentRuntimeHandle {
    pub orchestrator: TaskOrchestrator,
    pub roster: AgentRoster,
    pub pipeline: MemoryPipeline,
    pub current_ihsan: IhsanScore,
}

impl AgentRuntimeHandle {
    pub fn new(user_hash: u32, timestamp: u64) -> Self {
        Self {
            orchestrator: TaskOrchestrator::new(),
            roster: AgentRoster::new(user_hash, timestamp),
            pipeline: MemoryPipeline::new(),
            current_ihsan: IhsanScore::from_raw(9900),
        }
    }

    pub fn health(&self) -> FfiHealth {
        let roster_snap = self.roster.snapshot();
        let summary = self.pipeline.knowledge_summary();

        FfiHealth {
            messages_processed: self.orchestrator.messages_processed(),
            total_tasks: self.orchestrator.total_tasks_created(),
            total_vetoes: self.orchestrator.total_vetoes(),
            agents_available: roster_snap.agents_available,
            agents_degraded: roster_snap.agents_degraded,
            team_health: roster_snap.team_health,
            knows_me_score: self.pipeline.profile().completeness(),
            memory_fragments: summary.total_fragments,
            memory_insights: summary.total_insights,
            ihsan: self.current_ihsan.raw(),
        }
    }
}

// ============================================================
// EXPORTED C-ABI FUNCTIONS
// ============================================================

/// Create a new agent runtime
/// Returns opaque handle. Caller must destroy with bizra_agent_destroy.
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_create(
    user_hash: u32,
    timestamp: u64,
) -> *mut AgentRuntimeHandle {
    let handle = Box::new(AgentRuntimeHandle::new(user_hash, timestamp));
    Box::into_raw(handle)
}

/// Destroy an agent runtime handle
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_destroy(handle: *mut AgentRuntimeHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Process a message through the agent runtime
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_process(
    handle: *mut AgentRuntimeHandle,
    message: *const FfiMessage,
    output: *mut FfiResponse,
) -> i32 {
    if handle.is_null() || message.is_null() || output.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let runtime = &mut *handle;
    let ffi_msg = &*message;

    let msg = match ffi_msg.to_message() {
        Some(m) => m,
        None => return FfiResult::ErrInvalidUtf8 as i32,
    };

    let result = runtime.orchestrator.process_message(
        &msg,
        &mut runtime.roster,
        &mut runtime.pipeline,
        runtime.current_ihsan,
    );

    *output = FfiResponse::from_result(&result);
    FfiResult::Ok as i32
}

/// Get runtime health snapshot
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_health(
    handle: *const AgentRuntimeHandle,
    output: *mut FfiHealth,
) -> i32 {
    if handle.is_null() || output.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let runtime = &*handle;
    *output = runtime.health();
    FfiResult::Ok as i32
}

/// Update system إحسان score
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_update_ihsan(
    handle: *mut AgentRuntimeHandle,
    ihsan: u16,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let runtime = &mut *handle;
    let score = IhsanScore::from_raw(ihsan);
    runtime.current_ihsan = score;
    runtime.roster.update_ihsan_all(score);
    // Note: omega MemoryPipeline has no update_ihsan; ihsan is per-fragment at ingest
    FfiResult::Ok as i32
}

/// Get knows-me score
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_knows_me_score(handle: *const AgentRuntimeHandle) -> f32 {
    if handle.is_null() {
        return 0.0;
    }
    let runtime = &*handle;
    runtime.pipeline.profile().completeness()
}

/// Start a new conversation session
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_start_session(
    handle: *mut AgentRuntimeHandle,
    session_id: u32,
    timestamp: u64,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let _runtime = &mut *handle;
    // Note: omega MemoryPipeline has no start_session; sessions are tracked
    // at the agent runtime level, not the memory pipeline level.
    let _ = (session_id, timestamp);
    FfiResult::Ok as i32
}

/// End a conversation session
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn bizra_agent_end_session(
    handle: *mut AgentRuntimeHandle,
    session_id: u32,
    timestamp: u64,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let _runtime = &mut *handle;
    // Note: omega MemoryPipeline has no end_session; sessions are tracked
    // at the agent runtime level, not the memory pipeline level.
    let _ = (session_id, timestamp);
    FfiResult::Ok as i32
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_string_buffer_write_and_read() {
        let mut buf = FfiStringBuffer::empty();
        assert!(buf.write("Hello, BIZRA!"));
        assert_eq!(buf.as_str(), "Hello, BIZRA!");
        assert_eq!(buf.len, 13);
    }

    #[test]
    fn ffi_string_buffer_truncation() {
        let mut buf = FfiStringBuffer::empty();
        let long_string = "x".repeat(5000);
        let success = buf.write(&long_string);
        assert!(!success); // Should indicate truncation
        assert_eq!(buf.len, 4096);
    }

    #[test]
    fn ffi_message_to_message() {
        let mut ffi_msg = FfiMessage {
            session_id: 1,
            sequence: 42,
            content: [0u8; 4096],
            content_len: 0,
            timestamp: 1000,
            ihsan: 9900,
        };

        let text = b"Help me build something amazing";
        ffi_msg.content[..text.len()].copy_from_slice(text);
        ffi_msg.content_len = text.len() as u32;

        let msg = ffi_msg.to_message().unwrap();
        assert_eq!(msg.id.session_id(), 1);
        assert_eq!(msg.id.sequence(), 42);
        assert_eq!(msg.content.as_str(), "Help me build something amazing");
    }

    #[test]
    fn runtime_handle_lifecycle() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        let msg = Message::inbound(
            MessageId::new(1, 1),
            "What is Rust?",
            1000,
            IhsanScore::from_raw(9900),
        );

        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            runtime.current_ihsan,
        );

        assert!(result.guardian_approved);
        assert!(!result.response.vetoed);
    }

    #[test]
    fn runtime_health_snapshot() {
        let runtime = AgentRuntimeHandle::new(0xBEEF, 1000);
        let health = runtime.health();

        assert_eq!(health.messages_processed, 0);
        assert_eq!(health.agents_available, 7);
        assert_eq!(health.agents_degraded, 0);
        assert!((health.team_health - 1.0).abs() < 0.001);
    }

    #[test]
    fn ffi_response_from_result() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Create a document",
            1000,
            IhsanScore::from_raw(9900),
        );

        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            runtime.current_ihsan,
        );

        let ffi_resp = FfiResponse::from_result(&result);
        assert!(ffi_resp.content_len > 0);
        assert_eq!(ffi_resp.vetoed, 0);
        assert!(ffi_resp.agents_consulted >= 1);
    }
}
