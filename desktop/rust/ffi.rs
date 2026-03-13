// bizra-memory/src/ffi.rs
// ============================================================
// FFI Bridge — C-ABI boundary for Python engines
// ============================================================
// Provides C-compatible functions that Python can call via ctypes/cffi.
// Bridges the sovereign Rust memory pipeline to:
//   - VectorSearchEngine (semantic similarity)
//   - CognitiveResonance (reasoning over context)
//   - HMM (state prediction)
//
// Design:
// - All functions are `extern "C"` with `#[no_mangle]`
// - Fixed-size buffers for string exchange (no heap across boundary)
// - Error codes instead of Result types
// - Opaque pointer for pipeline state
//
// Gated by `ffi` feature flag — not compiled in pure Rust mode.
// ============================================================

use crate::pipeline::MemoryPipeline;
use crate::types::*;
use bizra_hooks::{ComponentId, IhsanScore};

// ============================================================
// FFI ERROR CODES
// ============================================================

/// FFI result codes
#[repr(i32)]
pub enum FfiResult {
    Ok = 0,
    ErrNull = -1,
    ErrFull = -2,
    ErrNotFound = -3,
    ErrDuplicate = -4,
    ErrIhsan = -5,
    ErrConfidence = -6,
    ErrBufferTooSmall = -7,
    ErrInvalidUtf8 = -8,
    ErrDegraded = -9,
}

// ============================================================
// FFI BUFFER — fixed-size string exchange
// ============================================================

pub const FFI_BUFFER_SIZE: usize = 1024;

/// Fixed-size buffer for string exchange across FFI boundary
#[repr(C)]
pub struct FfiBuffer {
    pub data: [u8; FFI_BUFFER_SIZE],
    pub len: u32,
}

impl FfiBuffer {
    pub fn empty() -> Self {
        Self {
            data: [0u8; FFI_BUFFER_SIZE],
            len: 0,
        }
    }

    pub fn from_str(s: &str) -> Self {
        let mut buf = Self::empty();
        let len = s.as_bytes().len().min(FFI_BUFFER_SIZE);
        buf.data[..len].copy_from_slice(&s.as_bytes()[..len]);
        buf.len = len as u32;
        buf
    }

    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(&self.data[..self.len as usize]).ok()
    }
}

// ============================================================
// FFI FRAGMENT — C-compatible fragment for ingestion
// ============================================================

#[repr(C)]
pub struct FfiFragment {
    pub conversation_hash: u32,
    pub sequence: u32,
    pub kind: u8,          // Maps to FragmentKind
    pub content: FfiBuffer,
    pub confidence: u16,   // 0-10000
    pub timestamp: u64,
    pub ihsan: u16,        // 0-10000
}

impl FfiFragment {
    pub fn to_memory_fragment(&self) -> Option<MemoryFragment> {
        let kind = match self.kind {
            0 => FragmentKind::Preference,
            1 => FragmentKind::Fact,
            2 => FragmentKind::Pattern,
            3 => FragmentKind::Emotion,
            4 => FragmentKind::Goal,
            5 => FragmentKind::Expertise,
            6 => FragmentKind::Relationship,
            7 => FragmentKind::Temporal,
            8 => FragmentKind::Domain,
            9 => FragmentKind::Style,
            _ => return None,
        };

        let content = self.content.as_str()?;

        Some(MemoryFragment::new(
            FragmentId::new(self.conversation_hash, self.sequence),
            kind,
            content,
            Confidence::new(self.confidence),
            ComponentId::new("ffi-bridge", "1.0"),
            self.timestamp,
            IhsanScore::new(self.ihsan),
        ))
    }
}

// ============================================================
// FFI HEALTH — C-compatible health snapshot
// ============================================================

#[repr(C)]
pub struct FfiHealth {
    pub state: u8,              // PipelineState as u8
    pub fragments_stored: u32,
    pub insights_stored: u32,
    pub profile_traits: u32,
    pub synthesis_rounds: u32,
    pub current_ihsan: u16,
    pub store_utilization: f32,
    pub knows_me_score: f32,
}

// ============================================================
// OPAQUE PIPELINE HANDLE
// ============================================================

/// Opaque handle to the memory pipeline
/// Python side holds a pointer to this
pub struct PipelineHandle {
    pub pipeline: MemoryPipeline,
}

// ============================================================
// FFI FUNCTIONS — exported when `ffi` feature is enabled
// ============================================================

/// Create a new memory pipeline, returns opaque handle
/// Caller must call `bizra_memory_destroy` to free
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_create() -> *mut PipelineHandle {
    let handle = Box::new(PipelineHandle {
        pipeline: MemoryPipeline::new(),
    });
    Box::into_raw(handle)
}

/// Destroy a memory pipeline handle
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_destroy(handle: *mut PipelineHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

/// Ingest a fragment into the pipeline
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_ingest(
    handle: *mut PipelineHandle,
    fragment: *const FfiFragment,
) -> i32 {
    if handle.is_null() || fragment.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let handle = unsafe { &mut *handle };
    let ffi_frag = unsafe { &*fragment };

    match ffi_frag.to_memory_fragment() {
        Some(mem_frag) => {
            match handle.pipeline.ingest(mem_frag, ffi_frag.timestamp) {
                Ok(_) => FfiResult::Ok as i32,
                Err(crate::store::StoreError::Full) => FfiResult::ErrFull as i32,
                Err(crate::store::StoreError::Duplicate) => FfiResult::ErrDuplicate as i32,
                Err(crate::store::StoreError::IhsanBelowThreshold) => FfiResult::ErrIhsan as i32,
                Err(crate::store::StoreError::ConfidenceBelowThreshold) => FfiResult::ErrConfidence as i32,
                Err(_) => FfiResult::ErrDegraded as i32,
            }
        }
        None => FfiResult::ErrInvalidUtf8 as i32,
    }
}

/// Run a synthesis round
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_synthesize(
    handle: *mut PipelineHandle,
    timestamp: u64,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }
    let handle = unsafe { &mut *handle };
    handle.pipeline.force_synthesis(timestamp) as i32
}

/// Query a profile trait, writes value to output buffer
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_query_trait(
    handle: *mut PipelineHandle,
    key: *const FfiBuffer,
    output: *mut FfiBuffer,
) -> i32 {
    if handle.is_null() || key.is_null() || output.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let handle = unsafe { &mut *handle };
    let key_buf = unsafe { &*key };
    let out_buf = unsafe { &mut *output };

    match key_buf.as_str() {
        Some(key_str) => {
            match handle.pipeline.query_trait(key_str) {
                Some((value, _confidence)) => {
                    *out_buf = FfiBuffer::from_str(value);
                    FfiResult::Ok as i32
                }
                None => FfiResult::ErrNotFound as i32,
            }
        }
        None => FfiResult::ErrInvalidUtf8 as i32,
    }
}

/// Get pipeline health
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_health(
    handle: *mut PipelineHandle,
    output: *mut FfiHealth,
) -> i32 {
    if handle.is_null() || output.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let handle = unsafe { &*handle };
    let health = handle.pipeline.health();

    let ffi_health = FfiHealth {
        state: health.state as u8,
        fragments_stored: health.fragments_stored as u32,
        insights_stored: health.insights_stored as u32,
        profile_traits: health.profile_traits as u32,
        synthesis_rounds: health.synthesis_rounds,
        current_ihsan: health.current_ihsan.raw(),
        store_utilization: health.store_utilization,
        knows_me_score: handle.pipeline.knows_me_score(),
    };

    unsafe {
        *output = ffi_health;
    }

    FfiResult::Ok as i32
}

/// Update إحسان score
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_update_ihsan(
    handle: *mut PipelineHandle,
    ihsan: u16,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let handle = unsafe { &mut *handle };
    handle.pipeline.update_ihsan(IhsanScore::new(ihsan));
    FfiResult::Ok as i32
}

/// Start a session
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_start_session(
    handle: *mut PipelineHandle,
    session_id: u64,
    timestamp: u64,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let handle = unsafe { &mut *handle };
    if handle.pipeline.start_session(session_id, timestamp) {
        FfiResult::Ok as i32
    } else {
        FfiResult::ErrFull as i32
    }
}

/// End a session
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_end_session(
    handle: *mut PipelineHandle,
    session_id: u64,
    timestamp: u64,
) -> i32 {
    if handle.is_null() {
        return FfiResult::ErrNull as i32;
    }

    let handle = unsafe { &mut *handle };
    handle.pipeline.end_session(session_id, timestamp);
    FfiResult::Ok as i32
}

/// Get the "knows me" score (0.0 - 1.0)
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn bizra_memory_knows_me_score(
    handle: *mut PipelineHandle,
) -> f32 {
    if handle.is_null() {
        return 0.0;
    }

    let handle = unsafe { &*handle };
    handle.pipeline.knows_me_score()
}

// ============================================================
// TESTS — FFI type sanity checks (no actual FFI calls in test)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_buffer_roundtrip() {
        let buf = FfiBuffer::from_str("Hello, BIZRA!");
        assert_eq!(buf.as_str(), Some("Hello, BIZRA!"));
        assert_eq!(buf.len, 13);
    }

    #[test]
    fn ffi_buffer_truncates_long_string() {
        let long = "x".repeat(2000);
        let buf = FfiBuffer::from_str(&long);
        assert_eq!(buf.len as usize, FFI_BUFFER_SIZE);
    }

    #[test]
    fn ffi_fragment_conversion() {
        let ffi_frag = FfiFragment {
            conversation_hash: 42,
            sequence: 1,
            kind: 1, // Fact
            content: FfiBuffer::from_str("User works at BIZRA"),
            confidence: 9000,
            timestamp: 1000,
            ihsan: 9900,
        };

        let mem_frag = ffi_frag.to_memory_fragment().unwrap();
        assert_eq!(mem_frag.kind, FragmentKind::Fact);
        assert_eq!(mem_frag.content.as_str(), "User works at BIZRA");
        assert_eq!(mem_frag.confidence.raw(), 9000);
    }

    #[test]
    fn ffi_fragment_invalid_kind_returns_none() {
        let ffi_frag = FfiFragment {
            conversation_hash: 42,
            sequence: 1,
            kind: 255, // Invalid
            content: FfiBuffer::from_str("Invalid"),
            confidence: 9000,
            timestamp: 1000,
            ihsan: 9900,
        };

        assert!(ffi_frag.to_memory_fragment().is_none());
    }

    #[test]
    fn ffi_buffer_empty() {
        let buf = FfiBuffer::empty();
        assert_eq!(buf.len, 0);
        assert_eq!(buf.as_str(), Some(""));
    }

    #[test]
    fn ffi_result_codes_are_distinct() {
        assert_ne!(FfiResult::Ok as i32, FfiResult::ErrNull as i32);
        assert_ne!(FfiResult::ErrFull as i32, FfiResult::ErrNotFound as i32);
        assert_ne!(FfiResult::ErrIhsan as i32, FfiResult::ErrConfidence as i32);
    }
}
