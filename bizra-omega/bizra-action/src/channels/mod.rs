//! # Action Channels — Where Actions Become Reality
//!
//! Each channel is a capability domain. The dispatcher routes actions
//! to the appropriate channel handler. Channel handlers translate
//! BizraAction variants into real-world effects.
//!
//! In this initial crate, all channels are stubs that return mock results.
//! Each stub follows the exact interface that production implementations
//! will use, so swapping is mechanical.
//!
//! ## Channel → Bridge Mapping (Production)
//! - Ahk      → JSON-RPC over stdio to AutoHotkey v2 runtime
//! - Llm      → HTTP to LM Studio / Ollama at localhost
//! - Memory   → SQLite + FAISS/hnswlib for vector search
//! - Mcp      → MCP protocol client (JSON-RPC over stdio/HTTP)
//! - FileSystem → std::fs (sandboxed by permit scope)
//! - Browser  → httpx/reqwest with SSRF protection
//! - Response → Stdout / WebSocket to UI
//! - Telescript → Serialization + encrypted transport

use crate::types::*;

/// The result of a channel executing an action.
pub type ChannelResult = Result<ActionPayload, ChannelError>;

/// Channel execution error.
#[derive(Debug, Clone)]
pub struct ChannelError {
    pub channel: Channel,
    pub message: String,
    pub recoverable: bool,
}

impl ChannelError {
    pub fn new(channel: Channel, message: impl Into<String>, recoverable: bool) -> Self {
        Self {
            channel,
            message: message.into(),
            recoverable,
        }
    }
}

/// Trait for channel handlers. Each channel implements this.
/// `Send` is required so the Dispatcher can be shared across threads
/// (e.g., inside an `Arc<Mutex<Node>>` for the MCP transport layer).
pub trait ChannelHandler: Send {
    /// Which channel this handler serves.
    fn channel(&self) -> Channel;

    /// Execute an action. Returns payload or error.
    fn execute(&mut self, action: &BizraAction) -> ChannelResult;

    /// Is this channel currently available?
    fn is_available(&self) -> bool;

    /// Human-readable status.
    fn status(&self) -> String;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Stub Implementations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// AHK channel stub — the hands.
pub struct AhkChannel {
    available: bool,
    actions_executed: u64,
}

impl AhkChannel {
    pub fn new() -> Self {
        Self {
            available: true,
            actions_executed: 0,
        }
    }
}

impl Default for AhkChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for AhkChannel {
    fn channel(&self) -> Channel {
        Channel::Ahk
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        self.actions_executed += 1;
        match action {
            BizraAction::AhkClick {
                window,
                element_path,
            } => {
                // STUB: In production, sends JSON-RPC to AHK v2 runtime
                Ok(ActionPayload::Text(format!(
                    "STUB: Clicked [{}] in '{}'",
                    element_path, window
                )))
            }
            BizraAction::AhkType { window, text, .. } => Ok(ActionPayload::Text(format!(
                "STUB: Typed {} chars in '{}'",
                text.len(),
                window
            ))),
            BizraAction::AhkRead {
                window,
                element_path,
            } => Ok(ActionPayload::Text(format!(
                "STUB: Read from [{}] in '{}' → (mock content)",
                element_path, window
            ))),
            BizraAction::AhkReflex {
                reflex_hash,
                params,
            } => Ok(ActionPayload::Text(format!(
                "STUB: Executed reflex {:02x}{:02x}.. with {} params",
                reflex_hash[0],
                reflex_hash[1],
                params.len()
            ))),
            BizraAction::AhkLaunch { executable, args } => Ok(ActionPayload::Text(format!(
                "STUB: Launched '{}' with {} args",
                executable,
                args.len()
            ))),
            BizraAction::AhkPerceive => {
                // Return a mock UIA tree
                Ok(ActionPayload::Structured {
                    entries: vec![
                        ("window".into(), "Notepad - Untitled".into()),
                        ("process".into(), "notepad.exe".into()),
                        ("element_count".into(), "12".into()),
                        ("focus".into(), "Edit1".into()),
                    ],
                })
            }
            _ => Err(ChannelError::new(
                Channel::Ahk,
                "Action not handled by AHK channel",
                false,
            )),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        format!("AHK: {} actions executed", self.actions_executed)
    }
}

/// LLM channel stub — the brain.
pub struct LlmChannel {
    available: bool,
    queries: u64,
}

impl LlmChannel {
    pub fn new() -> Self {
        Self {
            available: true,
            queries: 0,
        }
    }
}

impl Default for LlmChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for LlmChannel {
    fn channel(&self) -> Channel {
        Channel::Llm
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        self.queries += 1;
        match action {
            BizraAction::LlmQuery {
                model, user_prompt, ..
            } => {
                // STUB: In production, HTTP POST to LM Studio at 192.168.56.1:1234
                Ok(ActionPayload::Text(format!(
                    "STUB: LLM response from '{}' to '{}...'",
                    model,
                    &user_prompt[..user_prompt.len().min(50)]
                )))
            }
            BizraAction::LlmStream { model, .. } => Ok(ActionPayload::Text(format!(
                "STUB: Stream from '{}'",
                model
            ))),
            _ => Err(ChannelError::new(Channel::Llm, "Not an LLM action", false)),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        format!("LLM: {} queries", self.queries)
    }
}

/// Memory channel stub.
pub struct MemoryChannel {
    available: bool,
    stores: u64,
    recalls: u64,
}

impl MemoryChannel {
    pub fn new() -> Self {
        Self {
            available: true,
            stores: 0,
            recalls: 0,
        }
    }
}

impl Default for MemoryChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for MemoryChannel {
    fn channel(&self) -> Channel {
        Channel::Memory
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        match action {
            BizraAction::MemoryStore { fragment_id, .. } => {
                self.stores += 1;
                Ok(ActionPayload::Text(format!(
                    "STUB: Stored '{}'",
                    fragment_id
                )))
            }
            BizraAction::MemoryRecall { top_k, .. } => {
                self.recalls += 1;
                Ok(ActionPayload::Structured {
                    entries: (0..*top_k)
                        .map(|i| {
                            (
                                format!("fragment_{}", i),
                                format!("Mock recall result {}", i),
                            )
                        })
                        .collect(),
                })
            }
            BizraAction::MemoryUpdateKnownMe { session_id, delta } => {
                Ok(ActionPayload::Text(format!(
                    "STUB: Known-Me updated by {:.4} for session '{}'",
                    delta, session_id
                )))
            }
            _ => Err(ChannelError::new(
                Channel::Memory,
                "Not a memory action",
                false,
            )),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        format!("Memory: {} stores, {} recalls", self.stores, self.recalls)
    }
}

/// File system channel stub.
pub struct FileSystemChannel {
    available: bool,
    ops: u64,
}

impl FileSystemChannel {
    pub fn new() -> Self {
        Self {
            available: true,
            ops: 0,
        }
    }
}

impl Default for FileSystemChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for FileSystemChannel {
    fn channel(&self) -> Channel {
        Channel::FileSystem
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        self.ops += 1;
        match action {
            BizraAction::FileCreate { path, content } => Ok(ActionPayload::Text(format!(
                "STUB: Created '{}' ({} bytes)",
                path,
                content.len()
            ))),
            BizraAction::FileRead { path } => Ok(ActionPayload::Text(format!(
                "STUB: Read from '{}' → (mock content)",
                path
            ))),
            BizraAction::FileDelete { path } => {
                Ok(ActionPayload::Text(format!("STUB: Deleted '{}'", path)))
            }
            _ => Err(ChannelError::new(
                Channel::FileSystem,
                "Not a file action",
                false,
            )),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        format!("FileSystem: {} ops", self.ops)
    }
}

/// Response channel stub — delivers output to the human.
pub struct ResponseChannel {
    available: bool,
    responses: u64,
}

impl ResponseChannel {
    pub fn new() -> Self {
        Self {
            available: true,
            responses: 0,
        }
    }
}

impl Default for ResponseChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for ResponseChannel {
    fn channel(&self) -> Channel {
        Channel::Response
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        self.responses += 1;
        match action {
            BizraAction::RespondToUser {
                content,
                ihsan_score,
            } => Ok(ActionPayload::Text(format!(
                "RESPONSE [Iḥsān {}]: {}",
                ihsan_score,
                &content[..content.len().min(100)]
            ))),
            _ => Err(ChannelError::new(
                Channel::Response,
                "Not a response action",
                false,
            )),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        format!("Response: {} delivered", self.responses)
    }
}

/// Browser channel stub.
pub struct BrowserChannel {
    available: bool,
}

impl BrowserChannel {
    pub fn new() -> Self {
        Self { available: true }
    }
}

impl Default for BrowserChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for BrowserChannel {
    fn channel(&self) -> Channel {
        Channel::Browser
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        match action {
            BizraAction::BrowserNavigate { url } => {
                Ok(ActionPayload::Text(format!("STUB: Navigated to '{}'", url)))
            }
            BizraAction::BrowserFetch { url, method, .. } => Ok(ActionPayload::Text(format!(
                "STUB: {} '{}' → 200 OK",
                method, url
            ))),
            _ => Err(ChannelError::new(
                Channel::Browser,
                "Not a browser action",
                false,
            )),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        "Browser: stub".into()
    }
}

/// MCP channel stub.
pub struct McpChannel {
    available: bool,
}

impl McpChannel {
    pub fn new() -> Self {
        Self { available: true }
    }
}

impl Default for McpChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for McpChannel {
    fn channel(&self) -> Channel {
        Channel::Mcp
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        match action {
            BizraAction::McpToolCall {
                tool_name, server, ..
            } => Ok(ActionPayload::Text(format!(
                "STUB: MCP {}@{} → success",
                tool_name, server
            ))),
            _ => Err(ChannelError::new(Channel::Mcp, "Not an MCP action", false)),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        "MCP: stub".into()
    }
}

/// Telescript channel stub — agent travel.
pub struct TelescriptChannel {
    available: bool,
}

impl TelescriptChannel {
    pub fn new() -> Self {
        Self { available: true }
    }
}

impl Default for TelescriptChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelHandler for TelescriptChannel {
    fn channel(&self) -> Channel {
        Channel::Telescript
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        match action {
            BizraAction::TelescriptGo {
                destination_node, ..
            } => Ok(ActionPayload::Text(format!(
                "STUB: Agent dispatched to '{}'",
                destination_node
            ))),
            BizraAction::TelescriptMeet {
                visitor_agent_id, ..
            } => Ok(ActionPayload::Text(format!(
                "STUB: Meeting with '{}'",
                visitor_agent_id
            ))),
            _ => Err(ChannelError::new(
                Channel::Telescript,
                "Not a Telescript action",
                false,
            )),
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
    fn status(&self) -> String {
        "Telescript: stub".into()
    }
}

/// Echo channel — returns a text summary of any action. Used for testing.
/// Adapted from bizra-action zip session (2026-02-24).
pub struct EchoChannel {
    chan: Channel,
    call_count: u64,
}

impl EchoChannel {
    pub fn new(chan: Channel) -> Self {
        Self { chan, call_count: 0 }
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }
}

impl Default for EchoChannel {
    fn default() -> Self {
        Self::new(Channel::Response)
    }
}

impl ChannelHandler for EchoChannel {
    fn channel(&self) -> Channel {
        self.chan
    }

    fn execute(&mut self, action: &BizraAction) -> ChannelResult {
        self.call_count += 1;
        Ok(ActionPayload::Text(format!(
            "[ECHO:{}] {}",
            self.chan.name(),
            action.summary()
        )))
    }

    fn is_available(&self) -> bool {
        true
    }

    fn status(&self) -> String {
        format!("Echo({}, calls={})", self.chan.name(), self.call_count)
    }
}
