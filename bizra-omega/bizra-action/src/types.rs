//! # BIZRA Action Types — The DNA of Every Action
//!
//! Actions are commands that change the world. Unlike events (observations),
//! actions have consequences. Every type here represents something that will
//! happen in reality: a click, an API call, a file write, a response to a human.
//!
//! ## Standing on Giants
//! - **Lamport (1978)**: Each action carries a monotonic timestamp for happens-before ordering
//! - **Shannon (1948)**: Channel typing maximizes routing efficiency (no wasted dispatch)
//! - **Al-Ghazali**: Every action type includes Iḥsān metadata for constitutional gating

use core::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Iḥsān Score (local definition for zero-dependency sovereignty)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// إحسان score — the Lyapunov certificate of quality.
/// Range: 0.0 (worst) to 1.0 (perfect).
/// Constitutional threshold: 0.95 for production actions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IhsanScore(f64);

impl IhsanScore {
    /// Floor: no action may execute with Iḥsān below this value.
    pub const PRODUCTION_FLOOR: f64 = 0.95;

    /// Create a new score, clamped to [0.0, 1.0].
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Raw value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Does this score meet the production constitutional threshold?
    pub fn meets_constitutional(&self) -> bool {
        self.0 >= Self::PRODUCTION_FLOOR
    }

    /// Margin above (or below) the constitutional floor.
    pub fn margin(&self) -> f64 {
        self.0 - Self::PRODUCTION_FLOOR
    }
}

impl fmt::Display for IhsanScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.4} {}",
            self.0,
            if self.meets_constitutional() {
                "✓"
            } else {
                "✗"
            }
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Action Identity
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Unique identifier for an action (monotonic counter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub u64);

/// Monotonic timestamp (nanoseconds since node boot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActionTimestamp(pub u64);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Channel — Where actions are routed
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The execution channel for an action.
/// Each channel represents a distinct capability domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    /// AHK Desktop Automation — the HANDS.
    /// UIA perception, window manipulation, keyboard/mouse actuation.
    /// This is the moat. Nobody else has this.
    Ahk,

    /// LLM Inference — the reasoning engine.
    /// Calls to local models via LM Studio / Ollama / vLLM.
    Llm,

    /// Memory — persistence and recall.
    /// Store fragments, retrieve by embedding similarity, update Known-Me.
    Memory,

    /// MCP Tool Use — external tool integration.
    /// Structured tool calls via Model Context Protocol.
    Mcp,

    /// File System — read, write, create, delete.
    /// All file operations gated by permit scope.
    FileSystem,

    /// Browser — web navigation and fetching.
    /// URL validation, SSRF protection (already implemented in security audit).
    Browser,

    /// Response — deliver output to the human.
    /// The final channel: content + Iḥsān score + signed receipt.
    Response,

    /// Telescript — agent travel to remote nodes.
    /// Serialize agent state, transport, execute at destination, return.
    Telescript,
}

impl Channel {
    /// Risk level of this channel. Higher risk = stricter Guardian scrutiny.
    pub fn risk_level(&self) -> RiskLevel {
        match self {
            Channel::Memory => RiskLevel::Low,        // Read/write own data
            Channel::Llm => RiskLevel::Low,           // Inference, no side effects
            Channel::Response => RiskLevel::Low,      // Deliver to own user
            Channel::FileSystem => RiskLevel::Medium, // Touches disk
            Channel::Browser => RiskLevel::Medium,    // Network access
            Channel::Mcp => RiskLevel::Medium,        // External tool
            Channel::Ahk => RiskLevel::High,          // Manipulates desktop
            Channel::Telescript => RiskLevel::High,   // Agent leaves node
        }
    }

    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            Channel::Ahk => "AHK Desktop",
            Channel::Llm => "LLM Inference",
            Channel::Memory => "Memory",
            Channel::Mcp => "MCP Tool",
            Channel::FileSystem => "File System",
            Channel::Browser => "Browser",
            Channel::Response => "User Response",
            Channel::Telescript => "Telescript Travel",
        }
    }
}

/// Risk classification for Guardian gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Safe: information-only, no external side effects.
    Low,
    /// Moderate: touches disk, network, or external tools.
    Medium,
    /// High: manipulates desktop or sends agent to foreign node.
    /// Requires explicit permit + higher Iḥsān threshold.
    High,
}

impl RiskLevel {
    /// Minimum Iḥsān score required for this risk level.
    pub fn min_ihsan(&self) -> f64 {
        match self {
            RiskLevel::Low => 0.90,
            RiskLevel::Medium => 0.95,
            RiskLevel::High => 0.98,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Constitutional Permit
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A permit constraining what an action may do.
/// Telescript's "capability" model applied to all channels.
#[derive(Debug, Clone)]
pub struct Permit {
    /// Which channels this permit authorizes.
    pub allowed_channels: u8, // Bitfield: bit 0 = Ahk, bit 1 = Llm, etc.

    /// Maximum resource budget (compute units).
    pub resource_limit: u64,

    /// File system scope (paths the action may touch).
    /// Empty = no file access. "*" = full access. Prefixes for scoping.
    pub fs_scope: Vec<String>,

    /// Time-to-live in seconds. After this, permit expires.
    pub ttl_seconds: u64,

    /// Whether this permit allows network egress.
    pub allow_network: bool,

    /// Whether this permit allows desktop manipulation.
    pub allow_desktop: bool,

    /// Human-in-the-loop required for this action?
    pub requires_hitl: bool,
}

impl Permit {
    /// Default permit for local user actions — broad but not unlimited.
    pub fn user_default() -> Self {
        Self {
            allowed_channels: 0xFF, // All channels
            resource_limit: 1_000_000,
            fs_scope: vec!["~".into()], // User home directory
            ttl_seconds: 3600,          // 1 hour
            allow_network: true,
            allow_desktop: true,
            requires_hitl: false,
        }
    }

    /// Restrictive permit for visiting agents (Telescript).
    pub fn visitor(fs_scope: Vec<String>, ttl_seconds: u64) -> Self {
        Self {
            allowed_channels: 0b0000_0110, // Only LLM + Memory
            resource_limit: 100_000,
            fs_scope,
            ttl_seconds,
            allow_network: false,
            allow_desktop: false,
            requires_hitl: true,
        }
    }

    /// Check if a channel is permitted.
    pub fn allows_channel(&self, channel: &Channel) -> bool {
        let bit = match channel {
            Channel::Ahk => 0,
            Channel::Llm => 1,
            Channel::Memory => 2,
            Channel::Mcp => 3,
            Channel::FileSystem => 4,
            Channel::Browser => 5,
            Channel::Response => 6,
            Channel::Telescript => 7,
        };
        (self.allowed_channels >> bit) & 1 == 1
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BizraAction — The Action Enum
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Every possible action the node can take.
/// Each variant maps to exactly one Channel.
#[derive(Debug, Clone)]
pub enum BizraAction {
    // ── AHK Channel (The Hands) ────────────────────────
    /// Click a UI element identified by UIA path.
    AhkClick {
        window: String,
        element_path: String,
    },

    /// Type text into a UI element.
    AhkType {
        window: String,
        element_path: String,
        text: String,
    },

    /// Read text from a UI element.
    AhkRead {
        window: String,
        element_path: String,
    },

    /// Execute a compiled reflex script (cached AHK automation).
    AhkReflex {
        reflex_hash: [u8; 32], // BLAKE3 hash of the reflex
        params: Vec<String>,
    },

    /// Launch an application.
    AhkLaunch {
        executable: String,
        args: Vec<String>,
    },

    /// Capture the UIA tree of the foreground window (Perception).
    AhkPerceive,

    // ── LLM Channel (The Brain) ────────────────────────
    /// Send a prompt to a local model and get a response.
    LlmQuery {
        provider: String,
        model: String,
        system_prompt: String,
        user_prompt: String,
        max_tokens: u32,
        temperature: f32,
    },

    /// Stream a response from a local model.
    LlmStream {
        provider: String,
        model: String,
        system_prompt: String,
        user_prompt: String,
        max_tokens: u32,
    },

    // ── Memory Channel (Persistence) ───────────────────
    /// Store a knowledge fragment.
    MemoryStore {
        fragment_id: String,
        content: String,
        embedding: Vec<f32>,
        metadata: Vec<(String, String)>,
    },

    /// Recall fragments by semantic similarity.
    MemoryRecall {
        query: String,
        query_embedding: Vec<f32>,
        top_k: usize,
        min_similarity: f32,
    },

    /// Update the Known-Me score after a session.
    MemoryUpdateKnownMe { session_id: String, delta: f64 },

    // ── MCP Channel (Tool Use) ─────────────────────────
    /// Call an MCP tool.
    McpToolCall {
        server: String,
        tool_name: String,
        arguments: String, // JSON string
    },

    // ── File System Channel ────────────────────────────
    /// Create a file.
    FileCreate { path: String, content: Vec<u8> },

    /// Read a file.
    FileRead { path: String },

    /// Delete a file.
    FileDelete { path: String },

    // ── Browser Channel ────────────────────────────────
    /// Navigate to a URL (SSRF-validated).
    BrowserNavigate { url: String },

    /// Fetch content from a URL.
    BrowserFetch {
        url: String,
        method: String,
        headers: Vec<(String, String)>,
    },

    // ── Response Channel (Back to Human) ───────────────
    /// Deliver a response to the user with constitutional receipt.
    RespondToUser {
        content: String,
        ihsan_score: IhsanScore,
    },

    // ── Telescript Channel (Agent Travel) ──────────────
    /// Send an agent to a remote node.
    TelescriptGo {
        destination_node: String,
        agent_state: Vec<u8>, // Serialized agent
        permit: Permit,
    },

    /// Meet with a visiting agent.
    TelescriptMeet {
        visitor_agent_id: String,
        meeting_context: String,
    },
}

impl BizraAction {
    /// Which channel handles this action?
    pub fn channel(&self) -> Channel {
        match self {
            Self::AhkClick { .. }
            | Self::AhkType { .. }
            | Self::AhkRead { .. }
            | Self::AhkReflex { .. }
            | Self::AhkLaunch { .. }
            | Self::AhkPerceive => Channel::Ahk,

            Self::LlmQuery { .. } | Self::LlmStream { .. } => Channel::Llm,

            Self::MemoryStore { .. }
            | Self::MemoryRecall { .. }
            | Self::MemoryUpdateKnownMe { .. } => Channel::Memory,

            Self::McpToolCall { .. } => Channel::Mcp,

            Self::FileCreate { .. } | Self::FileRead { .. } | Self::FileDelete { .. } => {
                Channel::FileSystem
            }

            Self::BrowserNavigate { .. } | Self::BrowserFetch { .. } => Channel::Browser,

            Self::RespondToUser { .. } => Channel::Response,

            Self::TelescriptGo { .. } | Self::TelescriptMeet { .. } => Channel::Telescript,
        }
    }

    /// Human-readable summary for logging and receipts.
    pub fn summary(&self) -> String {
        match self {
            Self::AhkClick {
                window,
                element_path,
            } => format!("Click [{element_path}] in '{window}'"),
            Self::AhkType { window, text, .. } => {
                format!("Type {} chars in '{}'", text.len(), window)
            }
            Self::AhkRead {
                window,
                element_path,
            } => format!("Read [{element_path}] from '{window}'"),
            Self::AhkReflex { reflex_hash, .. } => format!(
                "Execute reflex {:02x}{:02x}...",
                reflex_hash[0], reflex_hash[1]
            ),
            Self::AhkLaunch { executable, .. } => format!("Launch '{executable}'"),
            Self::AhkPerceive => "Capture UIA tree (perception)".into(),
            Self::LlmQuery { model, .. } => format!("Query model '{model}'"),
            Self::LlmStream { model, .. } => format!("Stream from '{model}'"),
            Self::MemoryStore { fragment_id, .. } => format!("Store fragment '{fragment_id}'"),
            Self::MemoryRecall { top_k, .. } => format!("Recall top-{top_k} fragments"),
            Self::MemoryUpdateKnownMe { delta, .. } => format!("Update Known-Me by {delta:.4}"),
            Self::McpToolCall {
                tool_name, server, ..
            } => format!("MCP: {tool_name}@{server}"),
            Self::FileCreate { path, content } => {
                format!("Create '{}' ({} bytes)", path, content.len())
            }
            Self::FileRead { path } => format!("Read '{path}'"),
            Self::FileDelete { path } => format!("Delete '{path}'"),
            Self::BrowserNavigate { url } => format!("Navigate to '{url}'"),
            Self::BrowserFetch { url, method, .. } => format!("{method} '{url}'"),
            Self::RespondToUser { ihsan_score, .. } => {
                format!("Respond to user (Iḥsān: {ihsan_score})")
            }
            Self::TelescriptGo {
                destination_node, ..
            } => format!("Agent GO → '{destination_node}'"),
            Self::TelescriptMeet {
                visitor_agent_id, ..
            } => format!("MEET agent '{visitor_agent_id}'"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ActionResult — What comes back
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The result of executing an action.
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// The action that was executed.
    pub action_id: ActionId,

    /// Whether the action succeeded.
    pub success: bool,

    /// The output payload (if any).
    pub payload: ActionPayload,

    /// Time taken to execute (nanoseconds).
    pub duration_ns: u64,

    /// Iḥsān score of the output.
    pub ihsan_score: IhsanScore,

    /// Constitutional receipt hash (BLAKE3).
    pub receipt_hash: [u8; 32],
}

/// Payload returned by an action.
#[derive(Debug, Clone)]
pub enum ActionPayload {
    /// No data returned.
    Empty,

    /// Text content (LLM response, file content, AHK read result).
    Text(String),

    /// Binary content (file bytes, serialized agent state).
    Bytes(Vec<u8>),

    /// Structured data (UIA tree, memory recall results).
    Structured { entries: Vec<(String, String)> },

    /// Error message.
    Error(String),
}

impl ActionPayload {
    /// Is this an error?
    pub fn is_error(&self) -> bool {
        matches!(self, ActionPayload::Error(_))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Guardian Verdict
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The Guardian's decision on whether an action may execute.
#[derive(Debug, Clone)]
pub enum GuardianVerdict {
    /// Action approved. Proceed.
    Approved { reason: &'static str },

    /// Action denied. Do not execute.
    Denied {
        reason: String,
        violation: GuardianViolation,
    },

    /// Action requires human confirmation before proceeding.
    RequiresHitl {
        reason: String,
        action_summary: String,
    },
}

impl GuardianVerdict {
    pub fn is_approved(&self) -> bool {
        matches!(self, GuardianVerdict::Approved { .. })
    }
}

/// Why the Guardian denied an action.
#[derive(Debug, Clone)]
pub enum GuardianViolation {
    /// Iḥsān score below threshold for this risk level.
    IhsanBelowThreshold { score: f64, required: f64 },

    /// Channel not permitted by current permit.
    ChannelNotPermitted { channel: Channel },

    /// File path outside permitted scope.
    PathOutOfScope { path: String, scope: Vec<String> },

    /// Permit expired.
    PermitExpired { ttl: u64, elapsed: u64 },

    /// Desktop manipulation not permitted.
    DesktopNotPermitted,

    /// Network egress not permitted.
    NetworkNotPermitted,

    /// Resource budget exceeded.
    ResourceExceeded { used: u64, limit: u64 },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Constitutional Receipt
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A signed, verifiable receipt for every action.
/// This is the "Third Fact" — immutable proof of what happened.
#[derive(Debug, Clone)]
pub struct ConstitutionalReceipt {
    /// Unique action ID.
    pub action_id: ActionId,

    /// Monotonic timestamp.
    pub timestamp: ActionTimestamp,

    /// BLAKE3 hash of: channel || action_summary || payload || ihsan_score
    pub content_hash: [u8; 32],

    /// Iḥsān score at time of execution.
    pub ihsan_score: IhsanScore,

    /// Guardian verdict.
    pub verdict: GuardianVerdict,

    /// Channel that executed the action.
    pub channel: Channel,

    /// Action summary (human-readable).
    pub action_summary: String,

    /// Ed25519 signature of content_hash (placeholder: 64 bytes).
    /// In production, this is signed by the node's sovereign key.
    pub signature: [u8; 64],

    /// Previous receipt hash (Merkle chain link).
    pub previous_hash: [u8; 32],
}

impl ConstitutionalReceipt {
    /// Verify the Merkle chain: does this receipt's previous_hash match
    /// the content_hash of the preceding receipt?
    pub fn chain_valid(&self, previous: &ConstitutionalReceipt) -> bool {
        self.previous_hash == previous.content_hash
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Action Envelope — The complete dispatch unit
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// An action wrapped with metadata for dispatch.
#[derive(Debug, Clone)]
pub struct ActionEnvelope {
    /// Unique action ID (monotonic).
    pub id: ActionId,

    /// When this action was created.
    pub timestamp: ActionTimestamp,

    /// The action to execute.
    pub action: BizraAction,

    /// The permit governing this action.
    pub permit: Permit,

    /// Iḥsān score of the plan that generated this action.
    pub plan_ihsan: IhsanScore,

    /// Source: which component dispatched this action.
    pub source: String,
}
