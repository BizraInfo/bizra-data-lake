//! # bizra-action — The Muscle System
//!
//! BIZRA's Action Bus. The counterpart to bizra-hooks (Event Bus).
//!
//! - **Events** (bizra-hooks): Observations. Safe. Information flows freely.
//! - **Actions** (bizra-action): Commands. Dangerous. Guardian gates every one.
//!
//! Together they form the cognitive loop:
//! ```text
//! EVENT → THINK → ACTION → EVENT
//! (sense)  (decide)  (do)   (observe)
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use bizra_action::*;
//!
//! // 1. Create dispatcher
//! let mut dispatcher = Dispatcher::new();
//!
//! // 2. Register channels (the capabilities)
//! dispatcher.register_channel(Box::new(channels::AhkChannel::new()));
//! dispatcher.register_channel(Box::new(channels::LlmChannel::new()));
//! dispatcher.register_channel(Box::new(channels::ResponseChannel::new()));
//!
//! // 3. Dispatch an action through the constitutional pipeline
//! let result = dispatcher.dispatch(
//!     BizraAction::AhkLaunch {
//!         executable: "notepad.exe".into(),
//!         args: vec![],
//!     },
//!     Permit::user_default(),
//!     IhsanScore::new(0.98),
//!     "sovereign_core",
//! );
//!
//! // 4. Every action produces a constitutional receipt
//! assert!(result.is_ok());
//! assert_eq!(dispatcher.receipt_chain().len(), 1);
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                  Dispatcher                      │
//! │                                                  │
//! │  ActionEnvelope ──→ Guardian ──→ Channel ──→ Receipt │
//! │                     (gate)      (execute)   (proof)  │
//! │                                                  │
//! │  Channels:                                       │
//! │    AHK ─── LLM ─── Memory ─── MCP               │
//! │    FileSystem ─── Browser ─── Response            │
//! │    Telescript                                     │
//! │                                                  │
//! │  Support:                                        │
//! │    ReflexLedger (System 1 cache)                 │
//! │    ReceiptChain (Merkle history)                 │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Zero Dependencies
//!
//! This crate has zero external dependencies. Pure Rust. Sovereign.
//! The muscle system depends on nothing; everything depends on it.

pub mod channels;
pub mod dispatcher;
pub mod guardian;
pub mod receipt;
pub mod reflex;
pub mod types;

// Re-exports for ergonomic usage
pub use dispatcher::{DispatchError, Dispatcher, DispatcherHealth};
pub use guardian::Guardian;
pub use receipt::{chain_hash, content_hash, hash_payload, ReceiptChain};
pub use reflex::{Reflex, ReflexError, ReflexHealth, ReflexLedger};
pub use types::*;
