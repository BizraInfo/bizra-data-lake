//! # Action Dispatcher — The Muscle System
//!
//! The Dispatcher is the central nervous pathway from decision to execution.
//! Every action follows the same constitutional pipeline:
//!
//! ```text
//! ActionEnvelope
//!   → Guardian.evaluate()     — constitutional gate
//!     → IF Denied:  receipt(denied) → return
//!     → IF HITL:    queue for human confirmation → return
//!     → IF Approved:
//!       → Channel.execute()   — real-world effect
//!       → ReceiptChain.record() — immutable proof
//!       → return ActionResult
//! ```
//!
//! The Dispatcher owns:
//! - The Guardian (constitutional gate)
//! - All Channel handlers (registered at boot)
//! - The Receipt Chain (immutable history)
//! - The monotonic ID counter (Lamport ordering)
//!
//! ## Standing on Giants
//! - **Boyd (1976)**: OODA — the Dispatcher IS the "Act" phase
//! - **Lamport (1978)**: Monotonic IDs ensure happens-before ordering
//! - **Dijkstra (1968)**: Structured dispatch eliminates goto-chaos

use crate::{
    channels::ChannelHandler,
    guardian::Guardian,
    receipt::{hash_payload, ReceiptChain},
    types::*,
};

/// The Action Dispatcher — BIZRA's muscle system.
pub struct Dispatcher {
    /// Constitutional gate.
    guardian: Guardian,

    /// Immutable receipt history.
    receipt_chain: ReceiptChain,

    /// Registered channel handlers.
    channels: Vec<Box<dyn ChannelHandler>>,

    /// Monotonic action ID counter.
    next_id: u64,

    /// Monotonic timestamp counter (nanoseconds, simulated).
    clock_ns: u64,

    /// Actions pending HITL confirmation.
    hitl_queue: Vec<ActionEnvelope>,

    /// Total actions dispatched.
    total_dispatched: u64,

    /// Total actions that completed successfully.
    total_completed: u64,

    /// Total actions denied by Guardian.
    total_denied: u64,
}

impl Dispatcher {
    /// Create a new Dispatcher with no channels registered.
    /// Channels must be registered via `register_channel` before dispatch.
    pub fn new() -> Self {
        Self {
            guardian: Guardian::new(),
            receipt_chain: ReceiptChain::new(),
            channels: Vec::new(),
            next_id: 1,
            clock_ns: 0,
            hitl_queue: Vec::new(),
            total_dispatched: 0,
            total_completed: 0,
            total_denied: 0,
        }
    }

    /// Create a Dispatcher with strict Guardian (for visiting agents).
    pub fn strict() -> Self {
        let mut d = Self::new();
        d.guardian = Guardian::strict();
        d
    }

    /// Register a channel handler. Each channel type should be registered once.
    pub fn register_channel(&mut self, handler: Box<dyn ChannelHandler>) {
        self.channels.push(handler);
    }

    /// Allocate the next action ID and timestamp.
    fn next_envelope(
        &mut self,
        action: BizraAction,
        permit: Permit,
        plan_ihsan: IhsanScore,
        source: String,
    ) -> ActionEnvelope {
        let id = ActionId(self.next_id);
        self.next_id += 1;
        self.clock_ns += 1_000_000; // Advance 1ms per action (simulated)
        let timestamp = ActionTimestamp(self.clock_ns);

        ActionEnvelope {
            id,
            timestamp,
            action,
            permit,
            plan_ihsan,
            source,
        }
    }

    /// Dispatch an action through the full constitutional pipeline.
    ///
    /// This is the core method. Every action in BIZRA flows through here.
    ///
    /// Returns:
    /// - Ok(ActionResult) on successful execution
    /// - Err(DispatchError) on Guardian denial, channel error, or missing channel
    pub fn dispatch(
        &mut self,
        action: BizraAction,
        permit: Permit,
        plan_ihsan: IhsanScore,
        source: impl Into<String>,
    ) -> Result<ActionResult, DispatchError> {
        let envelope = self.next_envelope(action, permit, plan_ihsan, source.into());
        self.total_dispatched += 1;

        // ── Phase 1: Guardian Gate ──────────────────────────
        let verdict = self.guardian.evaluate(&envelope);

        match &verdict {
            GuardianVerdict::Denied { reason, .. } => {
                // Record denial receipt
                let payload_hash = [0u8; 32]; // No payload for denied actions
                self.receipt_chain.record(
                    envelope.id,
                    envelope.timestamp,
                    &envelope.action,
                    verdict.clone(),
                    envelope.plan_ihsan,
                    payload_hash,
                );
                self.total_denied += 1;

                return Err(DispatchError::GuardianDenied {
                    action_id: envelope.id,
                    reason: reason.clone(),
                });
            }

            GuardianVerdict::RequiresHitl {
                reason,
                action_summary,
            } => {
                let err = DispatchError::HitlRequired {
                    action_id: envelope.id,
                    reason: reason.clone(),
                    summary: action_summary.clone(),
                };
                self.hitl_queue.push(envelope);
                return Err(err);
            }

            GuardianVerdict::Approved { .. } => {
                // Proceed to execution
            }
        }

        // ── Phase 2: Channel Routing ────────────────────────
        let target_channel = envelope.action.channel();

        let handler = self
            .channels
            .iter_mut()
            .find(|h| h.channel() == target_channel);

        let handler = match handler {
            Some(h) => h,
            None => {
                return Err(DispatchError::ChannelNotRegistered {
                    channel: target_channel,
                });
            }
        };

        if !handler.is_available() {
            return Err(DispatchError::ChannelUnavailable {
                channel: target_channel,
                status: handler.status(),
            });
        }

        // ── Phase 3: Execute ────────────────────────────────
        let start_ns = self.clock_ns;

        let (success, payload) = match handler.execute(&envelope.action) {
            Ok(p) => (true, p),
            Err(e) => (false, ActionPayload::Error(e.message)),
        };

        self.clock_ns += 100_000; // Simulated execution time
        let duration_ns = self.clock_ns - start_ns;

        // ── Phase 4: Receipt ────────────────────────────────
        let payload_hash = hash_payload(&payload);
        let receipt = self.receipt_chain.record(
            envelope.id,
            envelope.timestamp,
            &envelope.action,
            verdict,
            envelope.plan_ihsan,
            payload_hash,
        );

        if success {
            self.total_completed += 1;
        }

        Ok(ActionResult {
            action_id: envelope.id,
            success,
            payload,
            duration_ns,
            ihsan_score: envelope.plan_ihsan,
            receipt_hash: receipt.content_hash,
        })
    }

    /// Dispatch a pre-built envelope (used for HITL-confirmed actions).
    pub fn dispatch_confirmed(
        &mut self,
        envelope: ActionEnvelope,
    ) -> Result<ActionResult, DispatchError> {
        let action = envelope.action.clone();
        let permit = envelope.permit.clone();
        let ihsan = envelope.plan_ihsan;
        let source = envelope.source.clone();
        self.dispatch(action, permit, ihsan, source)
    }

    /// Get and clear the HITL queue.
    pub fn drain_hitl_queue(&mut self) -> Vec<ActionEnvelope> {
        core::mem::take(&mut self.hitl_queue)
    }

    // ── Accessors ──────────────────────────────────────────

    /// Get the receipt chain (for verification/audit).
    pub fn receipt_chain(&self) -> &ReceiptChain {
        &self.receipt_chain
    }

    /// Get Guardian health.
    pub fn guardian_health(&self) -> crate::guardian::GuardianHealth {
        self.guardian.health()
    }

    /// List registered channels and their status.
    pub fn channel_status(&self) -> Vec<(Channel, bool, String)> {
        self.channels
            .iter()
            .map(|h| (h.channel(), h.is_available(), h.status()))
            .collect()
    }

    /// Dispatcher health snapshot.
    pub fn health(&self) -> DispatcherHealth {
        DispatcherHealth {
            total_dispatched: self.total_dispatched,
            total_completed: self.total_completed,
            total_denied: self.total_denied,
            hitl_pending: self.hitl_queue.len() as u64,
            receipt_chain_length: self.receipt_chain.len(),
            channels_registered: self.channels.len() as u64,
            completion_rate: if self.total_dispatched == 0 {
                1.0
            } else {
                self.total_completed as f64 / self.total_dispatched as f64
            },
        }
    }
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Dispatch errors.
#[derive(Debug)]
pub enum DispatchError {
    /// Guardian denied the action.
    GuardianDenied { action_id: ActionId, reason: String },

    /// Action requires human confirmation.
    HitlRequired {
        action_id: ActionId,
        reason: String,
        summary: String,
    },

    /// No handler registered for this channel.
    ChannelNotRegistered { channel: Channel },

    /// Channel exists but is unavailable.
    ChannelUnavailable { channel: Channel, status: String },
}

/// Dispatcher health snapshot.
#[derive(Debug, Clone)]
pub struct DispatcherHealth {
    pub total_dispatched: u64,
    pub total_completed: u64,
    pub total_denied: u64,
    pub hitl_pending: u64,
    pub receipt_chain_length: u64,
    pub channels_registered: u64,
    pub completion_rate: f64,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Event Emitter Trait (bridge back to Event Bus)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Bridge to the Event Bus for emitting completion events.
/// When the Action Bus completes an action, it emits an event
/// back to the Event Bus, closing the cognitive loop:
///
/// `EVENT → THINK → **ACTION** → **EVENT** (completion)`
pub trait EventEmitter {
    /// Emit an action completion event.
    fn emit_action_completed(
        &mut self,
        action_id: ActionId,
        channel: Channel,
        success: bool,
        ihsan_score: IhsanScore,
        receipt_hash: [u8; 32],
    );

    /// Emit an action denied event.
    fn emit_action_denied(&mut self, action_id: ActionId, channel: Channel, reason: String);
}

/// An event emitter that records events in memory. Used for testing.
pub struct RecordingEmitter {
    pub completed: Vec<(ActionId, Channel, bool, IhsanScore)>,
    pub denied: Vec<(ActionId, Channel, String)>,
}

impl RecordingEmitter {
    pub fn new() -> Self {
        Self {
            completed: Vec::new(),
            denied: Vec::new(),
        }
    }
}

impl Default for RecordingEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEmitter for RecordingEmitter {
    fn emit_action_completed(
        &mut self,
        action_id: ActionId,
        channel: Channel,
        success: bool,
        ihsan_score: IhsanScore,
        _receipt_hash: [u8; 32],
    ) {
        self.completed
            .push((action_id, channel, success, ihsan_score));
    }

    fn emit_action_denied(&mut self, action_id: ActionId, channel: Channel, reason: String) {
        self.denied.push((action_id, channel, reason));
    }
}
