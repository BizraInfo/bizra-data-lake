// bizra-node/src/node.rs
// ============================================================
// Node — the sovereign living process
// ============================================================
//
// The Node is the top-level struct that owns everything:
//   - AgentRuntime (which owns MemoryPipeline, AgentRoster, etc.)
//   - IhsanScore (the Lyapunov certificate)
//   - Protocol state (counters, session tracking)
//
// External code interacts with Node through exactly two methods:
//   node.execute(line) -> String    (single command)
//   node.run()          -> !        (stdin/stdout loop)
//
// The Node is self-contained. No global state. No singletons.
// ============================================================

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use bizra_agent::runtime::{AgentRuntime, RuntimeConfig};
use bizra_hooks::IhsanScore;

use bizra_memory::bridge::export_atoms_as_turns;

use crate::action_executor::ActionExecutor;
use crate::handler::{self, NodeInternals, SapSessionState};
use crate::protocol::{self, Response, NODE_NAME, NODE_VERSION};

// ============================================================
// NODE STATE
// ============================================================

/// The state of the node process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Node is running and accepting commands.
    Running,
    /// Node has been shut down.
    Stopped,
}

// ============================================================
// NODE CONFIG
// ============================================================

/// Configuration for a Node instance.
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// User identity hash (determines state directory, agent identity).
    pub user_hash: u32,
    /// The ihsan floor (raw u16, 0-65535). Below this the system degrades.
    pub ihsan_floor: u16,
    /// Whether to print the startup banner to stderr.
    pub show_banner: bool,
    /// Whether to auto-start a conversation session on first message.
    pub auto_start_session: bool,
    /// AgentRuntime configuration.
    pub runtime_config: RuntimeConfig,
}

impl Default for NodeConfig {
    fn default() -> Self {
        NodeConfig {
            user_hash: 1,
            ihsan_floor: 9500,
            show_banner: true,
            auto_start_session: true,
            runtime_config: RuntimeConfig::default(),
        }
    }
}

// ============================================================
// SELF-COMPILATION — Conversation Genesis Feedback Loop
// ============================================================

/// Number of commands between automatic self-compilation passes.
///
/// Every N commands the node exports its accumulated memory atoms as
/// ConversationTurnWire records suitable for the Python stereoscopic
/// compiler.  This closes the identity loop: use BIZRA -> atoms
/// extracted -> exported as turns -> stereoscopic engine compiles ->
/// identity grows.
const SELF_COMPILE_INTERVAL: usize = 50;

// ============================================================
// NODE — the living process
// ============================================================

/// The sovereign node. Owns all state, processes commands, speaks protocol.
pub struct Node {
    /// Configuration (immutable after construction).
    config: NodeConfig,
    /// The unified agent runtime (owns memory pipeline, roster, orchestrator).
    runtime: AgentRuntime,
    /// Current ihsan score.
    ihsan: IhsanScore,
    /// Current node state.
    state: NodeState,
    /// Total commands processed (including errors).
    commands_processed: usize,
    /// Total error responses returned.
    errors_encountered: usize,
    /// Session counter (monotonic).
    session_counter: u64,
    /// Message counter (monotonic, used for MessageId generation).
    message_counter: u64,
    /// Whether a session has been auto-started.
    session_auto_started: bool,
    /// Action-layer executor and state.
    action_executor: ActionExecutor,
    /// SAP v0 active sessions.
    sap_sessions: HashMap<String, SapSessionState>,
}

impl Node {
    /// Create a new Node from configuration.
    pub fn new(config: NodeConfig) -> Self {
        let runtime = AgentRuntime::with_config(config.runtime_config.clone());
        let ihsan = IhsanScore::from_raw(config.ihsan_floor.max(9500));

        let mut node = Node {
            config,
            runtime,
            ihsan,
            state: NodeState::Running,
            commands_processed: 0,
            errors_encountered: 0,
            session_counter: 0,
            message_counter: 0,
            session_auto_started: false,
            action_executor: ActionExecutor::default(),
            sap_sessions: HashMap::new(),
        };

        // Register PostDeliver audit hook for action.receipt events.
        if let Err(err) = node.action_executor.register_post_deliver_hook(
            "audit.action_receipt",
            0,
            crate::audit_hook::audit_receipt_hook,
        ) {
            eprintln!("[WARN] Failed to register action.receipt audit hook: {err:?}");
        }
        // Keep transitional direct-write fallback disabled to avoid duplicates.
        node.action_executor
            .set_direct_audit_fallback_on_eventbus(false);

        // Auto-start a session if configured
        if node.config.auto_start_session {
            node.runtime.start_conversation(0);
            node.session_counter = 1;
            node.session_auto_started = true;
        }

        node
    }

    // ================================================================
    // PUBLIC API
    // ================================================================

    /// Execute a single protocol command line.
    ///
    /// Parses the line, dispatches to the handler, serializes the response.
    /// Always returns a valid wire-format string. Never panics.
    pub fn execute(&mut self, line: &str) -> String {
        self.commands_processed += 1;

        // Parse
        let cmd = match protocol::parse_command(line) {
            Ok(cmd) => cmd,
            Err((code, msg)) => {
                self.errors_encountered += 1;
                return Response::err(code, &msg).to_wire();
            }
        };

        // Build the handler's borrowed view of our state
        let mut stopped = false;
        let response = {
            let mut internals = NodeInternals {
                runtime: &mut self.runtime,
                ihsan: &mut self.ihsan,
                session_counter: &mut self.session_counter,
                message_counter: &mut self.message_counter,
                ihsan_floor: self.config.ihsan_floor,
                user_hash: self.config.user_hash,
                stopped: &mut stopped,
                action_executor: &mut self.action_executor,
                sap_sessions: &mut self.sap_sessions,
            };
            handler::handle(cmd, &mut internals)
        };

        // Apply side effects that the handler signaled
        if stopped {
            self.state = NodeState::Stopped;
        }

        // Track errors
        if response.is_err() {
            self.errors_encountered += 1;
        }

        // Periodic self-compilation: export atoms for stereoscopic identity
        if self.commands_processed > 0
            && self
                .commands_processed
                .is_multiple_of(SELF_COMPILE_INTERVAL)
        {
            self.trigger_self_compilation();
        }

        response.to_wire()
    }

    /// Execute a pre-parsed Command directly (for MCP transport).
    ///
    /// Skips the parse step. Returns a Response object.
    pub fn handle_command(&mut self, cmd: protocol::Command) -> Response {
        self.commands_processed += 1;

        let mut stopped = false;
        let response = {
            let mut internals = NodeInternals {
                runtime: &mut self.runtime,
                ihsan: &mut self.ihsan,
                session_counter: &mut self.session_counter,
                message_counter: &mut self.message_counter,
                ihsan_floor: self.config.ihsan_floor,
                user_hash: self.config.user_hash,
                stopped: &mut stopped,
                action_executor: &mut self.action_executor,
                sap_sessions: &mut self.sap_sessions,
            };
            handler::handle(cmd, &mut internals)
        };

        if stopped {
            self.state = NodeState::Stopped;
        }

        if response.is_err() {
            self.errors_encountered += 1;
        }

        response
    }

    /// Run the stdin/stdout protocol loop.
    ///
    /// Reads lines from stdin, processes each through `execute()`,
    /// writes the response to stdout. Exits on SHUTDOWN or EOF.
    pub fn run(&mut self) -> Result<(), io::Error> {
        if self.config.show_banner {
            self.print_banner();
        }

        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut out = io::BufWriter::new(stdout.lock());

        for line_result in stdin.lock().lines() {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }

            let response = self.execute(&line);
            writeln!(out, "{response}")?;
            out.flush()?;

            if self.state == NodeState::Stopped {
                break;
            }
        }

        Ok(())
    }

    // ================================================================
    // ACCESSORS
    // ================================================================

    /// Current node state.
    pub fn state(&self) -> NodeState {
        self.state
    }

    /// Total commands processed (including errors).
    pub fn commands_processed(&self) -> usize {
        self.commands_processed
    }

    /// Total error responses returned.
    pub fn errors_encountered(&self) -> usize {
        self.errors_encountered
    }

    /// The user hash this node is configured for.
    pub fn user_hash(&self) -> u32 {
        self.config.user_hash
    }

    /// Immutable reference to the node configuration.
    pub fn config_ref(&self) -> &NodeConfig {
        &self.config
    }

    /// Mutable access to the AgentRuntime (used by persistence).
    pub fn runtime_mut(&mut self) -> &mut AgentRuntime {
        &mut self.runtime
    }

    /// Immutable access to the AgentRuntime (used by persistence).
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    pub fn action_executor_mut(&mut self) -> &mut ActionExecutor {
        &mut self.action_executor
    }

    pub fn action_executor(&self) -> &ActionExecutor {
        &self.action_executor
    }

    // ================================================================
    // INTERNAL
    // ================================================================

    /// Trigger periodic self-compilation of memory atoms.
    ///
    /// Exports all atoms currently in the memory pipeline as
    /// `ConversationTurnWire` records compatible with the Python
    /// stereoscopic engine. In the current implementation the
    /// turns are logged to stderr (Node0 bootstrap). In production
    /// this will write to a JSONL file or call the Python engine
    /// via FFI for full identity compilation.
    fn trigger_self_compilation(&self) {
        let summary = self.runtime.pipeline().knowledge_summary();
        if summary.total_atoms == 0 {
            return;
        }

        // Collect atoms from the store that have not been superseded.
        // We iterate all AtomKind variants and gather (kind, content, confidence, timestamp).
        let store = self.runtime.pipeline().store();
        let mut atom_tuples: Vec<(bizra_memory::types::AtomKind, String, f32, u64)> = Vec::new();

        let kinds = [
            bizra_memory::types::AtomKind::Fact,
            bizra_memory::types::AtomKind::Preference,
            bizra_memory::types::AtomKind::Pattern,
            bizra_memory::types::AtomKind::Relationship,
            bizra_memory::types::AtomKind::Goal,
            bizra_memory::types::AtomKind::Expertise,
            bizra_memory::types::AtomKind::Context,
            bizra_memory::types::AtomKind::Principle,
            bizra_memory::types::AtomKind::Temporal,
            bizra_memory::types::AtomKind::Negation,
        ];

        for kind in &kinds {
            for atom in store.atoms_by_kind(*kind) {
                if let Some(content) = store.atom_content(atom) {
                    atom_tuples.push((
                        atom.header.kind,
                        content.to_owned(),
                        atom.header.confidence.base,
                        atom.header.provenance.extracted_at,
                    ));
                }
            }
        }

        if atom_tuples.is_empty() {
            return;
        }

        // Build borrow-compatible slice of (&str) references for export
        let refs: Vec<(bizra_memory::types::AtomKind, &str, f32, u64)> = atom_tuples
            .iter()
            .map(|(k, c, conf, ts)| (*k, c.as_str(), *conf, *ts))
            .collect();

        let turns = export_atoms_as_turns(&refs, self.session_counter);

        eprintln!(
            "[self-compile] exported {} atoms as {} turns (session={}, commands={})",
            summary.total_atoms,
            turns.len(),
            self.session_counter,
            self.commands_processed,
        );
    }

    /// Print the startup banner to stderr.
    fn print_banner(&self) {
        eprintln!();
        eprintln!(
            "  {NODE_NAME} v{NODE_VERSION} | protocol v{}",
            protocol::PROTOCOL_VERSION,
        );
        eprintln!(
            "  user: {} | ihsan floor: {}",
            self.config.user_hash, self.config.ihsan_floor
        );
        eprintln!("  ready.");
        eprintln!();
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_creation_defaults() {
        let node = Node::new(NodeConfig::default());
        assert_eq!(node.state(), NodeState::Running);
        assert_eq!(node.commands_processed(), 0);
        assert_eq!(node.errors_encountered(), 0);
        assert_eq!(node.user_hash(), 1);
    }

    #[test]
    fn node_execute_ping() {
        let mut node = Node::new(NodeConfig::default());
        let resp = node.execute("PING");
        assert!(resp.contains("pong=true"));
        assert_eq!(node.commands_processed(), 1);
        assert_eq!(node.errors_encountered(), 0);
    }

    #[test]
    fn node_execute_bad_command() {
        let mut node = Node::new(NodeConfig::default());
        let resp = node.execute("BOGUS");
        assert!(resp.starts_with("ERR\t"));
        assert!(resp.contains("BAD_COMMAND"));
        assert_eq!(node.errors_encountered(), 1);
    }

    #[test]
    fn node_execute_shutdown() {
        let mut node = Node::new(NodeConfig::default());
        let resp = node.execute("SHUTDOWN");
        assert!(resp.contains("shutdown=true"));
        assert_eq!(node.state(), NodeState::Stopped);
    }

    #[test]
    fn node_command_counter() {
        let mut node = Node::new(NodeConfig::default());
        node.execute("PING");
        node.execute("VERSION");
        node.execute("BOGUS");
        assert_eq!(node.commands_processed(), 3);
        assert_eq!(node.errors_encountered(), 1);
    }

    #[test]
    fn node_no_auto_session() {
        let node = Node::new(NodeConfig {
            auto_start_session: false,
            ..Default::default()
        });
        assert_eq!(node.state(), NodeState::Running);
    }

    #[test]
    fn node_registers_action_receipt_post_deliver_hook() {
        let node = Node::new(NodeConfig::default());
        assert!(node.action_executor().post_deliver_hook_count() >= 1);
    }

    #[test]
    fn self_compile_interval_constant() {
        // Verify the constant is a reasonable value (not 0, not too large).
        let interval = SELF_COMPILE_INTERVAL;
        assert!(interval >= 10);
        assert!(interval <= 1000);
    }

    #[test]
    fn self_compile_does_not_panic_on_empty_pipeline() {
        // The trigger should be a no-op when no atoms exist.
        let node = Node::new(NodeConfig::default());
        node.trigger_self_compilation(); // must not panic
    }

    #[test]
    fn self_compile_triggers_at_interval() {
        // Feed SELF_COMPILE_INTERVAL PINGs and verify the node survives.
        // Self-compilation fires at exactly SELF_COMPILE_INTERVAL commands.
        let mut node = Node::new(NodeConfig::default());
        for _ in 0..SELF_COMPILE_INTERVAL {
            node.execute("PING");
        }
        assert_eq!(node.commands_processed(), SELF_COMPILE_INTERVAL);
        // No panic, no errors from self-compilation on an empty pipeline
        assert_eq!(node.errors_encountered(), 0);
    }

    #[test]
    fn self_compile_with_atoms() {
        // Teach some atoms, then hit the interval to trigger self-compilation.
        let mut node = Node::new(NodeConfig::default());

        // Teach a few atoms via the protocol (format: TEACH\tkind\tcontent\tihsan\ttimestamp)
        node.execute("TEACH\tfact\tI am the founder of BIZRA\t9500\t1000");
        node.execute("TEACH\tpreference\tI prefer Rust for sovereignty\t9500\t2000");
        node.execute("TEACH\tpattern\tBuilding a decentralized future every day\t9500\t3000");

        // Fill up to the interval with PINGs
        let remaining = SELF_COMPILE_INTERVAL - 3;
        for _ in 0..remaining {
            node.execute("PING");
        }

        assert_eq!(node.commands_processed(), SELF_COMPILE_INTERVAL);
        assert_eq!(node.errors_encountered(), 0);
    }
}
