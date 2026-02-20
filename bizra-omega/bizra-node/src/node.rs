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

use std::io::{self, BufRead, Write};

use bizra_agent::runtime::{AgentRuntime, RuntimeConfig};
use bizra_hooks::IhsanScore;

use crate::handler::{self, NodeInternals};
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
        };

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

        response.to_wire()
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
            writeln!(out, "{}", response)?;
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

    // ================================================================
    // INTERNAL
    // ================================================================

    /// Print the startup banner to stderr.
    fn print_banner(&self) {
        eprintln!();
        eprintln!(
            "  {} v{} | protocol v{}",
            NODE_NAME,
            NODE_VERSION,
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
}
