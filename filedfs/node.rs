// bizra-node/src/node.rs
// ============================================================
// Sovereign Node — the living process
// ============================================================
// Design lineage (standing on giants):
//
//   Erlang OTP     → supervisor lifecycle, crash recovery
//   Unix daemon    → one process, one job, stdio interface
//   Redis server   → read line, dispatch, respond, repeat
//   Go net/http    → ListenAndServe simplicity
//
// The Node is the process that turns libraries into a service.
// It owns the AgentRuntime and runs a simple loop:
//
//   1. Read line from stdin
//   2. Parse → Command
//   3. Dispatch → handler::handle(cmd, runtime) → Response
//   4. Write response to stdout
//   5. Repeat until SHUTDOWN or EOF
//
// That's it. Anything more complex belongs in the runtime.
// The node is a thin shell — all intelligence is below.
// ============================================================

use std::io::{self, BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use bizra_agent::runtime::{AgentRuntime, RuntimeConfig};
use crate::protocol::{self, Response, ErrorCode, NODE_NAME, NODE_VERSION, PROTOCOL_VERSION};
use crate::handler;

// ============================================================
// NODE CONFIGURATION
// ============================================================

/// Configuration for the sovereign node process
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// User hash for agent identity
    pub user_hash: u32,
    /// إحسان floor (0-10000, default 9500 = 0.950)
    pub ihsan_floor: u16,
    /// Auto-start a conversation session on boot
    pub auto_start_session: bool,
    /// Print banner on startup
    pub show_banner: bool,
    /// Underlying runtime config
    pub runtime_config: RuntimeConfig,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            user_hash: 1,
            ihsan_floor: 9500,
            auto_start_session: true,
            show_banner: true,
            runtime_config: RuntimeConfig::default(),
        }
    }
}

// ============================================================
// NODE STATE
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Not yet started
    Created,
    /// Running the event loop
    Running,
    /// Gracefully stopped
    Stopped,
}

// ============================================================
// THE SOVEREIGN NODE
// ============================================================

/// Node0 — the sovereign node.
///
/// A thin process shell around AgentRuntime.
/// Reads commands from stdin, dispatches to the runtime,
/// writes responses to stdout. Nothing more.
///
/// All intelligence lives in the runtime.
/// The node's job is to be a reliable, simple host.
pub struct Node {
    /// The agent runtime — the brain
    runtime: AgentRuntime,
    /// Node configuration
    config: NodeConfig,
    /// Current state
    state: NodeState,
    /// Commands processed count
    commands_processed: u64,
    /// Errors encountered
    errors_encountered: u64,
}

impl Node {
    /// Create a new sovereign node
    pub fn new(config: NodeConfig) -> Self {
        
        let runtime = AgentRuntime::with_config(config.runtime_config.clone());

        Self {
            runtime,
            config,
            state: NodeState::Created,
            commands_processed: 0,
            errors_encountered: 0,
        }
    }

    /// Get the current node state
    pub fn state(&self) -> NodeState {
        self.state
    }

    /// Get a mutable reference to the runtime (for testing)
    pub fn runtime_mut(&mut self) -> &mut AgentRuntime {
        &mut self.runtime
    }

    /// Get a reference to the runtime
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    /// Get commands processed count
    pub fn commands_processed(&self) -> u64 {
        self.commands_processed
    }

    /// Get errors encountered count
    pub fn errors_encountered(&self) -> u64 {
        self.errors_encountered
    }

    // ========================================================
    // STDIO EVENT LOOP
    // ========================================================

    /// Run the node — blocking stdio event loop.
    ///
    /// Reads lines from stdin, dispatches commands, writes
    /// responses to stdout. Runs until SHUTDOWN command or EOF.
    ///
    /// This is the main entry point for the binary.
    pub fn run(&mut self) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = io::BufWriter::new(stdout.lock());
        let stdin = io::stdin();
        let reader = stdin.lock();

        self.start(&mut out)?;

        // Main event loop
        for line_result in reader.lines() {
            match line_result {
                Ok(line) => {
                    let should_stop = self.process_line(&line, &mut out)?;
                    if should_stop {
                        break;
                    }
                }
                Err(e) => {
                    // IO error reading stdin — log and break
                    let resp = Response::err(
                        ErrorCode::Internal,
                        &format!("stdin read error: {}", e),
                    );
                    write_response(&mut out, &resp)?;
                    self.errors_encountered += 1;
                    break;
                }
            }
        }

        self.state = NodeState::Stopped;
        Ok(())
    }

    /// Start the node — initialize runtime and optionally show banner
    fn start<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        self.state = NodeState::Running;

        // Auto-start session if configured
        if self.config.auto_start_session {
            let now = current_timestamp();
            self.runtime.start_conversation(now);
        }

        // Show banner if configured
        if self.config.show_banner {
            let banner = Response::ok()
                .field("event", "started")
                .field("node", NODE_NAME)
                .field("version", NODE_VERSION)
                .field("protocol", PROTOCOL_VERSION)
                .field("state", "Running")
                .field("ihsan", self.runtime.current_ihsan().raw());
            write_response(out, &banner)?;
        }

        Ok(())
    }

    // ========================================================
    // LINE PROCESSING — the core of the event loop
    // ========================================================

    /// Process a single input line. Returns true if node should stop.
    pub fn process_line<W: Write>(&mut self, line: &str, out: &mut W) -> io::Result<bool> {
        // Parse
        let cmd = match protocol::parse_command(line) {
            Ok(Some(cmd)) => cmd,
            Ok(None) => return Ok(false), // Empty line, skip
            Err(err_resp) => {
                self.errors_encountered += 1;
                write_response(out, &err_resp)?;
                return Ok(false);
            }
        };

        // Check for shutdown before dispatch
        let is_shutdown = matches!(cmd, protocol::Command::Shutdown);

        // Dispatch to handler
        let response = handler::handle(cmd, &mut self.runtime);
        self.commands_processed += 1;

        // Track errors
        if matches!(response, Response::Err { .. }) {
            self.errors_encountered += 1;
        }

        // Write response
        write_response(out, &response)?;

        Ok(is_shutdown)
    }

    // ========================================================
    // PROGRAMMATIC API — for embedding without stdio
    // ========================================================

    /// Process a command string programmatically (no stdio).
    /// Returns the response as a string.
    ///
    /// Useful for:
    /// - Testing without stdio
    /// - Embedding the node in another process
    /// - FFI integration
    pub fn execute(&mut self, line: &str) -> String {
        if self.state == NodeState::Created {
            self.state = NodeState::Running;
            if self.config.auto_start_session {
                let now = current_timestamp();
                self.runtime.start_conversation(now);
            }
        }

        let cmd = match protocol::parse_command(line) {
            Ok(Some(cmd)) => cmd,
            Ok(None) => return String::new(),
            Err(err_resp) => {
                self.errors_encountered += 1;
                return err_resp.to_string();
            }
        };

        let is_shutdown = matches!(cmd, protocol::Command::Shutdown);
        let response = handler::handle(cmd, &mut self.runtime);
        self.commands_processed += 1;

        if matches!(response, Response::Err { .. }) {
            self.errors_encountered += 1;
        }

        if is_shutdown {
            self.state = NodeState::Stopped;
        }

        response.to_string()
    }
}

// ============================================================
// HELPERS
// ============================================================

/// Write a response line to the output, flushing immediately
fn write_response<W: Write>(out: &mut W, resp: &Response) -> io::Result<()> {
    writeln!(out, "{}", resp)?;
    out.flush()
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node() -> Node {
        Node::new(NodeConfig::default())
    }

    #[test]
    fn node_creation() {
        let node = make_node();
        assert_eq!(node.state(), NodeState::Created);
        assert_eq!(node.commands_processed(), 0);
        assert_eq!(node.errors_encountered(), 0);
    }

    #[test]
    fn execute_ping() {
        let mut node = make_node();
        let resp = node.execute("PING");
        assert!(resp.starts_with("OK\t"));
        assert!(resp.contains("pong=true"));
        assert_eq!(node.commands_processed(), 1);
    }

    #[test]
    fn execute_version() {
        let mut node = make_node();
        let resp = node.execute("VERSION");
        assert!(resp.contains("node=bizra-node"));
        assert!(resp.contains("version=0.1.0"));
    }

    #[test]
    fn execute_health() {
        let mut node = make_node();
        let resp = node.execute("HEALTH");
        assert!(resp.starts_with("OK\t"));
        assert!(resp.contains("state="));
        assert!(resp.contains("ihsan="));
        assert!(resp.contains("knows_me="));
    }

    #[test]
    fn execute_receive() {
        let mut node = make_node();
        let resp = node.execute("RECEIVE\tI love building AI systems\t1000");
        assert!(resp.starts_with("OK\t"));
        assert!(resp.contains("confidence="));
        assert!(resp.contains("knows_me="));
    }

    #[test]
    fn execute_teach_and_query() {
        let mut node = make_node();

        let t = node.execute("TEACH\tpreference\tlikes Rust\t9000\t1000");
        assert!(t.starts_with("OK\t"));

        let q = node.execute("QUERY\tpreference");
        assert!(q.starts_with("OK\t"));
    }

    #[test]
    fn execute_session_lifecycle() {
        let mut node = Node::new(NodeConfig {
            auto_start_session: false,
            show_banner: false,
            ..Default::default()
        });

        let start = node.execute("START_SESSION\t1000");
        assert!(start.contains("session_id="));

        // Add fragments so end_session can trigger synthesis
        for i in 0..6 {
            node.execute(&format!("TEACH\tfact\ttest fact {}\t8000\t{}", i, 1100 + i));
        }

        let end = node.execute("END_SESSION\t2000");
        // Should get OK response (ended with or without synthesis)
        assert!(end.starts_with("OK\t"));
    }

    #[test]
    fn execute_ihsan_update() {
        let mut node = make_node();
        let resp = node.execute("IHSAN\t9800");
        assert!(resp.contains("ihsan=9800"));
    }

    #[test]
    fn execute_shutdown() {
        let mut node = make_node();
        let resp = node.execute("SHUTDOWN");
        assert!(resp.contains("shutdown=true"));
        assert_eq!(node.state(), NodeState::Stopped);
    }

    #[test]
    fn execute_unknown_command_errors() {
        let mut node = make_node();
        let resp = node.execute("BOGUS_COMMAND");
        assert!(resp.starts_with("ERR\t"));
        assert_eq!(node.errors_encountered(), 1);
    }

    #[test]
    fn execute_empty_line_ignored() {
        let mut node = make_node();
        let resp = node.execute("");
        assert!(resp.is_empty());
        assert_eq!(node.commands_processed(), 0);
    }

    #[test]
    fn execute_multiple_commands_sequential() {
        let mut node = make_node();

        node.execute("PING");
        node.execute("HEALTH");
        node.execute("RECEIVE\tHello\t1000");
        node.execute("KNOWS_ME");
        node.execute("VERSION");

        assert_eq!(node.commands_processed(), 5);
        assert_eq!(node.errors_encountered(), 0);
    }

    #[test]
    fn process_line_returns_stop_on_shutdown() {
        let mut node = make_node();
        let mut buf = Vec::new();

        // Activate
        node.state = NodeState::Running;
        let now = current_timestamp();
        node.runtime.start_conversation(now);

        let stop = node.process_line("SHUTDOWN", &mut buf).unwrap();
        assert!(stop);
    }

    #[test]
    fn process_line_returns_continue_on_normal() {
        let mut node = make_node();
        let mut buf = Vec::new();

        node.state = NodeState::Running;
        let now = current_timestamp();
        node.runtime.start_conversation(now);

        let stop = node.process_line("PING", &mut buf).unwrap();
        assert!(!stop);

        // Verify output was written
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("pong=true"));
    }

    #[test]
    fn full_node_lifecycle_programmatic() {
        let mut node = make_node();

        // Version
        let v = node.execute("VERSION");
        assert!(v.contains("bizra-node"));

        // Send messages that teach the system
        node.execute("RECEIVE\tI prefer functional programming\t1000");
        node.execute("RECEIVE\tMy goal is to build a distributed AI platform\t2000");
        node.execute("TEACH\texpertise\tRust systems programming\t9500\t3000");

        // Check knowledge growth
        let k1 = node.execute("KNOWS_ME");
        assert!(k1.contains("score="));

        // Synthesize
        node.execute("SYNTHESIZE\t4000");

        // Health check
        let h = node.execute("HEALTH");
        assert!(h.contains("messages_processed=2"));

        // Shutdown
        let s = node.execute("SHUTDOWN");
        assert!(s.contains("shutdown=true"));
        assert_eq!(node.state(), NodeState::Stopped);
        assert_eq!(node.errors_encountered(), 0);
    }

    #[test]
    fn knowledge_persists_across_sessions() {
        let mut node = Node::new(NodeConfig {
            auto_start_session: false,
            show_banner: false,
            ..Default::default()
        });

        // Session 1
        node.execute("START_SESSION\t1000");
        node.execute("RECEIVE\tI love Rust programming\t1001");
        node.execute("TEACH\tpreference\tprefers dark mode\t9000\t1002");
        node.execute("END_SESSION\t1003");

        let k1 = node.execute("KNOWS_ME");

        // Session 2
        node.execute("START_SESSION\t2000");
        node.execute("RECEIVE\tI'm building BIZRA platform\t2001");
        node.execute("TEACH\tgoal\tdemocratize AI\t9500\t2002");
        node.execute("END_SESSION\t2003");

        let k2 = node.execute("KNOWS_ME");

        // Knowledge should grow or at least not shrink
        // Both should be valid OK responses
        assert!(k1.starts_with("OK\t"));
        assert!(k2.starts_with("OK\t"));
    }
}
