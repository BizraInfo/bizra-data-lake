// bizra-node/src/main.rs
// ============================================================
// Node0 — the sovereign binary
// ============================================================
//
//   بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
//
// Usage:
//   bizra-node                         # Defaults
//   bizra-node --user <hash>           # User identity
//   bizra-node --ihsan <floor>         # إحسان floor
//   bizra-node --seed <file>           # Pre-load knowledge
//   bizra-node --state-dir <path>      # Persist across sessions
//   bizra-node --mcp-port 9741         # JSON-RPC 2.0 over TCP
//   bizra-node --no-banner             # Suppress banner
//   bizra-node --no-auto-session       # Manual session control
//
// The seed file: your knowledge as text. You own it.
// ============================================================

use bizra_agent::reflex_cache::ReflexMode;
use bizra_agent::runtime::ActionMode;
use bizra_hooks::IhsanScore;
use bizra_node::mcp_transport::{self, McpTransportConfig};
use bizra_node::node::{Node, NodeConfig};
use bizra_node::persistence;
use std::path::PathBuf;
use std::process;
use std::sync::{Arc, Mutex};

struct CliConfig {
    node_config: NodeConfig,
    seed_file: Option<PathBuf>,
    state_dir: Option<PathBuf>,
    auto_persist: bool,
    mcp_port: Option<u16>,
}

fn main() {
    let cli = parse_args();

    let mut node = Node::new(cli.node_config);

    // Determine state directory (explicit or default)
    let state_dir = cli
        .state_dir
        .unwrap_or_else(|| persistence::state_dir(node.user_hash()));

    // Auto-persist is on by default if not explicitly off
    let auto_persist = cli.auto_persist;

    // Load seed file if provided
    if let Some(seed_path) = &cli.seed_file {
        if seed_path.exists() {
            match persistence::load_seed(&mut node, seed_path) {
                Ok((loaded, errors)) => {
                    if node.config_ref().show_banner {
                        eprintln!(
                            "  seed: loaded {} commands ({} errors) from {}",
                            loaded,
                            errors,
                            seed_path.display()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("  seed: error reading {}: {}", seed_path.display(), e);
                }
            }
        } else {
            eprintln!("  seed: file not found: {}", seed_path.display());
        }
    } else if auto_persist {
        // Try loading from default state path
        let default_seed = state_dir.join("knowledge.seed");
        if default_seed.exists() {
            match persistence::load_seed(&mut node, &default_seed) {
                Ok((loaded, _errors)) => {
                    if node.config_ref().show_banner && loaded > 0 {
                        eprintln!(
                            "  state: restored {} fragments from {}",
                            loaded,
                            default_seed.display()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("  state: error restoring: {}", e);
                }
            }
        }
    }

    if auto_persist {
        // Restore compiled reflex cache
        let reflex_cache_path = state_dir.join("reflex.cache");
        if reflex_cache_path.exists() {
            match persistence::load_reflex_cache(&mut node, &reflex_cache_path) {
                Ok((loaded, quarantined)) => {
                    if node.config_ref().show_banner && (loaded > 0 || quarantined > 0) {
                        eprintln!(
                            "  reflex: restored {} rules ({} quarantined) from {}",
                            loaded,
                            quarantined,
                            reflex_cache_path.display()
                        );
                    }
                }
                Err(e) => eprintln!("  reflex: error restoring cache: {}", e),
            }
        }

        // Restore action receipt history
        let actions_log_path = state_dir.join("actions.log");
        if actions_log_path.exists() {
            match persistence::load_action_log(&mut node, &actions_log_path) {
                Ok((loaded, rejected)) => {
                    if node.config_ref().show_banner && (loaded > 0 || rejected > 0) {
                        eprintln!(
                            "  actions: restored {} receipts ({} rejected) from {}",
                            loaded,
                            rejected,
                            actions_log_path.display()
                        );
                    }
                }
                Err(e) => eprintln!("  actions: error restoring log: {}", e),
            }
        }
    }

    // Run the node — either MCP TCP mode or stdin/stdout mode
    if let Some(port) = cli.mcp_port {
        // MCP JSON-RPC 2.0 over TCP
        let config = McpTransportConfig {
            port,
            ..McpTransportConfig::default()
        };
        let node_mutex = Arc::new(Mutex::new(node));
        let handler_ref = Arc::clone(&node_mutex);
        let handler = Arc::new(move |cmd| {
            let mut n = handler_ref.lock().expect("node lock poisoned");
            n.handle_command(cmd)
        });

        mcp_transport::start_tcp_listener(config, handler);

        // TCP listener returns on bind failure — save state
        let node = Arc::try_unwrap(node_mutex)
            .unwrap_or_else(|_| panic!("node still referenced"))
            .into_inner()
            .unwrap();
        if auto_persist {
            let _ = save_state_quietly(&node, &state_dir);
        }
    } else {
        // Standard stdin/stdout protocol loop
        if let Err(e) = node.run() {
            eprintln!("bizra-node: fatal error: {}", e);

            // Still try to save state on error
            if auto_persist {
                let _ = save_state_quietly(&node, &state_dir);
            }

            process::exit(1);
        }

        // Save state on shutdown
        if auto_persist {
            match save_state_quietly(&node, &state_dir) {
                Ok(count) => {
                    if count > 0 {
                        eprintln!(
                            "  state: saved {} fragments to {}",
                            count,
                            state_dir.join("knowledge.seed").display()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("  state: error saving: {}", e);
                }
            }

            if let Err(e) = save_reflex_cache_quietly(&node, &state_dir) {
                eprintln!("  reflex: error saving cache: {}", e);
            }
            if let Err(e) = save_action_log_quietly(&node, &state_dir) {
                eprintln!("  actions: error saving log: {}", e);
            }
        }
    }

    process::exit(0);
}

fn save_state_quietly(node: &Node, state_dir: &std::path::Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(state_dir)?;
    let seed_path = state_dir.join("knowledge.seed");
    persistence::save_state(node, &seed_path)
}

fn save_reflex_cache_quietly(node: &Node, state_dir: &std::path::Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(state_dir)?;
    let cache_path = state_dir.join("reflex.cache");
    persistence::save_reflex_cache(node, &cache_path)
}

fn save_action_log_quietly(node: &Node, state_dir: &std::path::Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(state_dir)?;
    let log_path = state_dir.join("actions.log");
    persistence::save_action_log(node, &log_path)
}

fn parse_args() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut node_config = NodeConfig::default();
    let mut seed_file = None;
    let mut state_dir = None;
    let mut auto_persist = true;
    let mut cli_policy_hash: Option<String> = None;
    let mut mcp_port: Option<u16> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--user" => {
                i += 1;
                if i < args.len() {
                    node_config.user_hash = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("bizra-node: invalid --user value: {}", args[i]);
                        process::exit(2);
                    });
                    node_config.runtime_config =
                        bizra_agent::runtime::RuntimeConfig::for_user(node_config.user_hash);
                }
            }
            "--ihsan" => {
                i += 1;
                if i < args.len() {
                    node_config.ihsan_floor = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("bizra-node: invalid --ihsan value: {}", args[i]);
                        process::exit(2);
                    });
                    node_config.runtime_config.ihsan_floor =
                        IhsanScore::from_f64(node_config.ihsan_floor as f64 / 10000.0);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed_file = Some(PathBuf::from(&args[i]));
                }
            }
            "--state-dir" => {
                i += 1;
                if i < args.len() {
                    state_dir = Some(PathBuf::from(&args[i]));
                }
            }
            "--reflex-mode" => {
                i += 1;
                if i < args.len() {
                    let mode = ReflexMode::parse(&args[i]).unwrap_or_else(|| {
                        eprintln!("bizra-node: invalid --reflex-mode value: {}", args[i]);
                        process::exit(2);
                    });
                    node_config.runtime_config.reflex_mode = mode;
                }
            }
            "--policy-hash" => {
                i += 1;
                if i < args.len() {
                    let hash = args[i].clone();
                    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                        eprintln!("bizra-node: invalid --policy-hash (expect 64 hex chars)");
                        process::exit(2);
                    }
                    cli_policy_hash = Some(hash);
                }
            }
            "--action-mode" => {
                i += 1;
                if i < args.len() {
                    let mode = match args[i].to_ascii_lowercase().as_str() {
                        "disabled" => ActionMode::Disabled,
                        "shadow" => ActionMode::Shadow,
                        "active" => ActionMode::Active,
                        _ => {
                            eprintln!("bizra-node: invalid --action-mode value: {}", args[i]);
                            process::exit(2);
                        }
                    };
                    node_config.runtime_config.action_mode = mode;
                }
            }
            "--mcp-port" => {
                i += 1;
                if i < args.len() {
                    mcp_port = Some(args[i].parse().unwrap_or_else(|_| {
                        eprintln!("bizra-node: invalid --mcp-port value: {}", args[i]);
                        process::exit(2);
                    }));
                }
            }
            "--no-persist" => {
                auto_persist = false;
            }
            "--no-banner" => {
                node_config.show_banner = false;
            }
            "--no-auto-session" => {
                node_config.auto_start_session = false;
            }
            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }
            "--version" | "-v" => {
                println!("bizra-node {}", bizra_node::protocol::NODE_VERSION);
                process::exit(0);
            }
            other => {
                eprintln!("bizra-node: unknown argument: {}", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let effective_policy_hash = cli_policy_hash.or_else(|| {
        std::env::var("BIZRA_GENESIS_POLICY_HASH")
            .ok()
            .filter(|v| v.len() == 64 && v.chars().all(|c| c.is_ascii_hexdigit()))
    });
    if let Some(hash) = effective_policy_hash {
        node_config.runtime_config.policy_hash_hex = hash;
    }

    CliConfig {
        node_config,
        seed_file,
        state_dir,
        auto_persist,
        mcp_port,
    }
}

fn print_help() {
    println!(
        r#"
bizra-node — Node0: The Sovereign AI Node

USAGE:
    bizra-node [OPTIONS]

OPTIONS:
    --user <hash>       User identity hash (default: 1)
    --ihsan <floor>     إحسان quality floor, 0-10000 (default: 9500)
    --seed <file>       Load knowledge from a .seed file at startup
    --state-dir <path>  Directory for persistent state (default: ~/.bizra/node-<hash>)
    --reflex-mode <m>   Reflex routing mode: disabled|shadow|active
    --action-mode <m>   Action execution mode: disabled|shadow|active
    --policy-hash <h>   Genesis policy hash (64 hex). Or env BIZRA_GENESIS_POLICY_HASH.
    --mcp-port <port>   Enable MCP JSON-RPC 2.0 TCP transport (default: stdin/stdout)
    --no-persist        Disable auto-save on shutdown
    --no-banner         Suppress startup banner
    --no-auto-session   Don't auto-start a conversation session
    --help, -h          Show this help
    --version, -v       Show version

PERSISTENCE:
    By default, knowledge auto-saves to ~/.bizra/node-<hash>/knowledge.seed
    reflex rules to ~/.bizra/node-<hash>/reflex.cache,
    and action receipts to ~/.bizra/node-<hash>/actions.log.
    State auto-loads on startup. Your knowledge is a text file.
    You own it. Back it up. Guard it. Share it — or don't.

PROTOCOL:
    stdin  → VERB<TAB>arg1<TAB>arg2<NEWLINE>
    stdout → OK<TAB>field=value<TAB>...<NEWLINE>

COMMANDS:
    RECEIVE <content> <ts>             Process a user message
    TEACH <kind> <content> <conf> <ts> Teach the node directly
    SYNTHESIZE <ts>                    Force memory synthesis
    QUERY <key>                        Query a user trait
    PROFILE                            Get full user profile
    KNOWS_ME                           Get "my AI knows me" score
    HEALTH                             Full system health
    PLAN_ACTION <json>                 Validate and stage an action plan
    RUN_ACTION <plan_id> <json>        Execute a staged or ad-hoc action
    ACTION_STATUS <action_id>          Fetch action execution status
    ACTION_HISTORY <limit> <cursor>    Fetch hash-chained action receipts
    START_SESSION <ts>                 Begin conversation session
    END_SESSION <ts>                   End conversation session
    IHSAN <score>                      Update إحسان score
    PING                               Keepalive check
    VERSION                            Node version info
    SHUTDOWN                           Graceful shutdown

  SAP v0 Protocol:
    SAP_MEET_OPEN <profile> <role> <ts>    Open SAP session
    SAP_MESSAGE <sid> <content> <ts>       Send message in session
    SAP_DISCLOSURE <sid>                   Request disclosure
    SAP_CONSENT_REQUEST <sid> <scopes>     Request consent
    SAP_CONSENT_REVOKE <sid> <receipt_id>  Revoke consent
    SAP_SESSION_CLOSE <sid> <ts>           Close session

    Every seed has infinite potential. ربي لا يعرف المستحيل
"#
    );
}
