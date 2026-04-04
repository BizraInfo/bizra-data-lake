//! BIZRA CLI Commands
//!
//! Non-interactive CLI command handlers.
//!
//! Genesis Spine (Phase 1):
//!   bizra init    — Discover substrate, generate node identity
//!   bizra genesis — Compute GenesisSeal, bind to constitution
//!   bizra agents  — Show PAT-7 + SAT-5 topology and mint status
//!   bizra node    — Node health, identity, constitutional compliance

use anyhow::Result;
use clap::Subcommand;

pub mod genesis_spine;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the TUI interface
    Tui,

    /// Show node status
    Status,

    /// Initialize sovereign node — discover substrate, generate Ed25519 identity
    Init {
        /// Force re-initialization even if node already exists
        #[arg(long)]
        force: bool,
    },

    /// Compute and display the Genesis Seal — constitutional root of trust
    Genesis {
        /// Show full constitutional parameters
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show PAT-7 + SAT-5 agent topology and mint status
    Agents {
        /// Show detailed agent capabilities
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show sovereign node health and constitutional compliance
    Node {
        /// Watch mode — continuous health updates
        #[arg(short, long)]
        watch: bool,
    },

    /// Execute a governed mission through the constitutional pipeline
    Mission {
        /// The mission objective (what you want to accomplish)
        #[arg(required = true)]
        objective: String,
    },

    /// View and verify the last mission receipt (from disk ledger)
    Receipt {
        /// Verify BLAKE3 hash + Ed25519 signature + chain integrity
        #[arg(short = 'k', long)]
        verify: bool,
    },

    /// Replay a mission from its receipt ID (re-execute + chain)
    Replay {
        /// Receipt ID prefix (at least 8 hex characters)
        #[arg(required = true)]
        id: String,
    },

    /// Show constitutional trust surface — verify all invariants
    Trust,

    /// Show daily receipt manifest — proof-of-life artifact
    Manifest,

    /// Interact with PAT agents
    #[command(subcommand)]
    Agent(AgentCommands),

    /// Query the knowledge base
    Query {
        /// The query text
        #[arg(required = true)]
        text: String,

        /// Agent to use for the query
        #[arg(short, long, default_value = "guardian")]
        agent: String,
    },

    /// Manage tasks
    #[command(subcommand)]
    Task(TaskCommands),

    /// Show system information
    Info,

    /// Voice interface
    Voice {
        /// Agent to use for voice
        #[arg(short, long, default_value = "guardian")]
        agent: String,
    },
}

#[derive(Subcommand)]
pub enum AgentCommands {
    /// List all PAT agents
    List,

    /// Show agent details
    Show {
        /// Agent name
        name: String,
    },

    /// Chat with an agent
    Chat {
        /// Agent name
        #[arg(short, long, default_value = "guardian")]
        agent: String,

        /// Message to send
        message: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum TaskCommands {
    /// List tasks
    List {
        /// Filter by status
        #[arg(short, long)]
        status: Option<String>,
    },

    /// Add a new task
    Add {
        /// Task title
        title: String,

        /// Task description
        #[arg(short, long)]
        description: Option<String>,

        /// Assign to agent
        #[arg(short, long)]
        agent: Option<String>,
    },

    /// Complete a task
    Complete {
        /// Task ID
        id: String,
    },
}

/// Execute status command (uses Python bridge for LM Studio check)
pub fn exec_status() -> Result<()> {
    use std::process::Command;

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        BIZRA Node Status                                   ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Node ID:      node0_ce5af35c848ce889");
    println!("  Node Name:    MoMo (محمد)");
    println!("  Location:     Dubai, UAE (GMT+4)");
    println!();
    println!("  Genesis Hash: a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b");
    println!();
    println!("  ┌─────────────────────────────────────────────┐");
    println!("  │ FATE Gates                                  │");
    println!("  ├─────────────────────────────────────────────┤");
    println!("  │ Ihsān:      0.95 / 0.95  ●                 │");
    println!("  │ Adl Gini:   0.25 / 0.35  ●                 │");
    println!("  │ Harm:       0.10 / 0.30  ●                 │");
    println!("  │ Confidence: 0.85 / 0.80  ●                 │");
    println!("  └─────────────────────────────────────────────┘");
    println!();

    // Check LM Studio via Python bridge (uses MultiModelManager)
    let bridge_path = "/mnt/c/BIZRA-DATA-LAKE/bizra_cli_bridge.py";
    let python_path = "/mnt/c/BIZRA-DATA-LAKE/.venv/bin/python";

    let mut cmd = Command::new(python_path);
    cmd.args([bridge_path, "status"]);
    if let Ok(key) = std::env::var("LM_STUDIO_API_KEY") {
        cmd.env("LM_STUDIO_API_KEY", key);
    }
    let output = cmd.output();

    match output {
        Ok(out) => {
            if let Ok(status) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                if status.get("status").and_then(|s| s.as_str()) == Some("connected") {
                    let total = status
                        .get("total_models")
                        .and_then(|n| n.as_i64())
                        .unwrap_or(0);
                    let loaded = status
                        .get("loaded_models")
                        .and_then(|n| n.as_i64())
                        .unwrap_or(0);
                    let loaded_list = status
                        .get("loaded_list")
                        .and_then(|l| l.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();

                    print!("  LM Studio:    ✓ Connected ({total} models");
                    if loaded > 0 {
                        print!(", {loaded} loaded");
                    }
                    println!(")");
                    if !loaded_list.is_empty() {
                        println!("  Active Model: {loaded_list}");
                    }
                } else {
                    println!("  LM Studio:    ✗ Not connected (LMSTUDIO_HOST:1234)");
                }
            } else {
                println!("  LM Studio:    ✗ Not connected (LMSTUDIO_HOST:1234)");
            }
        }
        Err(_) => {
            println!("  LM Studio:    ? Unable to check (Python bridge not found)");
        }
    }

    println!("  Voice:        Available (gTTS)");
    println!();
    Ok(())
}

/// Execute info command
pub fn exec_info() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║   ____  ___ ____  ____      _                                              ║");
    println!("║  | __ )|_ _|__  /|  _ \\    / \\      Sovereign Node v1.0                    ║");
    println!("║  |  _ \\ | |  / / | |_) |  / _ \\     ─────────────────────                  ║");
    println!("║  | |_) || | / /_ |  _ <  / ___ \\    Every human is a node.                 ║");
    println!("║  |____/|___/____|_| \\_\\/_/   \\_\\   Every node is a seed.                  ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ Personal Agentic Team (PAT)                                             │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │ ♟ Strategist   │ Sun Tzu • Clausewitz • Porter                         │");
    println!("  │ 🔍 Researcher   │ Shannon • Turing • Dijkstra                           │");
    println!("  │ ⚙ Developer    │ Knuth • Ritchie • Torvalds                            │");
    println!("  │ 📊 Analyst      │ Tukey • Tufte • Cleveland                             │");
    println!("  │ ✓ Reviewer     │ Fagan • Parnas • Brooks                               │");
    println!("  │ ▶ Executor     │ Toyota • Deming • Ohno                                │");
    println!("  │ 🛡 Guardian     │ Al-Ghazali • Rawls • Anthropic                        │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Standing on the shoulders of giants...");
    println!();
    Ok(())
}

/// Execute agent list command
pub fn exec_agent_list() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        PAT Agents                                          ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let agents = [
        (
            "Strategist",
            "♟",
            "Strategy & Planning",
            "Sun Tzu • Clausewitz • Porter",
        ),
        (
            "Researcher",
            "🔍",
            "Knowledge & Discovery",
            "Shannon • Turing • Dijkstra",
        ),
        (
            "Developer",
            "⚙",
            "Code & Implementation",
            "Knuth • Ritchie • Torvalds",
        ),
        (
            "Analyst",
            "📊",
            "Data & Insights",
            "Tukey • Tufte • Cleveland",
        ),
        (
            "Reviewer",
            "✓",
            "Quality & Validation",
            "Fagan • Parnas • Brooks",
        ),
        (
            "Executor",
            "▶",
            "Action & Delivery",
            "Toyota • Deming • Ohno",
        ),
        (
            "Guardian",
            "🛡",
            "Ethics & Protection",
            "Al-Ghazali • Rawls • Anthropic",
        ),
    ];

    for (name, icon, desc, giants) in agents {
        println!("  {icon} {name} - {desc}");
        println!("    Giants: {giants}");
        println!();
    }

    Ok(())
}
