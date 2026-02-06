//! BIZRA CLI Commands
//!
//! Non-interactive CLI command handlers.

use clap::{Args, Subcommand};
use anyhow::Result;


#[derive(Subcommand)]
pub enum Commands {
    /// Start the TUI interface
    Tui,

    /// Show node status
    Status,

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
                    let total = status.get("total_models").and_then(|n| n.as_i64()).unwrap_or(0);
                    let loaded = status.get("loaded_models").and_then(|n| n.as_i64()).unwrap_or(0);
                    let loaded_list = status.get("loaded_list")
                        .and_then(|l| l.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
                        .unwrap_or_default();

                    print!("  LM Studio:    ✓ Connected ({} models", total);
                    if loaded > 0 {
                        print!(", {} loaded", loaded);
                    }
                    println!(")");
                    if !loaded_list.is_empty() {
                        println!("  Active Model: {}", loaded_list);
                    }
                } else {
                    println!("  LM Studio:    ✗ Not connected (192.168.56.1:1234)");
                }
            } else {
                println!("  LM Studio:    ✗ Not connected (192.168.56.1:1234)");
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
        ("Strategist", "♟", "Strategy & Planning", "Sun Tzu • Clausewitz • Porter"),
        ("Researcher", "🔍", "Knowledge & Discovery", "Shannon • Turing • Dijkstra"),
        ("Developer", "⚙", "Code & Implementation", "Knuth • Ritchie • Torvalds"),
        ("Analyst", "📊", "Data & Insights", "Tukey • Tufte • Cleveland"),
        ("Reviewer", "✓", "Quality & Validation", "Fagan • Parnas • Brooks"),
        ("Executor", "▶", "Action & Delivery", "Toyota • Deming • Ohno"),
        ("Guardian", "🛡", "Ethics & Protection", "Al-Ghazali • Rawls • Anthropic"),
    ];

    for (name, icon, desc, giants) in agents {
        println!("  {} {} - {}", icon, name, desc);
        println!("    Giants: {}", giants);
        println!();
    }

    Ok(())
}
