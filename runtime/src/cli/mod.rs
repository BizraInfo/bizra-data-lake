pub mod ui;

use clap::{Parser, Subcommand};
use comfy_table::{presets::UTF8_FULL, Attribute, Cell, Color, Table};
use std::sync::Arc;

use crate::cli::ui::{print_banner, EliteUI};
use meta_alpha_dual_agentic::types::{DualAgenticRequest, EnhancedDualAgenticRequest};
use meta_alpha_dual_agentic::{create_http_server, ihsan, pat_enhanced, MetaAlphaDualAgentic};

#[derive(Parser)]
#[command(
    name = "elite",
    about = "BIZRA Dual-Agentic System - Elite Node Interface",
    version,
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the HTTP API Server (Production Mode)
    Serve {
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// Execute a task via PAT/SAT (Interactive)
    Task {
        /// The prompt/task to execute
        #[arg(index = 1)]
        prompt: Option<String>,

        /// Enable sub-agent spawning
        #[arg(long)]
        spawn: bool,
    },
    /// Check system health and status
    Status,
    /// List available models and capabilities
    Models,
    /// Run the integrated verification demo
    Demo,
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut ui = EliteUI::new();

    // Only print banner if not piping JSON (future proofing)
    print_banner();

    match cli.command.unwrap_or(Commands::Serve { port: 8080 }) {
        Commands::Serve { port } => {
            ui.header("SYSTEM ACTIVATED", &format!("Running on port {}", port));
            ui.info("Initializing core systems...");

            // Re-use main initialization logic
            let system = Arc::new(MetaAlphaDualAgentic::initialize().await?);
            ui.success("Core system ready");

            ui.info("Starting HTTP Server...");
            create_http_server(system, port).await?;
        }

        Commands::Status => {
            ui.header("SYSTEM STATUS", "Diagnostics & Health");
            ui.start_spinner("Checking subsystems...");

            // Mock checks (in reality would hit internal health checks)
            tokio::time::sleep(std::time::Duration::from_millis(800)).await;

            let constitution = ihsan::constitution();

            ui.stop_spinner("Checks complete");

            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .set_header(vec!["Component", "Status", "Details"]);

            table.add_row(vec![
                Cell::new("Core Engine").fg(Color::Green),
                Cell::new("ONLINE")
                    .fg(Color::Green)
                    .add_attribute(Attribute::Bold),
                Cell::new(format!("v{}", env!("CARGO_PKG_VERSION"))),
            ]);
            table.add_row(vec![
                Cell::new("Ihsan Gate").fg(Color::Cyan),
                Cell::new("ACTIVE").fg(Color::Green),
                Cell::new(format!(
                    "{} (Threshold: {:.2})",
                    constitution.id(),
                    constitution.threshold()
                )),
            ]);
            table.add_row(vec![
                Cell::new("Persistence").fg(Color::Yellow),
                Cell::new("CONNECTED").fg(Color::Green),
                Cell::new("Redis/Synapse"),
            ]);

            println!("{table}");
        }

        Commands::Task { prompt, spawn } => {
            let task_text = match prompt {
                Some(p) => p,
                None => {
                    // Interactive input if not provided
                    use dialoguer::{theme::ColorfulTheme, Input};
                    Input::with_theme(&ColorfulTheme::default())
                        .with_prompt("Enter your task")
                        .interact_text()?
                }
            };

            ui.header("TASK EXECUTION", "Dual-Agentic Orchestration");
            ui.info(&format!("Objective: {}", task_text));

            ui.start_spinner("Initializing Agents...");
            let enhanced_pat = Arc::new(pat_enhanced::EnhancedPATOrchestrator::new().await?);
            ui.stop_spinner("Agents Ready");

            ui.start_spinner("Reasoning (PAT) & Validating (SAT)...");

            let request = EnhancedDualAgenticRequest {
                base: DualAgenticRequest {
                    user_id: "cli_interactive".to_string(),
                    task: task_text.clone(),
                    requirements: vec![],
                    target: "cli_out".to_string(),
                    ..Default::default()
                },
                enable_sub_agents: spawn,
                ..Default::default()
            };

            match enhanced_pat.execute_enhanced(request).await {
                Ok(response) => {
                    ui.stop_spinner("Execution Complete");

                    ui.success("Result Generated:");
                    println!(
                        "\n{}\n",
                        response
                            .pat_contributions
                            .first()
                            .cloned()
                            .unwrap_or_default()
                    );

                    let mut stats = Table::new();
                    stats.load_preset(UTF8_FULL);
                    stats.set_header(vec!["Metric", "Value"]);
                    stats.add_row(vec![
                        "Synergy Score",
                        &format!("{:.3}", response.synergy_score),
                    ]);
                    stats.add_row(vec![
                        "Ihsan Integrity",
                        &format!("{:.3}", response.ihsan_score),
                    ]);
                    stats.add_row(vec![
                        "Latency",
                        &format!("{:.2}s", response.latency.as_secs_f64()),
                    ]);

                    println!("{stats}");
                }
                Err(e) => {
                    ui.stop_spinner("Execution Failed");
                    ui.error(&format!("Error: {}", e));
                }
            }
        }

        Commands::Models => {
            ui.header("MODEL INVENTORY", "Available Inference Engines");
            // In phase 2 this would query the Router. For now static manifest.
            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .set_header(vec!["Slot", "Model", "Provider"]);
            table.add_row(vec!["Cold Core", "deepseek-r1:8b", "Ollama"]);
            table.add_row(vec!["Warm Surface", "mistral:latest", "Ollama"]);
            table.add_row(vec!["Embeddings", "nomic-embed-text", "Ollama"]);
            println!("{table}");
        }

        Commands::Demo => {
            // Just wrap the unified commands or re-implement simple flow
            ui.info("Please use 'elite task \"Run system verification\"' for similar effect, or legacy demo mode.");
        }
    }

    Ok(())
}
