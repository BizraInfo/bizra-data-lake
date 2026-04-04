//! BIZRA CLI — Your Personal Command Center
//!
//! بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
//!
//! Usage:
//!   bizra            - Start TUI interface
//!   bizra status     - Show node status
//!   bizra agent list - List PAT agents
//!   bizra query "?"  - Query the knowledge base

use anyhow::Result;
use clap::Parser;

mod app;
mod commands;
// TUI scaffolding -- inference module used when direct Rust-to-LMStudio path replaces Python bridge
#[allow(dead_code)]
mod inference;
// TUI scaffolding -- theme constants/functions used as UI views are fully wired
#[allow(dead_code)]
mod theme;
mod widgets;

use commands::{AgentCommands, Commands, TaskCommands};

#[derive(Parser)]
#[command(name = "bizra")]
#[command(author = "MoMo (محمد) <sovereign@bizra.node0>")]
#[command(version = "1.0.0")]
#[command(about = "BIZRA Sovereign Node CLI — Your Personal Command Center")]
#[command(long_about = r#"
╔════════════════════════════════════════════════════════════════════════════╗
║   ____  ___ ____  ____      _        CLI v1.0.0                            ║
║  | __ )|_ _|__  /|  _ \    / \       Sovereign Node                        ║
║  |  _ \ | |  / / | |_) |  / _ \      ─────────────────                     ║
║  | |_) || | / /_ |  _ <  / ___ \     Standing on the                       ║
║  |____/|___/____|_| \_\/_/   \_\     shoulders of giants                   ║
╚════════════════════════════════════════════════════════════════════════════╝

EXAMPLES:
    bizra                    Start the TUI interface
    bizra status             Show node status
    bizra agent list         List all PAT agents
    bizra agent chat -a guardian "Hello"
    bizra query "What is BIZRA?"
    bizra voice              Start voice interface
"#)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("bizra_cli=info".parse()?),
        )
        .init();

    let cli = Cli::parse();

    // Create tokio runtime for async operations
    let _rt = tokio::runtime::Runtime::new()?;

    match cli.command {
        None | Some(Commands::Tui) => {
            // Start TUI
            run_tui()
        }
        Some(Commands::Status) => commands::exec_status(),
        Some(Commands::Info) => commands::exec_info(),

        // Genesis Spine (Phase 1)
        Some(Commands::Init { force }) => commands::genesis_spine::exec_init(force),
        Some(Commands::Genesis { verbose }) => commands::genesis_spine::exec_genesis(verbose),
        Some(Commands::Agents { verbose }) => commands::genesis_spine::exec_agents(verbose),
        Some(Commands::Node { watch }) => commands::genesis_spine::exec_node(watch),
        Some(Commands::Mission { objective }) => commands::genesis_spine::exec_mission(&objective),
        Some(Commands::Receipt { verify }) => commands::genesis_spine::exec_receipt(verify),
        Some(Commands::Replay { id }) => commands::genesis_spine::exec_replay(&id),
        Some(Commands::Trust) => commands::genesis_spine::exec_trust(),
        Some(Commands::Manifest) => commands::genesis_spine::exec_manifest(),
        Some(Commands::Brief) => commands::genesis_spine::exec_brief(),

        Some(Commands::Agent(cmd)) => match cmd {
            AgentCommands::List => commands::exec_agent_list(),
            AgentCommands::Show { name } => {
                println!("Agent: {name}");
                Ok(())
            }
            AgentCommands::Chat { agent, message } => exec_agent_chat(&agent, message.as_deref()),
        },
        Some(Commands::Query { text, agent }) => exec_query(&text, &agent),
        Some(Commands::Task(cmd)) => match cmd {
            TaskCommands::List { status } => {
                println!("Tasks (filter: {status:?})");
                Ok(())
            }
            TaskCommands::Add {
                title,
                description,
                agent,
            } => {
                println!("Add task: {title} ({description:?}, {agent:?})");
                Ok(())
            }
            TaskCommands::Complete { id } => {
                println!("Complete task: {id}");
                Ok(())
            }
        },
        Some(Commands::Voice { agent }) => {
            println!("Voice mode with agent: {agent}");
            println!("Note: Voice requires PersonaPlex server running at https://localhost:8998");
            Ok(())
        }
    }
}

/// Execute a query via Python bridge (uses MultiModelManager)
fn exec_query(text: &str, agent: &str) -> Result<()> {
    use std::process::Command;

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Query via {} {:>52}║", agent, "");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("  Query: {text}");
    println!();
    println!("  ─────────────────────────────────────────────────────────────────────────");
    println!();

    // Use Python bridge with existing MultiModelManager infrastructure
    let bridge_path = "/mnt/c/BIZRA-DATA-LAKE/bizra_cli_bridge.py";
    let python_path = "/mnt/c/BIZRA-DATA-LAKE/.venv/bin/python";

    // Pass LM Studio API key from environment
    let mut cmd = Command::new(python_path);
    cmd.args([bridge_path, "agent", agent, text]);

    // Inherit LM_STUDIO_API_KEY from environment
    if let Ok(key) = std::env::var("LM_STUDIO_API_KEY") {
        cmd.env("LM_STUDIO_API_KEY", key);
    }

    let output = cmd.output();

    match output {
        Ok(out) => {
            if out.status.success() {
                let response: serde_json::Value = serde_json::from_slice(&out.stdout)
                    .unwrap_or_else(|_| serde_json::json!({"content": String::from_utf8_lossy(&out.stdout).to_string()}));

                if let Some(content) = response.get("content").and_then(|c| c.as_str()) {
                    // Word wrap the response
                    for line in content.lines() {
                        if line.len() > 76 {
                            let mut start = 0;
                            while start < line.len() {
                                let end = std::cmp::min(start + 76, line.len());
                                println!("  {}", &line[start..end]);
                                start = end;
                            }
                        } else {
                            println!("  {line}");
                        }
                    }
                } else if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                    println!("  Error: {error}");
                }
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // Try to parse JSON error from stdout
                if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                    if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                        println!("  LM Studio: {error}");
                        println!();
                        println!("  Please start LM Studio and load a model.");
                    }
                } else if !stderr.is_empty() {
                    println!("  Error: {stderr}");
                }
            }
        }
        Err(e) => {
            println!("  Error: Failed to run Python bridge: {e}");
            println!("  Make sure Python venv is set up at /mnt/c/BIZRA-DATA-LAKE/.venv");
        }
    }

    println!();
    Ok(())
}

/// Execute agent chat via Python bridge
fn exec_agent_chat(agent: &str, message: Option<&str>) -> Result<()> {
    use std::{
        io::{self, Write},
        process::Command,
    };

    let agent_lower = agent.to_lowercase();
    let agent_display = match agent_lower.as_str() {
        "strategist" => ("♟", "Strategist"),
        "researcher" => ("🔍", "Researcher"),
        "developer" => ("⚙", "Developer"),
        "analyst" => ("📊", "Analyst"),
        "reviewer" => ("✓", "Reviewer"),
        "executor" => ("▶", "Executor"),
        _ => ("🛡", "Guardian"),
    };

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  {} {} Chat {:>56}║",
        agent_display.0, agent_display.1, ""
    );
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let bridge_path = "/mnt/c/BIZRA-DATA-LAKE/bizra_cli_bridge.py";
    let python_path = "/mnt/c/BIZRA-DATA-LAKE/.venv/bin/python";

    // Helper to create command with API key
    let create_cmd = |args: &[&str]| {
        let mut cmd = Command::new(python_path);
        cmd.args(args);
        if let Ok(key) = std::env::var("LM_STUDIO_API_KEY") {
            cmd.env("LM_STUDIO_API_KEY", key);
        }
        cmd
    };

    // If message provided, single response mode
    if let Some(msg) = message {
        println!("  You: {msg}");
        println!();

        let output = create_cmd(&[bridge_path, "agent", &agent_lower, msg]).output();

        match output {
            Ok(out) => {
                if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                    if let Some(content) = response.get("content").and_then(|c| c.as_str()) {
                        println!("  {}: {}", agent_display.1, content);
                    } else if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                        println!("  Error: {error}");
                    }
                }
            }
            Err(e) => println!("  Error: {e}"),
        }
    } else {
        // Interactive mode
        println!("  Type your message (or 'exit' to quit):");
        println!();

        loop {
            print!("  You: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }
            if input == "exit" || input == "quit" {
                println!();
                println!("  Goodbye!");
                break;
            }

            let output = create_cmd(&[bridge_path, "agent", &agent_lower, input]).output();

            match output {
                Ok(out) => {
                    println!();
                    if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                        if let Some(content) = response.get("content").and_then(|c| c.as_str()) {
                            println!("  {}: {}", agent_display.1, content);
                        } else if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                            println!("  Error: {error}");
                        }
                    }
                    println!();
                }
                Err(e) => {
                    println!();
                    println!("  Error: {e}");
                    println!();
                }
            }
        }
    }

    println!();
    Ok(())
}

fn run_tui() -> Result<()> {
    use std::io;

    use crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{backend::CrosstermBackend, Terminal};

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state with initial data gather
    let mut app = app::App::new();
    app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
    app.last_refresh = Some(std::time::Instant::now());

    // Welcome message with clean ASCII art
    app.add_message(
        "system",
        r#"
    ╔═══════════════════════════════════════════════════════════╗
    ║   ____  ___ ____  ____      _                             ║
    ║  | __ )|_ _|__  /|  _ \    / \                            ║
    ║  |  _ \ | |  / / | |_) |  / _ \                           ║
    ║  | |_) || | / /_ |  _ <  / ___ \                          ║
    ║  |____/|___/____|_| \_\/_/   \_\  Sovereign Node          ║
    ║                                                           ║
    ║  Standing on the shoulders of giants...                   ║
    ║  Your Personal Agentic Team (PAT) is ready.               ║
    ║                                                           ║
    ║  Press [i] to type, /help for commands                    ║
    ╚═══════════════════════════════════════════════════════════╝
"#,
        None,
    );

    // Main loop
    let res = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {err}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut app::App,
) -> Result<()>
where
    B::Error: Send + Sync + 'static,
{
    use std::time::Duration;

    use crossterm::event::{self, Event, KeyCode};

    loop {
        terminal.draw(|f| ui(f, app))?;

        // Poll for events with timeout
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match app.input_mode {
                    app::InputMode::Normal => match key.code {
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('i') => {
                            app.input_mode = app::InputMode::Editing;
                        }
                        KeyCode::Char('/') => {
                            app.input_mode = app::InputMode::Command;
                            app.input = "/".to_string();
                        }
                        KeyCode::Tab => {
                            app.next_view();
                        }
                        KeyCode::BackTab => {
                            app.prev_view();
                        }
                        KeyCode::Char('j') | KeyCode::Down => {
                            if app.active_view == app::ActiveView::Dashboard {
                                app.next_receipt();
                            } else {
                                app.next_agent();
                            }
                        }
                        KeyCode::Char('k') | KeyCode::Up => {
                            if app.active_view == app::ActiveView::Dashboard {
                                app.prev_receipt();
                            } else {
                                app.prev_agent();
                            }
                        }
                        KeyCode::Esc => {
                            // Deselect receipt on Dashboard
                            if app.active_view == app::ActiveView::Dashboard {
                                app.selected_receipt = None;
                            }
                        }
                        KeyCode::Char('r') => {
                            let mut new_data = commands::genesis_spine::gather_dashboard_data();
                            if let Some(ref prev) = app.dashboard_data {
                                let new_events =
                                    commands::genesis_spine::detect_events(prev, &new_data);
                                let mut log = prev.event_log.clone();
                                log.extend(new_events);
                                if log.len() > 20 {
                                    log.drain(..log.len() - 20);
                                }
                                new_data.event_log = log;
                            }
                            app.dashboard_data = Some(new_data);
                            app.last_refresh = Some(std::time::Instant::now());
                            app.set_status("Dashboard refreshed");
                        }
                        KeyCode::Char('m') => {
                            if app.active_view == app::ActiveView::Dashboard {
                                app.input_mode = app::InputMode::MissionInput;
                                app.input.clear();
                            }
                        }
                        KeyCode::Char('1') => app.active_view = app::ActiveView::Dashboard,
                        KeyCode::Char('2') => app.active_view = app::ActiveView::Agents,
                        KeyCode::Char('3') => app.active_view = app::ActiveView::Chat,
                        KeyCode::Char('4') => app.active_view = app::ActiveView::Tasks,
                        KeyCode::Char('5') => app.active_view = app::ActiveView::Treasury,
                        KeyCode::Char('6') => app.active_view = app::ActiveView::Settings,
                        _ => {}
                    },
                    app::InputMode::MissionInput => match key.code {
                        KeyCode::Esc => {
                            app.input_mode = app::InputMode::Normal;
                            app.input.clear();
                        }
                        KeyCode::Enter => {
                            let objective = app.input.trim().to_string();
                            if !objective.is_empty() {
                                app.set_status(format!("Submitting mission: {}...", &objective));
                                app.input_mode = app::InputMode::Normal;
                                app.input.clear();

                                // Execute governed mission through mission_bridge
                                match commands::genesis_spine::submit_mission_from_tui(&objective) {
                                    Ok(msg) => app.set_status(msg),
                                    Err(e) => app.set_status(format!("Mission failed: {e}")),
                                }

                                // Refresh dashboard to show new receipt
                                let mut new_data = commands::genesis_spine::gather_dashboard_data();
                                if let Some(ref prev) = app.dashboard_data {
                                    let new_events =
                                        commands::genesis_spine::detect_events(prev, &new_data);
                                    let mut log = prev.event_log.clone();
                                    log.extend(new_events);
                                    if log.len() > 20 {
                                        log.drain(..log.len() - 20);
                                    }
                                    new_data.event_log = log;
                                }
                                app.dashboard_data = Some(new_data);
                                app.last_refresh = Some(std::time::Instant::now());
                            } else {
                                app.input_mode = app::InputMode::Normal;
                                app.input.clear();
                            }
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        _ => {}
                    },
                    app::InputMode::Editing | app::InputMode::Command => {
                        match key.code {
                            KeyCode::Esc => {
                                app.input_mode = app::InputMode::Normal;
                                if app.input.starts_with('/') {
                                    app.input.clear();
                                }
                            }
                            KeyCode::Enter => {
                                app.process_input();
                                app.input_mode = app::InputMode::Normal;
                            }
                            KeyCode::Char(c) => {
                                app.input.push(c);
                            }
                            KeyCode::Backspace => {
                                app.input.pop();
                            }
                            KeyCode::Up => {
                                // History navigation
                                if !app.command_history.is_empty() {
                                    let idx = app
                                        .history_index
                                        .map_or(app.command_history.len() - 1, |i| {
                                            i.saturating_sub(1)
                                        });
                                    app.history_index = Some(idx);
                                    app.input = app.command_history[idx].clone();
                                }
                            }
                            KeyCode::Down => {
                                if let Some(idx) = app.history_index {
                                    if idx + 1 < app.command_history.len() {
                                        app.history_index = Some(idx + 1);
                                        app.input = app.command_history[idx + 1].clone();
                                    } else {
                                        app.history_index = None;
                                        app.input.clear();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Periodic dashboard refresh (every 5 seconds) with event detection
        if app
            .last_refresh
            .map(|t| t.elapsed() > Duration::from_secs(5))
            .unwrap_or(true)
        {
            let mut new_data = commands::genesis_spine::gather_dashboard_data();

            // Detect events by diffing previous state
            if let Some(ref prev) = app.dashboard_data {
                let new_events = commands::genesis_spine::detect_events(prev, &new_data);
                // Carry forward existing event log + append new events (cap at 20)
                let mut log = prev.event_log.clone();
                log.extend(new_events);
                if log.len() > 20 {
                    log.drain(..log.len() - 20);
                }
                new_data.event_log = log;
            }

            app.dashboard_data = Some(new_data);
            app.last_refresh = Some(std::time::Instant::now());
        }

        // Clear expired status messages
        app.clear_expired_status();

        if app.should_quit {
            return Ok(());
        }
    }
}

fn ui(f: &mut ratatui::Frame, app: &app::App) {
    use ratatui::{
        layout::{Constraint, Direction, Layout, Rect},
        text::Span,
        widgets::{Block, Borders, Clear, Paragraph},
    };

    use crate::{
        theme::Theme,
        widgets::{Header, StatusBar},
    };

    let size = f.area();

    // Main layout: header, content, status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Header
            Constraint::Min(10),   // Content
            Constraint::Length(2), // Status bar
        ])
        .split(size);

    // Header — augmented with trust verdict + model count
    let (trust_ok, model_ct) = app
        .dashboard_data
        .as_ref()
        .map(|d| {
            (
                d.trust_verdict == commands::genesis_spine::TrustVerdict::Sovereign,
                d.model_count,
            )
        })
        .unwrap_or((false, 0));

    let header = Header::new(&app.node_name, app.active_view)
        .lmstudio(app.lmstudio_connected)
        .voice(app.voice_active)
        .trust(trust_ok)
        .models(model_ct);
    f.render_widget(header, chunks[0]);

    // Content based on active view
    match app.active_view {
        app::ActiveView::Dashboard => render_dashboard(f, app, chunks[1]),
        app::ActiveView::Agents => render_agents(f, app, chunks[1]),
        app::ActiveView::Chat => render_chat(f, app, chunks[1]),
        app::ActiveView::Tasks => render_tasks(f, app, chunks[1]),
        app::ActiveView::Treasury => render_treasury(f, app, chunks[1]),
        app::ActiveView::Settings => render_settings(f, app, chunks[1]),
    }

    // Status bar — augmented with manifest summary
    let manifest_text = app.dashboard_data.as_ref().map(|d| {
        format!(
            "{}/{}{}",
            d.today_count,
            d.today_complete,
            crate::theme::symbols::SUCCESS
        )
    });
    let status = StatusBar::new(app.input_mode)
        .agent(app.selected_agent.map(|a| a.name()))
        .message(app.status_message.as_ref().map(|(m, _)| m.as_str()))
        .manifest(manifest_text.as_deref());
    f.render_widget(status, chunks[2]);

    // Input box (when in editing mode)
    if app.input_mode != app::InputMode::Normal {
        let input_area = Rect {
            x: 1,
            y: chunks[1].height.saturating_sub(3),
            width: size.width.saturating_sub(2),
            height: 3,
        };

        let input_title = match app.input_mode {
            app::InputMode::Command => " Command ",
            app::InputMode::MissionInput => " Mission Objective (Enter to submit, Esc to cancel) ",
            _ => " Message ",
        };
        let input_block = Block::default()
            .title(Span::styled(input_title, Theme::highlight()))
            .borders(Borders::ALL)
            .border_style(Theme::panel_border_focused())
            .style(Theme::panel_focused());

        let input = Paragraph::new(app.input.as_str())
            .style(Theme::text())
            .block(input_block);

        f.render_widget(Clear, input_area);
        f.render_widget(input, input_area);

        // Cursor position
        f.set_cursor_position((input_area.x + app.input.len() as u16 + 1, input_area.y + 1));
    }
}

fn render_dashboard(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use ratatui::layout::{Constraint, Direction, Layout};

    use crate::widgets::{
        GhostFeed, ParliamentPanel, ReceiptDetail, ReceiptRail, SubstratePanel, TrustRail,
    };

    // Guard: need data to render
    let Some(data) = &app.dashboard_data else {
        use ratatui::widgets::{Block, Borders, Paragraph};
        let block = Block::default()
            .title(" Dashboard ")
            .borders(Borders::ALL)
            .style(crate::theme::Theme::panel());
        let msg = Paragraph::new("Gathering sovereign intelligence...").block(block);
        f.render_widget(msg, area);
        return;
    };

    // ── 3-Column Layout ──
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30), // Left: Parliament + Substrate
            Constraint::Percentage(35), // Center: Ghost + Receipt
            Constraint::Percentage(35), // Right: Trust or Receipt Detail
        ])
        .split(area);

    // ── Left Column: Parliament (60%) + Substrate (40%) ──
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(columns[0]);

    // Zone 2: Parliament
    f.render_widget(
        ParliamentPanel::new(&data.pat_agents, &data.sat_agents),
        left[0],
    );

    // Zone 5: Substrate
    f.render_widget(SubstratePanel::from_data(data), left[1]);

    // ── Center Column: Ghost (50%) + Receipt (50%) ──
    let center = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(columns[1]);

    // Zone 3: Ghost Feed
    f.render_widget(GhostFeed::from_data(data), center[0]);

    // Zone 6: Receipt Rail
    f.render_widget(ReceiptRail::from_data(data), center[1]);

    // ── Right Column: Receipt Detail (if selected) or Trust Rail ──
    if let Some(idx) = app.selected_receipt {
        if let Some(receipt) = data.all_receipts.get(idx) {
            // Sprint 7.2: Receipt detail replaces trust rail when selected
            let right = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
                .split(columns[2]);

            f.render_widget(
                ReceiptDetail::new(receipt, idx, data.all_receipts.len()),
                right[0],
            );
            f.render_widget(TrustRail::from_data(data), right[1]);
        } else {
            f.render_widget(TrustRail::from_data(data), columns[2]);
        }
    } else {
        // Zone 4: Trust Rail (full height, default)
        f.render_widget(TrustRail::from_data(data), columns[2]);
    }
}

fn render_agents(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use ratatui::layout::{Constraint, Direction, Layout};

    use crate::widgets::AgentCard;

    // Full agent cards in a grid
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(area);

    let roles: Vec<_> = app::PATRole::all().to_vec();

    // Row 1: 3 agents
    let row1_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(rows[0]);

    for (i, col) in row1_cols.iter().enumerate() {
        if i < roles.len() {
            if let Some(state) = app.agents.get(&roles[i]) {
                let selected = app.selected_agent == Some(roles[i]);
                let card = AgentCard::new(state).selected(selected);
                f.render_widget(card, *col);
            }
        }
    }

    // Row 2: 3 agents
    let row2_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(rows[1]);

    for (i, col) in row2_cols.iter().enumerate() {
        let idx = 3 + i;
        if idx < roles.len() {
            if let Some(state) = app.agents.get(&roles[idx]) {
                let selected = app.selected_agent == Some(roles[idx]);
                let card = AgentCard::new(state).selected(selected);
                f.render_widget(card, *col);
            }
        }
    }

    // Row 3: 1 agent (Guardian, centered)
    if roles.len() > 6 {
        let row3_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(rows[2]);

        if let Some(state) = app.agents.get(&roles[6]) {
            let selected = app.selected_agent == Some(roles[6]);
            let card = AgentCard::new(state).selected(selected);
            f.render_widget(card, row3_cols[1]); // Center column
        }
    }
}

fn render_chat(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use ratatui::{
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph, Wrap},
    };

    use crate::theme::Theme;

    let agent_name = app.selected_agent.map(|a| a.name()).unwrap_or("Guardian");
    let title = format!(" Chat with {agent_name} ");

    let block = Block::default()
        .title(Span::styled(title, Theme::title()))
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .style(Theme::panel());

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Build all chat lines
    let mut all_lines: Vec<Line> = Vec::new();

    for msg in &app.chat_messages {
        let style = match msg.role.as_str() {
            "user" => Theme::highlight(),
            "system" => Theme::muted(),
            _ => {
                if let Some(agent) = msg.agent {
                    Theme::pat_agent(agent.name())
                } else {
                    Theme::text()
                }
            }
        };

        let prefix = match msg.role.as_str() {
            "user" => "You",
            "system" => "SYS",
            _ => &msg.role,
        };

        for (i, line) in msg.content.lines().enumerate() {
            if i == 0 {
                all_lines.push(Line::from(vec![
                    Span::styled(format!("[{prefix}] "), style),
                    Span::styled(line, Theme::text()),
                ]));
            } else {
                all_lines.push(Line::from(Span::styled(
                    format!("      {line}"),
                    Theme::text(),
                )));
            }
        }
        all_lines.push(Line::from("")); // Blank line between messages
    }

    // Auto-scroll: calculate how many lines we can show and scroll from bottom
    let visible_height = inner.height as usize;
    let total_lines = all_lines.len();
    let scroll_offset = if total_lines > visible_height {
        (total_lines - visible_height) as u16
    } else {
        0
    };

    let chat = Paragraph::new(all_lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));

    f.render_widget(chat, inner);
}

fn render_tasks(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use ratatui::{
        text::Span,
        widgets::{Block, Borders, Paragraph},
    };

    use crate::theme::Theme;

    let block = Block::default()
        .title(Span::styled(" Tasks ", Theme::title()))
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .style(Theme::panel());

    let text = if app.tasks.is_empty() {
        "No tasks. Use /task add <title> to create one."
    } else {
        "Tasks list..."
    };

    let para = Paragraph::new(text).block(block);
    f.render_widget(para, area);
}

fn render_treasury(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use ratatui::layout::{Constraint, Direction, Layout};

    use crate::widgets::{SkillTree, Wallet};

    let Some(data) = &app.dashboard_data else {
        use ratatui::{
            text::Span,
            widgets::{Block, Borders, Paragraph},
        };
        let block = Block::default()
            .title(Span::styled(" Treasury ", crate::theme::Theme::title()))
            .borders(Borders::ALL)
            .style(crate::theme::Theme::panel());
        let msg = Paragraph::new("Gathering treasury data...").block(block);
        f.render_widget(msg, area);
        return;
    };

    // Two-column: Wallet (50%) + Skills (50%)
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    f.render_widget(Wallet::from_data(data), columns[0]);
    f.render_widget(SkillTree::from_data(data), columns[1]);
}

fn render_settings(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use ratatui::{
        layout::{Constraint, Direction, Layout},
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph},
    };

    use crate::{theme::Theme, widgets::MemoryPanel};

    // Two-column: Memory (50%) + Config (50%)
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left: Memory panel (needs dashboard data)
    if let Some(data) = &app.dashboard_data {
        f.render_widget(MemoryPanel::from_data(data), columns[0]);
    } else {
        let block = Block::default()
            .title(Span::styled(" Memory ", Theme::title()))
            .borders(Borders::ALL)
            .style(Theme::panel());
        let msg = Paragraph::new("Gathering memory data...").block(block);
        f.render_widget(msg, columns[0]);
    }

    // Right: Configuration settings
    let block = Block::default()
        .title(Span::styled(" Settings ", Theme::title()))
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .style(Theme::panel());

    let lines = vec![
        Line::from(vec![
            Span::styled("LM Studio: ", Theme::muted()),
            Span::styled("auto-detected:1234", Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Voice Port: ", Theme::muted()),
            Span::styled("8998", Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Ihsan Threshold: ", Theme::muted()),
            Span::styled("0.95", Theme::ihsan()),
        ]),
        Line::from(vec![
            Span::styled("SNR Threshold:   ", Theme::muted()),
            Span::styled("0.85", Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Gini Ceiling:    ", Theme::muted()),
            Span::styled("0.35", Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Zakat Rate:      ", Theme::muted()),
            Span::styled("2.5%", Theme::ihsan()),
        ]),
    ];

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, columns[1]);
}

// ── Phase 6 TUI Smoke Tests (Headless) ───────────────────
#[cfg(test)]
mod tests {
    use ratatui::{backend::TestBackend, Terminal};

    use super::*;

    #[test]
    fn test_app_with_dashboard_data_no_panic() {
        // Construct App with live dashboard data — must not panic
        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());

        // Verify state
        assert!(app.dashboard_data.is_some());
        assert_eq!(app.active_view, app::ActiveView::Dashboard);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_render_dashboard_headless() {
        // Render the full dashboard to a test backend — must not panic
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());

        terminal.draw(|f| ui(f, &app)).unwrap();

        // Verify the buffer was written to (not blank)
        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            content.contains("BIZRA"),
            "Dashboard must render BIZRA header"
        );
        assert!(
            content.contains("Parliament"),
            "Dashboard must render Parliament panel"
        );
        assert!(
            content.contains("Ghost"),
            "Dashboard must render Ghost feed"
        );
        assert!(
            content.contains("Trust"),
            "Dashboard must render Trust rail"
        );
        assert!(
            content.contains("Substrate"),
            "Dashboard must render Substrate panel"
        );
        assert!(
            content.contains("Receipts"),
            "Dashboard must render Receipt rail"
        );
    }

    #[test]
    fn test_render_all_views_headless() {
        // Every view must render without panic
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());

        for view in app::ActiveView::all() {
            app.active_view = *view;
            terminal
                .draw(|f| ui(f, &app))
                .unwrap_or_else(|e| panic!("Render failed for {:?}: {}", view, e));
        }
    }

    #[test]
    fn test_view_navigation_cycle() {
        let mut app = app::App::new();
        assert_eq!(app.active_view, app::ActiveView::Dashboard);

        // Cycle through all views
        let views = app::ActiveView::all();
        for i in 0..views.len() {
            app.next_view();
            assert_eq!(app.active_view, views[(i + 1) % views.len()]);
        }

        // Back to Dashboard after full cycle
        assert_eq!(app.active_view, app::ActiveView::Dashboard);
    }

    #[test]
    fn test_agent_navigation_cycle() {
        let mut app = app::App::new();
        let initial = app.selected_agent;

        // Cycle through all agents
        for _ in 0..7 {
            app.next_agent();
        }
        // After 7 next_agent calls, should be back to initial
        assert_eq!(app.selected_agent, initial);
    }

    #[test]
    fn test_dashboard_no_data_renders_loading() {
        // When dashboard_data is None, should show loading message
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let app = app::App::new(); // No dashboard_data set
        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            content.contains("Gathering"),
            "No-data dashboard must show loading message"
        );
    }

    #[test]
    fn test_status_message_lifecycle() {
        let mut app = app::App::new();
        assert!(app.status_message.is_none());

        app.set_status("Dashboard refreshed");
        assert!(app.status_message.is_some());
        assert_eq!(
            app.status_message.as_ref().unwrap().0,
            "Dashboard refreshed"
        );

        // Status should not clear within 5 seconds
        app.clear_expired_status();
        assert!(app.status_message.is_some());
    }

    // ── Sprint 7.1: Event Rendering Test ──────────────────────

    #[test]
    fn test_ghost_feed_renders_injected_events() {
        use commands::genesis_spine::{EventKind, NodeEvent};

        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        let mut data = commands::genesis_spine::gather_dashboard_data();

        // Inject events into DashboardData
        data.event_log = vec![
            NodeEvent {
                kind: EventKind::ReceiptCreated,
                message: "Receipt abc123 — Test mission (Complete)".into(),
                timestamp: "14:30:00".into(),
            },
            NodeEvent {
                kind: EventKind::TrustChanged,
                message: "Trust degraded — checks failing".into(),
                timestamp: "14:30:05".into(),
            },
            NodeEvent {
                kind: EventKind::MissionCompleted,
                message: "1 mission completed today (total: 5)".into(),
                timestamp: "14:30:10".into(),
            },
        ];

        app.dashboard_data = Some(data);
        app.last_refresh = Some(std::time::Instant::now());

        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();

        // Ghost feed must render the Events section
        assert!(
            content.contains("Events"),
            "Ghost feed must render Events header"
        );
        // At least one event message should appear in the buffer
        assert!(
            content.contains("abc123") || content.contains("mission"),
            "Ghost feed must render injected event content"
        );
    }

    // ── Sprint 7.2: Receipt Detail + Navigation Tests ─────────

    #[test]
    fn test_receipt_detail_renders_with_selection() {
        use commands::genesis_spine::ReceiptSummary;

        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        let mut data = commands::genesis_spine::gather_dashboard_data();

        // Inject test receipts
        data.all_receipts = vec![ReceiptSummary {
            id_short: "deadbeef01234567".into(),
            objective: "Sovereign proof test".into(),
            state_label: "Complete",
            is_success: true,
            is_degraded: false,
            signed: true,
            ihsan_score: Some(0.97),
            snr_score: Some(0.93),
            chosen_model: Some("qwen2.5-coder:3b".into()),
            degradation_tier: 0,
            states_traversed: 7,
            chain_link: Some("prev0123456789ab".into()),
        }];

        app.dashboard_data = Some(data);
        app.last_refresh = Some(std::time::Instant::now());
        app.selected_receipt = Some(0); // Select first receipt

        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();

        // Receipt detail panel must render
        assert!(
            content.contains("deadbeef"),
            "Receipt detail must show receipt ID"
        );
        assert!(
            content.contains("0.97") || content.contains("Ihsan"),
            "Receipt detail must show Ihsan score or label"
        );
    }

    #[test]
    fn test_receipt_navigation_cycle() {
        let mut app = app::App::new();
        let mut data = commands::genesis_spine::gather_dashboard_data();

        // Inject 3 receipts
        data.all_receipts = vec![
            commands::genesis_spine::ReceiptSummary {
                id_short: "r1".into(),
                objective: "First".into(),
                state_label: "Complete",
                is_success: true,
                is_degraded: false,
                signed: true,
                ihsan_score: Some(0.96),
                snr_score: Some(0.90),
                chosen_model: None,
                degradation_tier: 0,
                states_traversed: 3,
                chain_link: None,
            },
            commands::genesis_spine::ReceiptSummary {
                id_short: "r2".into(),
                objective: "Second".into(),
                state_label: "Complete",
                is_success: true,
                is_degraded: false,
                signed: true,
                ihsan_score: Some(0.95),
                snr_score: Some(0.88),
                chosen_model: None,
                degradation_tier: 0,
                states_traversed: 4,
                chain_link: Some("r1hash".into()),
            },
            commands::genesis_spine::ReceiptSummary {
                id_short: "r3".into(),
                objective: "Third".into(),
                state_label: "Degraded",
                is_success: false,
                is_degraded: true,
                signed: true,
                ihsan_score: Some(0.80),
                snr_score: Some(0.70),
                chosen_model: None,
                degradation_tier: 2,
                states_traversed: 5,
                chain_link: Some("r2hash".into()),
            },
        ];
        app.dashboard_data = Some(data);

        assert!(app.selected_receipt.is_none());

        // Navigate forward
        app.next_receipt();
        assert_eq!(app.selected_receipt, Some(0));
        app.next_receipt();
        assert_eq!(app.selected_receipt, Some(1));
        app.next_receipt();
        assert_eq!(app.selected_receipt, Some(2));
        // Wraps around
        app.next_receipt();
        assert_eq!(app.selected_receipt, Some(0));

        // Navigate backward from 0 wraps to end
        app.prev_receipt();
        assert_eq!(app.selected_receipt, Some(2));
    }

    #[test]
    fn test_mission_input_mode() {
        let mut app = app::App::new();

        // Start in Normal mode
        assert_eq!(app.input_mode, app::InputMode::Normal);
        assert!(app.input.is_empty());

        // Simulate entering mission input mode
        app.input_mode = app::InputMode::MissionInput;
        app.input = "Test sovereign mission".into();

        assert_eq!(app.input_mode, app::InputMode::MissionInput);
        assert_eq!(app.input, "Test sovereign mission");

        // Simulate cancel
        app.input_mode = app::InputMode::Normal;
        app.input.clear();
        assert!(app.input.is_empty());
    }

    #[test]
    fn test_dashboard_renders_without_selection() {
        // Dashboard with no receipt selected should render trust rail (not detail)
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());
        app.selected_receipt = None;

        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();

        assert!(
            content.contains("Trust"),
            "Must show Trust rail when no receipt selected"
        );
    }

    // ── Sprint 7.3+7.4: Treasury + Memory + Skills Rendering ──

    #[test]
    fn test_treasury_view_renders_wallet_and_skills() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());
        app.active_view = app::ActiveView::Treasury;

        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();

        assert!(
            content.contains("Treasury"),
            "Treasury view must render Wallet panel"
        );
        assert!(
            content.contains("Skills") || content.contains("Reflex"),
            "Treasury view must render Skills panel"
        );
    }

    #[test]
    fn test_settings_view_renders_memory_panel() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());
        app.active_view = app::ActiveView::Settings;

        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();

        assert!(
            content.contains("Memory"),
            "Settings view must render Memory panel"
        );
        assert!(
            content.contains("Settings"),
            "Settings view must render Settings panel"
        );
    }

    #[test]
    fn test_treasury_no_data_renders_loading() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.active_view = app::ActiveView::Treasury;
        // No dashboard_data

        terminal.draw(|f| ui(f, &app)).unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();

        assert!(
            content.contains("Gathering") || content.contains("Treasury"),
            "Treasury with no data must show loading"
        );
    }

    #[test]
    fn test_all_views_render_with_data() {
        // Every view must render without panic when data is present
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut app = app::App::new();
        app.dashboard_data = Some(commands::genesis_spine::gather_dashboard_data());
        app.last_refresh = Some(std::time::Instant::now());

        for view in app::ActiveView::all() {
            app.active_view = *view;
            terminal
                .draw(|f| ui(f, &app))
                .unwrap_or_else(|e| panic!("Render failed for {:?}: {}", view, e));
        }
    }
}
