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
mod inference;
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
    let rt = tokio::runtime::Runtime::new()?;

    match cli.command {
        None | Some(Commands::Tui) => {
            // Start TUI
            run_tui()
        }
        Some(Commands::Status) => commands::exec_status(),
        Some(Commands::Info) => commands::exec_info(),
        Some(Commands::Agent(cmd)) => match cmd {
            AgentCommands::List => commands::exec_agent_list(),
            AgentCommands::Show { name } => {
                println!("Agent: {}", name);
                Ok(())
            }
            AgentCommands::Chat { agent, message } => exec_agent_chat(&agent, message.as_deref()),
        },
        Some(Commands::Query { text, agent }) => exec_query(&text, &agent),
        Some(Commands::Task(cmd)) => match cmd {
            TaskCommands::List { status } => {
                println!("Tasks (filter: {:?})", status);
                Ok(())
            }
            TaskCommands::Add {
                title,
                description,
                agent,
            } => {
                println!("Add task: {} ({:?}, {:?})", title, description, agent);
                Ok(())
            }
            TaskCommands::Complete { id } => {
                println!("Complete task: {}", id);
                Ok(())
            }
        },
        Some(Commands::Voice { agent }) => {
            println!("Voice mode with agent: {}", agent);
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

    println!("  Query: {}", text);
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
                            println!("  {}", line);
                        }
                    }
                } else if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                    println!("  Error: {}", error);
                }
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // Try to parse JSON error from stdout
                if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                    if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                        println!("  LM Studio: {}", error);
                        println!();
                        println!("  Please start LM Studio and load a model.");
                    }
                } else if !stderr.is_empty() {
                    println!("  Error: {}", stderr);
                }
            }
        }
        Err(e) => {
            println!("  Error: Failed to run Python bridge: {}", e);
            println!("  Make sure Python venv is set up at /mnt/c/BIZRA-DATA-LAKE/.venv");
        }
    }

    println!();
    Ok(())
}

/// Execute agent chat via Python bridge
fn exec_agent_chat(agent: &str, message: Option<&str>) -> Result<()> {
    use std::io::{self, Write};
    use std::process::Command;

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
        println!("  You: {}", msg);
        println!();

        let output = create_cmd(&[bridge_path, "agent", &agent_lower, msg]).output();

        match output {
            Ok(out) => {
                if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                    if let Some(content) = response.get("content").and_then(|c| c.as_str()) {
                        println!("  {}: {}", agent_display.1, content);
                    } else if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                        println!("  Error: {}", error);
                    }
                }
            }
            Err(e) => println!("  Error: {}", e),
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
                            println!("  Error: {}", error);
                        }
                    }
                    println!();
                }
                Err(e) => {
                    println!();
                    println!("  Error: {}", e);
                    println!();
                }
            }
        }
    }

    println!();
    Ok(())
}

fn run_tui() -> Result<()> {
    use crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{backend::CrosstermBackend, Terminal};
    use std::io;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = app::App::new();

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
        eprintln!("Error: {}", err);
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut app::App,
) -> Result<()> {
    use crossterm::event::{self, Event, KeyCode};
    use std::time::Duration;

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
                            app.next_agent();
                        }
                        KeyCode::Char('k') | KeyCode::Up => {
                            app.prev_agent();
                        }
                        KeyCode::Char('1') => app.active_view = app::ActiveView::Dashboard,
                        KeyCode::Char('2') => app.active_view = app::ActiveView::Agents,
                        KeyCode::Char('3') => app.active_view = app::ActiveView::Chat,
                        KeyCode::Char('4') => app.active_view = app::ActiveView::Tasks,
                        KeyCode::Char('5') => app.active_view = app::ActiveView::Treasury,
                        KeyCode::Char('6') => app.active_view = app::ActiveView::Settings,
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

        // Clear expired status messages
        app.clear_expired_status();

        if app.should_quit {
            return Ok(());
        }
    }
}

fn ui(f: &mut ratatui::Frame, app: &app::App) {
    use crate::theme::Theme;
    use crate::widgets::{Header, StatusBar};
    use ratatui::{
        layout::{Constraint, Direction, Layout, Rect},
        text::Span,
        widgets::{Block, Borders, Clear, Paragraph},
    };

    let size = f.size();

    // Main layout: header, content, status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Header
            Constraint::Min(10),   // Content
            Constraint::Length(2), // Status bar
        ])
        .split(size);

    // Header
    let header = Header::new(&app.node_name, app.active_view)
        .lmstudio(app.lmstudio_connected)
        .voice(app.voice_active);
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

    // Status bar
    let status = StatusBar::new(app.input_mode)
        .agent(app.selected_agent.map(|a| a.name()))
        .message(app.status_message.as_ref().map(|(m, _)| m.as_str()));
    f.render_widget(status, chunks[2]);

    // Input box (when in editing mode)
    if app.input_mode != app::InputMode::Normal {
        let input_area = Rect {
            x: 1,
            y: chunks[1].height.saturating_sub(3),
            width: size.width.saturating_sub(2),
            height: 3,
        };

        let input_block = Block::default()
            .title(Span::styled(
                if app.input_mode == app::InputMode::Command {
                    " Command "
                } else {
                    " Message "
                },
                Theme::highlight(),
            ))
            .borders(Borders::ALL)
            .border_style(Theme::panel_border_focused())
            .style(Theme::panel_focused());

        let input = Paragraph::new(app.input.as_str())
            .style(Theme::text())
            .block(input_block);

        f.render_widget(Clear, input_area);
        f.render_widget(input, input_area);

        // Cursor position
        f.set_cursor(input_area.x + app.input.len() as u16 + 1, input_area.y + 1);
    }
}

fn render_dashboard(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use crate::theme::Theme;
    use crate::widgets::{AgentCard, FateGauge};
    use ratatui::layout::{Constraint, Direction, Layout};

    // Split into left (agents) and right (FATE + info)
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Left: Agent grid (2 columns)
    let agent_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
        ])
        .split(chunks[0]);

    let roles: Vec<_> = app::PATRole::all().to_vec();
    for (i, row) in agent_rows.iter().enumerate() {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(*row);

        for (j, col) in cols.iter().enumerate() {
            let idx = i * 2 + j;
            if idx < roles.len() {
                let role = roles[idx];
                if let Some(state) = app.agents.get(&role) {
                    let selected = app.selected_agent == Some(role);
                    let card = AgentCard::new(state).selected(selected).compact(true);
                    f.render_widget(card, *col);
                }
            }
        }
    }

    // Right: FATE gates and node info
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(5)])
        .split(chunks[1]);

    // FATE gauge
    let fate = FateGauge::new(&app.fate_gates);
    f.render_widget(fate, right_chunks[0]);

    // Node info
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph};

    let info_block = Block::default()
        .title(Span::styled(" Node Info ", Theme::title()))
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .style(Theme::panel());

    let info_text = vec![
        Line::from(vec![
            Span::styled("ID: ", Theme::muted()),
            Span::styled(&app.node_id, Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Name: ", Theme::muted()),
            Span::styled(&app.node_name, Theme::highlight()),
        ]),
        Line::from(vec![
            Span::styled("Genesis: ", Theme::muted()),
            Span::styled(&app.genesis_hash[..16], Theme::text()),
            Span::styled("...", Theme::muted()),
        ]),
    ];

    let info = Paragraph::new(info_text).block(info_block);
    f.render_widget(info, right_chunks[1]);
}

fn render_agents(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use crate::widgets::AgentCard;
    use ratatui::layout::{Constraint, Direction, Layout};

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
    use crate::theme::Theme;
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

    let agent_name = app.selected_agent.map(|a| a.name()).unwrap_or("Guardian");
    let title = format!(" Chat with {} ", agent_name);

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
                    Span::styled(format!("[{}] ", prefix), style),
                    Span::styled(line, Theme::text()),
                ]));
            } else {
                all_lines.push(Line::from(Span::styled(
                    format!("      {}", line),
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
    use crate::theme::Theme;
    use ratatui::text::Span;
    use ratatui::widgets::{Block, Borders, Paragraph};

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
    use crate::theme::Theme;
    use ratatui::text::Span;
    use ratatui::widgets::{Block, Borders, Paragraph};

    let block = Block::default()
        .title(Span::styled(" Treasury ", Theme::title()))
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .style(Theme::panel());

    let para = Paragraph::new("Treasury management coming soon...").block(block);
    f.render_widget(para, area);
}

fn render_settings(f: &mut ratatui::Frame, app: &app::App, area: ratatui::layout::Rect) {
    use crate::theme::Theme;
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph};

    let block = Block::default()
        .title(Span::styled(" Settings ", Theme::title()))
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .style(Theme::panel());

    let lines = vec![
        Line::from(vec![
            Span::styled("LM Studio: ", Theme::muted()),
            Span::styled("192.168.56.1:1234", Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Voice Port: ", Theme::muted()),
            Span::styled("8998", Theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Ihsān Threshold: ", Theme::muted()),
            Span::styled("0.95", Theme::ihsan()),
        ]),
    ];

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
}
