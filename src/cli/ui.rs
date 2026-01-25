use console::{style, Emoji, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct EliteUI {
    term: Term,
    spinner: Option<ProgressBar>,
}

impl EliteUI {
    pub fn new() -> Self {
        Self {
            term: Term::stdout(),
            spinner: None,
        }
    }

    pub fn header(&self, title: &str, subtitle: &str) {
        self.term.clear_screen().ok();
        println!("\n{}", style(title).bold().cyan().underlined());
        println!("{}\n", style(subtitle).dim());
    }

    pub fn info(&self, msg: &str) {
        println!("{} {}", Emoji("ℹ️ ", "*"), msg);
    }

    pub fn success(&self, msg: &str) {
        println!("{} {}", Emoji("✅ ", "+"), style(msg).green().bold());
    }

    pub fn warning(&self, msg: &str) {
        println!("{} {}", Emoji("⚠️ ", "!"), style(msg).yellow());
    }

    pub fn error(&self, msg: &str) {
        eprintln!("{} {}", Emoji("❌ ", "x"), style(msg).red().bold());
    }

    pub fn start_spinner(&mut self, msg: &str) {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        pb.set_message(msg.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        self.spinner = Some(pb);
    }

    pub fn stop_spinner(&mut self, final_msg: &str) {
        if let Some(pb) = &self.spinner {
            pb.finish_with_message(final_msg.to_string());
        }
        self.spinner = None;
    }

    pub fn table_header(&self, _columns: Vec<&str>) {
        // Simple header printer if needed outside comfy-table
        // Comfy table handles this better mostly
    }
}

pub fn print_banner() {
    let banner = r#"
    ██████╗ ██╗███████╗██████╗  █████╗ 
    ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗
    ██████╔╝██║  ███╔╝ ██████╔╝███████║
    ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║
    ██████╔╝██║███████╗██║  ██║██║  ██║
    ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
           ELITE META ALPHA
    "#;
    println!("{}", style(banner).cyan());
}
