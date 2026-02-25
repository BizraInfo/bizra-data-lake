//! Alpha-100 subcommand handler
//!
//! Implements the `bizra alpha100` family of subcommands for bootstrapping
//! a new Alpha-100 node: doctor (pre-flight), install, launch, and status.
//!
//! CRITICAL: Fail-closed rule — if reflex_mode is "active" but the policy
//! hash is empty or invalid, the mode is automatically downgraded to "shadow".

use crate::binary_fetch;
use crate::config::{self, Alpha100Config};
use crate::policy;
use crate::provider;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Alpha100Commands {
    /// Pre-flight checks for Alpha-100 install
    Doctor,
    /// Install Node0 for Alpha-100
    Install(InstallArgs),
    /// Launch Node0 after install
    Launch,
    /// Show Alpha-100 installation status
    Status,
}

#[derive(Args)]
pub struct InstallArgs {
    /// Run without interactive prompts
    #[arg(long)]
    pub non_interactive: bool,
    /// Provider: "local", "anthropic", "openai"
    #[arg(long, default_value = "local")]
    pub provider: String,
    /// Local backend: "ollama", "lmstudio"
    #[arg(long, default_value = "ollama")]
    pub local_backend: String,
    /// Model to use (auto-detected if omitted)
    #[arg(long)]
    pub model: Option<String>,
    /// Reflex mode: "disabled", "shadow", "active"
    #[arg(long, default_value = "shadow")]
    pub reflex_mode: String,
    /// Path to the policy file for hash computation
    #[arg(long)]
    pub policy_file: Option<PathBuf>,
    /// Custom state directory
    #[arg(long)]
    pub state_dir: Option<PathBuf>,
}

/// Dispatch an Alpha100 subcommand.
pub fn dispatch(cmd: Alpha100Commands) -> Result<()> {
    match cmd {
        Alpha100Commands::Doctor => run_doctor(),
        Alpha100Commands::Install(args) => run_install(&args),
        Alpha100Commands::Launch => run_launch(),
        Alpha100Commands::Status => run_status(),
    }
}

/// Pre-flight checks: verify that all required components are available.
pub fn run_doctor() -> Result<()> {
    println!("\n  Alpha-100 Doctor — Pre-flight Checks\n");

    // 1. Rust toolchain
    let rust_ok = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    print_check("Rust toolchain", rust_ok);

    // 2. Node binary
    let node_bin = binary_fetch::locate_node_binary();
    print_check("bizra-node binary", node_bin.is_some());
    if let Some(ref path) = node_bin {
        println!("       path: {}", path.display());
    }

    // 3. Policy file
    let default_policy = config::alpha100_dir().join("policy.txt");
    let policy_exists = default_policy.exists();
    print_check("Policy file", policy_exists);
    if !policy_exists {
        println!("       expected: {}", default_policy.display());
    }

    // 4. Provider reachable
    let ollama = provider::probe_ollama();
    let lmstudio = provider::probe_lmstudio();
    let provider_ok = ollama.available || lmstudio.available;
    print_check("LLM provider reachable", provider_ok);
    if ollama.available {
        println!("       ollama: {} model(s)", ollama.models.len());
    }
    if lmstudio.available {
        println!("       lmstudio: {} model(s)", lmstudio.models.len());
    }

    // Summary
    let all_pass = rust_ok && node_bin.is_some() && policy_exists && provider_ok;
    println!();
    if all_pass {
        println!("  All checks passed. Ready for install.");
    } else {
        println!("  Some checks failed. Resolve issues before installing.");
    }

    Ok(())
}

/// Install Node0 for Alpha-100.
///
/// FAIL-CLOSED RULE: If reflex_mode is "active" and the policy_hash is
/// empty or not a valid 64-hex string, downgrade to "shadow" with a warning.
pub fn run_install(args: &InstallArgs) -> Result<()> {
    println!("\n  Alpha-100 Install\n");

    let base_dir = args.state_dir.clone().unwrap_or_else(config::alpha100_dir);

    // Compute policy hash
    let policy_hash = if let Some(ref pf) = args.policy_file {
        let hash = policy::compute_policy_hash_from_file(pf)
            .with_context(|| format!("Policy file: {}", pf.display()))?;
        println!("  Policy hash: {hash}");
        hash
    } else {
        // Try default location
        let default_path = base_dir.join("policy.txt");
        if default_path.exists() {
            let hash = policy::compute_policy_hash_from_file(&default_path)?;
            println!("  Policy hash: {hash} (default)");
            hash
        } else {
            println!("  Policy hash: (none — no policy file found)");
            String::new()
        }
    };

    // FAIL-CLOSED: validate reflex_mode vs policy_hash
    let reflex_mode = enforce_fail_closed(&args.reflex_mode, &policy_hash);

    // Detect provider
    let (det_provider, det_backend, det_model) = if args.provider == "local" {
        provider::detect_best_provider()
    } else {
        (
            args.provider.clone(),
            args.local_backend.clone(),
            args.model.clone().unwrap_or_else(|| "default".to_string()),
        )
    };

    let model = args.model.clone().unwrap_or(det_model);
    let provider_name = if args.provider != "local" {
        args.provider.clone()
    } else {
        det_provider
    };
    let backend = if args.local_backend != "ollama" || args.provider != "local" {
        args.local_backend.clone()
    } else {
        det_backend
    };

    // Build user hash from timestamp (deterministic enough for a local install)
    let user_hash = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0);

    // Assemble config
    let mut cfg = Alpha100Config::default_for_user(user_hash);
    cfg.provider = provider_name;
    cfg.local_backend = backend;
    cfg.model = model;
    cfg.policy_hash = policy_hash;
    cfg.reflex_mode = reflex_mode;
    cfg.state_dir = base_dir.to_string_lossy().to_string();

    // Locate or note the node binary path
    if let Some(node_path) = binary_fetch::locate_node_binary() {
        cfg.node_binary_path = node_path.to_string_lossy().to_string();
    }

    // Save
    config::save_config(&cfg, &base_dir)?;
    config::save_provider_env(&cfg, &base_dir)?;

    // Print summary
    println!();
    println!("  Install Summary");
    println!("  ---------------");
    println!("  User Hash:     {}", cfg.user_hash);
    println!("  Provider:      {}", cfg.provider);
    println!("  Backend:       {}", cfg.local_backend);
    println!("  Model:         {}", cfg.model);
    println!("  Reflex Mode:   {}", cfg.reflex_mode);
    println!(
        "  Policy Hash:   {}",
        if cfg.policy_hash.is_empty() {
            "(none)"
        } else {
            &cfg.policy_hash
        }
    );
    println!("  State Dir:     {}", cfg.state_dir);
    println!("  Installed At:  {}", cfg.installed_at);
    println!();
    println!(
        "  Config saved to: {}",
        base_dir.join("install.toml").display()
    );
    println!(
        "  Env saved to:    {}",
        base_dir.join("provider.env").display()
    );
    println!();

    Ok(())
}

/// Launch the installed bizra-node.
pub fn run_launch() -> Result<()> {
    println!("\n  Alpha-100 Launch\n");

    let base_dir = config::alpha100_dir();
    let cfg = config::load_config(&base_dir)
        .context("No install found. Run 'bizra alpha100 install' first.")?;

    let node_path = PathBuf::from(&cfg.node_binary_path);
    if !node_path.exists() {
        anyhow::bail!(
            "Node binary not found at: {}\nRun 'bizra alpha100 doctor' to diagnose.",
            node_path.display()
        );
    }

    println!("  Starting bizra-node...");
    println!("  Binary:  {}", cfg.node_binary_path);
    println!("  Model:   {}", cfg.model);
    println!("  Backend: {}", cfg.local_backend);
    println!("  Reflex:  {}", cfg.reflex_mode);

    let mut cmd = std::process::Command::new(&node_path);
    cmd.arg("--provider").arg(&cfg.provider);
    cmd.arg("--model").arg(&cfg.model);
    cmd.arg("--state-dir").arg(&cfg.state_dir);

    if cfg.reflex_mode != "disabled" {
        cmd.arg("--reflex-mode").arg(&cfg.reflex_mode);
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("Failed to spawn bizra-node at {}", node_path.display()))?;

    println!();
    println!("  Node launched (PID: {})", child.id());
    println!("  Connection: local via {}", cfg.local_backend);
    println!();

    Ok(())
}

/// Show the current Alpha-100 installation status.
pub fn run_status() -> Result<()> {
    println!("\n  Alpha-100 Status\n");

    let base_dir = config::alpha100_dir();

    let cfg = match config::load_config(&base_dir) {
        Ok(c) => c,
        Err(_) => {
            println!("  Not installed.");
            println!("  Run 'bizra alpha100 install' to set up.");
            return Ok(());
        }
    };

    println!("  Installed:     yes");
    println!("  User Hash:     {}", cfg.user_hash);
    println!("  Provider:      {}", cfg.provider);
    println!("  Backend:       {}", cfg.local_backend);
    println!("  Model:         {}", cfg.model);
    println!("  Reflex Mode:   {}", cfg.reflex_mode);
    println!(
        "  Policy Hash:   {}",
        if cfg.policy_hash.is_empty() {
            "(none)"
        } else {
            &cfg.policy_hash
        }
    );
    println!("  State Dir:     {}", cfg.state_dir);
    println!("  Installed At:  {}", cfg.installed_at);

    // Ping node if running
    let node_path = PathBuf::from(&cfg.node_binary_path);
    let node_exists = node_path.exists();
    println!(
        "  Node Binary:   {} ({})",
        cfg.node_binary_path,
        if node_exists { "found" } else { "missing" }
    );

    // Quick check if provider is reachable
    let (provider_status, model_count) = match cfg.local_backend.as_str() {
        "ollama" => {
            let probe = provider::probe_ollama();
            (probe.available, probe.models.len())
        }
        "lmstudio" => {
            let probe = provider::probe_lmstudio();
            (probe.available, probe.models.len())
        }
        _ => (false, 0),
    };

    println!(
        "  Provider:      {} ({} model(s))",
        if provider_status { "online" } else { "offline" },
        model_count
    );

    println!();
    Ok(())
}

// ── Internal helpers ──────────────────────────────────────────────────

fn print_check(label: &str, pass: bool) {
    let symbol = if pass { "[PASS]" } else { "[FAIL]" };
    println!("  {symbol}  {label}");
}

/// Enforce the fail-closed rule: if reflex_mode is "active" but the
/// policy hash is empty or not a valid 64-hex string, downgrade to "shadow".
fn enforce_fail_closed(requested_mode: &str, policy_hash: &str) -> String {
    if requested_mode == "active" && !is_valid_policy_hash(policy_hash) {
        println!("  WARNING: reflex_mode='active' requires a valid policy hash.");
        println!("           Downgrading to 'shadow' (fail-closed).");
        "shadow".to_string()
    } else {
        requested_mode.to_string()
    }
}

/// Check if a string is a valid 64-character lowercase hex hash.
fn is_valid_policy_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_closed_downgrades_active_with_empty_hash() {
        let result = enforce_fail_closed("active", "");
        assert_eq!(result, "shadow");
    }

    #[test]
    fn fail_closed_downgrades_active_with_invalid_hash() {
        let result = enforce_fail_closed("active", "not-a-valid-hash");
        assert_eq!(result, "shadow");
    }

    #[test]
    fn fail_closed_allows_active_with_valid_hash() {
        let valid = "a".repeat(64);
        let result = enforce_fail_closed("active", &valid);
        assert_eq!(result, "active");
    }

    #[test]
    fn fail_closed_does_not_affect_shadow_mode() {
        let result = enforce_fail_closed("shadow", "");
        assert_eq!(result, "shadow");
    }

    #[test]
    fn fail_closed_does_not_affect_disabled_mode() {
        let result = enforce_fail_closed("disabled", "");
        assert_eq!(result, "disabled");
    }

    #[test]
    fn is_valid_policy_hash_accepts_64_hex() {
        let hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert!(is_valid_policy_hash(hash));
    }

    #[test]
    fn is_valid_policy_hash_rejects_short() {
        assert!(!is_valid_policy_hash("abcd"));
    }

    #[test]
    fn is_valid_policy_hash_rejects_non_hex() {
        let bad = "g".repeat(64);
        assert!(!is_valid_policy_hash(&bad));
    }
}
