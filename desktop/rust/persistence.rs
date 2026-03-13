// ============================================================
// PERSISTENCE — Sovereign State Management
// ============================================================
//
// Your knowledge lives as a text file. You can read it, edit it,
// back it up, or delete it. No cloud. No database. No lock-in.
//
// Format: one protocol command per line.
// Lines starting with # are comments.
// Empty lines are ignored.
//
// This is the simplest possible persistence that preserves
// full sovereignty. A .seed file IS your digital identity.
// ============================================================

use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::node::Node;

/// Load a seed file and replay commands into the node.
/// Returns (commands_loaded, errors_encountered)
pub fn load_seed(node: &mut Node, path: &Path) -> io::Result<(usize, usize)> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut loaded = 0usize;
    let mut errors = 0usize;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        // Skip comments and empty lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let response = node.execute(trimmed);
        if response.starts_with("OK") {
            loaded += 1;
        } else {
            errors += 1;
            eprintln!("  seed: error on line: {} → {}", trimmed, response);
        }
    }

    Ok((loaded, errors))
}

/// Save current node knowledge to a seed file.
/// This file can be loaded to restore the node's knowledge.
pub fn save_state(node: &Node, path: &Path) -> io::Result<usize> {
    let commands = node.runtime().export_seed_commands();

    let mut file = std::fs::File::create(path)?;

    writeln!(file, "# ══════════════════════════════════════════════")?;
    writeln!(file, "# BIZRA Node0 — Sovereign State File")?;
    writeln!(file, "# User: {}", node.user_hash())?;
    writeln!(file, "# Fragments: {}", commands.len())?;
    writeln!(file, "# Generated: (runtime export)")?;
    writeln!(file, "# ")?;
    writeln!(file, "# This file IS your digital identity.")?;
    writeln!(file, "# Guard it. Back it up. It belongs to YOU.")?;
    writeln!(file, "# ══════════════════════════════════════════════")?;
    writeln!(file)?;

    for cmd in &commands {
        writeln!(file, "{}", cmd)?;
    }

    Ok(commands.len())
}

/// Get the default state directory for a user
pub fn state_dir(user_hash: u32) -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    Path::new(&home).join(".bizra").join(format!("node-{}", user_hash))
}

/// Ensure the state directory exists
pub fn ensure_state_dir(user_hash: u32) -> io::Result<std::path::PathBuf> {
    let dir = state_dir(user_hash);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Default seed file path for a user
pub fn seed_path(user_hash: u32) -> std::path::PathBuf {
    state_dir(user_hash).join("knowledge.seed")
}
