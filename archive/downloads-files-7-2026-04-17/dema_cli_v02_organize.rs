//! Dema CLI v0.2 — Cycle-6 additions
//!
//! بسم الله الرحمن الرحيم
//!
//! Add these to the existing dema.rs Clap subcommands.
//! Integration: merge into src/bin/dema.rs on NODE0.

// ============================================================================
// Add to the existing Commands enum:
// ============================================================================

/*
    /// Organize a directory — categorize and move files with receipts
    Organize {
        /// Target directory (default: ~/Downloads)
        #[arg(default_value_t = default_downloads_dir())]
        path: String,

        /// Quality score (default: 0.98)
        #[arg(long, default_value_t = 0.98)]
        quality: f64,

        /// Dry run — show what would happen without moving files
        #[arg(long)]
        dry_run: bool,
    },

    /// Compile any operation through the trust compiler
    Compile {
        /// What you want to do
        intent: String,

        /// Operation kind: mission, filesystem, tool, lifecycle
        #[arg(long, default_value = "mission")]
        kind: String,

        /// Quality score
        #[arg(long, default_value_t = 0.98)]
        quality: f64,

        /// Target path (for filesystem operations)
        #[arg(long)]
        target: Option<String>,
    },
*/

// ============================================================================
// Handler implementations (add to the match block):
// ============================================================================

// --- `dema organize` handler ---

use std::path::PathBuf;

fn default_downloads_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());
    format!("{}/Downloads", home)
}

async fn handle_organize(
    base: &str,
    path: &str,
    quality: f64,
    dry_run: bool,
) -> anyhow::Result<()> {
    let resolved = if path.starts_with('~') {
        let home = std::env::var("HOME")?;
        path.replacen('~', &home, 1)
    } else {
        path.to_string()
    };

    // Verify directory exists before sending to gateway
    let dir = PathBuf::from(&resolved);
    if !dir.exists() {
        eprintln!("DEMA — directory does not exist: {}", resolved);
        std::process::exit(1);
    }
    if !dir.is_dir() {
        eprintln!("DEMA — not a directory: {}", resolved);
        std::process::exit(1);
    }

    // Count files first
    let file_count = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .count();

    if file_count == 0 {
        println!("DEMA — {} is already clean (0 files)", resolved);
        return Ok(());
    }

    if dry_run {
        println!("DEMA — DRY RUN: would organize {} files in {}", file_count, resolved);
        println!();

        // Show what categories files would go to
        let mut categories: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                let ext = entry.path().extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                let cat = categorize_ext(&ext);
                categories.entry(cat.to_string())
                    .or_default()
                    .push(entry.file_name().to_string_lossy().to_string());
            }
        }

        for (cat, files) in &categories {
            println!("  {}/ ({} files)", cat, files.len());
            for f in files.iter().take(5) {
                println!("    └─ {}", f);
            }
            if files.len() > 5 {
                println!("    └─ ... and {} more", files.len() - 5);
            }
        }
        println!();
        println!("Run without --dry-run to execute.");
        return Ok(());
    }

    println!("DEMA — organizing {} files in {}", file_count, resolved);
    println!("  quality: {:.2} (IHSAN floor: 0.95)", quality);
    println!();

    // Call the gateway
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/organize", base))
        .json(&serde_json::json!({
            "path": resolved,
            "quality_score": quality,
        }))
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        let rejected = body["rejected"].as_bool().unwrap_or(false);

        if rejected {
            println!("  ✗ REJECTED");
            if let Some(reason) = body["rejection_reason"].as_str() {
                println!("    reason: {}", reason);
            }
            if let Some(remediation) = body["remediation"].as_str() {
                println!("    fix:    {}", remediation);
            }
            std::process::exit(2);
        }

        let sub_ops = body["sub_operations"].as_u64().unwrap_or(0);
        let receipt_id = body["receipt_id"].as_str().unwrap_or("—");
        let chain_len = body["chain_length"].as_u64().unwrap_or(0);

        println!("  ✓ ORGANIZED");
        println!("    files moved:  {}", sub_ops);
        println!("    receipt:      {}...{}", &receipt_id[..16.min(receipt_id.len())],
            &receipt_id[receipt_id.len().saturating_sub(8)..]);
        println!("    chain length: {}", chain_len);
        println!("    verdict:      {} (5/5 gates PERMIT)", body["verdict"].as_str().unwrap_or("?"));
        println!();

        // Show operations
        if let Some(ops) = body["operations"].as_array() {
            for op in ops {
                let tool = op["tool"].as_str().unwrap_or("?");
                println!("    ├─ {}", tool);
            }
        }
        println!();
        println!("  Run `dema chain` to see the full receipt chain.");
        println!("  Run `dema receipt {}` to inspect.", &receipt_id[..16.min(receipt_id.len())]);
    } else {
        // Error response
        if let Some(error) = body.get("error") {
            eprintln!("DEMA — compilation failed");
            eprintln!("  code:    {}", error["code"].as_str().unwrap_or("?"));
            eprintln!("  message: {}", error["message"].as_str().unwrap_or("?"));
        } else {
            eprintln!("DEMA — unexpected error: {}", body);
        }
        std::process::exit(1);
    }

    Ok(())
}

/// Mirror of FilesystemExecutor::categorize for dry-run display
fn categorize_ext(ext: &str) -> &'static str {
    match ext {
        "pdf" | "doc" | "docx" | "txt" | "md" | "rtf" | "odt" => "documents",
        "xls" | "xlsx" | "csv" | "tsv" | "ods" => "spreadsheets",
        "ppt" | "pptx" | "key" | "odp" => "presentations",
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "svg" | "webp" | "ico" => "images",
        "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm" => "videos",
        "mp3" | "wav" | "flac" | "aac" | "ogg" | "wma" | "m4a" => "audio",
        "zip" | "tar" | "gz" | "bz2" | "7z" | "rar" | "xz" => "archives",
        "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "c" | "cpp" | "h"
        | "java" | "go" | "rb" | "swift" | "kt" => "code",
        "toml" | "yaml" | "yml" | "json" | "xml" | "ini" | "env" => "config",
        "sql" | "db" | "sqlite" | "parquet" | "feather" => "data",
        "exe" | "msi" | "deb" | "rpm" | "appimage" | "dmg" => "executables",
        "ttf" | "otf" | "woff" | "woff2" => "fonts",
        "onnx" | "pt" | "safetensors" | "gguf" | "obj" | "stl" | "fbx" => "models",
        _ => "other",
    }
}

// --- `dema compile` handler ---

async fn handle_compile_cmd(
    base: &str,
    intent: &str,
    kind: &str,
    quality: f64,
    target: Option<&str>,
) -> anyhow::Result<()> {
    println!("DEMA — compiling: \"{}\"", intent);
    println!("  kind:    {}", kind);
    println!("  quality: {:.2}", quality);
    if let Some(t) = target {
        println!("  target:  {}", t);
    }
    println!();

    let client = reqwest::Client::new();
    let mut body = serde_json::json!({
        "intent": intent,
        "kind": kind,
        "quality_score": quality,
    });

    if let Some(t) = target {
        body["target_path"] = serde_json::Value::String(t.to_string());
    }

    let resp = client
        .post(format!("{}/compile", base))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let result: serde_json::Value = resp.json().await?;

    if status.is_success() {
        let rejected = result["rejected"].as_bool().unwrap_or(false);

        if rejected {
            println!("  ✗ REJECTED");
            if let Some(reason) = result["rejection_reason"].as_str() {
                println!("    reason: {}", reason);
            }
            if let Some(rem) = result["remediation"].as_str() {
                println!("    fix:    {}", rem);
            }
            std::process::exit(2);
        }

        let receipt = result["receipt_id"].as_str().unwrap_or("—");
        let subs = result["sub_operations"].as_u64().unwrap_or(0);
        let chain = result["chain_length"].as_u64().unwrap_or(0);

        println!("  ✓ COMPILED");
        println!("    receipt:    {}...{}", &receipt[..16.min(receipt.len())],
            &receipt[receipt.len().saturating_sub(8)..]);
        println!("    sub-ops:    {}", subs);
        println!("    chain:      {} records", chain);
        println!("    verdict:    {}", result["verdict"].as_str().unwrap_or("?"));
        println!("    gates:      {}/5 passed", result["gates_passed"].as_u64().unwrap_or(0));
    } else {
        eprintln!("DEMA — error: {}", result);
        std::process::exit(1);
    }

    Ok(())
}
