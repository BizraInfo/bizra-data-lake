//! Binary acquisition
//!
//! Locates, verifies, and optionally builds the bizra-node binary
//! for Alpha-100 installation.

use anyhow::{Context, Result};
use blake3::Hasher;
use std::path::{Path, PathBuf};

/// Paths to key binaries managed by the installer.
#[derive(Debug, Clone)]
pub struct BinaryPaths {
    /// Path to the bizra-node binary
    pub node: PathBuf,
    /// Path to the installer binary itself
    pub installer: PathBuf,
}

/// Attempt to locate an existing bizra-node binary by searching, in order:
/// 1. `./target/release/bizra-node` (workspace build)
/// 2. `~/.bizra/bin/bizra-node` (installed location)
/// 3. `bizra-node` on the system PATH
///
/// Returns `None` if no binary is found.
pub fn locate_node_binary() -> Option<PathBuf> {
    // 1. Workspace build
    let workspace_path = PathBuf::from("./target/release/bizra-node");
    if workspace_path.exists() {
        return Some(workspace_path);
    }

    // 2. Installed location
    if let Some(home) = dirs::home_dir() {
        let installed = home.join(".bizra").join("bin").join("bizra-node");
        if installed.exists() {
            return Some(installed);
        }
    }

    // 3. System PATH via `which`
    if let Ok(output) = std::process::Command::new("which")
        .arg("bizra-node")
        .output()
    {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
    }

    // Windows fallback: try `where` command
    #[cfg(target_os = "windows")]
    if let Ok(output) = std::process::Command::new("where")
        .arg("bizra-node")
        .output()
    {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
    }

    None
}

/// Compute the BLAKE3 checksum of a file and compare it to the expected hex.
/// Returns `Ok(true)` if they match, `Ok(false)` if they differ.
pub fn verify_checksum(path: &Path, expected_hex: &str) -> Result<bool> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read file for checksum: {}", path.display()))?;

    let mut hasher = Hasher::new();
    hasher.update(b"bizra-installer-v1:binary-verify:");
    hasher.update(&bytes);
    let result = hasher.finalize();
    let actual_hex = hex::encode(result.as_bytes());

    Ok(actual_hex == expected_hex.to_lowercase())
}

/// Build the bizra-node binary from source using `cargo build`.
/// `workspace_dir` should point to the root of the bizra-omega workspace.
///
/// Returns paths to the built binaries on success.
pub fn fallback_build(workspace_dir: &Path) -> Result<BinaryPaths> {
    println!("  Building bizra-node from source...");
    println!("  Workspace: {}", workspace_dir.display());

    let status = std::process::Command::new("cargo")
        .args(["build", "--release", "-p", "bizra-node"])
        .current_dir(workspace_dir)
        .status()
        .context("Failed to execute cargo build")?;

    if !status.success() {
        anyhow::bail!(
            "cargo build --release -p bizra-node failed with exit code: {}",
            status.code().unwrap_or(-1)
        );
    }

    let target_dir = workspace_dir.join("target").join("release");
    let node_path = target_dir.join("bizra-node");
    let installer_path = target_dir.join("bizra-install");

    if !node_path.exists() {
        anyhow::bail!(
            "Build succeeded but bizra-node binary not found at {}",
            node_path.display()
        );
    }

    Ok(BinaryPaths {
        node: node_path,
        installer: installer_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_checksum_correct_hash() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        let content = b"deterministic content for checksum test";
        std::fs::write(&file_path, content).expect("write");

        // Compute expected BLAKE3
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-installer-v1:binary-verify:");
        hasher.update(content);
        let expected = hex::encode(hasher.finalize().as_bytes());

        let result = verify_checksum(&file_path, &expected).expect("verify");
        assert!(result, "Checksum should match");
    }

    #[test]
    fn verify_checksum_wrong_hash() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"some content").expect("write");

        let wrong_hash = "0".repeat(64);
        let result = verify_checksum(&file_path, &wrong_hash).expect("verify");
        assert!(!result, "Checksum should not match");
    }

    #[test]
    fn verify_checksum_missing_file_returns_error() {
        let result = verify_checksum(Path::new("/nonexistent/file.bin"), "aabbccdd");
        assert!(result.is_err());
    }

    #[test]
    fn verify_checksum_case_insensitive() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        let content = b"case test";
        std::fs::write(&file_path, content).expect("write");

        let mut hasher = Hasher::new();
        hasher.update(b"bizra-installer-v1:binary-verify:");
        hasher.update(content);
        let expected_lower = hex::encode(hasher.finalize().as_bytes());
        let expected_upper = expected_lower.to_uppercase();

        let result = verify_checksum(&file_path, &expected_upper).expect("verify");
        assert!(result, "Uppercase hex should also match");
    }

    #[test]
    fn locate_node_binary_returns_option() {
        // This is a best-effort test — the binary may or may not exist
        let result = locate_node_binary();
        // Just verify it returns cleanly without panic
        if let Some(path) = &result {
            assert!(!path.to_string_lossy().is_empty());
        }
    }
}
