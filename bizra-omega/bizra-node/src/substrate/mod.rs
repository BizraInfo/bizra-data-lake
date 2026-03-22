// bizra-node/src/substrate/mod.rs
// ============================================================
// Sovereign Substrate — Backend-Abstracted Self-Awareness
// ============================================================
//
// Shared types live here. Discovery implementations live in
// platform-specific backends (windows.rs, linux.rs).
//
// The Node calls `ResourceManifest::discover()` and gets a
// complete view of its body regardless of OS. The backend
// is selected at compile time via #[cfg(target_os)].
//
// Standing on Giants:
// - Popek & Goldberg (1974): VM resource partitioning
// - Lampson (1974): Complete mediation
// - Al-Ghazali: Self-knowledge as prerequisite for self-governance
// ============================================================

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "windows")]
mod windows;

use std::{collections::HashMap, path::PathBuf};

// ── Shared Types ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HardwareManifest {
    pub cpu_name: String,
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    pub gpus: Vec<GpuInfo>,
    pub disks: Vec<DiskInfo>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub driver_version: String,
}

#[derive(Debug, Clone)]
pub struct DiskInfo {
    pub mount: String,
    pub total_gb: f64,
    pub free_gb: f64,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct LocalModel {
    pub name: String,
    pub size_bytes: u64,
    pub quantization: String,
    pub runtime: ModelRuntime,
    pub path: PathBuf,
    pub parameter_count: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelRuntime {
    Ollama,
    LmStudio,
    HuggingFace,
    Standalone,
}

impl ModelRuntime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "lm-studio",
            Self::HuggingFace => "huggingface",
            Self::Standalone => "standalone",
        }
    }
}

// ── The Sovereign Manifest ─────────────────────────────────

#[derive(Debug, Clone)]
pub struct ResourceManifest {
    pub hardware: HardwareManifest,
    pub models: Vec<LocalModel>,
    pub model_count_by_runtime: HashMap<ModelRuntime, usize>,
    pub total_model_storage_gb: f64,
    pub discovered_at: u64,
    pub platform: &'static str,
}

impl ResourceManifest {
    /// Discover everything — delegates to platform-specific backend.
    pub fn discover() -> Self {
        let hardware = discover_hardware();
        let models = discover_all_models();

        let mut by_runtime: HashMap<ModelRuntime, usize> = HashMap::new();
        let mut total_bytes: u64 = 0;
        for m in &models {
            *by_runtime.entry(m.runtime).or_insert(0) += 1;
            total_bytes += m.size_bytes;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            hardware,
            models,
            model_count_by_runtime: by_runtime,
            total_model_storage_gb: total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            discovered_at: now,
            platform: current_platform(),
        }
    }

    pub fn empty() -> Self {
        Self {
            hardware: HardwareManifest {
                cpu_name: String::new(),
                cpu_cores: 0,
                cpu_threads: 0,
                ram_total_gb: 0.0,
                ram_available_gb: 0.0,
                gpus: Vec::new(),
                disks: Vec::new(),
            },
            models: Vec::new(),
            model_count_by_runtime: HashMap::new(),
            total_model_storage_gb: 0.0,
            discovered_at: 0,
            platform: current_platform(),
        }
    }

    pub fn total_models(&self) -> usize {
        self.models.len()
    }

    pub fn summary(&self) -> String {
        let gpu_summary = if self.hardware.gpus.is_empty() {
            "no GPU".to_string()
        } else {
            self.hardware
                .gpus
                .iter()
                .map(|g| format!("{} ({}MB)", g.name, g.vram_total_mb))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "CPU: {} ({} cores) | RAM: {:.1}/{:.1} GB | GPU: {} | Models: {} ({:.1} GB) | Disks: {} | {}",
            self.hardware.cpu_name, self.hardware.cpu_cores,
            self.hardware.ram_available_gb, self.hardware.ram_total_gb,
            gpu_summary, self.models.len(), self.total_model_storage_gb,
            self.hardware.disks.len(), self.platform,
        )
    }
}

// ── Platform dispatch ──────────────────────────────────────

fn current_platform() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "windows"
    }
    #[cfg(target_os = "linux")]
    {
        "linux"
    }
    #[cfg(target_os = "android")]
    {
        "android"
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
    {
        "unknown"
    }
}

fn discover_hardware() -> HardwareManifest {
    #[cfg(target_os = "windows")]
    {
        windows::discover_hardware()
    }
    #[cfg(target_os = "linux")]
    {
        linux::discover_hardware()
    }
    #[cfg(target_os = "android")]
    {
        android::discover_hardware()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
    {
        HardwareManifest {
            cpu_name: "unknown".into(),
            cpu_cores: 1,
            cpu_threads: 1,
            ram_total_gb: 0.0,
            ram_available_gb: 0.0,
            gpus: Vec::new(),
            disks: Vec::new(),
        }
    }
}

fn discover_all_models() -> Vec<LocalModel> {
    let mut all = Vec::new();
    all.extend(discover_ollama_models());
    #[cfg(target_os = "windows")]
    all.extend(windows::discover_lmstudio_models());
    #[cfg(target_os = "linux")]
    all.extend(linux::discover_lmstudio_models());
    #[cfg(target_os = "android")]
    all.extend(android::discover_lmstudio_models());
    all.extend(discover_huggingface_models());
    all
}

// ── Cross-platform model discovery ─────────────────────────

fn discover_ollama_models() -> Vec<LocalModel> {
    let output = std::process::Command::new("ollama").arg("list").output();
    let Ok(out) = output else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&out.stdout);

    text.lines()
        .skip(1)
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                Some(LocalModel {
                    name: parts[0].to_string(),
                    size_bytes: parse_human_size(parts[2], parts.get(3).copied()),
                    quantization: String::new(),
                    runtime: ModelRuntime::Ollama,
                    path: home_dir().join(".ollama").join("models"),
                    parameter_count: String::new(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn discover_huggingface_models() -> Vec<LocalModel> {
    let cache = home_dir().join(".cache").join("huggingface").join("hub");
    if !cache.exists() {
        return Vec::new();
    }

    let mut models = Vec::new();
    let Ok(entries) = std::fs::read_dir(&cache) else {
        return Vec::new();
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let dir_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if !dir_name.starts_with("models--") {
            continue;
        }

        let model_name = dir_name
            .strip_prefix("models--")
            .unwrap_or(&dir_name)
            .replace("--", "/");
        let weight_exts = ["safetensors", "bin", "gguf", "pt", "pth"];
        let mut total_size: u64 = 0;
        let mut found = false;

        if let Ok(files) = walk_files_recursive(&dir) {
            for path in files {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if weight_exts.contains(&ext) {
                        found = true;
                        total_size += path.metadata().map(|m| m.len()).unwrap_or(0);
                    }
                }
            }
        }
        if found {
            models.push(LocalModel {
                name: model_name,
                size_bytes: total_size,
                quantization: String::new(),
                runtime: ModelRuntime::HuggingFace,
                path: dir,
                parameter_count: String::new(),
            });
        }
    }
    models
}

// ── Shared utilities ───────────────────────────────────────

pub fn home_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        PathBuf::from(std::env::var("USERPROFILE").unwrap_or_else(|_| "C:\\Users\\default".into()))
    }
    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
    }
}

pub fn walk_files_recursive(dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    walk_inner(dir, &mut files, 0)?;
    Ok(files)
}

fn walk_inner(dir: &std::path::Path, out: &mut Vec<PathBuf>, depth: u32) -> std::io::Result<()> {
    if depth > 10 {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)?.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            let _ = walk_inner(&path, out, depth + 1);
        } else {
            out.push(path);
        }
    }
    Ok(())
}

pub fn scan_gguf_recursive(
    dir: &std::path::Path,
    runtime: ModelRuntime,
    out: &mut Vec<LocalModel>,
) {
    let Ok(files) = walk_files_recursive(dir) else {
        return;
    };
    for path in files {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "gguf" {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem.starts_with("mmproj") {
            continue;
        }
        let size = path.metadata().map(|m| m.len()).unwrap_or(0);
        if size < 1_000_000 {
            continue;
        }
        let name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or(stem)
            .to_string();
        out.push(LocalModel {
            name,
            size_bytes: size,
            quantization: extract_quantization(stem),
            runtime,
            path: path.clone(),
            parameter_count: String::new(),
        });
    }
}

pub fn extract_quantization(stem: &str) -> String {
    let quants = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1",
        "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32", "IQ1_S", "IQ2_S", "IQ2_XS", "IQ3_S",
        "IQ3_XS", "IQ4_NL",
    ];
    let upper = stem.to_uppercase();
    for q in &quants {
        if upper.contains(q) {
            return q.to_string();
        }
    }
    String::new()
}

pub fn parse_human_size(num_str: &str, unit: Option<&str>) -> u64 {
    let num: f64 = num_str.parse().unwrap_or(0.0);
    match unit.unwrap_or("") {
        "GB" => (num * 1024.0 * 1024.0 * 1024.0) as u64,
        "MB" => (num * 1024.0 * 1024.0) as u64,
        "KB" => (num * 1024.0) as u64,
        _ => (num * 1024.0 * 1024.0 * 1024.0) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_quant_from_filenames() {
        assert_eq!(
            extract_quantization("Qwen2.5-14B_Uncensored_Instruct-Q4_K_S"),
            "Q4_K_S"
        );
        assert_eq!(
            extract_quantization("DeepSeek-R1-0528-Qwen3-8B-Q8_0"),
            "Q8_0"
        );
        assert_eq!(extract_quantization("model-F16"), "F16");
        assert_eq!(extract_quantization("nomic-embed-text-v1.5"), "");
        assert_eq!(extract_quantization("AgentFlow-Planner-7B.i1-Q6_K"), "Q6_K");
    }

    #[test]
    fn parse_sizes_correctly() {
        assert_eq!(parse_human_size("4.4", Some("GB")), 4724464025);
        assert_eq!(parse_human_size("274", Some("MB")), 287309824);
        assert_eq!(parse_human_size("1.7", Some("GB")), 1825361100);
    }

    #[test]
    fn model_runtime_strings() {
        assert_eq!(ModelRuntime::Ollama.as_str(), "ollama");
        assert_eq!(ModelRuntime::LmStudio.as_str(), "lm-studio");
        assert_eq!(ModelRuntime::HuggingFace.as_str(), "huggingface");
        assert_eq!(ModelRuntime::Standalone.as_str(), "standalone");
    }

    #[test]
    fn discover_does_not_panic() {
        let manifest = ResourceManifest::discover();
        assert!(!manifest.hardware.cpu_name.is_empty());
        assert!(manifest.hardware.cpu_cores >= 1);
        assert!(manifest.hardware.ram_total_gb > 0.0);
        assert!(manifest.discovered_at > 0);
        assert!(!manifest.platform.is_empty());
        let summary = manifest.summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn empty_manifest_is_valid() {
        let m = ResourceManifest::empty();
        assert_eq!(m.total_models(), 0);
        assert_eq!(m.total_model_storage_gb, 0.0);
        assert_eq!(m.discovered_at, 0);
        assert!(!m.summary().is_empty());
    }
}
