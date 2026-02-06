//! Hardware Detection

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub ram_gb: f64,
    pub gpu_name: Option<String>,
    pub vram_gb: f64,
    pub disk_free_gb: f64,
    pub os: String,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            ram_gb: 8.0,
            gpu_name: None,
            vram_gb: 0.0,
            disk_free_gb: 50.0,
            os: std::env::consts::OS.into(),
        }
    }
}

impl HardwareProfile {
    pub fn recommended_tier(&self) -> &'static str {
        if self.vram_gb >= 8.0 {
            "LOCAL"
        } else if self.vram_gb >= 4.0 {
            "EDGE+"
        } else {
            "EDGE"
        }
    }

    pub fn recommended_model(&self) -> &'static str {
        if self.vram_gb >= 8.0 {
            "Qwen2.5-7B-Q4"
        } else if self.vram_gb >= 4.0 {
            "Qwen2.5-3B-Q4"
        } else if self.ram_gb >= 8.0 {
            "Qwen2.5-1.5B-Q4"
        } else {
            "Qwen2.5-0.5B-Q4"
        }
    }
}

pub fn detect_hardware() -> HardwareProfile {
    // Simplified detection
    let cpu_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    let (gpu_name, vram_gb) = detect_gpu();

    HardwareProfile {
        cpu_cores,
        ram_gb: 16.0, // Placeholder
        gpu_name,
        vram_gb,
        disk_free_gb: 100.0,
        os: std::env::consts::OS.into(),
    }
}

fn detect_gpu() -> (Option<String>, f64) {
    #[cfg(target_os = "linux")]
    if std::path::Path::new("/dev/nvidia0").exists() {
        return (Some("NVIDIA GPU".into()), 8.0);
    }
    #[cfg(target_os = "windows")]
    if std::path::Path::new("C:\\Windows\\System32\\nvml.dll").exists() {
        return (Some("NVIDIA GPU".into()), 8.0);
    }
    (None, 0.0)
}
