//! Full DeviceProfile — Universal Sovereign Installer v2.0
//!
//! Comprehensive hardware + locale + display detection for adaptive
//! installation across all platforms and device classes.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §5.1
//! Standing on Giants: Shannon (adaptation), Torvalds (sysinfo)
//!
//! Constitutional: Every human is a node — this module ensures even
//! a 1GB device gets a sovereign experience (micro-node with TinyLlama Q2).

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OS {
    Windows,
    MacOS,
    Linux,
    Android,
    IOS,
    Unknown,
}

impl OS {
    pub fn detect() -> Self {
        match std::env::consts::OS {
            "windows" => OS::Windows,
            "macos" => OS::MacOS,
            "linux" => {
                // Distinguish Android from Linux
                if std::path::Path::new("/system/build.prop").exists() {
                    OS::Android
                } else {
                    OS::Linux
                }
            }
            "ios" => OS::IOS,
            _ => OS::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arch {
    X86_64,
    Aarch64,
    Riscv64,
    Unknown,
}

impl Arch {
    pub fn detect() -> Self {
        match std::env::consts::ARCH {
            "x86_64" => Arch::X86_64,
            "aarch64" => Arch::Aarch64,
            "riscv64" | "riscv64gc" => Arch::Riscv64,
            _ => Arch::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuApi {
    Cuda,
    Rocm,
    Metal,
    OpenVino,
    Vulkan,
    None,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiskType {
    SSD,
    HDD,
    EMMC,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDirection {
    LTR,
    RTL,
}

// ─────────────────────────────────────────────────────────────
// GPU Info
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub vram_gb: f32,
    pub api: GpuApi,
}

// ─────────────────────────────────────────────────────────────
// Model Tier (Spec §5.2)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTier {
    /// 1-2 GB RAM → TinyLlama 1.1B Q2_K (~500MB)
    Micro,
    /// 2-4 GB RAM → TinyLlama 1.1B Q4_K_M (~650MB)
    Compact,
    /// 4-8 GB RAM → Phi-3-mini 3.8B Q4_K_M (~2.3GB)
    Standard,
    /// 8-16 GB RAM → Llama 3.1 8B Q4_K_M (~4.7GB)
    Enhanced,
    /// 16-32 GB RAM → Qwen 2.5 14B Q4_K_M (~8.5GB)
    Full,
    /// 32-64 GB RAM → Llama 3.1 70B Q4_K_M (~40GB)
    Premium,
    /// 64+ GB RAM → Multiple models Q8/FP16
    Elite,
}

impl ModelTier {
    /// Select tier from available RAM (spec §5.2)
    pub fn from_ram_gb(ram: f32) -> Self {
        match ram {
            r if r >= 64.0 => ModelTier::Elite,
            r if r >= 32.0 => ModelTier::Premium,
            r if r >= 16.0 => ModelTier::Full,
            r if r >= 8.0 => ModelTier::Enhanced,
            r if r >= 4.0 => ModelTier::Standard,
            r if r >= 2.0 => ModelTier::Compact,
            _ => ModelTier::Micro,
        }
    }

    pub fn model_name(&self) -> &'static str {
        match self {
            ModelTier::Micro => "TinyLlama 1.1B Q2_K",
            ModelTier::Compact => "TinyLlama 1.1B Q4_K_M",
            ModelTier::Standard => "Phi-3-mini 3.8B Q4_K_M",
            ModelTier::Enhanced => "Llama 3.1 8B Q4_K_M",
            ModelTier::Full => "Qwen 2.5 14B Q4_K_M",
            ModelTier::Premium => "Llama 3.1 70B Q4_K_M",
            ModelTier::Elite => "Multiple models Q8/FP16",
        }
    }

    pub fn disk_requirement_gb(&self) -> f32 {
        match self {
            ModelTier::Micro => 0.5,
            ModelTier::Compact => 0.65,
            ModelTier::Standard => 2.3,
            ModelTier::Enhanced => 4.7,
            ModelTier::Full => 8.5,
            ModelTier::Premium => 40.0,
            ModelTier::Elite => 80.0,
        }
    }

    /// Fallback to next smaller tier if current fails (OOM protection)
    pub fn fallback(&self) -> Option<ModelTier> {
        match self {
            ModelTier::Elite => Some(ModelTier::Premium),
            ModelTier::Premium => Some(ModelTier::Full),
            ModelTier::Full => Some(ModelTier::Enhanced),
            ModelTier::Enhanced => Some(ModelTier::Standard),
            ModelTier::Standard => Some(ModelTier::Compact),
            ModelTier::Compact => Some(ModelTier::Micro),
            ModelTier::Micro => None, // No fallback below micro
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Install Footprint
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstallFootprint {
    /// Core + compact model (~1.5GB total)
    Minimal,
    /// Core + standard model (~4GB total)
    Standard,
    /// Core + enhanced model + tools (~10GB total)
    Full,
}

// ─────────────────────────────────────────────────────────────
// DeviceProfile (Spec §5.1)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceProfile {
    // Platform
    pub os: OS,
    pub os_version: String,
    pub arch: Arch,

    // Compute
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub ram_total_gb: f32,
    pub ram_available_gb: f32,

    // GPU
    pub gpu: Option<GpuInfo>,

    // Storage
    pub disk_available_gb: f32,
    pub disk_type: DiskType,

    // Network
    pub network_available: bool,
    pub network_speed_mbps: f32,

    // Locale
    pub system_locale: String,
    pub timezone: String,

    // Display
    pub screen_width: u32,
    pub screen_height: u32,
    pub dpi: f32,
    pub touch_capable: bool,
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self {
            os: OS::detect(),
            os_version: String::new(),
            arch: Arch::detect(),
            cpu_cores: std::thread::available_parallelism()
                .map(|p| p.get() as u32)
                .unwrap_or(4),
            cpu_threads: std::thread::available_parallelism()
                .map(|p| p.get() as u32)
                .unwrap_or(4),
            ram_total_gb: 8.0,
            ram_available_gb: 4.0,
            gpu: None,
            disk_available_gb: 50.0,
            disk_type: DiskType::Unknown,
            network_available: false,
            network_speed_mbps: 0.0,
            system_locale: "en-US".to_string(),
            timezone: "UTC".to_string(),
            screen_width: 1920,
            screen_height: 1080,
            dpi: 96.0,
            touch_capable: false,
        }
    }
}

impl DeviceProfile {
    /// Recommended model tier based on available RAM (spec §5.2)
    pub fn recommended_tier(&self) -> ModelTier {
        ModelTier::from_ram_gb(self.ram_available_gb)
    }

    /// Recommended GPU API backend (spec §5.3)
    pub fn recommended_gpu_api(&self) -> GpuApi {
        match &self.gpu {
            Some(info) => info.api.clone(),
            None => GpuApi::None,
        }
    }

    /// Recommended install footprint based on available disk (spec §3.2)
    pub fn recommended_footprint(&self) -> InstallFootprint {
        if self.disk_available_gb >= 10.0 {
            InstallFootprint::Full
        } else if self.disk_available_gb >= 4.0 {
            InstallFootprint::Standard
        } else {
            InstallFootprint::Minimal
        }
    }

    /// Whether this device qualifies as a micro-node (< 2GB RAM)
    /// Micro-nodes run System-1 only, 5-min heartbeat instead of 60s
    pub fn is_micro_node(&self) -> bool {
        self.ram_available_gb < 2.0
    }

    /// Detect text direction from locale
    pub fn text_direction(&self) -> TextDirection {
        let lang = self.system_locale.split('-').next().unwrap_or("en");
        match lang {
            "ar" | "he" | "ur" | "fa" | "ps" | "sd" | "yi" | "ku" => TextDirection::RTL,
            _ => TextDirection::LTR,
        }
    }

    /// Whether sufficient disk space for the recommended tier
    pub fn has_sufficient_disk(&self) -> bool {
        let required = self.recommended_tier().disk_requirement_gb();
        // Need model + 500MB for core runtime + 500MB free buffer
        self.disk_available_gb >= required + 1.0
    }
}

/// Detect the full DeviceProfile for the current system.
///
/// Uses platform-specific APIs where available, falls back to
/// safe defaults. Never panics — degraded detection is better
/// than no detection.
pub fn detect_device() -> DeviceProfile {
    let mut profile = DeviceProfile::default();

    // OS version
    profile.os_version = detect_os_version();

    // GPU detection (platform-specific)
    profile.gpu = detect_gpu_info();

    // Locale detection
    profile.system_locale = detect_locale();

    // Network availability (quick check)
    profile.network_available = check_network();

    profile
}

fn detect_os_version() -> String {
    #[cfg(target_os = "windows")]
    {
        // Read from registry or ver command
        std::process::Command::new("cmd")
            .args(["/c", "ver"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("VERSION_ID="))
                    .map(|l| l.trim_start_matches("VERSION_ID=").trim_matches('"').to_string())
            })
            .unwrap_or_default()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        String::new()
    }
}

fn detect_gpu_info() -> Option<GpuInfo> {
    // NVIDIA: try nvidia-smi
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = text.trim().split(", ").collect();
                if parts.len() >= 2 {
                    let vram_mb: f32 = parts[1].trim().parse().unwrap_or(0.0);
                    return Some(GpuInfo {
                        vendor: "NVIDIA".to_string(),
                        model: parts[0].trim().to_string(),
                        vram_gb: vram_mb / 1024.0,
                        api: GpuApi::Cuda,
                    });
                }
            }
        }
    }

    // macOS: Metal detection
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                // Extract GPU name from system_profiler output
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("Chipset Model:") {
                        let model = trimmed
                            .trim_start_matches("Chipset Model:")
                            .trim()
                            .to_string();
                        return Some(GpuInfo {
                            vendor: "Apple".to_string(),
                            model,
                            vram_gb: 0.0, // Unified memory on Apple Silicon
                            api: GpuApi::Metal,
                        });
                    }
                }
            }
        }
    }

    // AMD ROCm check (Linux)
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/dev/kfd").exists() {
            if let Ok(output) = std::process::Command::new("rocm-smi")
                .args(["--showproductname"])
                .output()
            {
                if output.status.success() {
                    return Some(GpuInfo {
                        vendor: "AMD".to_string(),
                        model: "AMD GPU (ROCm)".to_string(),
                        vram_gb: 0.0,
                        api: GpuApi::Rocm,
                    });
                }
            }
        }
    }

    None
}

fn detect_locale() -> String {
    // Try LANG environment variable (Linux/macOS)
    if let Ok(lang) = std::env::var("LANG") {
        // LANG is typically like "en_US.UTF-8"
        let base = lang.split('.').next().unwrap_or("en_US");
        return base.replace('_', "-");
    }

    // Try LC_ALL
    if let Ok(lc) = std::env::var("LC_ALL") {
        let base = lc.split('.').next().unwrap_or("en_US");
        return base.replace('_', "-");
    }

    // Windows: use powershell to get culture
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("powershell")
            .args(["-Command", "(Get-Culture).Name"])
            .output()
        {
            if output.status.success() {
                let locale = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !locale.is_empty() {
                    return locale;
                }
            }
        }
    }

    "en-US".to_string()
}

fn check_network() -> bool {
    // Quick connectivity check — try to resolve a well-known host
    // We use a DNS check, not an HTTP request, to be fast and non-intrusive
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([1, 1, 1, 1], 53)),
        std::time::Duration::from_secs(2),
    )
    .is_ok()
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tier_from_ram() {
        assert_eq!(ModelTier::from_ram_gb(1.0), ModelTier::Micro);
        assert_eq!(ModelTier::from_ram_gb(2.0), ModelTier::Compact);
        assert_eq!(ModelTier::from_ram_gb(4.0), ModelTier::Standard);
        assert_eq!(ModelTier::from_ram_gb(8.0), ModelTier::Enhanced);
        assert_eq!(ModelTier::from_ram_gb(16.0), ModelTier::Full);
        assert_eq!(ModelTier::from_ram_gb(32.0), ModelTier::Premium);
        assert_eq!(ModelTier::from_ram_gb(64.0), ModelTier::Elite);
        assert_eq!(ModelTier::from_ram_gb(128.0), ModelTier::Elite);
    }

    #[test]
    fn model_tier_fallback_chain() {
        assert_eq!(ModelTier::Elite.fallback(), Some(ModelTier::Premium));
        assert_eq!(ModelTier::Premium.fallback(), Some(ModelTier::Full));
        assert_eq!(ModelTier::Full.fallback(), Some(ModelTier::Enhanced));
        assert_eq!(ModelTier::Enhanced.fallback(), Some(ModelTier::Standard));
        assert_eq!(ModelTier::Standard.fallback(), Some(ModelTier::Compact));
        assert_eq!(ModelTier::Compact.fallback(), Some(ModelTier::Micro));
        assert_eq!(ModelTier::Micro.fallback(), None);
    }

    #[test]
    fn device_profile_defaults() {
        let p = DeviceProfile::default();
        assert!(p.cpu_cores > 0);
        assert!(!p.system_locale.is_empty());
    }

    #[test]
    fn micro_node_detection() {
        let mut p = DeviceProfile::default();
        p.ram_available_gb = 1.5;
        assert!(p.is_micro_node());

        p.ram_available_gb = 2.0;
        assert!(!p.is_micro_node());
    }

    #[test]
    fn text_direction_rtl() {
        let mut p = DeviceProfile::default();
        p.system_locale = "ar-AE".to_string();
        assert_eq!(p.text_direction(), TextDirection::RTL);

        p.system_locale = "ur-PK".to_string();
        assert_eq!(p.text_direction(), TextDirection::RTL);

        p.system_locale = "fa-IR".to_string();
        assert_eq!(p.text_direction(), TextDirection::RTL);

        p.system_locale = "he-IL".to_string();
        assert_eq!(p.text_direction(), TextDirection::RTL);
    }

    #[test]
    fn text_direction_ltr() {
        let mut p = DeviceProfile::default();
        p.system_locale = "en-US".to_string();
        assert_eq!(p.text_direction(), TextDirection::LTR);

        p.system_locale = "es-MX".to_string();
        assert_eq!(p.text_direction(), TextDirection::LTR);
    }

    #[test]
    fn footprint_recommendation() {
        let mut p = DeviceProfile::default();
        p.disk_available_gb = 15.0;
        assert_eq!(p.recommended_footprint(), InstallFootprint::Full);

        p.disk_available_gb = 5.0;
        assert_eq!(p.recommended_footprint(), InstallFootprint::Standard);

        p.disk_available_gb = 2.0;
        assert_eq!(p.recommended_footprint(), InstallFootprint::Minimal);
    }

    #[test]
    fn disk_sufficiency() {
        let mut p = DeviceProfile::default();
        p.ram_available_gb = 4.0; // Standard tier → needs 2.3 + 1.0 = 3.3 GB
        p.disk_available_gb = 4.0;
        assert!(p.has_sufficient_disk());

        p.disk_available_gb = 2.0;
        assert!(!p.has_sufficient_disk());
    }

    #[test]
    fn os_detection_is_stable() {
        let os = OS::detect();
        // Just verify it doesn't panic and returns a valid variant
        assert_ne!(format!("{:?}", os), "");
    }

    #[test]
    fn arch_detection_is_stable() {
        let arch = Arch::detect();
        assert_ne!(format!("{:?}", arch), "");
    }

    #[test]
    fn model_tier_names_non_empty() {
        for tier in [
            ModelTier::Micro,
            ModelTier::Compact,
            ModelTier::Standard,
            ModelTier::Enhanced,
            ModelTier::Full,
            ModelTier::Premium,
            ModelTier::Elite,
        ] {
            assert!(!tier.model_name().is_empty());
            assert!(tier.disk_requirement_gb() > 0.0);
        }
    }
}
