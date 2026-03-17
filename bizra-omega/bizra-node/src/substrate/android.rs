// bizra-node/src/substrate/android.rs
// ============================================================
// Android Substrate Backend — Phone as Sovereign Node
// ============================================================
//
// The phone is not a thin client. It is a sovereign node with
// unique capabilities (sensors, always-on, biometrics) that
// NODE0 does not have. Together with the desktop node, they
// form a Personal Sovereign Cluster — one human identity,
// two complementary bodies.
//
// Discovery uses Android system properties and /proc where
// available (Android is Linux-based).
//
// Standing on Giants:
// - Android is Linux kernel underneath
// - /proc/cpuinfo, /proc/meminfo work on Android
// - GPU discovery via getprop and Vulkan
// ============================================================

use super::*;
use std::process::Command;

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd).args(args).output().ok()
        .and_then(|o| if o.status.success() {
            String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
        } else { None })
        .filter(|s| !s.is_empty())
}

pub fn discover_hardware() -> HardwareManifest {
    HardwareManifest {
        cpu_name: discover_cpu_name(),
        cpu_cores: discover_cpu_cores(),
        cpu_threads: discover_cpu_threads(),
        ram_total_gb: discover_ram_total_gb(),
        ram_available_gb: discover_ram_available_gb(),
        gpus: discover_gpus(),
        disks: discover_disks(),
    }
}

fn discover_cpu_name() -> String {
    // Android: read from /proc/cpuinfo (same as Linux)
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("Hardware") || l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .or_else(|| run_cmd("getprop", &["ro.hardware"]))
        .unwrap_or_else(|| "Unknown Mobile CPU".into())
}

fn discover_cpu_cores() -> u32 {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|info| info.lines().filter(|l| l.starts_with("processor")).count() as u32)
        .unwrap_or(4) // Most modern phones have at least 4
}

fn discover_cpu_threads() -> u32 {
    discover_cpu_cores() // On ARM, cores ≈ threads typically
}

fn discover_ram_total_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .map(|kb| kb / (1024.0 * 1024.0))
        })
        .unwrap_or(0.0)
}

fn discover_ram_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .map(|kb| kb / (1024.0 * 1024.0))
        })
        .unwrap_or(0.0)
}

fn discover_gpus() -> Vec<GpuInfo> {
    // Android GPU: use getprop for Adreno/Mali/Vulkan info
    let gpu_name = run_cmd("getprop", &["ro.hardware.egl"])
        .or_else(|| run_cmd("getprop", &["ro.board.platform"]))
        .unwrap_or_else(|| "Mobile GPU".into());
    vec![GpuInfo { name: gpu_name, vram_total_mb: 0, vram_used_mb: 0, driver_version: String::new() }]
}

fn discover_disks() -> Vec<DiskInfo> {
    // Android: internal storage + SD card if present
    let mut disks = Vec::new();
    // Internal storage via /data
    if let Ok(out) = std::process::Command::new("df").args(["-BG", "/data"]).output() {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines().skip(1) {
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() >= 4 {
                disks.push(DiskInfo {
                    mount: "/data".into(),
                    total_gb: p[1].trim_end_matches('G').parse().unwrap_or(0.0),
                    free_gb: p[3].trim_end_matches('G').parse().unwrap_or(0.0),
                    label: "Internal".into(),
                });
            }
        }
    }
    // SD card via /storage/emulated/0 or /sdcard
    for path in &["/storage/emulated/0", "/sdcard"] {
        if std::path::Path::new(path).exists() {
            if let Ok(out) = std::process::Command::new("df").args(["-BG", *path]).output() {
                let text = String::from_utf8_lossy(&out.stdout);
                for line in text.lines().skip(1) {
                    let p: Vec<&str> = line.split_whitespace().collect();
                    if p.len() >= 4 && !disks.iter().any(|d: &DiskInfo| d.mount == *path) {
                        disks.push(DiskInfo {
                            mount: path.to_string(),
                            total_gb: p[1].trim_end_matches('G').parse().unwrap_or(0.0),
                            free_gb: p[3].trim_end_matches('G').parse().unwrap_or(0.0),
                            label: "Storage".into(),
                        });
                    }
                }
            }
        }
    }
    disks
}

/// Discover GGUF models stored on the Android device.
/// Models are expected in /data/local/tmp/bizra/models/ or
/// /storage/emulated/0/BIZRA/models/ — user-configurable.
pub fn discover_lmstudio_models() -> Vec<LocalModel> {
    let search_dirs = [
        std::path::PathBuf::from("/data/local/tmp/bizra/models"),
        std::path::PathBuf::from("/storage/emulated/0/BIZRA/models"),
        home_dir().join(".bizra").join("models"),
    ];
    let mut models = Vec::new();
    for dir in &search_dirs {
        if dir.exists() {
            scan_gguf_recursive(dir, ModelRuntime::Standalone, &mut models);
        }
    }
    models
}
