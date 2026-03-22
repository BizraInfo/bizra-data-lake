// bizra-node/src/substrate/linux.rs
// Linux-specific substrate discovery via /proc + nvidia-smi + filesystem

use std::process::Command;

use super::*;

fn read_proc(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
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
    read_proc("/proc/cpuinfo")
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "Unknown CPU".into())
}

fn discover_cpu_cores() -> u32 {
    // Physical cores from /proc/cpuinfo "cpu cores" field
    read_proc("/proc/cpuinfo")
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("cpu cores"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|s| s.trim().parse().ok())
        })
        .unwrap_or(1)
}

fn discover_cpu_threads() -> u32 {
    // Logical processors = count of "processor" lines
    read_proc("/proc/cpuinfo")
        .map(|info| info.lines().filter(|l| l.starts_with("processor")).count() as u32)
        .unwrap_or(1)
}

fn discover_ram_total_gb() -> f64 {
    read_proc("/proc/meminfo")
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
    read_proc("/proc/meminfo")
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
    // nvidia-smi is the same binary on Linux and Windows
    if let Some(smi) = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
    ) {
        let gpus: Vec<GpuInfo> = smi
            .lines()
            .filter_map(|line| {
                let p: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if p.len() >= 4 {
                    Some(GpuInfo {
                        name: p[0].into(),
                        vram_total_mb: p[1].parse().unwrap_or(0),
                        vram_used_mb: p[2].parse().unwrap_or(0),
                        driver_version: p[3].into(),
                    })
                } else {
                    None
                }
            })
            .collect();
        if !gpus.is_empty() {
            return gpus;
        }
    }
    // Fallback: lspci for GPU name (no VRAM info)
    if let Some(lspci) = run_cmd("lspci", &[]) {
        return lspci
            .lines()
            .filter(|l| l.contains("VGA") || l.contains("3D controller"))
            .map(|l| {
                let name = l.split(':').next_back().unwrap_or(l).trim().to_string();
                GpuInfo {
                    name,
                    vram_total_mb: 0,
                    vram_used_mb: 0,
                    driver_version: String::new(),
                }
            })
            .collect();
    }
    Vec::new()
}

fn discover_disks() -> Vec<DiskInfo> {
    // Use df -BG for gigabyte output, filter real filesystems
    run_cmd("df", &["-BG", "--output=target,size,avail"])
        .map(|out| {
            out.lines()
                .skip(1)
                .filter_map(|line| {
                    let p: Vec<&str> = line.split_whitespace().collect();
                    if p.len() >= 3 {
                        let mount = p[0].to_string();
                        // Skip virtual filesystems
                        if mount.starts_with("/dev")
                            || mount == "/"
                            || mount.starts_with("/home")
                            || mount.starts_with("/mnt")
                        {
                            let total = p[1].trim_end_matches('G').parse().unwrap_or(0.0);
                            let free = p[2].trim_end_matches('G').parse().unwrap_or(0.0);
                            return Some(DiskInfo {
                                mount,
                                total_gb: total,
                                free_gb: free,
                                label: String::new(),
                            });
                        }
                    }
                    None
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn discover_lmstudio_models() -> Vec<LocalModel> {
    let base = home_dir().join(".lmstudio");
    let dirs = [
        base.join("models"),
        base.join("hub").join("models"),
        base.join(".internal").join("bundled-models"),
    ];
    let mut models = Vec::new();
    for dir in &dirs {
        if dir.exists() {
            scan_gguf_recursive(dir, ModelRuntime::LmStudio, &mut models);
        }
    }
    models
}
