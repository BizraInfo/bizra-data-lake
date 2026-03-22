// bizra-node/src/substrate/windows.rs
// Windows-specific substrate discovery via PowerShell + WMI + nvidia-smi

use std::process::Command;

use super::*;

fn run_powershell(script: &str) -> Option<String> {
    Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
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
        cpu_name: run_powershell("(Get-CimInstance Win32_Processor).Name")
            .unwrap_or_else(|| "Unknown".into()),
        cpu_cores: run_powershell("(Get-CimInstance Win32_Processor).NumberOfCores")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        cpu_threads: run_powershell("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        ram_total_gb: run_powershell(
            "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)",
        )
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0),
        ram_available_gb: run_powershell(
            "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)",
        )
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0),
        gpus: discover_gpus(),
        disks: discover_disks(),
    }
}

fn discover_gpus() -> Vec<GpuInfo> {
    if let Some(smi) = run_powershell(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits 2>$null"
    ) {
        let gpus: Vec<GpuInfo> = smi.lines().filter_map(|line| {
            let p: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if p.len() >= 4 {
                Some(GpuInfo { name: p[0].into(), vram_total_mb: p[1].parse().unwrap_or(0), vram_used_mb: p[2].parse().unwrap_or(0), driver_version: p[3].into() })
            } else { None }
        }).collect();
        if !gpus.is_empty() { return gpus; }
    }
    if let Some(wmi) = run_powershell("Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name),$([math]::Round($_.AdapterRAM/1MB))\" }") {
        return wmi.lines().filter_map(|line| {
            let p: Vec<&str> = line.split(',').collect();
            if p.len() >= 2 { Some(GpuInfo { name: p[0].into(), vram_total_mb: p[1].parse().unwrap_or(0), vram_used_mb: 0, driver_version: String::new() }) } else { None }
        }).collect();
    }
    Vec::new()
}

fn discover_disks() -> Vec<DiskInfo> {
    run_powershell(r#"Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object { "$($_.DeviceID),$([math]::Round($_.Size/1GB,2)),$([math]::Round($_.FreeSpace/1GB,2)),$($_.VolumeName)" }"#)
        .map(|out| out.lines().filter_map(|line| {
            let p: Vec<&str> = line.split(',').collect();
            if p.len() >= 3 { Some(DiskInfo { mount: p[0].into(), total_gb: p[1].parse().unwrap_or(0.0), free_gb: p[2].parse().unwrap_or(0.0), label: p.get(3).unwrap_or(&"").to_string() }) } else { None }
        }).collect()).unwrap_or_default()
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
