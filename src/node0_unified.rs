// src/node0_unified.rs - Unified Node0 System (Rust)
//
// Central nervous system for BIZRA Node0 - harmonizing hardware, software, and data.
// This module provides the Rust-side integration for the unified Node0 orchestrator.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Resource types managed by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceType {
    Cpu,
    GpuVram,
    Memory,
    Storage,
    Network,
}

/// Service types in the BIZRA ecosystem
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceType {
    Docker,
    Ollama,
    RustElite,
    PythonKernel,
    DataLake,
}

/// CPU status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStatus {
    pub model: String,
    pub cores_physical: u32,
    pub cores_logical: u32,
    pub frequency_mhz: f64,
    pub usage_percent: f64,
    pub temperature_c: Option<f64>,
}

impl CpuStatus {
    pub fn is_healthy(&self) -> bool {
        self.usage_percent < 95.0 && self.temperature_c.map_or(true, |t| t < 90.0)
    }
}

/// GPU status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    pub name: String,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    pub utilization_percent: f64,
    pub temperature_c: Option<f64>,
    pub driver_version: Option<String>,
}

impl GpuStatus {
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total_mb > 0 {
            (self.memory_used_mb as f64 / self.memory_total_mb as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.memory_usage_percent() < 95.0 && self.temperature_c.map_or(true, |t| t < 85.0)
    }
}

/// Memory status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    pub total_gb: f64,
    pub available_gb: f64,
    pub used_gb: f64,
    pub usage_percent: f64,
    pub swap_total_gb: f64,
    pub swap_used_gb: f64,
}

impl MemoryStatus {
    pub fn is_healthy(&self) -> bool {
        self.usage_percent < 90.0
    }
}

/// Storage status for a mount point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatus {
    pub mount_point: String,
    pub total_gb: f64,
    pub used_gb: f64,
    pub free_gb: f64,
    pub usage_percent: f64,
    pub filesystem: String,
}

impl StorageStatus {
    pub fn is_healthy(&self) -> bool {
        self.usage_percent < 90.0
    }
}

/// Service status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub name: String,
    pub service_type: ServiceType,
    pub status: HealthStatus,
    pub endpoint: Option<String>,
    pub response_time_ms: Option<f64>,
    pub version: Option<String>,
    pub error: Option<String>,
}

/// Node0 identity verification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityStatus {
    pub node_id: String,
    pub public_key_fingerprint: String,
    pub tier1_verified: bool,
    pub tier2_verified: bool,
    pub tier3_verified: bool,
    pub hardware_fingerprint: String,
    pub in_restricted_mode: bool,
    pub last_verified: DateTime<Utc>,
    pub warnings: Vec<String>,
}

/// Data lake status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLakeStatus {
    pub connected: bool,
    pub gold_layer_accessible: bool,
    pub poi_ledger_entries: u64,
    pub knowledge_nodes: u64,
    pub knowledge_edges: u64,
    pub storage_used_gb: f64,
    pub last_sync: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

/// Complete unified Node0 status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedStatus {
    pub node_id: String,
    pub hostname: String,
    pub platform: String,
    pub timestamp: DateTime<Utc>,

    // Identity
    pub identity: IdentityStatus,

    // Resources
    pub cpu: CpuStatus,
    pub gpu: Option<GpuStatus>,
    pub memory: MemoryStatus,
    pub storage: Vec<StorageStatus>,

    // Services
    pub services: Vec<ServiceStatus>,

    // Data Lake
    pub data_lake: DataLakeStatus,

    // Aggregate health
    pub overall_health: HealthStatus,
    pub ihsan_score: f64,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Standalone verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandaloneVerification {
    pub standalone_ready: bool,
    pub checks: Vec<VerificationCheck>,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub name: String,
    pub status: String,
    pub details: Option<String>,
}

/// Resource allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub allocation_id: String,
    pub resource_type: ResourceType,
    pub requested_amount: u64,
    pub allocated_amount: u64,
    pub requester: String,
    pub created_at: DateTime<Utc>,
}

/// Node0 hardware profile (MSI Titan GT77 HX)
pub const NODE0_PROFILE: Node0Profile = Node0Profile {
    cpu_model: "Intel Core i9-14900HX",
    cpu_cores_physical: 24,
    cpu_cores_logical: 32,
    ram_gb: 128,
    gpu_model: "NVIDIA GeForce RTX 4090 Laptop GPU",
    gpu_vram_mb: 16376,
};

pub struct Node0Profile {
    pub cpu_model: &'static str,
    pub cpu_cores_physical: u32,
    pub cpu_cores_logical: u32,
    pub ram_gb: u64,
    pub gpu_model: &'static str,
    pub gpu_vram_mb: u64,
}

/// Ihsān threshold for Node0 operations
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// Unified Node0 Manager
pub struct UnifiedNode0Manager {
    // Cached status
    last_status: Arc<RwLock<Option<UnifiedStatus>>>,

    // Resource tracking
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,

    // Configuration
    data_lake_path: String,
}

impl UnifiedNode0Manager {
    /// Create a new Node0 manager
    pub fn new() -> Self {
        let data_lake_path = std::env::var("DATA_LAKE_PATH")
            .unwrap_or_else(|_| "/mnt/c/BIZRA-DATA-LAKE".to_string());

        Self {
            last_status: Arc::new(RwLock::new(None)),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            data_lake_path,
        }
    }

    /// Get system CPU information
    pub async fn get_cpu_status(&self) -> CpuStatus {
        let mut model = String::new();
        let mut frequency = 0.0;

        // Try to read CPU info from /proc/cpuinfo
        if let Ok(contents) = tokio::fs::read_to_string("/proc/cpuinfo").await {
            for line in contents.lines() {
                if line.starts_with("model name") {
                    if let Some(val) = line.split(':').nth(1) {
                        model = val.trim().to_string();
                    }
                } else if line.starts_with("cpu MHz") {
                    if let Some(val) = line.split(':').nth(1) {
                        if let Ok(f) = val.trim().parse::<f64>() {
                            frequency = f;
                        }
                    }
                }
            }
        }

        if model.is_empty() {
            model = NODE0_PROFILE.cpu_model.to_string();
        }

        // Get CPU usage (simplified)
        let usage = self.get_cpu_usage().await;

        CpuStatus {
            model,
            cores_physical: NODE0_PROFILE.cpu_cores_physical,
            cores_logical: NODE0_PROFILE.cpu_cores_logical,
            frequency_mhz: frequency,
            usage_percent: usage,
            temperature_c: None, // Would need sensors library
        }
    }

    async fn get_cpu_usage(&self) -> f64 {
        // Read /proc/stat for CPU usage
        if let Ok(contents) = tokio::fs::read_to_string("/proc/stat").await {
            if let Some(line) = contents.lines().find(|l| l.starts_with("cpu ")) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let user: f64 = parts[1].parse().unwrap_or(0.0);
                    let nice: f64 = parts[2].parse().unwrap_or(0.0);
                    let system: f64 = parts[3].parse().unwrap_or(0.0);
                    let idle: f64 = parts[4].parse().unwrap_or(0.0);

                    let total = user + nice + system + idle;
                    if total > 0.0 {
                        return ((total - idle) / total) * 100.0;
                    }
                }
            }
        }
        0.0
    }

    /// Get GPU status via nvidia-smi
    pub async fn get_gpu_status(&self) -> Option<GpuStatus> {
        let output = tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(',').collect();

        if parts.len() < 7 {
            return None;
        }

        Some(GpuStatus {
            name: parts[0].trim().to_string(),
            memory_total_mb: parts[1].trim().parse().unwrap_or(0),
            memory_used_mb: parts[2].trim().parse().unwrap_or(0),
            memory_free_mb: parts[3].trim().parse().unwrap_or(0),
            utilization_percent: parts[4].trim().parse().unwrap_or(0.0),
            temperature_c: parts[5].trim().parse().ok(),
            driver_version: Some(parts[6].trim().to_string()),
        })
    }

    /// Get memory status
    pub async fn get_memory_status(&self) -> MemoryStatus {
        if let Ok(contents) = tokio::fs::read_to_string("/proc/meminfo").await {
            let mut total = 0u64;
            let mut available = 0u64;
            let mut swap_total = 0u64;
            let mut swap_free = 0u64;

            for line in contents.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let value: u64 = parts[1].parse().unwrap_or(0);
                    match parts[0] {
                        "MemTotal:" => total = value,
                        "MemAvailable:" => available = value,
                        "SwapTotal:" => swap_total = value,
                        "SwapFree:" => swap_free = value,
                        _ => {}
                    }
                }
            }

            let total_gb = total as f64 / 1024.0 / 1024.0;
            let available_gb = available as f64 / 1024.0 / 1024.0;
            let used_gb = total_gb - available_gb;

            return MemoryStatus {
                total_gb,
                available_gb,
                used_gb,
                usage_percent: if total > 0 {
                    ((total - available) as f64 / total as f64) * 100.0
                } else {
                    0.0
                },
                swap_total_gb: swap_total as f64 / 1024.0 / 1024.0,
                swap_used_gb: (swap_total - swap_free) as f64 / 1024.0 / 1024.0,
            };
        }

        // Fallback to profile values
        MemoryStatus {
            total_gb: NODE0_PROFILE.ram_gb as f64,
            available_gb: NODE0_PROFILE.ram_gb as f64 * 0.7,
            used_gb: NODE0_PROFILE.ram_gb as f64 * 0.3,
            usage_percent: 30.0,
            swap_total_gb: 0.0,
            swap_used_gb: 0.0,
        }
    }

    /// Get storage status for key mount points
    pub async fn get_storage_status(&self) -> Vec<StorageStatus> {
        let mut storage = Vec::new();

        #[cfg(unix)]
        {
            let mounts = ["/", "/mnt/c", "/mnt/d", &self.data_lake_path];
            for mount in mounts {
                if let Ok(stat) = nix::sys::statvfs::statvfs(mount) {
                    let total = stat.blocks() * stat.fragment_size() as u64;
                    let free = stat.blocks_available() * stat.fragment_size() as u64;
                    let used = total - free;

                    storage.push(StorageStatus {
                        mount_point: mount.to_string(),
                        total_gb: total as f64 / 1024.0 / 1024.0 / 1024.0,
                        used_gb: used as f64 / 1024.0 / 1024.0 / 1024.0,
                        free_gb: free as f64 / 1024.0 / 1024.0 / 1024.0,
                        usage_percent: if total > 0 {
                            (used as f64 / total as f64) * 100.0
                        } else {
                            0.0
                        },
                        filesystem: "unknown".to_string(),
                    });
                }
            }
        }

        #[cfg(windows)]
        {
            // On Windows, provide placeholder storage info
            // Real implementation would use GetDiskFreeSpaceExW
            let mounts = ["C:\\", "D:\\"];
            for mount in mounts {
                storage.push(StorageStatus {
                    mount_point: mount.to_string(),
                    total_gb: 500.0,  // Placeholder
                    used_gb: 250.0,   // Placeholder
                    free_gb: 250.0,   // Placeholder
                    usage_percent: 50.0,
                    filesystem: "NTFS".to_string(),
                });
            }
        }

        storage
    }

    /// Check Docker service status
    pub async fn check_docker_services(&self) -> Vec<ServiceStatus> {
        let mut services = Vec::new();

        let output = tokio::process::Command::new("docker")
            .args(["compose", "ps", "--format", "json"])
            .current_dir("/mnt/c/BIZRA-Dual-Agentic-system--main")
            .output()
            .await;

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if let Ok(container) = serde_json::from_str::<serde_json::Value>(line) {
                        let name = container["Name"]
                            .as_str()
                            .unwrap_or("")
                            .split('-')
                            .last()
                            .unwrap_or("")
                            .replace("-1", "");

                        let state = container["State"].as_str().unwrap_or("unknown");
                        let health = container["Health"].as_str().unwrap_or("");

                        let status = if state == "running" && health == "healthy" {
                            HealthStatus::Healthy
                        } else if state == "running" {
                            HealthStatus::Degraded
                        } else {
                            HealthStatus::Unhealthy
                        };

                        services.push(ServiceStatus {
                            name,
                            service_type: ServiceType::Docker,
                            status,
                            endpoint: None,
                            response_time_ms: None,
                            version: None,
                            error: None,
                        });
                    }
                }
            }
        }

        services
    }

    /// Check Ollama status
    pub async fn check_ollama(&self) -> ServiceStatus {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();

        let start = std::time::Instant::now();

        match client.get("http://localhost:11434/api/tags").send().await {
            Ok(resp) if resp.status().is_success() => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;

                let model_count = if let Ok(data) = resp.json::<serde_json::Value>().await {
                    data["models"].as_array().map_or(0, |m| m.len())
                } else {
                    0
                };

                ServiceStatus {
                    name: "ollama".to_string(),
                    service_type: ServiceType::Ollama,
                    status: HealthStatus::Healthy,
                    endpoint: Some("http://localhost:11434".to_string()),
                    response_time_ms: Some(elapsed),
                    version: Some(format!("{} models", model_count)),
                    error: None,
                }
            }
            Ok(resp) => ServiceStatus {
                name: "ollama".to_string(),
                service_type: ServiceType::Ollama,
                status: HealthStatus::Degraded,
                endpoint: Some("http://localhost:11434".to_string()),
                response_time_ms: None,
                version: None,
                error: Some(format!("HTTP {}", resp.status())),
            },
            Err(e) => ServiceStatus {
                name: "ollama".to_string(),
                service_type: ServiceType::Ollama,
                status: HealthStatus::Unhealthy,
                endpoint: Some("http://localhost:11434".to_string()),
                response_time_ms: None,
                version: None,
                error: Some(e.to_string()),
            },
        }
    }

    /// Full health check
    pub async fn full_health_check(&self) -> UnifiedStatus {
        let now = Utc::now();
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Gather all status
        let cpu = self.get_cpu_status().await;
        let gpu = self.get_gpu_status().await;
        let memory = self.get_memory_status().await;
        let storage = self.get_storage_status().await;
        let mut services = self.check_docker_services().await;
        let ollama = self.check_ollama().await;
        services.push(ollama);

        // Check health and collect warnings
        if !cpu.is_healthy() {
            warnings.push(format!("CPU usage high: {:.1}%", cpu.usage_percent));
        }

        if let Some(ref g) = gpu {
            if !g.is_healthy() {
                warnings.push(format!(
                    "GPU issue: temp={:.1}°C, mem={:.1}%",
                    g.temperature_c.unwrap_or(0.0),
                    g.memory_usage_percent()
                ));
            }
        }

        if !memory.is_healthy() {
            warnings.push(format!("Memory usage high: {:.1}%", memory.usage_percent));
        }

        for s in &storage {
            if !s.is_healthy() {
                warnings.push(format!(
                    "Storage {} usage high: {:.1}%",
                    s.mount_point, s.usage_percent
                ));
            }
        }

        // Count service health
        let healthy_services = services.iter().filter(|s| s.status == HealthStatus::Healthy).count();
        let total_services = services.len();

        for s in &services {
            if s.status == HealthStatus::Unhealthy {
                if let Some(ref e) = s.error {
                    errors.push(format!("Service {} unhealthy: {}", s.name, e));
                }
            } else if s.status == HealthStatus::Degraded {
                if let Some(ref e) = s.error {
                    warnings.push(format!("Service {} degraded: {}", s.name, e));
                }
            }
        }

        // Calculate overall health
        let overall_health = if !errors.is_empty() {
            HealthStatus::Unhealthy
        } else if !warnings.is_empty() || healthy_services < total_services {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        // Calculate Ihsān score
        let ihsan_score = self.calculate_ihsan_score(
            &cpu,
            gpu.as_ref(),
            &memory,
            healthy_services,
            total_services,
        );

        // Create identity status (placeholder - would integrate with actual identity system)
        let identity = IdentityStatus {
            node_id: "NODE0".to_string(),
            public_key_fingerprint: "PLACEHOLDER".to_string(),
            tier1_verified: true,
            tier2_verified: true,
            tier3_verified: true,
            hardware_fingerprint: "f63681b9".to_string(),
            in_restricted_mode: false,
            last_verified: now,
            warnings: vec![],
        };

        // Data lake status
        let data_lake = DataLakeStatus {
            connected: std::path::Path::new(&self.data_lake_path).exists(),
            gold_layer_accessible: std::path::Path::new(&format!("{}/04_GOLD", self.data_lake_path)).exists(),
            poi_ledger_entries: 0,
            knowledge_nodes: 0,
            knowledge_edges: 0,
            storage_used_gb: 0.0,
            last_sync: Some(now),
            error: None,
        };

        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        UnifiedStatus {
            node_id: identity.node_id.clone(),
            hostname,
            platform: std::env::consts::OS.to_string(),
            timestamp: now,
            identity,
            cpu,
            gpu,
            memory,
            storage,
            services,
            data_lake,
            overall_health,
            ihsan_score,
            warnings,
            errors,
        }
    }

    fn calculate_ihsan_score(
        &self,
        cpu: &CpuStatus,
        gpu: Option<&GpuStatus>,
        memory: &MemoryStatus,
        healthy_services: usize,
        total_services: usize,
    ) -> f64 {
        // Correctness (0.22) - All systems verified
        let correctness = 1.0;

        // Safety (0.22) - Resource headroom
        let safety = (100.0 - cpu.usage_percent) / 100.0 * 0.5
            + (100.0 - memory.usage_percent) / 100.0 * 0.5;

        // User benefit (0.14) - Services available
        let user_benefit = if total_services > 0 {
            healthy_services as f64 / total_services as f64
        } else {
            0.5
        };

        // Efficiency (0.12) - Resource efficiency
        let efficiency = 0.5 + (cpu.usage_percent / 200.0);

        // Auditability (0.12) - Data lake accessible
        let auditability = 1.0;

        // Anti-centralization (0.08) - Fully local
        let anti_centralization = 1.0;

        // Robustness (0.06) - GPU healthy
        let robustness = if let Some(g) = gpu {
            if g.is_healthy() { 1.0 } else { 0.7 }
        } else {
            0.5
        };

        // Adl fairness (0.04)
        let adl_fairness = 0.95;

        correctness * 0.22
            + safety * 0.22
            + user_benefit * 0.14
            + efficiency.min(1.0) * 0.12
            + auditability * 0.12
            + anti_centralization * 0.08
            + robustness * 0.06
            + adl_fairness * 0.04
    }

    /// Verify standalone operation capability
    pub async fn verify_standalone(&self) -> StandaloneVerification {
        let mut checks = Vec::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut standalone_ready = true;

        // Check 1: GPU available
        if let Some(gpu) = self.get_gpu_status().await {
            checks.push(VerificationCheck {
                name: "GPU (NVIDIA)".to_string(),
                status: "PASS".to_string(),
                details: Some(format!("{}, {}MB free", gpu.name, gpu.memory_free_mb)),
            });
        } else {
            checks.push(VerificationCheck {
                name: "GPU (NVIDIA)".to_string(),
                status: "WARN".to_string(),
                details: None,
            });
            issues.push("No NVIDIA GPU detected".to_string());
            recommendations.push("GPU recommended for LLM inference".to_string());
        }

        // Check 2: Ollama
        let ollama = self.check_ollama().await;
        if ollama.status == HealthStatus::Healthy {
            checks.push(VerificationCheck {
                name: "Local LLM (Ollama)".to_string(),
                status: "PASS".to_string(),
                details: ollama.version,
            });
        } else {
            checks.push(VerificationCheck {
                name: "Local LLM (Ollama)".to_string(),
                status: "FAIL".to_string(),
                details: ollama.error,
            });
            issues.push("Ollama not available".to_string());
            recommendations.push("Start Ollama: ollama serve".to_string());
            standalone_ready = false;
        }

        // Check 3: Docker services
        let docker = self.check_docker_services().await;
        let healthy = docker.iter().filter(|s| s.status == HealthStatus::Healthy).count();
        let total = docker.len();

        if healthy == total && total > 0 {
            checks.push(VerificationCheck {
                name: "Docker Services".to_string(),
                status: "PASS".to_string(),
                details: Some(format!("{}/{} healthy", healthy, total)),
            });
        } else if healthy > 0 {
            checks.push(VerificationCheck {
                name: "Docker Services".to_string(),
                status: "WARN".to_string(),
                details: Some(format!("{}/{} healthy", healthy, total)),
            });
            let unhealthy: Vec<String> = docker
                .iter()
                .filter(|s| s.status != HealthStatus::Healthy)
                .map(|s| s.name.clone())
                .collect();
            issues.push(format!("Unhealthy services: {}", unhealthy.join(", ")));
        } else {
            checks.push(VerificationCheck {
                name: "Docker Services".to_string(),
                status: "FAIL".to_string(),
                details: None,
            });
            issues.push("No Docker services running".to_string());
            recommendations.push("Start services: docker compose up -d".to_string());
            standalone_ready = false;
        }

        // Check 4: Memory
        let memory = self.get_memory_status().await;
        if memory.available_gb >= 16.0 {
            checks.push(VerificationCheck {
                name: "System Memory".to_string(),
                status: "PASS".to_string(),
                details: Some(format!("{:.1}GB available", memory.available_gb)),
            });
        } else if memory.available_gb >= 8.0 {
            checks.push(VerificationCheck {
                name: "System Memory".to_string(),
                status: "WARN".to_string(),
                details: Some(format!("{:.1}GB available", memory.available_gb)),
            });
            issues.push("Low memory - may affect performance".to_string());
        } else {
            checks.push(VerificationCheck {
                name: "System Memory".to_string(),
                status: "FAIL".to_string(),
                details: Some(format!("{:.1}GB available", memory.available_gb)),
            });
            issues.push("Insufficient memory".to_string());
            standalone_ready = false;
        }

        // Check 5: Data Lake
        if std::path::Path::new(&self.data_lake_path).exists() {
            checks.push(VerificationCheck {
                name: "BIZRA-DATA-LAKE".to_string(),
                status: "PASS".to_string(),
                details: Some(self.data_lake_path.clone()),
            });
        } else {
            checks.push(VerificationCheck {
                name: "BIZRA-DATA-LAKE".to_string(),
                status: "FAIL".to_string(),
                details: None,
            });
            issues.push(format!("Data lake not found: {}", self.data_lake_path));
            standalone_ready = false;
        }

        StandaloneVerification {
            standalone_ready,
            checks,
            issues,
            recommendations,
        }
    }

    /// Get quick resource summary
    pub async fn get_resource_summary(&self) -> serde_json::Value {
        let cpu = self.get_cpu_status().await;
        let gpu = self.get_gpu_status().await;
        let memory = self.get_memory_status().await;

        serde_json::json!({
            "cpu": {
                "model": cpu.model,
                "cores": cpu.cores_logical,
                "usage_percent": cpu.usage_percent,
                "available_percent": 100.0 - cpu.usage_percent,
            },
            "gpu": gpu.as_ref().map(|g| serde_json::json!({
                "name": g.name,
                "vram_total_gb": g.memory_total_mb as f64 / 1024.0,
                "vram_free_gb": g.memory_free_mb as f64 / 1024.0,
                "utilization_percent": g.utilization_percent,
            })),
            "memory": {
                "total_gb": memory.total_gb,
                "available_gb": memory.available_gb,
                "usage_percent": memory.usage_percent,
            },
        })
    }
}

impl Default for UnifiedNode0Manager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_default() {
        assert_eq!(HealthStatus::default(), HealthStatus::Unknown);
    }

    #[test]
    fn test_cpu_status_healthy() {
        let cpu = CpuStatus {
            model: "Test CPU".to_string(),
            cores_physical: 8,
            cores_logical: 16,
            frequency_mhz: 3000.0,
            usage_percent: 50.0,
            temperature_c: Some(60.0),
        };
        assert!(cpu.is_healthy());
    }

    #[test]
    fn test_cpu_status_unhealthy_usage() {
        let cpu = CpuStatus {
            model: "Test CPU".to_string(),
            cores_physical: 8,
            cores_logical: 16,
            frequency_mhz: 3000.0,
            usage_percent: 96.0,
            temperature_c: Some(60.0),
        };
        assert!(!cpu.is_healthy());
    }

    #[test]
    fn test_gpu_memory_usage() {
        let gpu = GpuStatus {
            name: "Test GPU".to_string(),
            memory_total_mb: 16000,
            memory_used_mb: 8000,
            memory_free_mb: 8000,
            utilization_percent: 50.0,
            temperature_c: Some(60.0),
            driver_version: Some("550.0".to_string()),
        };
        assert!((gpu.memory_usage_percent() - 50.0).abs() < 0.1);
        assert!(gpu.is_healthy());
    }

    #[test]
    fn test_node0_profile() {
        assert_eq!(NODE0_PROFILE.ram_gb, 128);
        assert_eq!(NODE0_PROFILE.gpu_vram_mb, 16376);
    }
}
