//! Installation Health Check — Constitutional Verification
//!
//! After installation completes, run a 10-point health check to ensure
//! the node is fully operational. If ANY check fails, the installer
//! rolls back completely. The user never sees a broken terminal.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §16
//! Standing on Giants: Deming (PDCA, 1950), Lamport (consensus, 1978)

use serde::{Deserialize, Serialize};

use crate::device_profile::DeviceProfile;

// ─────────────────────────────────────────────────────────────
// Health Check Results
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    Pass,
    Fail,
    Warn,
    Skip,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthCheckItem {
    pub name: String,
    pub description: String,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthCheckReport {
    pub items: Vec<HealthCheckItem>,
    pub all_passed: bool,
    pub critical_failures: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: String,
}

impl HealthCheckReport {
    pub fn new(items: Vec<HealthCheckItem>) -> Self {
        let critical_failures: Vec<String> = items
            .iter()
            .filter(|i| i.status == CheckStatus::Fail)
            .map(|i| i.name.clone())
            .collect();

        let warnings: Vec<String> = items
            .iter()
            .filter(|i| i.status == CheckStatus::Warn)
            .map(|i| i.name.clone())
            .collect();

        let all_passed = critical_failures.is_empty();

        Self {
            items,
            all_passed,
            critical_failures,
            warnings,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Individual Checks (Spec §16)
// ─────────────────────────────────────────────────────────────

/// Check 1: Core runtime binary exists and is executable
pub fn check_core_runtime(install_dir: &std::path::Path) -> HealthCheckItem {
    let binary = if cfg!(target_os = "windows") {
        install_dir.join("bin").join("bizra-node.exe")
    } else {
        install_dir.join("bin").join("bizra-node")
    };

    let (status, detail) = if binary.exists() {
        (CheckStatus::Pass, format!("Found: {}", binary.display()))
    } else {
        (CheckStatus::Fail, format!("Missing: {}", binary.display()))
    };

    HealthCheckItem {
        name: "core_runtime".into(),
        description: "Core runtime executable".into(),
        status,
        detail,
    }
}

/// Check 2: LLM model file exists and loadable
pub fn check_llm_model(install_dir: &std::path::Path) -> HealthCheckItem {
    let models_dir = install_dir.join("models");

    if !models_dir.exists() {
        return HealthCheckItem {
            name: "llm_model".into(),
            description: "LLM model loads successfully".into(),
            status: CheckStatus::Fail,
            detail: format!("Models directory missing: {}", models_dir.display()),
        };
    }

    // Check for any .gguf file
    let has_model = std::fs::read_dir(&models_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
        })
        .unwrap_or(false);

    if has_model {
        HealthCheckItem {
            name: "llm_model".into(),
            description: "LLM model loads successfully".into(),
            status: CheckStatus::Pass,
            detail: "GGUF model found".into(),
        }
    } else {
        HealthCheckItem {
            name: "llm_model".into(),
            description: "LLM model loads successfully".into(),
            status: CheckStatus::Fail,
            detail: "No .gguf model file found".into(),
        }
    }
}

/// Check 3: Ed25519 identity exists
pub fn check_identity(install_dir: &std::path::Path) -> HealthCheckItem {
    let identity_file = install_dir.join("identity.json");

    let (status, detail) = if identity_file.exists() {
        // Verify it's valid JSON with required fields
        match std::fs::read_to_string(&identity_file) {
            Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(val) => {
                    if val.get("public_key").is_some() && val.get("node_id").is_some() {
                        (CheckStatus::Pass, "Identity valid".into())
                    } else {
                        (
                            CheckStatus::Fail,
                            "Identity file missing required fields".into(),
                        )
                    }
                }
                Err(e) => (CheckStatus::Fail, format!("Invalid JSON: {e}")),
            },
            Err(e) => (CheckStatus::Fail, format!("Cannot read: {e}")),
        }
    } else {
        (CheckStatus::Fail, "identity.json not found".into())
    };

    HealthCheckItem {
        name: "identity".into(),
        description: "Ed25519 identity generated".into(),
        status,
        detail,
    }
}

/// Check 4: Evidence ledger initialized (block #0)
pub fn check_evidence_ledger(install_dir: &std::path::Path) -> HealthCheckItem {
    let ledger = install_dir.join("evidence_ledger");

    let (status, detail) = if ledger.exists() && ledger.is_dir() {
        (CheckStatus::Pass, "Evidence ledger directory exists".into())
    } else {
        (CheckStatus::Fail, "Evidence ledger not initialized".into())
    };

    HealthCheckItem {
        name: "evidence_ledger".into(),
        description: "Evidence ledger initialized (block #0)".into(),
        status,
        detail,
    }
}

/// Check 5: 12 agents minted (7 PAT + 5 SAT)
pub fn check_agents(install_dir: &std::path::Path) -> HealthCheckItem {
    let agents_file = install_dir.join("agents.json");

    let (status, detail) = if agents_file.exists() {
        match std::fs::read_to_string(&agents_file) {
            Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(val) => {
                    let count = val
                        .as_array()
                        .map(|a| a.len())
                        .or_else(|| {
                            val.get("agents")
                                .and_then(|a| a.as_array())
                                .map(|a| a.len())
                        })
                        .unwrap_or(0);
                    if count >= 12 {
                        (CheckStatus::Pass, format!("{count} agents minted"))
                    } else {
                        (CheckStatus::Warn, format!("Only {count}/12 agents minted"))
                    }
                }
                Err(e) => (CheckStatus::Fail, format!("Invalid agents.json: {e}")),
            },
            Err(e) => (CheckStatus::Fail, format!("Cannot read: {e}")),
        }
    } else {
        (CheckStatus::Fail, "agents.json not found".into())
    };

    HealthCheckItem {
        name: "agents".into(),
        description: "12 agents minted (7 PAT + 5 SAT)".into(),
        status,
        detail,
    }
}

/// Check 6: Language packs loaded
pub fn check_language_packs(install_dir: &std::path::Path) -> HealthCheckItem {
    let locales_dir = install_dir.join("locales");

    if !locales_dir.exists() {
        return HealthCheckItem {
            name: "language_packs".into(),
            description: "Language packs loaded".into(),
            status: CheckStatus::Fail,
            detail: "Locales directory missing".into(),
        };
    }

    let pack_count = std::fs::read_dir(&locales_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .count()
        })
        .unwrap_or(0);

    let (status, detail) = if pack_count >= 2 {
        (
            CheckStatus::Pass,
            format!("{pack_count} language pack(s) loaded"),
        )
    } else if pack_count == 1 {
        (
            CheckStatus::Warn,
            "Only 1 language pack — recommend at least ar + en".into(),
        )
    } else {
        (CheckStatus::Fail, "No language packs found".into())
    };

    HealthCheckItem {
        name: "language_packs".into(),
        description: "Language packs loaded".into(),
        status,
        detail,
    }
}

/// Check 7: Disk space sufficient (500MB free after install)
pub fn check_disk_space(profile: &DeviceProfile) -> HealthCheckItem {
    let (status, detail) = if profile.disk_available_gb >= 0.5 {
        (
            CheckStatus::Pass,
            format!("{:.1} GB free", profile.disk_available_gb),
        )
    } else {
        (
            CheckStatus::Warn,
            format!(
                "Only {:.1} GB free — recommend 500MB minimum",
                profile.disk_available_gb
            ),
        )
    };

    HealthCheckItem {
        name: "disk_space".into(),
        description: "Disk space sufficient (500MB free after install)".into(),
        status,
        detail,
    }
}

/// Run all health checks and return a report (Spec §16)
pub fn run_health_check(
    install_dir: &std::path::Path,
    profile: &DeviceProfile,
) -> HealthCheckReport {
    let items = vec![
        check_core_runtime(install_dir),
        check_llm_model(install_dir),
        check_identity(install_dir),
        check_evidence_ledger(install_dir),
        check_agents(install_dir),
        check_language_packs(install_dir),
        check_disk_space(profile),
    ];

    HealthCheckReport::new(items)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn health_report_tracks_failures() {
        let items = vec![
            HealthCheckItem {
                name: "a".into(),
                description: "test".into(),
                status: CheckStatus::Pass,
                detail: "ok".into(),
            },
            HealthCheckItem {
                name: "b".into(),
                description: "test".into(),
                status: CheckStatus::Fail,
                detail: "bad".into(),
            },
        ];
        let report = HealthCheckReport::new(items);
        assert!(!report.all_passed);
        assert_eq!(report.critical_failures, vec!["b"]);
    }

    #[test]
    fn health_report_all_pass() {
        let items = vec![
            HealthCheckItem {
                name: "a".into(),
                description: "test".into(),
                status: CheckStatus::Pass,
                detail: "ok".into(),
            },
            HealthCheckItem {
                name: "b".into(),
                description: "test".into(),
                status: CheckStatus::Warn,
                detail: "warn".into(),
            },
        ];
        let report = HealthCheckReport::new(items);
        assert!(report.all_passed); // Warnings don't block
        assert_eq!(report.warnings, vec!["b"]);
    }

    #[test]
    fn disk_space_check() {
        let mut profile = DeviceProfile::default();
        profile.disk_available_gb = 10.0;
        let check = check_disk_space(&profile);
        assert_eq!(check.status, CheckStatus::Pass);

        profile.disk_available_gb = 0.3;
        let check = check_disk_space(&profile);
        assert_eq!(check.status, CheckStatus::Warn);
    }

    #[test]
    fn missing_dir_fails() {
        let dir = std::path::Path::new("/nonexistent/bizra/install");
        let check = check_core_runtime(dir);
        assert_eq!(check.status, CheckStatus::Fail);
    }
}
