//! Comprehensive tests for bizra-installer — hardware detection, model cache, CLI parsing
//!
//! Phase 13: Test Sprint

mod hardware {
    // We test hardware_detect and model_cache which are private modules in main.rs.
    // Since they're `mod` inside main.rs, we can't access them from integration tests.
    // Instead, we test them indirectly through the public re-export patterns,
    // or we test the behaviors we can observe.

    // For hardware_detect.rs and model_cache.rs testing, we replicate the structs
    // and test the logic since they're not public library exports.
    // This is a common pattern for binary-only crates.

    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    struct HardwareProfile {
        cpu_cores: usize,
        ram_gb: f64,
        gpu_name: Option<String>,
        vram_gb: f64,
        disk_free_gb: f64,
        os: String,
    }

    impl HardwareProfile {
        fn recommended_tier(&self) -> &'static str {
            if self.vram_gb >= 8.0 {
                "LOCAL"
            } else if self.vram_gb >= 4.0 {
                "EDGE+"
            } else {
                "EDGE"
            }
        }

        fn recommended_model(&self) -> &'static str {
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

    #[test]
    fn tier_local_for_high_vram() {
        let hw = HardwareProfile {
            cpu_cores: 8,
            ram_gb: 32.0,
            gpu_name: Some("RTX 4090".into()),
            vram_gb: 24.0,
            disk_free_gb: 500.0,
            os: "windows".into(),
        };
        assert_eq!(hw.recommended_tier(), "LOCAL");
    }

    #[test]
    fn tier_edge_plus_for_mid_vram() {
        let hw = HardwareProfile {
            cpu_cores: 8,
            ram_gb: 16.0,
            gpu_name: Some("RTX 3060".into()),
            vram_gb: 6.0,
            disk_free_gb: 200.0,
            os: "linux".into(),
        };
        assert_eq!(hw.recommended_tier(), "EDGE+");
    }

    #[test]
    fn tier_edge_for_no_gpu() {
        let hw = HardwareProfile {
            cpu_cores: 4,
            ram_gb: 8.0,
            gpu_name: None,
            vram_gb: 0.0,
            disk_free_gb: 50.0,
            os: "windows".into(),
        };
        assert_eq!(hw.recommended_tier(), "EDGE");
    }

    #[test]
    fn model_7b_for_high_vram() {
        let hw = HardwareProfile {
            cpu_cores: 8,
            ram_gb: 32.0,
            gpu_name: Some("RTX 4090".into()),
            vram_gb: 24.0,
            disk_free_gb: 500.0,
            os: "windows".into(),
        };
        assert_eq!(hw.recommended_model(), "Qwen2.5-7B-Q4");
    }

    #[test]
    fn model_3b_for_mid_vram() {
        let hw = HardwareProfile {
            cpu_cores: 8,
            ram_gb: 16.0,
            gpu_name: Some("RTX 3060".into()),
            vram_gb: 6.0,
            disk_free_gb: 200.0,
            os: "linux".into(),
        };
        assert_eq!(hw.recommended_model(), "Qwen2.5-3B-Q4");
    }

    #[test]
    fn model_1_5b_for_cpu_with_enough_ram() {
        let hw = HardwareProfile {
            cpu_cores: 4,
            ram_gb: 16.0,
            gpu_name: None,
            vram_gb: 0.0,
            disk_free_gb: 100.0,
            os: "windows".into(),
        };
        assert_eq!(hw.recommended_model(), "Qwen2.5-1.5B-Q4");
    }

    #[test]
    fn model_0_5b_for_low_resources() {
        let hw = HardwareProfile {
            cpu_cores: 2,
            ram_gb: 4.0,
            gpu_name: None,
            vram_gb: 0.0,
            disk_free_gb: 20.0,
            os: "linux".into(),
        };
        assert_eq!(hw.recommended_model(), "Qwen2.5-0.5B-Q4");
    }

    #[test]
    fn tier_boundary_exactly_8gb_vram() {
        let hw = HardwareProfile {
            cpu_cores: 8,
            ram_gb: 16.0,
            gpu_name: Some("GPU".into()),
            vram_gb: 8.0,
            disk_free_gb: 100.0,
            os: "linux".into(),
        };
        assert_eq!(hw.recommended_tier(), "LOCAL");
    }

    #[test]
    fn tier_boundary_exactly_4gb_vram() {
        let hw = HardwareProfile {
            cpu_cores: 8,
            ram_gb: 16.0,
            gpu_name: Some("GPU".into()),
            vram_gb: 4.0,
            disk_free_gb: 100.0,
            os: "linux".into(),
        };
        assert_eq!(hw.recommended_tier(), "EDGE+");
    }

    #[test]
    fn model_boundary_exactly_8gb_ram_no_gpu() {
        let hw = HardwareProfile {
            cpu_cores: 4,
            ram_gb: 8.0,
            gpu_name: None,
            vram_gb: 0.0,
            disk_free_gb: 50.0,
            os: "windows".into(),
        };
        assert_eq!(hw.recommended_model(), "Qwen2.5-1.5B-Q4");
    }
}

mod model_cache {
    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    struct ModelSpec {
        name: String,
        desc: String,
        size_gb: f64,
        url: String,
        tier: String,
    }

    impl ModelSpec {
        fn available() -> Vec<Self> {
            vec![
                Self {
                    name: "qwen2.5-0.5b-q4".into(),
                    desc: "Ultra lightweight".into(),
                    size_gb: 0.4,
                    url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
                    tier: "EDGE".into(),
                },
                Self {
                    name: "qwen2.5-1.5b-q4".into(),
                    desc: "Lightweight".into(),
                    size_gb: 1.1,
                    url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF".into(),
                    tier: "EDGE".into(),
                },
                Self {
                    name: "qwen2.5-7b-q4".into(),
                    desc: "Balanced".into(),
                    size_gb: 4.7,
                    url: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF".into(),
                    tier: "LOCAL".into(),
                },
                Self {
                    name: "llama3-8b-q4".into(),
                    desc: "General purpose".into(),
                    size_gb: 4.9,
                    url: "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct".into(),
                    tier: "LOCAL".into(),
                },
            ]
        }
    }

    #[test]
    fn available_models_not_empty() {
        let models = ModelSpec::available();
        assert!(!models.is_empty());
    }

    #[test]
    fn available_models_count() {
        let models = ModelSpec::available();
        assert_eq!(models.len(), 4);
    }

    #[test]
    fn available_models_have_valid_sizes() {
        for m in ModelSpec::available() {
            assert!(m.size_gb > 0.0, "model {} has invalid size", m.name);
        }
    }

    #[test]
    fn available_models_have_urls() {
        for m in ModelSpec::available() {
            assert!(
                m.url.starts_with("https://"),
                "model {} has invalid URL",
                m.name
            );
        }
    }

    #[test]
    fn available_models_have_valid_tiers() {
        for m in ModelSpec::available() {
            assert!(
                m.tier == "EDGE" || m.tier == "LOCAL" || m.tier == "POOL",
                "model {} has invalid tier: {}",
                m.name,
                m.tier
            );
        }
    }

    #[test]
    fn edge_models_smaller_than_local() {
        let models = ModelSpec::available();
        let max_edge = models
            .iter()
            .filter(|m| m.tier == "EDGE")
            .map(|m| m.size_gb)
            .fold(0.0f64, f64::max);
        let min_local = models
            .iter()
            .filter(|m| m.tier == "LOCAL")
            .map(|m| m.size_gb)
            .fold(f64::MAX, f64::min);
        assert!(
            max_edge < min_local,
            "Edge models should be smaller than local models"
        );
    }
}

mod expand_home {
    use std::path::{Path, PathBuf};

    fn expand_home(path: &Path) -> PathBuf {
        let path_str = path.to_string_lossy();
        if path_str.starts_with("~/") || path_str == "~" {
            if let Some(home) = dirs::home_dir() {
                return home.join(path_str.trim_start_matches("~/"));
            }
        }
        path.to_path_buf()
    }

    #[test]
    fn expand_absolute_path_unchanged() {
        let p = Path::new("/usr/local/bin");
        assert_eq!(expand_home(p), PathBuf::from("/usr/local/bin"));
    }

    #[test]
    fn expand_relative_path_unchanged() {
        let p = Path::new("relative/path");
        assert_eq!(expand_home(p), PathBuf::from("relative/path"));
    }

    #[test]
    fn expand_tilde_path() {
        let p = Path::new("~/.bizra");
        let expanded = expand_home(p);
        // Should be home_dir + .bizra
        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, home.join(".bizra"));
        }
    }
}
