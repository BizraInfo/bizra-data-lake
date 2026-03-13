# 01 — Device Detection & Hardware Adaptation

> Module: `bizra-installer/src/device/`
> Language: Rust (Tauri backend)
> Constitutional Anchor: Law 3 (Hardware Adaptation) + Law 5 (Progressive Capability)

## 1. DeviceProfile Struct

```
STRUCT DeviceProfile:
    # Platform
    os: OS                     # Windows | macOS | Linux | Android | iOS
    os_version: String         # "11", "14.2", "24.04"
    arch: Arch                 # x86_64 | aarch64 | riscv64

    # Compute
    cpu_cores: u32             # Physical cores
    cpu_threads: u32           # Logical threads
    ram_total_gb: f32          # Total RAM
    ram_available_gb: f32      # Available at install time

    # GPU
    gpu: Option<GPUInfo>       # {vendor, model, vram_gb, api}

    # Storage
    disk_available_gb: f32
    disk_type: DiskType        # SSD | HDD | eMMC

    # Network
    network_available: bool
    network_speed_mbps: f32

    # Locale
    system_locale: String      # "ar-AE", "en-US", "pt-BR"
    timezone: String           # "Asia/Dubai"

    # Display
    screen_width: u32
    screen_height: u32
    dpi: f32
    touch_capable: bool
```

## 2. Detection Pseudocode

```
FUNCTION detect_device() -> DeviceProfile:
    profile = DeviceProfile::default()

    # Platform detection (compile-time + runtime)
    profile.os = detect_os()           # cfg!(target_os) + runtime version
    profile.arch = detect_arch()       # cfg!(target_arch)
    profile.os_version = os_version()  # Platform-specific API

    # CPU detection
    profile.cpu_cores = num_cpus::get_physical()
    profile.cpu_threads = num_cpus::get()

    # RAM detection
    profile.ram_total_gb = sys_info::mem_total() / 1024^3
    profile.ram_available_gb = sys_info::mem_available() / 1024^3

    # GPU detection (graceful — never fails)
    profile.gpu = detect_gpu()  # Returns None if no GPU

    # Storage
    profile.disk_available_gb = disk_free(install_path) / 1024^3
    profile.disk_type = detect_disk_type(install_path)

    # Network (async, timeout 3s)
    profile.network_available = ping_download_server(timeout=3s)
    IF profile.network_available:
        profile.network_speed_mbps = estimate_speed(sample=1MB)

    # Locale
    profile.system_locale = sys_locale::get_locale() OR "en-US"
    profile.timezone = iana_time_zone::get_timezone() OR "UTC"

    # Display
    (w, h, dpi) = detect_display()
    profile.screen_width = w
    profile.screen_height = h
    profile.dpi = dpi
    profile.touch_capable = detect_touch()

    RETURN profile
```

## 3. GPU Detection

```
FUNCTION detect_gpu() -> Option<GPUInfo>:
    # Try each API in priority order
    # Never crash — return None on any failure

    IF cfg!(target_os = "windows") OR cfg!(target_os = "linux"):
        TRY:
            info = nvml_detect()       # NVIDIA via NVML
            RETURN Some(GPUInfo {
                vendor: NVIDIA,
                model: info.name,
                vram_gb: info.memory / 1024^3,
                api: CUDA
            })

        TRY:
            info = rocm_detect()       # AMD via ROCm
            RETURN Some(GPUInfo { vendor: AMD, api: ROCm, ... })

        TRY:
            info = vulkan_detect()     # Generic Vulkan
            RETURN Some(GPUInfo { vendor: infer(info), api: Vulkan, ... })

    IF cfg!(target_os = "macos"):
        TRY:
            info = metal_detect()      # Apple Metal
            RETURN Some(GPUInfo { vendor: Apple, api: Metal, ... })

    IF cfg!(target_os = "android"):
        TRY:
            info = android_gpu_detect() # Android GPU info
            RETURN Some(GPUInfo { vendor: infer(info), api: Vulkan, ... })

    RETURN None  # CPU-only fallback — always valid
```

## 4. Adaptive Model Selection

```
FUNCTION select_model_tier(profile: &DeviceProfile) -> ModelTier:
    ram = profile.ram_available_gb
    gpu_vram = profile.gpu.map(|g| g.vram_gb).unwrap_or(0.0)

    # RAM-based tier selection
    tier = MATCH ram:
        r IF r < 2.0  => ModelTier::Micro      # TinyLlama 1.1B Q2_K, ~500MB
        r IF r < 4.0  => ModelTier::Compact     # TinyLlama 1.1B Q4_K_M, ~650MB
        r IF r < 8.0  => ModelTier::Standard    # Phi-3-mini 3.8B Q4_K_M, ~2.3GB
        r IF r < 16.0 => ModelTier::Enhanced    # Llama 3.1 8B Q4_K_M, ~4.7GB
        r IF r < 32.0 => ModelTier::Full        # Qwen 2.5 14B Q4_K_M, ~8.5GB
        r IF r < 64.0 => ModelTier::Elite       # Llama 3.1 70B Q4_K_M, ~40GB
        _             => ModelTier::Premium     # Multiple models, Q8/FP16

    # Disk space guard — downgrade if insufficient
    WHILE tier.disk_requirement() > profile.disk_available_gb - 0.5:
        tier = tier.downgrade()  # Never goes below Micro

    RETURN tier
```

```
FUNCTION select_backend(profile: &DeviceProfile) -> InferenceBackend:
    MATCH profile.gpu:
        Some(gpu) IF gpu.api == CUDA   => LlamaCppCuda
        Some(gpu) IF gpu.api == ROCm   => LlamaCppROCm
        Some(gpu) IF gpu.api == Metal  => LlamaCppMetal
        Some(gpu) IF gpu.api == Vulkan => LlamaCppVulkan
        _                              => LlamaCppCPU  # AVX2/NEON auto
```

## 5. Model Tier Data

```
ENUM ModelTier:
    Micro:
        model: "tinyllama-1.1b"
        quant: "Q2_K"
        disk_gb: 0.5
        min_ram_gb: 1.0
        quality: "S1 reflex only, sovereign"
        heartbeat_interval: 300s  # 5 min (degraded)

    Compact:
        model: "tinyllama-1.1b"
        quant: "Q4_K_M"
        disk_gb: 0.65
        min_ram_gb: 2.0
        quality: "Basic tasks"

    Standard:
        model: "phi-3-mini-3.8b"
        quant: "Q4_K_M"
        disk_gb: 2.3
        min_ram_gb: 4.0
        quality: "Most tasks"

    Enhanced:
        model: "llama-3.1-8b"
        quant: "Q4_K_M"
        disk_gb: 4.7
        min_ram_gb: 8.0
        quality: "Complex tasks"

    Full:
        model: "qwen-2.5-14b"
        quant: "Q4_K_M"
        disk_gb: 8.5
        min_ram_gb: 16.0
        quality: "Excellent"

    Elite:
        model: "llama-3.1-70b"
        quant: "Q4_K_M"
        disk_gb: 40.0
        min_ram_gb: 32.0
        quality: "Elite"

    Premium:
        model: "multiple"
        quant: "Q8/FP16"
        disk_gb: 80.0
        min_ram_gb: 64.0
        quality: "MoE routing"
```

## 6. OOM Fallback Chain

```
FUNCTION load_model_with_fallback(tier: ModelTier) -> Result<Model>:
    current = tier
    LOOP:
        TRY:
            model = llama_cpp_load(current.model_path(), current.quant)
            LOG info "Model loaded: {current} ({current.disk_gb}GB)"
            RETURN Ok(model)
        CATCH OutOfMemory:
            IF current == ModelTier::Micro:
                RETURN Err("Device cannot run any model — insufficient resources")
            current = current.downgrade()
            LOG warn "OOM: falling back to {current}"
```

## TDD Anchors

```
TEST detect_profile_never_panics:
    # DeviceProfile::detect() must succeed on ANY platform
    profile = detect_device()
    ASSERT profile.os IS_NOT None
    ASSERT profile.ram_total_gb > 0
    ASSERT profile.cpu_cores > 0

TEST gpu_detection_returns_none_without_gpu:
    # On CPU-only CI, detect_gpu() returns None (not error)
    result = detect_gpu()
    # Must not panic, may return None

TEST model_tier_1gb_gets_micro:
    profile = DeviceProfile { ram_available_gb: 1.0, ... }
    tier = select_model_tier(&profile)
    ASSERT tier == ModelTier::Micro

TEST model_tier_16gb_gets_full:
    profile = DeviceProfile { ram_available_gb: 16.0, ... }
    tier = select_model_tier(&profile)
    ASSERT tier == ModelTier::Full

TEST model_tier_downgrades_on_low_disk:
    profile = DeviceProfile { ram_available_gb: 16.0, disk_available_gb: 3.0, ... }
    tier = select_model_tier(&profile)
    ASSERT tier.disk_requirement() <= 2.5  # Can't fit Full (8.5GB)

TEST fallback_chain_never_panics:
    # Even with 512MB RAM, fallback reaches Micro without crash
    result = load_model_with_fallback(ModelTier::Premium)
    # Either succeeds with a tier or returns Err — never panics

TEST backend_selects_cpu_without_gpu:
    profile = DeviceProfile { gpu: None, ... }
    backend = select_backend(&profile)
    ASSERT backend == LlamaCppCPU

TEST locale_detection_defaults_to_en:
    # If system locale unavailable, default to "en-US"
    # Never returns empty string
    locale = detect_locale()
    ASSERT locale.len() >= 2
```
