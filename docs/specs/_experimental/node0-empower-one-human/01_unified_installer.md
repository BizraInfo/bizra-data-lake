# Phase 01 — Unified Installer (Any OS)

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Tauri (native web apps) · Wasmtime (WASM sandbox) · Firecracker (microVM) · Docker (containers) · 12-Factor App (config)

## 1.1 Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| I-01 | Install on Windows, macOS, Linux | Signed binary for each platform |
| I-02 | Complete in ≤ 10 minutes | Timer from download to first OODA cycle |
| I-03 | Zero terminal interaction | GUI wizard handles all steps |
| I-04 | Create local wallet | Ed25519 keypair stored encrypted |
| I-05 | Create encrypted data store | AES-256-GCM or XChaCha20-Poly1305 |
| I-06 | Run as background service + tray | System service + tray icon with status |
| I-07 | Resource slider (CPU/GPU/RAM) | User controls allocation, enforced by cgroup/WASM limits |

## 1.2 Two-Layer Architecture

### Host App (Tauri + Rust)

```pseudocode
MODULE HostApp:
  STRUCT HostApp:
    tauri_window:  TauriWebview
    tray:          SystemTray
    capsule:       RuntimeCapsule
    wallet:        LocalWallet
    config:        AppConfig

  FUNCTION main():
    config = load_or_create_config()
    IF NOT config.setup_complete:
      run_install_wizard(config)
    capsule = RuntimeCapsule.connect(config.capsule_type)
    tray = SystemTray.create(icon="bizra", menu=[
      "Status"   -> show_status_window,
      "Goals"    -> show_goals_window,
      "Settings" -> show_settings_window,
      "---",
      "Quit"     -> graceful_shutdown
    ])
    capsule.start()
    tray.run()

  FUNCTION graceful_shutdown():
    capsule.stop(timeout=30_seconds)
    tray.destroy()
    EXIT 0
```

### Runtime Capsule Interface

```pseudocode
INTERFACE RuntimeCapsule:
  """
  Uniform interface regardless of backend (WASM, microVM, Docker).
  Standing on Giants: Adapter pattern (GoF, 1994)
  """

  FUNCTION start() -> Result<(), CapsuleError>
  FUNCTION stop(timeout: Duration) -> Result<(), CapsuleError>
  FUNCTION health() -> HealthStatus
  FUNCTION execute(command: CapsuleCommand) -> CapsuleResult
  FUNCTION mount_volume(host_path: Path, guest_path: Path, readonly: bool)
  FUNCTION set_resource_limits(cpu_percent: u8, memory_mb: u32, gpu_percent: u8)
  FUNCTION expose_port(host: u16, guest: u16)

ENUM CapsuleBackend:
  WASM       # Primary: Wasmtime sandbox (portable, lowest overhead)
  MICROVM    # Secondary: Firecracker (Linux only, strongest isolation)
  DOCKER     # Fallback: Docker/Containerd (widely available)

FUNCTION select_capsule_backend(platform: Platform) -> CapsuleBackend:
  IF platform.has_kvm AND platform.is_linux:
    RETURN CapsuleBackend.MICROVM
  ELSE IF platform.has_wasm_runtime:
    RETURN CapsuleBackend.WASM
  ELSE IF platform.has_docker:
    RETURN CapsuleBackend.DOCKER
  ELSE:
    ABORT "No supported isolation backend available"
```

## 1.3 Install Wizard Flow

```pseudocode
MODULE InstallWizard:
  """
  GUI wizard — zero terminal interaction.
  Standing on Giants: Tauri (frontend) · 12-Factor App (config)
  """

  FUNCTION run_install_wizard(config: AppConfig) -> Result<(), InstallError>:

    # ── Step 1: System Check ─────────────────────────────────
    show_page("Checking your system...")
    checks = SystemChecks {
      cpu_virt:    check_cpu_virtualization(),    # VT-x / AMD-V
      disk_space:  check_disk_space(min=10_GB),
      gpu:         detect_gpu(),                  # Optional
      ram:         check_ram(min=8_GB),
      platform:    detect_platform(),
    }

    IF checks.has_blocker():
      show_page("System Requirements", blockers=checks.blockers())
      RETURN Err(InstallError::SystemRequirements)

    show_page("System OK", summary=checks.summary())
    await user_click("Continue")

    # ── Step 2: Pull Runtime Capsule ──────────────────────────
    show_page("Downloading BIZRA Runtime...")
    backend = select_capsule_backend(checks.platform)
    capsule_image = download_signed_image(
      backend=backend,
      progress_callback=update_progress_bar
    )
    verify_signature(capsule_image, BIZRA_PUBLIC_KEY)
    config.capsule_type = backend
    config.capsule_image = capsule_image.path

    # ── Step 3: Generate Credentials ──────────────────────────
    show_page("Creating your secure identity...")
    passphrase = prompt_passphrase(
      label="Set a passphrase to protect your data",
      min_length=12,
      strength_meter=true
    )

    keypair = generate_ed25519_keypair()                # core.pci.crypto
    encrypted_store = create_encrypted_volume(
      path=config.data_dir / "vault",
      cipher="XChaCha20-Poly1305",
      key=derive_key(passphrase, salt=random_salt())    # Argon2id KDF
    )
    wallet = LocalWallet {
      public_key:  keypair.public,
      private_key: encrypt(keypair.private, derived_key),
      node_id:     sha256(keypair.public)[:16],
    }
    store_wallet(encrypted_store, wallet)
    config.wallet_fingerprint = wallet.node_id

    # ── Step 4: Register System Service ───────────────────────
    show_page("Setting up background service...")
    register_system_service(
      name="bizra-node0",
      binary=config.host_app_path,
      args=["--daemon"],
      start_on_boot=true,
      restart_policy="on-failure"
    )
    register_tray_app(config)

    # ── Step 5: Onboarding (PAT Personalization) ──────────────
    show_page("Let's set up your Personal Agent Team")
    onboarding = run_onboarding_flow()   # See Phase 02
    config.pat_config = onboarding.pat_config
    config.initial_goals = onboarding.goals
    config.setup_complete = true
    save_config(config)

    show_page("Setup Complete!", message="BIZRA Node0 is ready.")
    RETURN Ok(())
```

## 1.4 Resource Slider

```pseudocode
MODULE ResourceSlider:
  """
  User-controlled resource allocation.
  Standing on Giants: cgroups v2 (Linux) · Job Objects (Windows) · WASM limits
  """

  STRUCT ResourceLimits:
    cpu_percent:    u8     # 0-100, % of available cores
    memory_mb:      u32    # Maximum RSS
    gpu_percent:    u8     # 0-100, % of available VRAM
    disk_quota_gb:  u16    # Maximum disk usage

  CONST PRESETS = {
    "minimal":   ResourceLimits(10, 2048, 0,  5),
    "balanced":  ResourceLimits(25, 4096, 25, 20),
    "generous":  ResourceLimits(50, 8192, 50, 50),
    "maximum":   ResourceLimits(75, 16384, 75, 100),
  }

  FUNCTION apply_limits(capsule: RuntimeCapsule, limits: ResourceLimits):
    capsule.set_resource_limits(
      cpu_percent=limits.cpu_percent,
      memory_mb=limits.memory_mb,
      gpu_percent=limits.gpu_percent
    )
    log("Resource limits applied", limits)
```

## 1.5 Security Properties

```pseudocode
SECURITY_INVARIANTS:
  # S-1: Signed binaries
  ASSERT host_app.signature.verify(BIZRA_PUBLIC_KEY) == true
  ASSERT capsule_image.signature.verify(BIZRA_PUBLIC_KEY) == true

  # S-2: No secrets in image
  ASSERT scan_for_secrets(capsule_image) == []

  # S-3: Encrypted at rest
  ASSERT wallet.private_key.is_encrypted() == true
  ASSERT data_store.cipher IN ["AES-256-GCM", "XChaCha20-Poly1305"]

  # S-4: Sandboxed execution
  ASSERT capsule.isolation_level >= IsolationLevel.WASM
  ASSERT capsule.can_access_host_filesystem() == false  # only mounted volumes

  # S-5: KDF is memory-hard
  ASSERT kdf_algorithm == "Argon2id"
  ASSERT kdf_memory_cost >= 64_MB
```

## 1.6 TDD Anchors

```pseudocode
TEST "install_completes_under_10_minutes":
  timer = start_timer()
  result = run_install_wizard(config=fresh_config())
  ASSERT result.is_ok()
  ASSERT timer.elapsed() <= 600_seconds

TEST "wizard_creates_encrypted_wallet":
  run_install_wizard(config)
  wallet = load_wallet(config.data_dir / "vault")
  ASSERT wallet.public_key.length == 32
  ASSERT wallet.private_key.is_encrypted()
  ASSERT wallet.node_id == sha256(wallet.public_key)[:16]

TEST "capsule_backend_selection_correct":
  ASSERT select_capsule_backend(linux_with_kvm) == MICROVM
  ASSERT select_capsule_backend(linux_no_kvm) == WASM
  ASSERT select_capsule_backend(macos) == WASM
  ASSERT select_capsule_backend(windows_with_docker) == DOCKER

TEST "resource_presets_within_bounds":
  FOR preset IN PRESETS.values():
    ASSERT preset.cpu_percent <= 75
    ASSERT preset.memory_mb <= 16384
    ASSERT preset.gpu_percent <= 75

TEST "capsule_isolation_enforced":
  capsule = start_capsule(backend=WASM)
  ASSERT capsule.can_read("/etc/passwd") == false
  ASSERT capsule.can_write("/tmp/host_file") == false
  ASSERT capsule.can_access_network(external=true) == false  # unless allowlisted

TEST "system_service_registers":
  register_system_service(name="bizra-test", binary="/tmp/test")
  ASSERT service_exists("bizra-test") == true
  unregister_system_service("bizra-test")

TEST "signed_image_verification_rejects_tampered":
  image = download_signed_image(backend=WASM)
  tamper(image.bytes, offset=100, value=0xFF)
  ASSERT verify_signature(image, BIZRA_PUBLIC_KEY) == false
```

## 1.7 Edge Cases

| Scenario | Behavior |
|----------|----------|
| No GPU detected | Skip GPU slider, set gpu_percent=0, continue |
| Disk space < 10GB | Block install, show actionable message |
| Docker not installed (no KVM) | Prompt to install Docker, link to instructions |
| Passphrase too weak | Reject, show strength feedback, retry |
| Download interrupted | Resume from last chunk (HTTP Range), retry 3x |
| Service already registered | Offer upgrade/replace with version check |
| Antivirus blocks capsule | Detect, guide user to whitelist |
