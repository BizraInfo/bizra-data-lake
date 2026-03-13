# 04 — Platform Adapters & Tauri Shell

> Module: `bizra-installer/src/platform/`
> Language: Rust (Tauri backend) + TypeScript (WebView frontend)
> Constitutional Anchor: Law 2 (Zero Prerequisites) + Law 3 (Hardware Adaptation)

## 1. Core Principle

One codebase, one shell, every platform. Tauri provides the Rust backend + system WebView.
No bundled browser (unlike Electron). No vendor telemetry. 30MB RAM vs 300MB.

## 2. Tauri Shell Architecture

```
STRUCT TauriShell:
    backend:    RustRuntime        # Hardware detection, LLM, identity, IPC
    webview:    SystemWebView      # OS-native (WebView2, WKWebView, WebKitGTK)
    ipc:        TauriCommandBridge # Rust ↔ JS invoke/listen
    i18n:       LocaleEngine       # react-intl + RTL
    updater:    SelfUpdateEngine   # Delta patches, sovereign choice
```

```
FUNCTION init_tauri_app() -> TauriApp:
    builder = tauri::Builder::default()

    # Register IPC commands (Rust functions callable from JS)
    builder.invoke_handler(generate_handler![
        cmd_detect_device,
        cmd_detect_language,
        cmd_select_model_tier,
        cmd_execute_install,
        cmd_create_identity,
        cmd_configure_urp,
        cmd_start_heartbeat,
        cmd_check_health,
        cmd_get_briefing,
    ])

    # Register event listeners
    builder.setup(|app| {
        emit_event(app, "install.ready", {})
        Ok(())
    })

    RETURN builder.build()
```

## 3. IPC Command Bridge

```
# Each Tauri command is a Rust function exposed to the JS frontend

#[tauri::command]
FUNCTION cmd_detect_device() -> Result<DeviceProfile, String>:
    TRY:
        profile = detect_device()
        RETURN Ok(profile)
    CATCH e:
        RETURN Err(format("Detection failed: {}", e))

#[tauri::command]
FUNCTION cmd_execute_install(
    path: String,
    tier: String,
    on_progress: tauri::Window
) -> Result<InstallResult, String>:
    # Progress updates sent via window events
    on_progress.emit("install.progress", { step: "dirs", pct: 5 })
    create_directory_structure(path)

    on_progress.emit("install.progress", { step: "core", pct: 15 })
    extract_core_runtime(path)

    # ... remaining steps ...
    RETURN Ok(result)

#[tauri::command]
FUNCTION cmd_create_identity(
    name: String,
    photo: Option<Vec<u8>>,
    lang: String,
    path: String
) -> Result<GenesisResult, String>:
    seed = generate_random_seed(32)
    keypair = ed25519_generate(seed)
    genesis = GenesisActivation(seed, data_dir = path / "sovereign_state")
    result = genesis.activate()
    write_profile(path / "profile.json", { name, photo, lang, node_id: result.node_id })
    RETURN Ok(result)
```

## 4. Windows Adapter

```
STRUCT WindowsAdapter:
    install_path:   "%LOCALAPPDATA%\\BIZRA\\"
    webview:        WebView2              # Built-in Windows 10+
    admin_required: false                 # Per-user install always
    shortcut:       Desktop + Start Menu
    autostart:      Task Scheduler (opt-in)
    cli_path:       "%LOCALAPPDATA%\\BIZRA\\bin\\bizra.exe"

FUNCTION windows_install(plan: InstallPlan) -> Result<()>:
    # 1. Create directory (no admin needed for LOCALAPPDATA)
    path = expand_env("%LOCALAPPDATA%\\BIZRA\\")
    create_dir_all(path)

    # 2. Extract runtime
    extract_self_extracting_pe(path)

    # 3. WebView2 check
    IF NOT webview2_installed():
        # WebView2 is included in Windows 10 21H2+ and Windows 11
        # For older Windows 10, download Evergreen Bootstrapper (~1.5MB)
        download_webview2_bootstrapper()
        run_silent_install(bootstrapper)

    # 4. GPU backend
    IF nvidia_gpu_detected():
        extract_cuda_libs(path / "lib")     # Bundled CUDA runtime
    ELSE IF amd_gpu_detected():
        extract_vulkan_libs(path / "lib")

    # 5. PATH registration (user-level, no admin)
    add_to_user_path_via_registry(path / "bin")
    write_file(path / "bin" / "bizra.cmd",
        "@echo off\n\"{path}\\bin\\bizra.exe\" %*")

    # 6. Shortcuts
    create_desktop_shortcut(path / "bin" / "bizra.exe", "BIZRA")
    create_start_menu_shortcut(path / "bin" / "bizra.exe", "BIZRA")

    # 7. Optional autostart
    # NOT enabled by default — user opts in via Settings
    # Uses Task Scheduler (no admin), NOT registry Run key

    RETURN Ok(())
```

```
FUNCTION windows_uninstall(path):
    remove_from_user_path(path / "bin")
    remove_desktop_shortcut("BIZRA")
    remove_start_menu_shortcut("BIZRA")
    remove_task_scheduler_entry("BIZRA-Autostart")
    # Prompt: "Delete your identity and evidence? [Keep / Delete]"
    IF user_chooses_delete:
        remove_dir_all(path)
    ELSE:
        # Keep sovereign_state/, delete everything else
        remove_runtime_only(path)
```

## 5. macOS Adapter

```
STRUCT MacOSAdapter:
    install_path:   "~/Library/Application Support/BIZRA/"
    app_bundle:     "/Applications/BIZRA.app"
    webview:        WKWebView             # Built-in on all macOS
    admin_required: false
    notarization:   Apple Developer ID required
    cli_path:       "~/.local/bin/bizra"

FUNCTION macos_install(plan: InstallPlan) -> Result<()>:
    # 1. DMG mount (standard macOS pattern)
    # User drags BIZRA.app to /Applications/ (or ~/Applications/)
    app_path = "/Applications/BIZRA.app"

    # 2. Data directory
    data_path = expand("~/Library/Application Support/BIZRA/")
    create_dir_all(data_path)

    # 3. Metal backend (Universal Binary handles x64 + ARM)
    IF apple_silicon():
        configure_metal_backend(data_path)
        # Apple Neural Engine available on M1+
        enable_ane_if_available()
    ELSE:
        configure_cpu_backend(data_path)  # Intel Mac

    # 4. CLI symlink
    cli_dir = expand("~/.local/bin")
    create_dir_all(cli_dir)
    symlink(app_path / "Contents/MacOS/bizra-cli", cli_dir / "bizra")
    append_path_if_missing("~/.zshrc", cli_dir)

    # 5. Gatekeeper / Notarization
    # Binary must be signed with Apple Developer ID
    # AND notarized via `xcrun notarytool`
    # Otherwise macOS shows "unidentified developer" dialog

    RETURN Ok(())

FUNCTION macos_build_universal() -> Binary:
    # Build for both architectures
    x64 = cargo_build("x86_64-apple-darwin")
    arm = cargo_build("aarch64-apple-darwin")
    universal = lipo_create(x64, arm)
    RETURN universal
```

## 6. Linux Adapter

```
STRUCT LinuxAdapter:
    install_path:   "~/.local/share/bizra/"
    format:         AppImage             # Primary: no root, any distro
    alt_formats:    [Flatpak, Snap, Deb, RPM]
    webview:        WebKitGTK            # libwebkit2gtk-4.0
    admin_required: false
    cli_path:       "~/.local/bin/bizra"

FUNCTION linux_install_appimage(plan: InstallPlan) -> Result<()>:
    # AppImage: single file, chmod +x, run. No root. Any distro.

    # 1. Check WebKitGTK dependency
    IF NOT webkitgtk_available():
        show_friendly_error(
            "BIZRA needs WebKitGTK to display its interface.\n"
            "Install it with:\n"
            "  Ubuntu/Debian: sudo apt install libwebkit2gtk-4.0-dev\n"
            "  Fedora: sudo dnf install webkit2gtk4.0-devel\n"
            "  Arch: sudo pacman -S webkit2gtk"
        )
        RETURN Err("WebKitGTK not found")

    # 2. Data directory (XDG standard)
    data_path = env("XDG_DATA_HOME") OR expand("~/.local/share/bizra/")
    create_dir_all(data_path)

    # 3. GPU detection
    IF nvidia_detected():
        # Check CUDA availability
        IF cuda_version() >= 12:
            configure_cuda_backend(data_path)
        ELSE:
            configure_vulkan_backend(data_path)
    ELSE IF amd_detected():
        IF rocm_available():
            configure_rocm_backend(data_path)
        ELSE:
            configure_vulkan_backend(data_path)
    ELSE:
        configure_cpu_backend(data_path)  # AVX2 or NEON auto-detected

    # 4. CLI symlink
    cli_dir = expand("~/.local/bin")
    create_dir_all(cli_dir)
    symlink(appimage_path, cli_dir / "bizra")

    # 5. Add to PATH
    shell_rc = detect_shell_rc()  # .bashrc, .zshrc, or .profile
    append_path_if_missing(shell_rc, cli_dir)

    # 6. Desktop entry (freedesktop.org standard)
    write_desktop_entry(
        expand("~/.local/share/applications/bizra.desktop"),
        {
            Name: "BIZRA",
            Exec: cli_dir / "bizra",
            Icon: data_path / "icons/bizra.png",
            Type: "Application",
            Categories: "Utility;Development;"
        }
    )

    RETURN Ok(())
```

## 7. Android Adapter (Phase 2)

```
STRUCT AndroidAdapter:
    install_path:   app_internal_storage
    distribution:   [APK_sideload, PlayStore]
    webview:        Android WebView       # System component
    llm_runtime:    LlamaCppVulkan OR LlamaCppCPU
    background:     ForegroundService     # Keeps heartbeat alive
    min_api:        26                    # Android 8.0+

FUNCTION android_adapt(profile: DeviceProfile) -> AndroidConfig:
    # Android-specific adaptations
    config = AndroidConfig::default()

    # 1. Storage strategy
    IF profile.disk_available_gb >= 4.0:
        config.model_location = InternalStorage
    ELSE IF sd_card_available():
        config.model_location = SDCard
        config.warn_user("Model on SD card may be slower")
    ELSE:
        config.model_tier = ModelTier::Micro  # Smallest possible

    # 2. Battery optimization
    config.heartbeat_interval = MATCH profile.ram_total_gb:
        r IF r < 3.0  => 600s   # 10 min (battery saving)
        r IF r < 6.0  => 300s   # 5 min
        _             => 60s    # Standard

    # 3. Background service
    # Android kills background apps aggressively
    # ForegroundService with persistent notification keeps heartbeat alive
    config.use_foreground_service = true
    config.notification_channel = "bizra_heartbeat"
    config.notification_text = i18n::heartbeat_active(lang)

    # 4. Permissions (minimal)
    # INTERNET — optional (for federation)
    # FOREGROUND_SERVICE — for heartbeat
    # No camera, contacts, location, etc.

    RETURN config
```

## 8. iOS Adapter (Phase 3)

```
STRUCT IOSAdapter:
    install_path:   app_sandbox
    distribution:   AppStore              # Apple requirement
    webview:        WKWebView             # iOS built-in
    llm_runtime:    LlamaCppMetal         # Apple Neural Engine
    background:     BackgroundAppRefresh  # iOS limits background
    min_ios:        16.0

FUNCTION ios_adapt(profile: DeviceProfile) -> IOSConfig:
    config = IOSConfig::default()

    # 1. Apple Neural Engine
    IF apple_neural_engine_available():
        config.inference_backend = ANE  # Fastest on iOS
    ELSE:
        config.inference_backend = Metal

    # 2. Background limitations
    # iOS allows ~30 seconds of background execution
    # BackgroundAppRefresh gives periodic wake-ups
    config.heartbeat_strategy = BackgroundAppRefresh
    config.heartbeat_interval = 900s  # 15 min (iOS constraint)

    # 3. Model size constraint
    # App Store has 4GB binary limit
    # Model must be downloaded post-install
    config.bundle_model = false
    config.download_model_on_first_launch = true

    # 4. Privacy (App Store requirements)
    config.privacy_manifest = generate_privacy_manifest()
    # Declare: no tracking, no third-party analytics
    # Declare: local-only data processing

    RETURN config
```

## 9. Build Matrix

```
STRUCT BuildTarget:
    platform:  String
    arch:      String
    command:   String
    artifact:  String
    ci_job:    String

BUILD_MATRIX = [
    BuildTarget {
        platform: "windows", arch: "x64",
        command: "cargo tauri build --target x86_64-pc-windows-msvc",
        artifact: "bizra-setup.exe",
        ci_job: "build-installer-windows"
    },
    BuildTarget {
        platform: "windows", arch: "arm64",
        command: "cargo tauri build --target aarch64-pc-windows-msvc",
        artifact: "bizra-setup-arm.exe",
        ci_job: "build-installer-windows-arm"
    },
    BuildTarget {
        platform: "macos", arch: "universal",
        command: "cargo tauri build --target universal-apple-darwin",
        artifact: "BIZRA.dmg",
        ci_job: "build-installer-macos"
    },
    BuildTarget {
        platform: "linux", arch: "x64",
        command: "cargo tauri build --target x86_64-unknown-linux-gnu",
        artifact: "bizra.AppImage",
        ci_job: "build-installer-linux"
    },
    BuildTarget {
        platform: "linux", arch: "arm64",
        command: "cargo tauri build --target aarch64-unknown-linux-gnu",
        artifact: "bizra-arm64.AppImage",
        ci_job: "build-installer-linux-arm"
    },
    BuildTarget {
        platform: "android", arch: "multi",
        command: "cargo tauri android build",
        artifact: "BIZRA.apk",
        ci_job: "build-installer-android"
    },
]
```

## 10. Offline Bundle Structure

```
STRUCT OfflineBundle:
    installer_binary:  Binary       # 50 MB (platform-specific)
    bundled_model:     GGUF         # 2.3 GB (Phi-3 Mini Q4)
    locale_packs:      [ar, en]     # 400 KB (minimum 2 languages)
    readme:            MultiLang    # Quick start in 10 languages

FUNCTION create_offline_bundle(platform, model_tier) -> Bundle:
    bundle = OfflineBundle::new()

    bundle.installer_binary = get_build_artifact(platform)
    bundle.bundled_model = download_model(model_tier)

    # Always include Arabic + English minimum
    bundle.locale_packs = [
        compress_locale("ar"),
        compress_locale("en"),
    ]

    # Total: ~2.5 GB for Standard tier
    # Fits on any USB stick
    RETURN bundle
```

## TDD Anchors

```
TEST tauri_commands_registered:
    app = init_tauri_app()
    # All 9 IPC commands must be registered
    ASSERT app.has_command("cmd_detect_device")
    ASSERT app.has_command("cmd_execute_install")
    ASSERT app.has_command("cmd_create_identity")
    ASSERT app.has_command("cmd_configure_urp")

TEST windows_install_no_admin:
    # Installation must succeed without elevation
    result = windows_install(mock_plan)
    ASSERT result.is_ok()
    ASSERT NOT was_admin_required()

TEST macos_universal_binary_both_archs:
    binary = macos_build_universal()
    ASSERT binary.contains_arch("x86_64")
    ASSERT binary.contains_arch("arm64")

TEST linux_appimage_no_root:
    result = linux_install_appimage(mock_plan)
    ASSERT result.is_ok()
    ASSERT NOT was_root_required()
    ASSERT file_exists("~/.local/bin/bizra")

TEST android_battery_adaptive_heartbeat:
    low_ram = DeviceProfile { ram_total_gb: 2.0, ... }
    config = android_adapt(low_ram)
    ASSERT config.heartbeat_interval >= 600  # 10 min minimum

TEST ios_model_not_bundled:
    config = ios_adapt(mock_profile)
    ASSERT config.bundle_model == false
    ASSERT config.download_model_on_first_launch == true

TEST offline_bundle_includes_ar_en:
    bundle = create_offline_bundle("linux_x64", ModelTier::Standard)
    ASSERT "ar" IN bundle.locale_packs
    ASSERT "en" IN bundle.locale_packs

TEST webkitgtk_missing_shows_friendly_error:
    # On Linux without WebKitGTK, show helpful install instructions
    mock_webkitgtk_unavailable()
    result = linux_install_appimage(mock_plan)
    ASSERT result.is_err()
    ASSERT "apt install" IN result.error_message OR "dnf install" IN result.error_message
```
