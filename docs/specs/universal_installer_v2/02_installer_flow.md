# 02 — Installer Flow & Genesis Pipeline

> Module: `bizra-installer/src/flow/`
> Language: Rust (Tauri) + TypeScript (UI screens)
> Constitutional Anchor: Law 1 (Mother Test) + Law 2 (Zero Prerequisites)

## 1. Flow Overview

```
PIPELINE install(binary_path) -> Result<InstallerReceipt>:
    # 6 steps, 3 mandatory taps, 1 optional tap
    # Total time target: < 180 seconds

    Step 1: DETECT   (0 taps, automatic)
    Step 2: GREET    (1 tap — language confirm)
    Step 3: ADAPT    (0 taps, automatic)
    Step 4: INSTALL  (1 tap — location + model confirm)
    Step 5: IDENTITY (1 tap — name + create node)
    Step 5.5: SHARE  (1 optional tap — URP dedication)
    Step 6: ALIVE    (0 taps — terminal opens)
```

## 2. Step 1: DETECT

```
FUNCTION step_detect() -> DeviceProfile:
    # Fully automatic — zero user interaction
    # See 01_device_detection.md for DeviceProfile struct

    profile = detect_device()

    # Emit detection event for telemetry (local only, never sent)
    emit_event("install.detect", {
        os: profile.os,
        arch: profile.arch,
        ram_gb: profile.ram_total_gb,
        gpu: profile.gpu.map(|g| g.model),
        locale: profile.system_locale
    })

    RETURN profile
```

## 3. Step 2: GREET

```
FUNCTION step_greet(profile: &DeviceProfile) -> Language:
    # 1 user interaction: confirm or change language
    detected = detect_language(profile.system_locale)

    # Show screen:
    #   [Flag] "Welcome to BIZRA" (in detected language)
    #   [Confirm] [Change Language ▼]
    screen = GreetScreen {
        detected_language: detected,
        greeting: i18n::greeting(detected),
        confirm_label: i18n::confirm(detected),
        change_label: i18n::change_language(detected),
        language_list: i18n::all_languages()  # 50+ options
    }

    choice = await show_screen(screen)

    MATCH choice:
        Confirm          => RETURN detected
        Change(language) => RETURN language
```

## 4. Step 3: ADAPT

```
FUNCTION step_adapt(profile: &DeviceProfile) -> InstallPlan:
    # Fully automatic — zero user interaction
    # Produces a plan the user confirms in Step 4

    model_tier = select_model_tier(profile)
    backend = select_backend(profile)

    footprint = MATCH model_tier:
        Micro | Compact => InstallFootprint::Minimal   # ~1.5GB
        Standard        => InstallFootprint::Standard  # ~4GB
        _               => InstallFootprint::Full      # ~10GB+

    install_path = default_install_path(profile.os)

    RETURN InstallPlan {
        model_tier,
        backend,
        footprint,
        install_path,
        estimated_size_gb: model_tier.disk_requirement() + 0.5,  # +500MB core
        estimated_time_s: estimate_install_time(profile),
        offline: NOT profile.network_available
    }
```

## 5. Step 4: INSTALL

```
FUNCTION step_install(plan: InstallPlan, lang: Language) -> InstallResult:
    # 1 user interaction: confirm location + model

    screen = InstallScreen {
        install_path: plan.install_path,
        recommended_model: plan.model_tier.display_name(lang),
        model_size: plan.model_tier.disk_requirement(),
        allow_change_model: true,
        allow_change_path: true,
        progress_messages: i18n::install_progress(lang)
    }

    user_choice = await show_screen(screen)
    final_path = user_choice.path OR plan.install_path
    final_tier = user_choice.model OR plan.model_tier

    # Execute installation with progress callbacks
    result = execute_install(final_path, final_tier, plan.backend,
        on_progress = |step, pct|:
            update_progress_bar(pct)
            update_message(plan.progress_messages[step])
    )

    RETURN result
```

```
FUNCTION execute_install(path, tier, backend, on_progress) -> InstallResult:
    create_directory_structure(path)           # on_progress("dirs", 5%)
    extract_core_runtime(path)                 # on_progress("core", 15%)
    extract_locale_packs(path)                 # on_progress("i18n", 20%)

    IF network_available AND NOT offline_bundle_has_model(tier):
        download_model(tier, path / "models")  # on_progress("model", 20-80%)
    ELSE:
        extract_bundled_model(tier, path / "models")  # on_progress("model", 80%)

    configure_backend(backend, path)           # on_progress("backend", 85%)
    install_cli_command(path)                  # on_progress("cli", 90%)
    create_desktop_shortcut(path)              # on_progress("shortcut", 95%)

    RETURN InstallResult { path, tier, backend, success: true }
```

## 6. Step 5: IDENTITY

```
FUNCTION step_identity(path, lang: Language) -> GenesisResult:
    # 1 user interaction: enter name + create node

    screen = IdentityScreen {
        prompt: i18n::whats_your_name(lang),
        name_input: "",
        optional_photo: true,
        create_label: i18n::create_my_node(lang)
    }

    user = await show_screen(screen)

    # Behind the scenes — Genesis Ceremony
    seed = generate_random_seed(32)            # 32 bytes entropy
    keypair = ed25519_generate(seed)

    genesis = GenesisActivation(seed, data_dir = path / "sovereign_state")
    result = genesis.activate()

    # Store user profile
    write_profile(path / "profile.json", {
        name: user.name,
        photo: user.photo,  # Optional
        language: lang,
        node_id: result.ceremony_result.node_id,
        created: now_iso8601()
    })

    # Generate first DEMA briefing
    briefing = generate_dema_briefing(lang, user.name)
    write_file(path / "briefings" / "first.md", briefing)

    RETURN result
```

## 7. Step 5.5: SHARE (Optional)

```
FUNCTION step_share(profile: &DeviceProfile, lang: Language) -> URPConfig:
    # 1 OPTIONAL interaction — skippable
    # See 05_urp_economy.md for full URP spec

    suggested = suggest_urp_tier(profile)

    screen = ShareScreen {
        heading: i18n::device_can_help(lang),
        cpu_slider: { current: suggested.cpu_cores, max: profile.cpu_cores },
        ram_slider: { current: suggested.ram_gb, max: profile.ram_total_gb },
        disk_slider: { current: suggested.disk_gb, max: profile.disk_available_gb },
        gpu_slider: if profile.gpu { current: suggested.vram_gb, max: gpu.vram_gb },
        schedule_picker: ["always", "when_idle", "scheduled", "manual", "never"],
        default_schedule: "when_idle",
        share_label: i18n::share_and_earn(lang),
        skip_label: i18n::skip_for_now(lang),
        earn_explanation: i18n::earn_explanation(lang)
    }

    choice = await show_screen(screen)

    MATCH choice:
        Skip  => RETURN URPConfig::disabled()
        Share => RETURN URPConfig {
            cpu_cores: choice.cpu,
            ram_gb: choice.ram,
            disk_gb: choice.disk,
            vram_gb: choice.vram,
            schedule: choice.schedule,
            enabled: true
        }
```

## 8. Step 6: ALIVE

```
FUNCTION step_alive(path, lang, user_name, genesis_result):
    # Zero interaction — terminal opens automatically

    # Start heartbeat
    heartbeat = start_heartbeat(path / "sovereign_state")

    # DEMA greeting (in user's language)
    greeting = i18n::dema_greeting(lang, user_name)
    show_terminal_with_greeting(greeting)

    # Load dashboard view
    load_terminal_view("dashboard", {
        node_id: genesis_result.node_id,
        agents: 12,  # 7 PAT + 5 SAT
        heartbeat: "alive",
        first_mission_prompt: true
    })
```

## 9. Health Check (Post-Install Gate)

```
FUNCTION run_health_check(path) -> HealthCheckResult:
    checks = [
        ("core_runtime",    verify_executable(path / "bin" / "bizra")),
        ("model_loads",     verify_model_loads(path / "models")),
        ("identity",        verify_identity(path / "sovereign_state")),
        ("evidence_ledger", verify_ledger_block0(path / "sovereign_state")),
        ("agents_minted",   verify_agent_count(path) == 12),
        ("heartbeat",       verify_heartbeat_alive(path)),
        ("first_briefing",  file_exists(path / "briefings" / "first.md")),
        ("terminal_ui",     verify_webview_renders()),
        ("locale_packs",    verify_locale_loaded()),
        ("disk_space",      disk_free(path) >= 500_MB),
    ]

    failures = [name FOR (name, ok) IN checks IF NOT ok]

    IF failures.len() > 0:
        rollback_install(path)
        show_error_report(failures, lang)
        RETURN HealthCheckResult::Failed(failures)

    RETURN HealthCheckResult::Passed
```

## 10. Installer Audit Receipt

```
FUNCTION generate_receipt(profile, plan, genesis, health) -> InstallerReceipt:
    receipt = InstallerReceipt {
        installer_version: env!("CARGO_PKG_VERSION"),
        install_date: now_iso8601(),
        device_profile: {
            os: profile.os.to_string(),
            arch: profile.arch.to_string(),
            ram_gb: profile.ram_total_gb,
            gpu: profile.gpu.map(|g| g.model),
            locale: profile.system_locale
        },
        selected_tier: plan.model_tier.display_name("en"),
        user_overrode_recommendation: plan.user_overrode,
        genesis_receipt: {
            node_id: genesis.ceremony_result.node_id,
            evidence_block_0: genesis.ceremony_result.genesis_hash,
            agents_minted: 12
        },
        health_check_passed: health.passed(),
        installation_time_seconds: elapsed.as_secs()
    }

    # Hash-chain the receipt
    receipt.receipt_hash = blake2b(canonical_json(receipt))

    write_json(path / "installation_receipt.json", receipt)
    RETURN receipt
```

## 11. CLI Installation

```
FUNCTION install_cli_command(install_path):
    # Make `bizra` available in any terminal

    IF os == Windows:
        # Add to user PATH via registry (no admin)
        add_to_user_path(install_path / "bin")
        # Also create .cmd wrapper
        write_file(install_path / "bin" / "bizra.cmd",
            "@echo off\n\"{install_path}\\bin\\bizra.exe\" %*")

    IF os == macOS:
        # Symlink to /usr/local/bin or ~/bin
        symlink(install_path / "bin" / "bizra", "~/.local/bin/bizra")
        # Add to PATH in .zshrc if not present
        append_path_if_missing("~/.zshrc", "~/.local/bin")

    IF os == Linux:
        # Symlink to ~/.local/bin (XDG standard, no root)
        symlink(install_path / "bin" / "bizra", "~/.local/bin/bizra")
        # Add to PATH in .bashrc/.zshrc if not present
        append_path_if_missing(shell_rc(), "~/.local/bin")
```

## TDD Anchors

```
TEST full_pipeline_completes_in_180s:
    # Mother Test: install must complete in < 3 minutes
    start = now()
    result = install(mock_binary, mock_profile_16gb)
    ASSERT now() - start < 180s
    ASSERT result.health_check_passed

TEST pipeline_works_offline:
    profile = mock_profile(network_available=false)
    result = install(offline_bundle, profile)
    ASSERT result.success
    ASSERT result.tier IN [Micro, Compact, Standard]  # Bundled models only

TEST health_check_rollback_on_failure:
    # If health check fails, nothing remains
    inject_fault("model_loads", false)
    result = install(mock_binary, mock_profile)
    ASSERT result == HealthCheckResult::Failed
    ASSERT NOT directory_exists(install_path)

TEST cli_command_available_after_install:
    install(mock_binary, mock_profile)
    output = run_command("bizra --version")
    ASSERT output.contains("BIZRA")

TEST receipt_contains_valid_hash:
    result = install(mock_binary, mock_profile)
    receipt = read_json(install_path / "installation_receipt.json")
    ASSERT receipt.receipt_hash == blake2b(canonical_json(receipt - "receipt_hash"))

TEST identity_creates_ed25519_keypair:
    result = step_identity(tmp_path, Language::English)
    ASSERT file_exists(tmp_path / "sovereign_state" / "node0_genesis.json")
    ASSERT result.ceremony_result.node_id.len() > 0
```
