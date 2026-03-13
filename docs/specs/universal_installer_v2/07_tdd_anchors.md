# 07 — Test Plan: TDD Anchors & Acceptance Criteria

> Module: `bizra-installer/tests/`
> Language: Rust (cargo test) + TypeScript (Vitest) + Python (pytest for PoI/PoT)
> Constitutional Anchor: All 8 Design Laws

## 1. Core Principle

Every module has inline TDD anchors (modules 01-06). This document defines
the cross-cutting test plan: integration, E2E, acceptance, and the 8 Billion Test.

## 2. Test Pyramid

```
STRUCT TestPyramid:
    unit:          "~200 tests — module-level (01-06 TDD anchors)"
    integration:   "~50 tests — cross-module wiring"
    e2e:           "~20 tests — full install pipeline"
    acceptance:    "15 tests — 8 Billion Test criteria"
    visual:        "~10 tests — RTL, layout regression"
```

## 3. Unit Test Summary (from modules 01-06)

Each module (01-06) contains inline TDD anchors. Total: ~175 unit tests.

| Module | Tests | Key Coverage |
|--------|-------|-------------|
| 01 Device Detection | ~30 | Profile never panics, GPU→None fallback, tier selection, OOM chain |
| 02 Installer Flow | ~25 | Pipeline < 180s, offline, health rollback, receipt hash, Ed25519 |
| 03 i18n Language | ~30 | RTL mirror, Tier 1 completeness, glossary keep/translate, PoT scoring |
| 04 Platform Adapters | ~25 | No-admin install, universal binary, AppImage, battery-adaptive |
| 05 URP Economy | ~35 | Ihsan gate, Zakat deduction, child mode, PoI verification, Genesis-1 |
| 06 Update Lifecycle | ~30 | Delta patch, signature, auto-rollback, disk alerts, profile isolation |

## 4. Integration Tests (~50 tests)

```
SUITE integration_detect_to_adapt:
    # Device detection feeds into model selection

    TEST detect_then_select_model:
        profile = detect_device()
        tier = select_model_tier(&profile)
        ASSERT tier.min_ram_gb <= profile.ram_available_gb

    TEST detect_then_select_backend:
        profile = detect_device()
        backend = select_backend(&profile)
        IF profile.gpu IS Some:
            ASSERT backend != LlamaCppCPU
        ELSE:
            ASSERT backend == LlamaCppCPU

    TEST detect_then_suggest_urp:
        profile = detect_device()
        suggestion = suggest_urp_tier(&profile)
        ASSERT suggestion.ram_gb <= profile.ram_total_gb * 0.5

SUITE integration_install_to_genesis:
    # Install pipeline feeds into identity creation

    TEST install_creates_genesis_artifacts:
        result = run_full_pipeline(mock_profile)
        ASSERT file_exists(result.path / "sovereign_state" / "node0_genesis.json")
        ASSERT file_exists(result.path / "sovereign_state" / "genesis_hash.txt")
        ASSERT file_exists(result.path / "briefings" / "first.md")

    TEST install_receipt_chains_to_genesis:
        result = run_full_pipeline(mock_profile)
        receipt = load_json(result.path / "installation_receipt.json")
        genesis = load_json(result.path / "sovereign_state" / "node0_genesis.json")
        ASSERT receipt.genesis_receipt.node_id == genesis.node_id

SUITE integration_i18n_to_install:
    # Language selection affects all subsequent screens

    TEST language_persists_through_pipeline:
        lang = Language::Arabic
        greet_result = step_greet_with(lang)
        install_result = step_install_with(greet_result.lang)
        identity_result = step_identity_with(greet_result.lang)
        ASSERT identity_result.profile.language == "ar"

    TEST rtl_language_mirrors_all_screens:
        lang = Language::Arabic
        FOR screen IN [GreetScreen, InstallScreen, IdentityScreen, ShareScreen]:
            rendered = render_screen(screen, lang)
            ASSERT rendered.direction == "rtl"
            ASSERT rendered.navigation_side == "right"

SUITE integration_urp_to_rewards:
    # URP contribution flows through PoI to SEED minting

    TEST contribution_verified_then_rewarded:
        config = URPConfig { cpu_cores: 2, ... }
        contribution = simulate_contribution(config, duration=1.hour)
        poi = verify_cpu_contribution(contribution)
        ASSERT poi == PoIResult::Verified
        reward = calculate_reward(contribution, 1.0, poi, 0.95)
        ASSERT reward > 0.0

    TEST contribution_rejected_no_reward:
        contribution = simulate_fake_contribution()
        poi = verify_cpu_contribution(contribution)
        ASSERT poi != PoIResult::Verified
        reward = calculate_reward(contribution, 1.0, poi, 0.95)
        ASSERT reward == 0.0

SUITE integration_update_preserves_state:
    # Updates preserve identity and evidence

    TEST update_preserves_identity:
        identity_before = read_identity()
        apply_update(mock_patch)
        identity_after = read_identity()
        ASSERT identity_before == identity_after

    TEST update_preserves_evidence_ledger:
        ledger_hash_before = evidence_ledger.last_hash()
        apply_update(mock_patch)
        ledger_hash_after = evidence_ledger.last_hash()
        ASSERT ledger_hash_before == ledger_hash_after

    TEST rollback_preserves_evidence:
        append_evidence(mock_receipt)
        apply_update(mock_patch)
        rollback_to_previous()
        ASSERT evidence_ledger.contains(mock_receipt)
```

## 5. End-to-End Tests (~20 tests)

```
SUITE e2e_full_install:
    # Complete install pipeline, automated

    TEST e2e_windows_x64_standard:
        profile = mock_windows_x64(ram=16, gpu="RTX 3060")
        result = full_install_pipeline(profile, lang=Language::English)
        ASSERT result.health_check_passed
        ASSERT result.installation_time_seconds < 180
        ASSERT result.tier == ModelTier::Full

    TEST e2e_linux_arm64_minimal:
        profile = mock_linux_arm64(ram=2, gpu=None)
        result = full_install_pipeline(profile, lang=Language::Arabic)
        ASSERT result.health_check_passed
        ASSERT result.tier == ModelTier::Compact

    TEST e2e_macos_m2_standard:
        profile = mock_macos_arm64(ram=8, gpu="Apple M2")
        result = full_install_pipeline(profile, lang=Language::Spanish)
        ASSERT result.health_check_passed
        ASSERT result.tier == ModelTier::Enhanced

    TEST e2e_offline_usb_install:
        profile = mock_linux_x64(ram=4, network=false)
        bundle = load_offline_bundle("usb/bizra-offline.tar")
        result = full_install_pipeline(profile, bundle=bundle)
        ASSERT result.health_check_passed
        ASSERT result.tier IN [ModelTier::Micro, ModelTier::Compact, ModelTier::Standard]

    TEST e2e_install_then_update:
        profile = mock_windows_x64(ram=16)
        install_result = full_install_pipeline(profile)
        ASSERT install_result.health_check_passed

        # Simulate update
        mock_update_available("2.0.0")
        update_result = apply_update_flow()
        ASSERT current_version() == "2.0.0"
        ASSERT identity_unchanged()

    TEST e2e_install_then_urp_enable:
        profile = mock_linux_x64(ram=16, gpu="RTX 3060")
        install_result = full_install_pipeline(profile)
        urp_config = configure_urp(cpu_cores=2, ram_gb=4, schedule="when_idle")
        ASSERT urp_config.enabled
        # Simulate idle → verify resources shared
        simulate_idle(minutes=6)
        ASSERT urp_active()

    TEST e2e_multi_profile_lifecycle:
        install_result = full_install_pipeline(mock_profile)
        dad = create_profile("Dad", Language::Arabic, "pass1")
        mom = create_profile("Mom", Language::Urdu, "pass2")

        # Dad uses the system
        switch_profile(dad.id, "pass1")
        run_mission("test mission for Dad")
        ASSERT evidence_ledger_has_entries(dad)

        # Switch to Mom
        switch_profile(mom.id, "pass2")
        ASSERT evidence_ledger_is_empty(mom)  # Mom's ledger is separate
        ASSERT current_language() == Language::Urdu
```

## 6. Visual Regression Tests (~10 tests)

```
SUITE visual_rtl:
    # Screenshot comparison for RTL languages

    TEST visual_arabic_dashboard:
        screenshot = render_full_terminal(Language::Arabic, view="dashboard")
        compare_to_baseline(screenshot, "baselines/ar_dashboard.png", tolerance=0.01)

    TEST visual_arabic_mission:
        compare_to_baseline(render_full_terminal(Language::Arabic, view="mission"), "baselines/ar_mission.png")

    TEST visual_urdu_installer:
        compare_to_baseline(render_installer(Language::Urdu, step="greet"), "baselines/ur_greet.png")

    TEST visual_english_dashboard:
        compare_to_baseline(render_full_terminal(Language::English, view="dashboard"), "baselines/en_dashboard.png")

    TEST visual_mixed_direction:
        # Arabic text with English technical terms
        screenshot = render_terminal_with_mixed_text(Language::Arabic)
        ASSERT text_direction_correct(screenshot)
        ASSERT numbers_are_ltr(screenshot)
```

## 7. Performance Tests

```
SUITE performance_install:
    TEST install_completes_under_3_minutes:
        # Mother Test: MUST complete in < 180 seconds
        FOR profile IN [low_end, mid_range, high_end]:
            start = now()
            result = full_install_pipeline(profile)
            elapsed = now() - start
            ASSERT elapsed < 180s, f"Install took {elapsed}s on {profile}"

    TEST model_loads_under_30_seconds:
        FOR tier IN [Micro, Compact, Standard, Enhanced]:
            start = now()
            model = load_model(tier)
            elapsed = now() - start
            ASSERT elapsed < 30s

    TEST heartbeat_starts_under_5_seconds:
        start = now()
        heartbeat = start_heartbeat(mock_state)
        elapsed = now() - start
        ASSERT elapsed < 5s

    TEST delta_patch_applies_under_10_seconds:
        patch = create_delta_patch(v1_binary, v2_binary)
        start = now()
        apply_delta_patch(patch, v1_binary)
        elapsed = now() - start
        ASSERT elapsed < 10s

    TEST memory_usage_under_threshold:
        # Tauri shell must use < 50MB RAM (vs Electron's 300MB)
        launch_app()
        mem = measure_process_memory()
        ASSERT mem < 50_MB
```

## 8. Security Tests

```
SUITE security_install:
    TEST no_admin_elevation_requested:
        # Installation must never request admin/root
        FOR platform IN [Windows, macOS, Linux]:
            result = install_on(platform)
            ASSERT NOT admin_was_requested()

    TEST ed25519_keys_not_in_plaintext:
        result = full_install_pipeline(mock_profile)
        # Private key must be encrypted, not stored as plaintext
        FOR file IN walk_files(result.path):
            content = read_file(file)
            ASSERT NOT contains_ed25519_private_key_pattern(content)

    TEST update_patch_signature_required:
        # Unsigned patches must be rejected
        patch = create_unsigned_patch(v1, v2)
        result = apply_delta_patch(patch, v1)
        ASSERT result.is_err()
        ASSERT "signature" IN result.error_message

    TEST profile_passphrase_not_stored_plaintext:
        create_profile("Test", Language::English, "my_secret_pass")
        FOR file IN walk_files(profiles_dir):
            content = read_file(file)
            ASSERT "my_secret_pass" NOT IN content

    TEST evidence_ledger_tamper_detected:
        ledger = create_test_ledger(entries=10)
        tamper_entry(ledger, index=5)
        (valid, errors) = ledger.verify_chain()
        ASSERT NOT valid
        ASSERT len(errors) > 0
```

## 9. Accessibility Tests

```
SUITE accessibility:
    TEST all_ui_elements_have_aria_labels:
        FOR screen IN all_installer_screens():
            elements = render_screen(screen, Language::English)
            FOR element IN elements.interactive():
                ASSERT element.has_aria_label()

    TEST keyboard_navigation_complete:
        FOR screen IN all_installer_screens():
            tab_order = get_tab_order(screen)
            ASSERT tab_order.covers_all_interactive_elements()
            ASSERT tab_order.is_logical()

    TEST high_contrast_mode_respected:
        enable_system_high_contrast()
        screenshot = render_installer(Language::English)
        ASSERT contrast_ratio_meets_wcag_aa(screenshot)

    TEST font_scaling_200pct:
        set_system_font_scale(2.0)
        screenshot = render_installer(Language::English)
        ASSERT no_text_overflow(screenshot)
        ASSERT no_text_clipping(screenshot)

    TEST minimum_click_target_48px:
        FOR screen IN all_installer_screens():
            elements = render_screen(screen, Language::English)
            FOR button IN elements.buttons():
                ASSERT button.width >= 48
                ASSERT button.height >= 48
```

## 10. The 8 Billion Test (Acceptance Criteria)

```
SUITE eight_billion_test:
    # THE acceptance gate. All 15 must pass before public release.

    TEST grandmother_cairo_windows_arabic_3min:
        profile = mock_windows_x64(ram=8, locale="ar-EG")
        start = now()
        result = full_install_pipeline(profile, lang=Language::Arabic,
            interactions=[confirm_language, confirm_install, create_identity])
        ASSERT now() - start < 180s
        ASSERT result.health_check_passed
        ASSERT result.user_interactions <= 3

    TEST student_lagos_android_english_3min:
        profile = mock_android(ram=4, locale="en-NG")
        start = now()
        result = full_install_pipeline(profile, lang=Language::English)
        ASSERT now() - start < 180s
        ASSERT result.health_check_passed

    TEST developer_saopaulo_linux_portuguese_3min:
        profile = mock_linux_x64(ram=16, locale="pt-BR")
        start = now()
        result = full_install_pipeline(profile, lang=Language::Portuguese)
        ASSERT now() - start < 180s
        ASSERT result.health_check_passed

    TEST farmer_java_offline_usb_indonesian:
        profile = mock_android(ram=2, network=false, locale="id-ID")
        bundle = load_offline_bundle()
        result = full_install_pipeline(profile, bundle=bundle, lang=Language::Indonesian)
        ASSERT result.health_check_passed

    TEST shopkeeper_karachi_urdu_rtl:
        profile = mock_windows_x64(ram=4, locale="ur-PK")
        result = full_install_pipeline(profile, lang=Language::Urdu)
        ASSERT result.health_check_passed
        screenshot = render_terminal(Language::Urdu)
        ASSERT screenshot.direction == "rtl"
        ASSERT screenshot.navigation_side == "right"

    TEST blind_user_screen_reader:
        profile = mock_windows_x64(ram=8, screen_reader=true)
        result = full_install_pipeline(profile)
        ASSERT result.health_check_passed
        ASSERT all_screens_have_aria_labels()
        ASSERT keyboard_navigation_complete()

    TEST micro_node_1gb_ram:
        profile = mock_android(ram=1.0)
        result = full_install_pipeline(profile)
        ASSERT result.health_check_passed
        ASSERT result.tier == ModelTier::Micro
        ASSERT heartbeat_alive()

    TEST premium_node_128gb_ram:
        profile = mock_linux_x64(ram=128.0, gpu="A100 80GB")
        result = full_install_pipeline(profile)
        ASSERT result.health_check_passed
        ASSERT result.tier == ModelTier::Premium
        ASSERT result.user_interactions <= 3  # Same 3-tap promise

    TEST works_without_internet:
        profile = mock_windows_x64(ram=8, network=false)
        result = full_install_pipeline(profile)
        ASSERT result.health_check_passed
        ASSERT result.offline == true

    TEST never_asks_admin:
        FOR platform IN [Windows, macOS, Linux]:
            profile = mock_profile(os=platform)
            result = full_install_pipeline(profile)
            ASSERT NOT admin_elevation_requested()

    TEST bizra_command_available:
        result = full_install_pipeline(mock_profile)
        output = run_shell_command("bizra --version")
        ASSERT "BIZRA" IN output

    TEST urp_earns_seed_within_24h:
        result = full_install_pipeline(mock_profile)
        configure_urp(cpu_cores=2, schedule="always")
        simulate_time(hours=24)
        balance = get_seed_balance()
        ASSERT balance > 0.0

    TEST no_share_loses_nothing:
        result = full_install_pipeline(mock_profile)
        configure_urp(enabled=false)
        ASSERT can_run_missions()
        ASSERT heartbeat_alive()
        ASSERT can_earn_seed_locally()

    TEST user_can_dedicate_100pct:
        profile = mock_linux_x64(ram=64, cpu_cores=16)
        result = full_install_pipeline(profile)
        config = configure_urp(cpu_cores=16, ram_gb=64, user_confirmed=true)
        ASSERT config.enabled
        ASSERT config.cpu_cores == 16  # User's sovereign choice

    TEST genesis_100_gate_passed:
        # Prerequisite: Genesis-100 (68 checks, 5 SAT agents) must pass L1-L3
        gate_result = run_genesis_100_gate()
        ASSERT gate_result.l1_structural.passed
        ASSERT gate_result.l2_constitutional.passed
        ASSERT gate_result.l3_economic.passed
```

## 11. CI Integration

```
PIPELINE installer_ci():
    # 1. Unit tests (fast, every PR)
    STAGE unit:
        cargo test --workspace -- --test-threads=4
        npm run test -- --run
        pytest tests/installer/ -x --timeout=60

    # 2. Integration tests (medium, every PR)
    STAGE integration:
        cargo test --workspace --test integration_*
        pytest tests/installer/integration/ -x

    # 3. E2E tests (slow, merge to main only)
    STAGE e2e:
        FOR target IN [windows_x64, linux_x64, macos_universal]:
            run_e2e_suite(target)

    # 4. Visual regression (slow, merge to main only)
    STAGE visual:
        FOR lang IN [ar, en, ur, he]:
            generate_screenshots(lang)
            compare_to_baselines(lang, tolerance=0.01)

    # 5. Acceptance gate (release only)
    STAGE acceptance:
        run_eight_billion_test()
        ASSERT all_15_criteria_pass()

    # 6. Build artifacts (release only)
    STAGE build:
        FOR target IN BUILD_MATRIX:
            build_installer(target)
            sign_binary(target)
            upload_artifact(target)
```

## 12. Test Infrastructure

```
STRUCT TestFixtures:
    mock_profiles = {
        "low_end":  DeviceProfile { ram: 1.0, cpu: 2, gpu: None },
        "budget":   DeviceProfile { ram: 4.0, cpu: 4, gpu: None },
        "mid":      DeviceProfile { ram: 8.0, cpu: 4, gpu: "GTX 1650" },
        "standard": DeviceProfile { ram: 16.0, cpu: 8, gpu: "RTX 3060" },
        "high":     DeviceProfile { ram: 64.0, cpu: 16, gpu: "RTX 4090" },
        "server":   DeviceProfile { ram: 128.0, cpu: 32, gpu: "A100" },
    }
    tmp_install_dir = temp_dir() / "bizra_test_{uuid}"
    mock_model_dir  = test_fixtures / "mock_models"
    mock_locales    = test_fixtures / "locales"   # ar + en minimum
```
