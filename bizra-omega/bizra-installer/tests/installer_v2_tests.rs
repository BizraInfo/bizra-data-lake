//! Integration tests for BIZRA Universal Installer v2.0 modules
//!
//! Tests the full installation flow: detect → greet → adapt → alive,
//! plus URP, profiles, i18n, receipts, health checks, and self-update.

use bizra_installer::device_profile::*;
use bizra_installer::health_check::*;
use bizra_installer::i18n::*;
use bizra_installer::install_flow::*;
use bizra_installer::install_receipt::*;
use bizra_installer::profiles::*;
use bizra_installer::self_update::*;
use bizra_installer::urp::*;

// ─────────────────────────────────────────────────────────────
// End-to-End Flow Tests
// ─────────────────────────────────────────────────────────────

#[test]
fn full_install_flow_state_machine() {
    let opts = InstallOptions::default();
    let mut state = InstallState::new(opts);

    // Step 1: Detect
    assert_eq!(state.current_step, InstallStep::Detect);
    let detect = execute_detect(&mut state);
    assert_eq!(state.current_step, InstallStep::Greet);
    assert!(state.profile.is_some());
    assert!(state.locale.is_some());

    // Step 2: Greet
    let i18n = I18nManager::new(&detect.locale.code, "en");
    let greet = execute_greet(&state, &i18n);
    assert!(!greet.locale_code.is_empty());

    // Step 3: Adapt (doesn't advance — user interaction needed)
    let adapt = execute_adapt(&state);
    assert!(!adapt.model_name.is_empty());
    assert!(adapt.model_size_gb > 0.0);
}

#[test]
fn arabic_locale_flow() {
    let opts = InstallOptions {
        locale_override: Some("ar".into()),
        ..Default::default()
    };
    let mut state = InstallState::new(opts);
    let detect = execute_detect(&mut state);

    assert_eq!(detect.locale.code, "ar");
    assert_eq!(detect.locale.direction, TextDir::RTL);

    let mut i18n = I18nManager::new("ar", "en");
    let ar_bundle = StringBundle {
        locale: "ar".into(),
        component: "installer".into(),
        strings: [("welcome".into(), "مرحبا بك في بذرة".into())]
            .into_iter()
            .collect(),
    };
    i18n.register(ar_bundle);

    let greet = execute_greet(&state, &i18n);
    assert!(greet.is_rtl);
    assert_eq!(greet.welcome_text, "مرحبا بك في بذرة");
}

// ─────────────────────────────────────────────────────────────
// DeviceProfile + ModelTier Integration
// ─────────────────────────────────────────────────────────────

#[test]
fn micro_node_flow() {
    let opts = InstallOptions {
        model_tier_override: Some(ModelTier::Micro),
        ..Default::default()
    };
    let mut state = InstallState::new(opts);
    let detect = execute_detect(&mut state);

    assert_eq!(detect.recommended_tier, ModelTier::Micro);
    assert_eq!(detect.recommended_tier.model_name(), "TinyLlama 1.1B Q2_K");
}

#[test]
fn model_tier_cascading_fallback() {
    // Test the full fallback chain
    let mut tier = ModelTier::Elite;
    let expected_chain = [
        ModelTier::Premium,
        ModelTier::Full,
        ModelTier::Enhanced,
        ModelTier::Standard,
        ModelTier::Compact,
        ModelTier::Micro,
    ];
    for expected in &expected_chain {
        tier = tier.fallback().unwrap();
        assert_eq!(&tier, expected);
    }
    assert!(tier.fallback().is_none());
}

// ─────────────────────────────────────────────────────────────
// URP Integration
// ─────────────────────────────────────────────────────────────

#[test]
fn urp_seed_accumulation_over_time() {
    let pledge = ResourcePledge {
        cpu_threads: 8,
        ram_gb: 16.0,
        vram_gb: 8.0,
        storage_gb: 50.0,
        bandwidth_mbps: 50.0,
        hours_per_day: 24,
        consent: true,
        consented_at: Some("2026-01-01T00:00:00Z".into()),
    };
    let mut urp = URPState::new(pledge);

    // Simulate 24 hours of contribution (in 1-hour intervals)
    let mut total_seed = 0.0;
    let mut total_zakat = 0.0;
    for _ in 0..24 {
        let (net, zakat) = urp.credit_contribution(3600);
        total_seed += net;
        total_zakat += zakat;
    }

    assert!(total_seed > 0.0, "Should earn SEED");
    assert!(total_zakat > 0.0, "Zakat should be deducted");

    // Verify Zakat rate
    let effective_rate = total_zakat / (total_seed + total_zakat);
    assert!(
        (effective_rate - ZAKAT_RATE).abs() < 0.001,
        "Zakat rate should be 2.5%, was {}%",
        effective_rate * 100.0
    );
}

// ─────────────────────────────────────────────────────────────
// Profile Integration
// ─────────────────────────────────────────────────────────────

#[test]
fn multi_user_profile_lifecycle() {
    let mut registry = ProfileRegistry::default();

    // Create primary
    let alice = registry.create_profile("Alice", "en").unwrap();
    assert!(alice.is_primary);

    // Create secondary with PIN
    let bob = registry.create_profile("Bob", "ar").unwrap();
    let bob_id = bob.profile_id.clone();
    if let Some(p) = registry
        .profiles
        .iter_mut()
        .find(|p| p.profile_id == bob_id)
    {
        p.set_pin("9999");
    }

    // Try switching without PIN
    assert!(registry.switch_profile(&bob_id, None).is_err());

    // Switch with correct PIN
    assert!(registry.switch_profile(&bob_id, Some("9999")).is_ok());
    assert_eq!(registry.active_profile().unwrap().display_name, "Bob");
    assert_eq!(registry.active_profile().unwrap().locale, "ar");

    // Remove Bob → auto-switches to Alice
    registry.remove_profile(&bob_id).unwrap();
    assert_eq!(registry.active_profile().unwrap().display_name, "Alice");
}

// ─────────────────────────────────────────────────────────────
// Receipt Chain Integration
// ─────────────────────────────────────────────────────────────

#[test]
fn receipt_chain_integrity() {
    let r1 = InstallReceipt::new(
        InstallReceipt::genesis_parent_hash(),
        InstallAction::FreshInstall,
        DeviceSummary {
            os: "Linux".into(),
            arch: "X86_64".into(),
            ram_gb: 16.0,
            gpu: None,
            tier: "Full".into(),
            locale: "en-US".into(),
        },
        ModelSelection::from_tier(&ModelTier::Full, true),
        vec![],
        30.0,
        true,
        0.97,
    );
    assert!(r1.verify());

    // Chain: r2 links to r1
    let r2 = InstallReceipt::new(
        r1.receipt_hash.clone(),
        InstallAction::Upgrade {
            from_version: "2.0.0".into(),
        },
        DeviceSummary {
            os: "Linux".into(),
            arch: "X86_64".into(),
            ram_gb: 16.0,
            gpu: None,
            tier: "Full".into(),
            locale: "en-US".into(),
        },
        ModelSelection::from_tier(&ModelTier::Full, true),
        vec![],
        15.0,
        true,
        0.98,
    );
    assert!(r2.verify());
    assert_eq!(r2.parent_hash, r1.receipt_hash);
}

// ─────────────────────────────────────────────────────────────
// Self-Update Integration
// ─────────────────────────────────────────────────────────────

#[test]
fn update_strategy_integration() {
    let manifest = UpdateManifest {
        from_version: "2.0.0".into(),
        to_version: "2.1.0".into(),
        released_at: "2026-06-01T00:00:00Z".into(),
        target_sha256: "deadbeef".into(),
        patch_url: Some("https://releases.bizra.ai/patches/2.0.0-2.1.0.bsdiff".into()),
        patch_sha256: Some("cafebabe".into()),
        full_url: "https://releases.bizra.ai/bin/bizra-node-2.1.0".into(),
        patch_size_bytes: Some(500_000),
        full_size_bytes: 15_000_000,
        mandatory: true,
        release_notes: "Security fix".into(),
        min_ihsan: 0.95,
    };

    // Current version matches from_version → delta
    let plan = plan_update(&manifest, "2.0.0");
    assert_eq!(plan.strategy, UpdateStrategy::DeltaPatch);
    assert!(plan.detail.contains("Delta"));

    // Different current version → full
    let plan = plan_update(&manifest, "1.9.0");
    assert_eq!(plan.strategy, UpdateStrategy::FullReplace);

    // Already up to date
    let plan = plan_update(&manifest, "2.1.0");
    assert_eq!(plan.strategy, UpdateStrategy::UpToDate);
}

// ─────────────────────────────────────────────────────────────
// i18n Integration
// ─────────────────────────────────────────────────────────────

#[test]
fn i18n_full_coverage_check() {
    let locales = supported_locales();
    let tier1_count = locales
        .iter()
        .filter(|l| l.tier == LanguageTier::Tier1)
        .count();
    let tier2_count = locales
        .iter()
        .filter(|l| l.tier == LanguageTier::Tier2)
        .count();

    // Spec §4.2 requires 10 Tier-1, 7+ Tier-2
    assert_eq!(tier1_count, 10, "Expected 10 Tier-1 languages");
    assert!(tier2_count >= 7, "Expected at least 7 Tier-2 languages");

    // RTL languages must include Arabic, Urdu, Persian
    let rtl: Vec<_> = locales
        .iter()
        .filter(|l| l.direction == TextDir::RTL)
        .map(|l| l.code.as_str())
        .collect();
    assert!(rtl.contains(&"ar"), "Arabic must be RTL");
    assert!(rtl.contains(&"ur"), "Urdu must be RTL");
    assert!(rtl.contains(&"fa"), "Persian must be RTL");
}
