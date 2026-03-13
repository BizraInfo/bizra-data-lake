# 06 — Update Lifecycle, Disk Management & Multi-User Profiles

> Module: `bizra-installer/src/update/` + `bizra-installer/src/profiles/`
> Language: Rust (updater, disk monitor) + TypeScript (profile UI)
> Constitutional Anchor: Law 5 (Progressive Capability) + Sovereignty

## 1. Core Principle

BIZRA never force-updates. The user chooses when. Delta patches minimize bandwidth.
Rollback is always available. Multi-user profiles share models but isolate identity.

## 2. Self-Update Flow

```
PIPELINE self_update():
    # 1. Check (daily, background, non-blocking)
    update_info = check_for_update()

    IF update_info IS None:
        RETURN  # Already up to date

    # 2. Notify (non-intrusive)
    show_notification({
        title: i18n::update_available(lang),
        body: format("BIZRA {} — {} MB", update_info.version, update_info.size_mb),
        actions: [i18n::update_now(lang), i18n::later(lang)]
    })

    # 3. User confirms → download delta patch
    IF user_chooses_update():
        patch = download_delta_patch(update_info.patch_url)
        verify_patch_signature(patch, BIZRA_PUBLIC_KEY)

        # 4. Apply on restart (or immediately)
        IF user_chooses_immediate():
            apply_patch_now(patch)
            restart_app()
        ELSE:
            stage_patch_for_restart(patch)

FUNCTION check_for_update() -> Option<UpdateInfo>:
    TRY:
        response = http_get(
            UPDATE_SERVER + "/v1/update/check",
            params = {
                current_version: app_version(),
                platform: platform_string(),
                arch: arch_string(),
            },
            timeout = 5s
        )

        IF response.latest_version > app_version():
            RETURN Some(UpdateInfo {
                version: response.latest_version,
                changelog: response.changelog,
                size_mb: response.delta_size_mb,
                patch_url: response.delta_url,
                signature: response.signature,
            })

        RETURN None

    CATCH NetworkError:
        # Offline — silently skip
        RETURN None
```

## 3. Delta Patch System

```
FUNCTION create_delta_patch(old_binary, new_binary) -> DeltaPatch:
    # Binary diff — only changed bytes are transmitted
    diff = bsdiff(old_binary, new_binary)

    patch = DeltaPatch {
        from_version: old_binary.version,
        to_version: new_binary.version,
        diff_bytes: compress_zstd(diff),
        old_hash: blake2b(old_binary),
        new_hash: blake2b(new_binary),
    }

    # Sign with BIZRA release key
    patch.signature = ed25519_sign(patch.new_hash, RELEASE_KEY)

    RETURN patch

FUNCTION apply_delta_patch(patch: DeltaPatch, current_binary) -> Result<()>:
    # 1. Verify current binary matches expected
    IF blake2b(current_binary) != patch.old_hash:
        RETURN Err("Current binary doesn't match patch base — full download needed")

    # 2. Apply diff
    new_binary = bspatch(current_binary, decompress_zstd(patch.diff_bytes))

    # 3. Verify result
    IF blake2b(new_binary) != patch.new_hash:
        RETURN Err("Patch produced unexpected result — corrupted")

    # 4. Verify signature
    IF NOT ed25519_verify(patch.signature, patch.new_hash, BIZRA_PUBLIC_KEY):
        RETURN Err("Invalid patch signature — refusing to apply")

    # 5. Stage the update
    write_file(staging_path / "bizra_new", new_binary)
    write_file(staging_path / "update_manifest.json", {
        from: patch.from_version,
        to: patch.to_version,
        applied_at: now_iso8601(),
    })

    RETURN Ok(())
```

## 4. Version History & Rollback

```
STRUCT VersionHistory:
    current:    Version
    previous:   Option<Version>    # One version back (rollback target)
    # Only keep 2 versions — current + previous
    # Older versions deleted after 2 successful updates

FUNCTION rollback_to_previous() -> Result<()>:
    IF version_history.previous IS None:
        RETURN Err("No previous version available")

    prev = version_history.previous

    # 1. Swap binaries
    rename(current_binary, staging / "bizra_rollback")
    rename(prev.binary_path, current_binary)

    # 2. Restore config (if version had different schema)
    IF prev.config_backup_exists:
        restore_config(prev.config_backup)

    # 3. Evidence ledger and identity are NEVER rolled back
    # They live in sovereign_state/ which is version-independent

    # 4. Restart
    restart_app()

    RETURN Ok(())

FUNCTION auto_rollback_on_crash():
    # If 3 consecutive starts fail within 60 seconds each, auto-rollback
    crash_count = read_crash_counter()

    IF crash_count >= 3:
        LOG warn "3 consecutive crashes detected — auto-rolling back"
        rollback_to_previous()
        reset_crash_counter()
        show_notification({
            title: i18n::auto_rollback_title(lang),
            body: i18n::auto_rollback_body(lang, version_history.previous.version),
        })

    # Increment on start, reset after 60s of successful operation
    increment_crash_counter()
    schedule_reset_crash_counter(delay = 60s)
```

## 5. Model Updates (Separate from System)

```
FUNCTION update_model(current_tier, target_tier) -> Result<()>:
    # Models are independent of system updates
    # User can upgrade/downgrade at any time

    IF target_tier.disk_requirement() > disk_available() + current_tier.disk_requirement():
        RETURN Err("Insufficient disk space for model upgrade")

    IF target_tier > current_tier:
        # Upgrade: download larger model
        download_model(target_tier, models_dir())
        # Keep old model until new one verified
        verify_model_loads(target_tier)
        delete_model(current_tier)
    ELSE:
        # Downgrade: just delete current, use smaller
        download_model(target_tier, models_dir())
        verify_model_loads(target_tier)
        delete_model(current_tier)

    update_config("model_tier", target_tier)
    RETURN Ok(())
```

## 6. Disk Space Management

```
STRUCT DiskMonitor:
    check_interval:  heartbeat_interval   # Piggyback on heartbeat
    thresholds: {
        normal:    0.85,   # < 85% usage — normal
        info:      0.85,   # 85-90% — info notification
        warning:   0.90,   # 90-95% — suggest cleanup
        alert:     0.95,   # 95-98% — pause non-critical writes
        emergency: 0.98,   # > 98% — pause heartbeat, preserve ledger
    }

FUNCTION check_disk_space(install_path) -> DiskAction:
    usage_pct = 1.0 - (disk_free(install_path) / disk_total(install_path))

    MATCH usage_pct:
        u IF u < 0.85 => RETURN DiskAction::Normal

        u IF u < 0.90 => RETURN DiskAction::Info(
            i18n::disk_getting_full(lang)
        )

        u IF u < 0.95 => RETURN DiskAction::Warning(
            suggestions = [
                suggest_model_downgrade(),
                suggest_log_cleanup(),
                suggest_reflex_cache_compress(),
            ]
        )

        u IF u < 0.98 => RETURN DiskAction::Alert(
            actions = [
                pause_log_writes(),
                pause_reflex_cache_growth(),
            ]
        )

        _ => RETURN DiskAction::Emergency(
            actions = [
                pause_heartbeat_writes(),
                preserve_evidence_ledger(),
                show_emergency_dialog(),
            ]
        )

FUNCTION suggest_space_recovery() -> Vec<RecoveryOption>:
    options = []

    # 1. Model downgrade
    current = get_current_model_tier()
    IF current > ModelTier::Micro:
        smaller = current.downgrade()
        savings = current.disk_requirement() - smaller.disk_requirement()
        options.push(RecoveryOption {
            action: "Switch to smaller model",
            savings_gb: savings,
            reversible: true,
        })

    # 2. Old evidence logs
    old_logs = find_evidence_logs_older_than(90.days)
    IF old_logs.total_size() > 0:
        options.push(RecoveryOption {
            action: "Archive evidence logs older than 90 days",
            savings_gb: old_logs.total_size_gb(),
            reversible: true,  # Can restore from archive
        })

    # 3. Reflex cache compression
    cache_size = reflex_cache_size()
    options.push(RecoveryOption {
        action: "Compress reflex cache (lossless)",
        savings_gb: cache_size * 0.4,  # ~40% compression typical
        reversible: true,
    })

    # 4. Move to external storage
    IF external_storage_available():
        options.push(RecoveryOption {
            action: "Move data to external drive",
            savings_gb: total_data_size(),
            reversible: true,
        })

    RETURN options
```

## 7. Multi-User Profile System

```
STRUCT ProfileManager:
    install_path:  PathBuf           # System-wide install
    profiles_dir:  PathBuf           # install_path / "profiles"
    shared_models: PathBuf           # install_path / "models" (shared)
    active:        Option<ProfileId>

STRUCT UserProfile:
    id:              ProfileId       # UUID
    name:            String
    language:        Language
    photo:           Option<Vec<u8>>
    identity:        Ed25519Keypair   # Per-profile sovereign identity
    evidence_ledger: PathBuf         # Per-profile evidence chain
    reflex_cache:    PathBuf         # Per-profile compiled patterns
    seed_balance:    f64             # Per-profile SEED wallet
    bloom_balance:   f64             # Per-profile BLOOM (soulbound)
    passphrase_hash: String          # Argon2id hash of passphrase
    created:         String          # ISO 8601

FUNCTION create_profile(name, language, passphrase) -> UserProfile:
    id = uuid_v4()
    profile_dir = profiles_dir / id

    create_dir_all(profile_dir / "sovereign_state")
    create_dir_all(profile_dir / "reflex_cache")
    create_dir_all(profile_dir / "briefings")

    # Generate unique identity for this profile
    seed = generate_random_seed(32)
    keypair = ed25519_generate(seed)

    # Run genesis ceremony for this profile
    genesis = GenesisActivation(seed, data_dir = profile_dir / "sovereign_state")
    result = genesis.activate()

    profile = UserProfile {
        id: id,
        name: name,
        language: language,
        identity: keypair,
        evidence_ledger: profile_dir / "sovereign_state" / "evidence_ledger.jsonl",
        reflex_cache: profile_dir / "reflex_cache",
        passphrase_hash: argon2id_hash(passphrase),
        created: now_iso8601(),
    }

    write_json(profile_dir / "profile.json", profile.to_public())  # No keys in JSON
    store_keypair_encrypted(profile_dir / "identity.enc", keypair, passphrase)

    RETURN profile

FUNCTION switch_profile(target_id, passphrase) -> Result<()>:
    profile = load_profile(target_id)

    # Verify passphrase
    IF NOT argon2id_verify(passphrase, profile.passphrase_hash):
        RETURN Err("Incorrect passphrase")

    # Unload current profile's state
    IF active_profile IS Some:
        save_current_state()
        unload_reflex_cache()

    # Load target profile's state
    decrypt_keypair = decrypt_identity(profile.identity_path, passphrase)
    load_reflex_cache(profile.reflex_cache)
    load_evidence_ledger(profile.evidence_ledger)
    set_language(profile.language)

    active_profile = Some(target_id)
    RETURN Ok(())
```

## 8. Profile Isolation Guarantees

```
STRUCT ProfileIsolation:
    # SHARED across profiles (saves disk):
    shared = [
        "models/",          # LLM model files
        "bin/",             # Runtime binaries
        "locales/",         # Language packs
    ]

    # ISOLATED per profile (sovereign):
    isolated = [
        "identity.enc",         # Ed25519 keypair (encrypted)
        "sovereign_state/",     # Evidence ledger, genesis
        "reflex_cache/",        # Compiled patterns
        "briefings/",           # DEMA briefings
        "profile.json",         # Name, language, settings
        "wallet.json",          # SEED/BLOOM balances
        "urp_config.json",      # Resource sharing settings
    ]

FUNCTION verify_profile_isolation(profiles_dir) -> IsolationResult:
    profiles = list_profiles(profiles_dir)

    FOR (a, b) IN combinations(profiles, 2):
        # No profile can read another's encrypted identity
        ASSERT NOT can_read(a, b.identity_path)

        # No profile shares evidence ledger entries
        ASSERT disjoint(a.evidence_entries(), b.evidence_entries())

        # No profile shares reflex cache
        ASSERT disjoint(a.reflex_files(), b.reflex_files())

    RETURN IsolationResult::Verified
```

## TDD Anchors

```
TEST update_check_offline_returns_none:
    mock_network_unavailable()
    result = check_for_update()
    ASSERT result IS None  # No crash, just skip

TEST delta_patch_smaller_than_full:
    old = mock_binary(version="1.0.0", size=50_MB)
    new = mock_binary(version="1.0.1", size=50.1_MB)  # Small change
    patch = create_delta_patch(old, new)
    ASSERT patch.diff_bytes.len() < 5_MB  # Much smaller than full binary

TEST patch_signature_verified:
    patch = create_delta_patch(old, new)
    # Tamper with patch
    patch.diff_bytes[0] ^= 0xFF
    result = apply_delta_patch(patch, old)
    ASSERT result.is_err()  # Signature mismatch

TEST auto_rollback_after_3_crashes:
    set_crash_counter(3)
    auto_rollback_on_crash()
    ASSERT current_version() == previous_version()

TEST model_upgrade_keeps_old_until_verified:
    upgrade_model(ModelTier::Standard, ModelTier::Enhanced)
    # Old model should exist until new one loads
    ASSERT model_loads_successfully(ModelTier::Enhanced)

TEST disk_alert_pauses_log_writes:
    mock_disk_usage(0.96)  # 96% full
    action = check_disk_space(install_path)
    ASSERT action == DiskAction::Alert
    ASSERT log_writes_paused()

TEST disk_emergency_preserves_ledger:
    mock_disk_usage(0.99)  # 99% full
    action = check_disk_space(install_path)
    ASSERT action == DiskAction::Emergency
    ASSERT evidence_ledger_preserved()

TEST profile_switch_requires_passphrase:
    create_profile("Dad", Language::Arabic, "password123")
    result = switch_profile("Dad", "wrong_password")
    ASSERT result.is_err()

TEST profile_isolation_no_cross_read:
    dad = create_profile("Dad", Language::Arabic, "pass1")
    mom = create_profile("Mom", Language::Urdu, "pass2")
    ASSERT NOT can_decrypt(dad.identity_path, "pass2")  # Mom's pass can't read Dad

TEST profiles_share_model_files:
    dad = create_profile("Dad", Language::Arabic, "pass1")
    mom = create_profile("Mom", Language::Urdu, "pass2")
    # Both profiles use the same model directory
    ASSERT dad.model_path() == mom.model_path()

TEST update_never_forces:
    # Simulate update available but user chooses "Later"
    mock_update_available("2.0.0")
    user_chooses("later")
    ASSERT current_version() == "1.0.0"  # Not updated
```
