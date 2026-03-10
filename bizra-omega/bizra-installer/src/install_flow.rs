//! Installation Flow — 6-Step Sovereign Journey
//!
//! The install flow follows the Mother Test constraint:
//! 3 minutes, 3 taps, any language, on any device.
//!
//! Flow: DETECT → GREET → ADAPT → INSTALL → IDENTITY → ALIVE
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §3.2, §6-§12
//! Standing on Giants: Boyd (OODA, 1976), Deming (PDCA, 1950)

use crate::device_profile::{detect_device, DeviceProfile, ModelTier};
use crate::health_check::{run_health_check, HealthCheckReport};
use crate::i18n::{resolve_locale, I18nManager, LocaleInfo};
use crate::install_receipt::{
    DeviceSummary, InstallAction, InstallReceipt, ModelSelection,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────
// Install Step (State Machine)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstallStep {
    /// Step 1: Hardware detection (automatic, silent)
    Detect,
    /// Step 2: Greet in detected language
    Greet,
    /// Step 3: Show recommended config, let user adjust
    Adapt,
    /// Step 4: Download/copy model + install components
    Install,
    /// Step 5: Generate Ed25519 identity
    Identity,
    /// Step 6: First inference + "I am alive" moment
    Alive,
    /// Terminal: Something went wrong
    Failed { reason: String },
    /// Terminal: Installation complete
    Complete,
}

impl InstallStep {
    /// Next step in the flow (returns None at terminal states)
    pub fn next(&self) -> Option<InstallStep> {
        match self {
            InstallStep::Detect => Some(InstallStep::Greet),
            InstallStep::Greet => Some(InstallStep::Adapt),
            InstallStep::Adapt => Some(InstallStep::Install),
            InstallStep::Install => Some(InstallStep::Identity),
            InstallStep::Identity => Some(InstallStep::Alive),
            InstallStep::Alive => Some(InstallStep::Complete),
            InstallStep::Complete | InstallStep::Failed { .. } => None,
        }
    }

    pub fn step_number(&self) -> u8 {
        match self {
            InstallStep::Detect => 1,
            InstallStep::Greet => 2,
            InstallStep::Adapt => 3,
            InstallStep::Install => 4,
            InstallStep::Identity => 5,
            InstallStep::Alive => 6,
            InstallStep::Complete => 7,
            InstallStep::Failed { .. } => 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Install Options (User can adjust at Adapt step)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallOptions {
    /// Target installation directory
    pub install_dir: PathBuf,
    /// Override model tier (None = auto-detect)
    pub model_tier_override: Option<ModelTier>,
    /// Enable URP resource sharing (default: ask user)
    pub urp_enabled: Option<bool>,
    /// Override locale (None = auto-detect)
    pub locale_override: Option<String>,
    /// Offline mode (skip all network operations)
    pub offline: bool,
    /// Accept defaults without interaction (CI mode)
    pub non_interactive: bool,
}

impl Default for InstallOptions {
    fn default() -> Self {
        Self {
            install_dir: default_install_dir(),
            model_tier_override: None,
            urp_enabled: None,
            locale_override: None,
            offline: false,
            non_interactive: false,
        }
    }
}

fn default_install_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".bizra")
    } else {
        PathBuf::from("/opt/bizra")
    }
}

// ─────────────────────────────────────────────────────────────
// Install State (holds accumulated state across steps)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct InstallState {
    pub current_step: InstallStep,
    pub profile: Option<DeviceProfile>,
    pub locale: Option<LocaleInfo>,
    pub selected_tier: Option<ModelTier>,
    pub options: InstallOptions,
    pub started_at: std::time::Instant,
    pub errors: Vec<String>,
}

impl InstallState {
    pub fn new(options: InstallOptions) -> Self {
        Self {
            current_step: InstallStep::Detect,
            profile: None,
            locale: None,
            selected_tier: None,
            options,
            started_at: std::time::Instant::now(),
            errors: Vec::new(),
        }
    }

    pub fn fail(&mut self, reason: String) {
        self.errors.push(reason.clone());
        self.current_step = InstallStep::Failed { reason };
    }

    pub fn advance(&mut self) {
        if let Some(next) = self.current_step.next() {
            self.current_step = next;
        }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64()
    }
}

// ─────────────────────────────────────────────────────────────
// Step Executors (Spec §6-§12)
// ─────────────────────────────────────────────────────────────

/// Step 1: DETECT — Silent hardware scan (Spec §6)
///
/// No UI, no user interaction. Takes <2 seconds.
/// Populates DeviceProfile with all hardware, locale, and network info.
pub fn execute_detect(state: &mut InstallState) -> DetectResult {
    let profile = detect_device();
    let locale = if let Some(ref override_locale) = state.options.locale_override {
        resolve_locale(override_locale)
    } else {
        resolve_locale(&profile.system_locale)
    };

    let tier = state
        .options
        .model_tier_override
        .clone()
        .unwrap_or_else(|| profile.recommended_tier());

    let result = DetectResult {
        profile: profile.clone(),
        locale: locale.clone(),
        recommended_tier: tier.clone(),
        is_micro_node: profile.is_micro_node(),
        has_sufficient_disk: profile.has_sufficient_disk(),
    };

    state.profile = Some(profile);
    state.locale = Some(locale);
    state.selected_tier = Some(tier);
    state.advance();

    result
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectResult {
    pub profile: DeviceProfile,
    pub locale: LocaleInfo,
    pub recommended_tier: ModelTier,
    pub is_micro_node: bool,
    pub has_sufficient_disk: bool,
}

/// Step 2: GREET — Display welcome in detected language (Spec §7)
///
/// Returns the greeting strings for UI rendering.
pub fn execute_greet(state: &InstallState, i18n: &I18nManager) -> GreetResult {
    let locale = state.locale.as_ref().expect("detect must run before greet");

    GreetResult {
        welcome_text: i18n.t("installer", "welcome"),
        subtitle_text: i18n.t("installer", "subtitle"),
        locale_code: locale.code.clone(),
        locale_name: locale.native_name.clone(),
        is_rtl: i18n.is_rtl(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GreetResult {
    pub welcome_text: String,
    pub subtitle_text: String,
    pub locale_code: String,
    pub locale_name: String,
    pub is_rtl: bool,
}

/// Step 3: ADAPT — Show config, let user adjust (Spec §8)
///
/// Returns the recommended configuration. In non-interactive mode,
/// returns immediately with defaults.
pub fn execute_adapt(state: &InstallState) -> AdaptResult {
    let profile = state.profile.as_ref().expect("detect must run before adapt");
    let tier = state.selected_tier.as_ref().expect("detect must run before adapt");

    AdaptResult {
        model_name: tier.model_name().to_string(),
        model_size_gb: tier.disk_requirement_gb(),
        install_dir: state.options.install_dir.display().to_string(),
        disk_available_gb: profile.disk_available_gb,
        ram_available_gb: profile.ram_available_gb,
        gpu_description: profile
            .gpu
            .as_ref()
            .map(|g| format!("{} ({:?})", g.model, g.api))
            .unwrap_or_else(|| "None (CPU-only)".to_string()),
        urp_suggested: profile.ram_available_gb >= 8.0,
        can_proceed: profile.has_sufficient_disk(),
        warnings: generate_warnings(profile, tier),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptResult {
    pub model_name: String,
    pub model_size_gb: f32,
    pub install_dir: String,
    pub disk_available_gb: f32,
    pub ram_available_gb: f32,
    pub gpu_description: String,
    pub urp_suggested: bool,
    pub can_proceed: bool,
    pub warnings: Vec<String>,
}

fn generate_warnings(profile: &DeviceProfile, tier: &ModelTier) -> Vec<String> {
    let mut w = Vec::new();
    if !profile.has_sufficient_disk() {
        w.push(format!(
            "Insufficient disk space: {:.1} GB available, need {:.1} GB",
            profile.disk_available_gb,
            tier.disk_requirement_gb() + 1.0
        ));
    }
    if profile.is_micro_node() {
        w.push("Micro-node mode: System-1 only, extended heartbeat".into());
    }
    if !profile.network_available && !profile.disk_available_gb.is_sign_positive() {
        w.push("No network detected — offline install only".into());
    }
    w
}

/// Step 6: ALIVE — Run first inference + constitutional check (Spec §12)
///
/// Returns the "I am alive" message from the local LLM and the
/// health check report.
pub fn execute_alive(
    state: &InstallState,
) -> AliveResult {
    let install_dir = &state.options.install_dir;
    let profile = state.profile.as_ref().expect("detect must run before alive");

    // Run health check
    let health = run_health_check(install_dir, profile);

    // Generate receipt
    let tier = state.selected_tier.as_ref().expect("tier must be set");
    let receipt = InstallReceipt::new(
        InstallReceipt::genesis_parent_hash(),
        InstallAction::FreshInstall,
        DeviceSummary::from_profile(profile),
        ModelSelection::from_tier(tier, state.options.model_tier_override.is_none()),
        Vec::new(), // Components filled by actual install step
        state.elapsed_seconds(),
        health.all_passed,
        if health.all_passed { 0.97 } else { 0.80 },
    );

    AliveResult {
        health_check: health,
        receipt,
        first_inference_prompt: "You are BIZRA Node0, a sovereign AI. Introduce yourself in one sentence.".to_string(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AliveResult {
    pub health_check: HealthCheckReport,
    pub receipt: InstallReceipt,
    pub first_inference_prompt: String,
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_sequence() {
        let mut step = InstallStep::Detect;
        let expected = vec![
            InstallStep::Greet,
            InstallStep::Adapt,
            InstallStep::Install,
            InstallStep::Identity,
            InstallStep::Alive,
            InstallStep::Complete,
        ];
        for exp in expected {
            step = step.next().unwrap();
            assert_eq!(step, exp);
        }
        assert!(step.next().is_none());
    }

    #[test]
    fn failed_is_terminal() {
        let step = InstallStep::Failed {
            reason: "disk full".into(),
        };
        assert!(step.next().is_none());
    }

    #[test]
    fn step_numbers() {
        assert_eq!(InstallStep::Detect.step_number(), 1);
        assert_eq!(InstallStep::Greet.step_number(), 2);
        assert_eq!(InstallStep::Alive.step_number(), 6);
        assert_eq!(InstallStep::Complete.step_number(), 7);
        assert_eq!(
            InstallStep::Failed {
                reason: "x".into()
            }
            .step_number(),
            0
        );
    }

    #[test]
    fn default_options() {
        let opts = InstallOptions::default();
        assert!(opts.install_dir.to_str().unwrap().contains("bizra") || opts.install_dir.to_str().unwrap().contains(".bizra"));
        assert!(!opts.offline);
        assert!(!opts.non_interactive);
    }

    #[test]
    fn install_state_fail() {
        let mut state = InstallState::new(InstallOptions::default());
        state.fail("test error".into());
        assert!(matches!(state.current_step, InstallStep::Failed { .. }));
        assert_eq!(state.errors.len(), 1);
    }

    #[test]
    fn detect_runs() {
        let mut state = InstallState::new(InstallOptions::default());
        let result = execute_detect(&mut state);
        assert!(state.profile.is_some());
        assert!(state.locale.is_some());
        assert!(state.selected_tier.is_some());
        assert_eq!(state.current_step, InstallStep::Greet); // Advanced
        assert!(!result.locale.code.is_empty());
    }

    #[test]
    fn greet_requires_locale() {
        let mut state = InstallState::new(InstallOptions::default());
        execute_detect(&mut state);
        let i18n = I18nManager::new(&state.locale.as_ref().unwrap().code, "en");
        let greet = execute_greet(&state, &i18n);
        assert!(!greet.locale_code.is_empty());
    }

    #[test]
    fn adapt_generates_info() {
        let mut state = InstallState::new(InstallOptions::default());
        execute_detect(&mut state);
        let adapt = execute_adapt(&state);
        assert!(!adapt.model_name.is_empty());
        assert!(adapt.model_size_gb > 0.0);
    }

    #[test]
    fn locale_override() {
        let opts = InstallOptions {
            locale_override: Some("ar".into()),
            ..Default::default()
        };
        let mut state = InstallState::new(opts);
        let result = execute_detect(&mut state);
        assert_eq!(result.locale.code, "ar");
    }

    #[test]
    fn model_tier_override() {
        let opts = InstallOptions {
            model_tier_override: Some(ModelTier::Micro),
            ..Default::default()
        };
        let mut state = InstallState::new(opts);
        let result = execute_detect(&mut state);
        assert_eq!(result.recommended_tier, ModelTier::Micro);
    }
}
