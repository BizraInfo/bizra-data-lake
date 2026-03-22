//! i18n Engine — Language Sovereignty for BIZRA Installer
//!
//! BIZRA does not "translate" English. It is language-native from the first
//! interaction. This module handles locale detection, RTL support, and
//! string loading from JSON bundles.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §4
//! Standing on Giants: Unicode (CLDR), Shannon (information encoding)
//!
//! Constitutional: Arabic is Tier 1. The Mother Test is in Arabic.
//! RTL is not optional — it is a constitutional requirement.

use std::{collections::HashMap, path::Path};

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// Language Tiers (Spec §4.2)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanguageTier {
    /// Full: Complete UI + DEMA persona + docs + onboarding
    Tier1,
    /// UI + LLM: Full UI translation + LLM responds in language
    Tier2,
    /// LLM Native: UI in closest Tier 1/2, LLM responds natively
    Tier3,
}

// ─────────────────────────────────────────────────────────────
// Locale Metadata
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocaleInfo {
    /// BCP-47 language code (e.g., "ar", "en", "zh-CN")
    pub code: String,
    /// Language name in its own script (e.g., "العربية", "English")
    pub native_name: String,
    /// Language name in English (e.g., "Arabic")
    pub english_name: String,
    /// Text direction
    pub direction: TextDir,
    /// Tier classification
    pub tier: LanguageTier,
    /// Fallback locale if strings are missing
    pub fallback: Option<String>,
    /// Flag emoji for UI display
    pub flag: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDir {
    #[serde(rename = "ltr")]
    LTR,
    #[serde(rename = "rtl")]
    RTL,
}

/// All supported locales with metadata (Spec §4.2)
pub fn supported_locales() -> Vec<LocaleInfo> {
    vec![
        // ─── Tier 1: Full Native (4.5B speakers) ───
        LocaleInfo {
            code: "ar".into(),
            native_name: "العربية".into(),
            english_name: "Arabic".into(),
            direction: TextDir::RTL,
            tier: LanguageTier::Tier1,
            fallback: None, // Arabic is root — no fallback
            flag: "🇸🇦".into(),
        },
        LocaleInfo {
            code: "en".into(),
            native_name: "English".into(),
            english_name: "English".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: None,
            flag: "🇺🇸".into(),
        },
        LocaleInfo {
            code: "es".into(),
            native_name: "Español".into(),
            english_name: "Spanish".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇪🇸".into(),
        },
        LocaleInfo {
            code: "fr".into(),
            native_name: "Français".into(),
            english_name: "French".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇫🇷".into(),
        },
        LocaleInfo {
            code: "zh-CN".into(),
            native_name: "简体中文".into(),
            english_name: "Chinese (Simplified)".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇨🇳".into(),
        },
        LocaleInfo {
            code: "hi".into(),
            native_name: "हिन्दी".into(),
            english_name: "Hindi".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇮🇳".into(),
        },
        LocaleInfo {
            code: "pt".into(),
            native_name: "Português".into(),
            english_name: "Portuguese".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇧🇷".into(),
        },
        LocaleInfo {
            code: "id".into(),
            native_name: "Bahasa Indonesia".into(),
            english_name: "Indonesian".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇮🇩".into(),
        },
        LocaleInfo {
            code: "bn".into(),
            native_name: "বাংলা".into(),
            english_name: "Bengali".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇧🇩".into(),
        },
        LocaleInfo {
            code: "ru".into(),
            native_name: "Русский".into(),
            english_name: "Russian".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: Some("en".into()),
            flag: "🇷🇺".into(),
        },
        // ─── Tier 2: UI + LLM (+1.5B speakers) ───
        LocaleInfo {
            code: "ur".into(),
            native_name: "اردو".into(),
            english_name: "Urdu".into(),
            direction: TextDir::RTL,
            tier: LanguageTier::Tier2,
            fallback: Some("ar".into()),
            flag: "🇵🇰".into(),
        },
        LocaleInfo {
            code: "fa".into(),
            native_name: "فارسی".into(),
            english_name: "Persian".into(),
            direction: TextDir::RTL,
            tier: LanguageTier::Tier2,
            fallback: Some("ar".into()),
            flag: "🇮🇷".into(),
        },
        LocaleInfo {
            code: "tr".into(),
            native_name: "Türkçe".into(),
            english_name: "Turkish".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier2,
            fallback: Some("en".into()),
            flag: "🇹🇷".into(),
        },
        LocaleInfo {
            code: "ja".into(),
            native_name: "日本語".into(),
            english_name: "Japanese".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier2,
            fallback: Some("en".into()),
            flag: "🇯🇵".into(),
        },
        LocaleInfo {
            code: "ko".into(),
            native_name: "한국어".into(),
            english_name: "Korean".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier2,
            fallback: Some("en".into()),
            flag: "🇰🇷".into(),
        },
        LocaleInfo {
            code: "de".into(),
            native_name: "Deutsch".into(),
            english_name: "German".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier2,
            fallback: Some("en".into()),
            flag: "🇩🇪".into(),
        },
        LocaleInfo {
            code: "sw".into(),
            native_name: "Kiswahili".into(),
            english_name: "Swahili".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier2,
            fallback: Some("en".into()),
            flag: "🇰🇪".into(),
        },
    ]
}

/// Find the best matching locale for a system locale string.
/// Handles partial matches (e.g., "ar-AE" → "ar", "pt-BR" → "pt").
pub fn resolve_locale(system_locale: &str) -> LocaleInfo {
    let locales = supported_locales();

    // Exact match
    if let Some(info) = locales.iter().find(|l| l.code == system_locale) {
        return info.clone();
    }

    // Language-only match (e.g., "ar-AE" → "ar")
    let lang = system_locale.split('-').next().unwrap_or("en");
    if let Some(info) = locales.iter().find(|l| l.code == lang) {
        return info.clone();
    }

    // Default to English
    locales
        .iter()
        .find(|l| l.code == "en")
        .cloned()
        .unwrap_or_else(|| LocaleInfo {
            code: "en".into(),
            native_name: "English".into(),
            english_name: "English".into(),
            direction: TextDir::LTR,
            tier: LanguageTier::Tier1,
            fallback: None,
            flag: "🇺🇸".into(),
        })
}

// ─────────────────────────────────────────────────────────────
// String Bundle (Spec §4.6)
// ─────────────────────────────────────────────────────────────

/// A loaded string bundle for one locale and one component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StringBundle {
    pub locale: String,
    pub component: String,
    pub strings: HashMap<String, String>,
}

impl StringBundle {
    /// Load a string bundle from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("Read error: {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("Parse error: {e}"))
    }

    /// Get a string by key, returning the key itself if not found
    /// (fail-open for display — never show a blank screen).
    pub fn get<'a>(&'a self, key: &'a str) -> &'a str {
        self.strings.get(key).map(|s| s.as_str()).unwrap_or(key)
    }
}

/// i18n manager that loads and resolves strings with fallback.
pub struct I18nManager {
    bundles: HashMap<String, StringBundle>,
    active_locale: String,
    fallback_locale: String,
}

impl I18nManager {
    pub fn new(locale: &str, fallback: &str) -> Self {
        Self {
            bundles: HashMap::new(),
            active_locale: locale.to_string(),
            fallback_locale: fallback.to_string(),
        }
    }

    /// Register a string bundle.
    pub fn register(&mut self, bundle: StringBundle) {
        let key = format!("{}:{}", bundle.locale, bundle.component);
        self.bundles.insert(key, bundle);
    }

    /// Get a translated string. Resolution order:
    /// 1. Active locale bundle
    /// 2. Fallback locale bundle
    /// 3. Key itself (never empty)
    pub fn t(&self, component: &str, key: &str) -> String {
        // Try active locale
        let active_key = format!("{}:{}", self.active_locale, component);
        if let Some(bundle) = self.bundles.get(&active_key) {
            if let Some(val) = bundle.strings.get(key) {
                return val.clone();
            }
        }

        // Try fallback
        let fallback_key = format!("{}:{}", self.fallback_locale, component);
        if let Some(bundle) = self.bundles.get(&fallback_key) {
            if let Some(val) = bundle.strings.get(key) {
                return val.clone();
            }
        }

        // Return key itself
        key.to_string()
    }

    pub fn set_locale(&mut self, locale: &str) {
        self.active_locale = locale.to_string();
    }

    pub fn active_locale(&self) -> &str {
        &self.active_locale
    }

    pub fn is_rtl(&self) -> bool {
        let info = resolve_locale(&self.active_locale);
        info.direction == TextDir::RTL
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier1_has_ten_languages() {
        let t1: Vec<_> = supported_locales()
            .into_iter()
            .filter(|l| l.tier == LanguageTier::Tier1)
            .collect();
        assert_eq!(t1.len(), 10);
    }

    #[test]
    fn arabic_is_first_tier1() {
        let t1: Vec<_> = supported_locales()
            .into_iter()
            .filter(|l| l.tier == LanguageTier::Tier1)
            .collect();
        assert_eq!(t1[0].code, "ar");
    }

    #[test]
    fn arabic_is_rtl() {
        let ar = resolve_locale("ar-AE");
        assert_eq!(ar.direction, TextDir::RTL);
        assert_eq!(ar.code, "ar");
    }

    #[test]
    fn urdu_is_rtl() {
        let ur = resolve_locale("ur-PK");
        assert_eq!(ur.direction, TextDir::RTL);
    }

    #[test]
    fn english_is_ltr() {
        let en = resolve_locale("en-US");
        assert_eq!(en.direction, TextDir::LTR);
    }

    #[test]
    fn partial_locale_match() {
        // "pt-BR" should resolve to "pt"
        let pt = resolve_locale("pt-BR");
        assert_eq!(pt.code, "pt");
    }

    #[test]
    fn unknown_locale_falls_back_to_english() {
        let zz = resolve_locale("zz-ZZ");
        assert_eq!(zz.code, "en");
    }

    #[test]
    fn i18n_manager_active_locale() {
        let mut mgr = I18nManager::new("ar", "en");

        let ar_bundle = StringBundle {
            locale: "ar".into(),
            component: "installer".into(),
            strings: [("welcome".into(), "مرحبا بك في بذرة".into())]
                .into_iter()
                .collect(),
        };
        let en_bundle = StringBundle {
            locale: "en".into(),
            component: "installer".into(),
            strings: [("welcome".into(), "Welcome to BIZRA".into())]
                .into_iter()
                .collect(),
        };

        mgr.register(ar_bundle);
        mgr.register(en_bundle);

        // Arabic active → gets Arabic
        assert_eq!(mgr.t("installer", "welcome"), "مرحبا بك في بذرة");

        // Switch to English
        mgr.set_locale("en");
        assert_eq!(mgr.t("installer", "welcome"), "Welcome to BIZRA");
    }

    #[test]
    fn i18n_manager_fallback() {
        let mut mgr = I18nManager::new("ar", "en");

        // Only English has the string
        let en_bundle = StringBundle {
            locale: "en".into(),
            component: "installer".into(),
            strings: [("missing_in_ar".into(), "Fallback text".into())]
                .into_iter()
                .collect(),
        };
        mgr.register(en_bundle);

        // Arabic is active but string missing → falls back to English
        assert_eq!(mgr.t("installer", "missing_in_ar"), "Fallback text");
    }

    #[test]
    fn i18n_manager_key_as_last_resort() {
        let mgr = I18nManager::new("ar", "en");
        // No bundles registered at all → returns key
        assert_eq!(mgr.t("installer", "some_key"), "some_key");
    }

    #[test]
    fn i18n_manager_rtl_detection() {
        let mgr = I18nManager::new("ar", "en");
        assert!(mgr.is_rtl());

        let mgr2 = I18nManager::new("en", "en");
        assert!(!mgr2.is_rtl());
    }

    #[test]
    fn all_locales_have_native_names() {
        for locale in supported_locales() {
            assert!(
                !locale.native_name.is_empty(),
                "Empty native_name for {}",
                locale.code
            );
            assert!(!locale.flag.is_empty(), "Empty flag for {}", locale.code);
        }
    }
}
