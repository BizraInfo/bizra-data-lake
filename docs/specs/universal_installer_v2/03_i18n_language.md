# 03 — Language Sovereignty & i18n Architecture

> Module: `bizra-installer/src/i18n/` + `frontend/src/i18n/`
> Language: Rust (detection) + TypeScript (react-intl)
> Constitutional Anchor: Law 4 (Language Sovereignty)

## 1. Core Principle

Language is not a setting. It is a **sovereignty attribute** stored in node identity.
BIZRA does not "translate English." It is language-native from first interaction.

## 2. Language Tiers

```
ENUM LanguageTier:
    Tier1_FullNative:
        # Complete UI + DEMA personality + docs + onboarding
        languages: [ar, en, es, fr, zh-CN, hi, pt, id, bn, ru]
        coverage: "4.5B speakers"

    Tier2_UIAndLLM:
        # Full UI translation + LLM responds in language
        languages: [ja, ko, de, tr, vi, it, th, pl, nl, uk, sw, ur, fa, ms]
        coverage: "+1.5B speakers"

    Tier3_LLMNative:
        # UI in closest Tier1/2 language, LLM responds natively
        languages: "100+ (all LLM-supported)"
        coverage: "+2B speakers"
```

## 3. Language Detection

```
FUNCTION detect_language(locale_string: &str) -> Language:
    # Priority: device locale > browser > geolocation > user choice

    # 1. Parse OS locale
    lang = parse_locale(locale_string)  # "ar-AE" -> Language::Arabic

    IF lang IS_NOT Unknown:
        RETURN lang

    # 2. Browser language (if web context)
    IF running_in_webview():
        browser_lang = navigator.language
        lang = parse_locale(browser_lang)
        IF lang IS_NOT Unknown:
            RETURN lang

    # 3. Default — NEVER English by assumption
    #    Default to detected, or show picker if ambiguous
    RETURN Language::English  # Last resort only
```

## 4. i18n File Structure

```
DIRECTORY locales/:
    locale-meta.json           # {lang: {name_native, direction, fallback}}

    ar/                        # Arabic (Tier 1) — the ORIGINAL, not a translation
        installer.json         # ~50 strings: installer screens
        terminal.json          # ~200 strings: 7-view terminal UI
        errors.json            # ~80 strings: error messages
        onboarding.json        # Onboarding flow text
        dema.json              # DEMA persona system prompts
        glossary.json          # Constitutional term handling

    en/                        # English (Tier 1)
        ...same structure...

    # Tier 1: ar, en, es, fr, zh-CN, hi, pt, id, bn, ru
    # Tier 2: ja, ko, de, tr, vi, it, th, pl, nl, uk, sw, ur, fa, ms
```

## 5. Translation Scope

```
# TRANSLATED (per-locale JSON bundles):
- Installer UI:        ~50 strings (6 screens)
- Terminal 7 views:    ~200 strings (labels, headers, buttons)
- Error messages:      ~80 strings
- Onboarding flow:     ~30 strings
- Constitutional terms: glossary per language

# DYNAMIC (LLM generates in user's language):
- DEMA greetings, briefings, mission prompts
- Receipt synthesis text
- Help responses

# NEVER TRANSLATED:
- API endpoints (always English)
- Code comments, log files
- Hash values, Ed25519 keys
- Constitutional constants (mathematical)
```

## 6. RTL (Right-to-Left) Support

```
FUNCTION apply_rtl_layout(lang: Language):
    IF lang.direction() == RTL:  # ar, he, ur, fa
        document.dir = "rtl"
        # Mirror entire layout
        swap(navigation.side, "right")
        swap(content.side, "left")
        text_align = "right"
        progress_bar.direction = "rtl"
        table.column_order = reversed
        tab.order = reversed
        # Numbers stay LTR within RTL context
        number_display.direction = "ltr"
```

### RTL Adaptation Matrix

```
STRUCT RTLAdaptation:
    terminal_layout:    "Full mirror (nav right, content left)"
    text_alignment:     "Right-aligned default"
    progress_bars:      "Right-to-left fill"
    tables:             "Right-to-left column order"
    navigation:         "Right-to-left tab order"
    keyboard_shortcuts: "Mirrored where applicable"
    numbers:            "LTR within RTL (standard Arabic numeral handling)"
```

## 7. Constitutional Terms Glossary

```
# Some terms are TAUGHT, not translated
GLOSSARY = {
    "BIZRA":    { strategy: "keep_original", reason: "brand" },
    "Ihsan":    { strategy: "keep_arabic_explain", reason: "constitutional" },
    "SEED":     { strategy: "translate", reason: "economic, needs local word" },
    "BLOOM":    { strategy: "translate", reason: "governance, needs local word" },
    "DEMA":     { strategy: "keep_original", reason: "named after founder's daughter" },
    "Node":     { strategy: "translate", reason: "technical needs local word" },
    "Mission":  { strategy: "translate", reason: "core UX concept" },
    "Receipt":  { strategy: "translate", reason: "core proof concept" },
    "Reflex":   { strategy: "translate", reason: "core learning concept" },
    "Gini":     { strategy: "keep_original", reason: "mathematical" },
    "Zakat":    { strategy: "keep_arabic_explain", reason: "Islamic economic" },
}

# For Arabic: many terms ARE Arabic (Ihsan, Zakat). Arabic is original.
```

## 8. DEMA Persona Localization

```
FUNCTION dema_system_prompt(lang: Language, user_name: &str) -> String:
    # DEMA speaks as a native speaker of the user's language
    # Personality is warm, direct, helpful — adapted culturally

    template = load_locale(lang, "dema.json")

    RETURN format(template.system_prompt, {
        user_name: user_name,
        greeting: template.greeting,        # "Marhaba" / "Hello" / "Hola"
        personality: template.personality,   # Culturally adapted warmth
        formality: template.formality_level  # Some cultures prefer formal
    })
```

## 9. Proof of Translation (PoT) Governance

```
PIPELINE pot_lifecycle(submission):
    # 1. Submit
    translator.stake(100 SEED)
    submission = upload_locale_bundle(lang, strings)
    submission.declare_native_speaker()

    # 2. Validate (7 days)
    FOR reviewer IN native_speaker_pool(lang):
        reviewer.stake(10 SEED)
        vote = reviewer.review(submission)  # Accept | Reject | RequestChanges

    # 3. Consensus
    quality = weighted_average(votes, weights=reviewer_reputation)

    IF quality >= 0.67:  # Accept threshold
        translator.earn(500 BLOOM)
        publish_locale(lang, submission)
    ELSE:
        translator.lose(50 SEED)
        reject_submission(submission)

    # 4. Dispute (optional)
    IF translator.contests():
        genesis_council_arbitrate(submission)
        loser.pay(50 SEED)  # Arbitration cost
```

```
FUNCTION translation_quality_score(votes, weights) -> f64:
    # Quality = sum(w_r * v_r) / sum(w_r)
    # w_r = reviewer reputation weight
    # v_r = 1.0 (accept) or 0.0 (reject)
    numerator = sum(w * v FOR (w, v) IN zip(weights, votes))
    denominator = sum(weights)
    RETURN numerator / denominator
```

## 10. i18n Build Pipeline

```
PIPELINE build_i18n():
    # 1. Extract source strings
    run("react-intl extract src/ --out messages/en.json")

    # 2. Validate completeness
    FOR lang IN tier1_languages:
        missing = diff(messages/en.json, locales/{lang}/terminal.json)
        IF missing.len() > 0:
            FAIL "Tier 1 language {lang} missing {missing.len()} strings"

    # 3. Compile binary bundles
    run("react-intl compile locales/ --out compiled/")

    # 4. RTL visual regression (Arabic, Hebrew, Urdu)
    FOR lang IN rtl_languages:
        screenshot = render_terminal(lang)
        compare_to_baseline(screenshot, "baselines/{lang}.png")
```

## TDD Anchors

```
TEST language_detection_defaults_safely:
    # Must return a valid Language, never crash
    result = detect_language("xx-XX")  # Unknown locale
    ASSERT result IS Language  # Not None, not crash

TEST arabic_is_rtl:
    ASSERT Language::Arabic.direction() == RTL
    ASSERT Language::English.direction() == LTR
    ASSERT Language::Urdu.direction() == RTL

TEST tier1_languages_have_all_strings:
    en_keys = load_keys("locales/en/terminal.json")
    FOR lang IN ["ar", "es", "fr", "zh-CN", "hi", "pt", "id", "bn", "ru"]:
        lang_keys = load_keys(f"locales/{lang}/terminal.json")
        missing = en_keys - lang_keys
        ASSERT missing.len() == 0, f"{lang} missing: {missing}"

TEST glossary_keep_original_terms_unchanged:
    FOR lang IN all_languages:
        glossary = load_glossary(lang)
        ASSERT glossary["BIZRA"] == "BIZRA"  # Never translated
        ASSERT glossary["DEMA"] == "DEMA"    # Never translated

TEST rtl_layout_mirrors_correctly:
    render_terminal(Language::Arabic)
    ASSERT navigation.side == "right"
    ASSERT content.side == "left"
    ASSERT progress_bar.direction == "rtl"

TEST pot_quality_score_calculation:
    votes = [1.0, 1.0, 0.0, 1.0]  # 3 accept, 1 reject
    weights = [1.0, 0.8, 0.5, 0.7]
    score = translation_quality_score(votes, weights)
    ASSERT abs(score - 0.833) < 0.01  # (1+0.8+0+0.7)/(1+0.8+0.5+0.7)

TEST dema_greeting_uses_correct_language:
    prompt_ar = dema_system_prompt(Language::Arabic, "محمد")
    ASSERT "مرحبا" IN prompt_ar
    prompt_en = dema_system_prompt(Language::English, "John")
    ASSERT "Hello" IN prompt_en
```
