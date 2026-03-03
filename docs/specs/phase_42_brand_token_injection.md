# Phase 42: Brand Token Injection

> Standing on Giants: Coyier (CSS custom properties, 2015) · Tailwind Labs (design tokens, 2020) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

`filedfs/` has no centralized design system. Brand colors (`#D4A547`, `#0A0B0F`),
typography (DM Sans, JetBrains Mono), and spacing are hardcoded inline across 9+
component files totaling ~300KB. Any brand evolution requires find-and-replace
across every file — violating DRY and creating visual inconsistency.

| Dimension | Current | Target |
|-----------|---------|--------|
| Color definitions | Inline hex in 12+ files | Single `tokens.css` + Tailwind config |
| Typography | Google Fonts link + inline `fontFamily` | CSS vars + Tailwind `fontFamily` |
| Spacing scale | Ad-hoc px values (4,8,10,12,14,16,20,24,28,32,36) | Standardized 4px grid |
| Dark mode | Hardcoded `#0A0B0F` background | `prefers-color-scheme` ready |
| Component reach | 0 files use tokens | 100% of components consume tokens |

### Audit: Hardcoded Values Found

| Token Category | Hardcoded Count | Files Affected |
|----------------|-----------------|----------------|
| `#D4A547` (gold) | 40+ occurrences | App.jsx, node0-dashboard, bizra-dashboard, onboarding/* |
| `#0A0B0F` (bg) | 15+ occurrences | App.jsx, LandingDemo, all dashboards |
| `rgba(255,255,255,0.88)` (text) | 25+ occurrences | Every component file |
| `DM Sans` font | 8+ occurrences | index.html, inline styles |
| `JetBrains Mono` font | 6+ occurrences | inline styles, code blocks |
| Spacing (px) | 200+ occurrences | All files |

---

## 2. Token Schema

### 2.1 Color Tokens

```
DEFINE color_tokens:
  # Core brand
  --bz-gold:          #D4A547    # sovereign accent
  --bz-gold-light:    #F0D68A    # highlights, hover
  --bz-gold-dark:     #8B6914    # pressed, darken
  --bz-gold-muted:    rgba(212, 165, 71, 0.15)  # subtle backgrounds

  # Surface
  --bz-bg-root:       #0A0B0F    # app background
  --bz-bg-card:       rgba(255, 255, 255, 0.03)  # card surface
  --bz-bg-elevated:   rgba(255, 255, 255, 0.06)  # elevated surface
  --bz-bg-overlay:    rgba(0, 0, 0, 0.6)         # modal overlay

  # Text
  --bz-text-primary:  rgba(255, 255, 255, 0.88)  # headings, body
  --bz-text-secondary: rgba(255, 255, 255, 0.50) # labels, hints
  --bz-text-muted:    rgba(255, 255, 255, 0.30)  # disabled, placeholders
  --bz-text-inverse:  #0A0B0F                     # text on gold bg

  # Border
  --bz-border-subtle: rgba(255, 255, 255, 0.04)  # card borders
  --bz-border-default: rgba(255, 255, 255, 0.06) # input borders
  --bz-border-gold:   rgba(212, 165, 71, 0.30)   # accent borders

  # Semantic (gauge segments)
  --bz-facts:         #6B9BF7    # blue
  --bz-preferences:   #A78BFA    # purple
  --bz-goals:         #F59E42    # orange
  --bz-expertise:     #38BDF8    # cyan
  --bz-patterns:      #F0D68A    # gold-light
  --bz-relationships: #5BBA6F    # green
  --bz-principles:    #D4A547    # gold
  --bz-context:       #FF6B9D    # pink

  # Status
  --bz-success:       #5BBA6F
  --bz-warning:       #F59E42
  --bz-error:         #EF4444
  --bz-info:          #6B9BF7
```

### 2.2 Typography Tokens

```
DEFINE typography_tokens:
  --bz-font-sans:  'DM Sans', system-ui, -apple-system, sans-serif
  --bz-font-mono:  'JetBrains Mono', 'Fira Code', monospace

  # Scale (major third: 1.25 ratio)
  --bz-text-xs:    0.75rem    # 12px — captions
  --bz-text-sm:    0.875rem   # 14px — labels
  --bz-text-base:  1rem       # 16px — body
  --bz-text-lg:    1.25rem    # 20px — subtitles
  --bz-text-xl:    1.5rem     # 24px — section heads
  --bz-text-2xl:   2rem       # 32px — page titles
  --bz-text-3xl:   2.5rem     # 40px — hero

  # Weight
  --bz-font-normal: 400
  --bz-font-medium: 500
  --bz-font-semibold: 600
  --bz-font-bold:   700

  # Line height
  --bz-leading-tight:  1.25
  --bz-leading-normal: 1.5
  --bz-leading-relaxed: 1.75
```

### 2.3 Spacing Tokens

```
DEFINE spacing_tokens:
  # 4px grid
  --bz-space-1:  0.25rem   # 4px
  --bz-space-2:  0.5rem    # 8px
  --bz-space-3:  0.75rem   # 12px
  --bz-space-4:  1rem      # 16px
  --bz-space-5:  1.25rem   # 20px
  --bz-space-6:  1.5rem    # 24px
  --bz-space-8:  2rem      # 32px
  --bz-space-10: 2.5rem    # 40px
  --bz-space-12: 3rem      # 48px
  --bz-space-16: 4rem      # 64px

  # Radius
  --bz-radius-sm:  4px
  --bz-radius-md:  8px
  --bz-radius-lg:  12px
  --bz-radius-xl:  16px
  --bz-radius-full: 9999px

  # Shadow
  --bz-shadow-sm:  0 1px 2px rgba(0,0,0,0.3)
  --bz-shadow-md:  0 4px 12px rgba(0,0,0,0.4)
  --bz-shadow-lg:  0 8px 24px rgba(0,0,0,0.5)
  --bz-shadow-gold: 0 0 20px rgba(212,165,71,0.15)
```

### 2.4 Animation Tokens

```
DEFINE animation_tokens:
  --bz-ease-default:  cubic-bezier(0.4, 0, 0.2, 1)
  --bz-ease-in:       cubic-bezier(0.4, 0, 1, 1)
  --bz-ease-out:      cubic-bezier(0, 0, 0.2, 1)
  --bz-ease-bounce:   cubic-bezier(0.34, 1.56, 0.64, 1)

  --bz-duration-fast:   150ms
  --bz-duration-normal: 300ms
  --bz-duration-slow:   600ms

  --bz-transition-default: all var(--bz-duration-normal) var(--bz-ease-default)
```

---

## 3. File Structure

```
filedfs/
├── src/
│   ├── tokens/
│   │   ├── tokens.css          # All CSS custom properties (:root)
│   │   ├── animations.css      # @keyframes definitions
│   │   └── index.css           # @import aggregator (tokens + animations + reset)
│   └── ...
├── tailwind.config.js          # Extends theme with token references
└── index.html                  # imports src/tokens/index.css
```

---

## 4. Pseudocode: Token File Generation

```
PROCEDURE create_token_files():
    # Step 1: Create tokens.css
    WRITE "filedfs/src/tokens/tokens.css":
        :root {
            FOR EACH token IN [color_tokens, typography_tokens, spacing_tokens, animation_tokens]:
                EMIT "--bz-{category}-{name}: {value};"
        }

    # Step 2: Create animations.css
    WRITE "filedfs/src/tokens/animations.css":
        MOVE all @keyframes from App.jsx inline styles
        MOVE all @keyframes from onboarding steps
        Standardize names: bz-fade-up, bz-slide-in, bz-pulse, bz-typing

    # Step 3: Create index.css (aggregator)
    WRITE "filedfs/src/tokens/index.css":
        @import './tokens.css';
        @import './animations.css';
        /* Reset */
        *, *::before, *::after { box-sizing: border-box; margin: 0; }
        body { font-family: var(--bz-font-sans); background: var(--bz-bg-root); color: var(--bz-text-primary); }
        ::placeholder { color: var(--bz-text-muted); }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: var(--bz-border-default); border-radius: var(--bz-radius-sm); }

    # Step 4: Update index.html
    REPLACE Google Fonts <link> → keep (still needed for font loading)
    ADD <link rel="stylesheet" href="/src/tokens/index.css"> BEFORE main.jsx
    REMOVE any inline <style> blocks that duplicate token values
```

---

## 5. Pseudocode: Tailwind Config Extension

```
PROCEDURE create_tailwind_config():
    WRITE "filedfs/tailwind.config.js":
        export default {
            content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
            theme: {
                extend: {
                    colors: {
                        bz: {
                            gold:       'var(--bz-gold)',
                            'gold-light': 'var(--bz-gold-light)',
                            'gold-dark':  'var(--bz-gold-dark)',
                            bg:         'var(--bz-bg-root)',
                            card:       'var(--bz-bg-card)',
                            elevated:   'var(--bz-bg-elevated)',
                        },
                        # Gauge segments
                        gauge: {
                            facts:      'var(--bz-facts)',
                            prefs:      'var(--bz-preferences)',
                            goals:      'var(--bz-goals)',
                            expertise:  'var(--bz-expertise)',
                            patterns:   'var(--bz-patterns)',
                            relations:  'var(--bz-relationships)',
                            principles: 'var(--bz-principles)',
                            context:    'var(--bz-context)',
                        },
                    },
                    fontFamily: {
                        sans: 'var(--bz-font-sans)',
                        mono: 'var(--bz-font-mono)',
                    },
                    borderRadius: {
                        bz:    'var(--bz-radius-md)',
                        'bz-lg': 'var(--bz-radius-lg)',
                    },
                    boxShadow: {
                        bz:      'var(--bz-shadow-md)',
                        'bz-gold': 'var(--bz-shadow-gold)',
                    },
                    transitionTimingFunction: {
                        bz: 'var(--bz-ease-default)',
                    },
                },
            },
        }
```

---

## 6. Pseudocode: Inline Style Migration

```
PROCEDURE migrate_inline_to_tokens(component_file):
    OPEN component_file

    # Color replacements
    REPLACE '#D4A547' → 'var(--bz-gold)'
    REPLACE '#F0D68A' → 'var(--bz-gold-light)'
    REPLACE '#8B6914' → 'var(--bz-gold-dark)'
    REPLACE '#0A0B0F' → 'var(--bz-bg-root)'
    REPLACE 'rgba(255,255,255,0.88)' → 'var(--bz-text-primary)'
    REPLACE 'rgba(255,255,255,0.5*)' → 'var(--bz-text-secondary)'
    REPLACE 'rgba(255,255,255,0.3*)' → 'var(--bz-text-muted)'
    REPLACE 'rgba(255,255,255,0.04)' → 'var(--bz-border-subtle)'
    REPLACE 'rgba(255,255,255,0.06)' → 'var(--bz-border-default)'

    # Font replacements
    REPLACE "'DM Sans', sans-serif"  → 'var(--bz-font-sans)'
    REPLACE "'JetBrains Mono', monospace" → 'var(--bz-font-mono)'

    # Gauge segment colors
    REPLACE '#6B9BF7' → 'var(--bz-facts)'
    REPLACE '#A78BFA' → 'var(--bz-preferences)'
    REPLACE '#F59E42' → 'var(--bz-goals)'
    REPLACE '#38BDF8' → 'var(--bz-expertise)'
    REPLACE '#5BBA6F' → 'var(--bz-relationships)'
    REPLACE '#FF6B9D' → 'var(--bz-context)'

    SAVE component_file

PROCEDURE run_migration():
    target_files = [
        'App.jsx',
        'LandingDemo.jsx',
        'bizra-dashboard.jsx',
        'node0-dashboard.jsx',
        'bizra-inventory.jsx',
        'bizra-status.jsx',
        'node0-mvp.jsx',
        'architecture.jsx',
        'self-modifying.jsx',
        'onboarding/OnboardingFlow.jsx',
        'onboarding/steps/VerifyStep.jsx',
        'onboarding/steps/ProviderStep.jsx',
        'onboarding/steps/TeachStep.jsx',
        'onboarding/steps/FirstChatStep.jsx',
        'onboarding/steps/DashboardStep.jsx',
    ]
    FOR EACH file IN target_files:
        migrate_inline_to_tokens(file)

    # Verify no orphan hardcoded values remain
    grep -r '#D4A547' filedfs/src/ → EXPECT 0 matches
    grep -r '#0A0B0F' filedfs/src/ → EXPECT 0 matches
```

---

## 7. TDD Anchors

```
TEST_SUITE brand_token_injection:

    TEST "tokens.css defines all required custom properties":
        css = READ 'src/tokens/tokens.css'
        ASSERT '--bz-gold' IN css
        ASSERT '--bz-bg-root' IN css
        ASSERT '--bz-font-sans' IN css
        ASSERT '--bz-space-4' IN css
        ASSERT count_of('--bz-') >= 60

    TEST "no hardcoded brand hex in component files":
        FOR EACH file IN glob('src/**/*.{jsx,tsx}'):
            content = READ file
            ASSERT '#D4A547' NOT IN content
            ASSERT '#0A0B0F' NOT IN content
            ASSERT '#F0D68A' NOT IN content
            ASSERT '#8B6914' NOT IN content

    TEST "tailwind config extends with bz- namespace":
        config = IMPORT 'tailwind.config.js'
        ASSERT 'bz' IN config.theme.extend.colors
        ASSERT 'gauge' IN config.theme.extend.colors
        ASSERT 'sans' IN config.theme.extend.fontFamily

    TEST "index.css imports tokens before reset":
        css = READ 'src/tokens/index.css'
        tokens_pos = css.indexOf("tokens.css")
        reset_pos = css.indexOf("box-sizing")
        ASSERT tokens_pos < reset_pos

    TEST "visual regression: KnowsMeGauge renders correct colors":
        render <KnowsMeGauge score={0.65} />
        segments = queryAll('[data-segment]')
        ASSERT segments[0].style.stroke == getComputedStyle('--bz-facts')

    TEST "dark mode: root background uses token":
        render <App />
        body = document.body
        ASSERT getComputedStyle(body).backgroundColor == rgb(10, 11, 15)
```

---

## 8. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `tokens.css` defines ≥60 custom properties | `grep -c '\-\-bz-' tokens.css` |
| 2 | Zero hardcoded `#D4A547` in `src/` | `grep -r '#D4A547' src/ \| wc -l` = 0 |
| 3 | Zero hardcoded `#0A0B0F` in `src/` | `grep -r '#0A0B0F' src/ \| wc -l` = 0 |
| 4 | Tailwind config compiles without errors | `npx tailwindcss --content ./src/**/*.jsx` exits 0 |
| 5 | All existing visual appearance preserved | Screenshot diff < 1% (Chromatic/Playwright) |
| 6 | Google Fonts still load (DM Sans, JetBrains Mono) | Lighthouse font audit passes |
| 7 | PWA theme_color matches `--bz-gold` | manifest.json `theme_color` = `#D4A547` |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Regex replacement breaks JS string containing hex | Use AST-aware codemods (jscodeshift) or manual review |
| CSS specificity conflicts with inline styles | `!important` on tokens only as last resort; prefer removing inline |
| Tailwind purge removes token classes | Ensure `content` globs cover all component paths |
| Performance: 60+ CSS vars on `:root` | Negligible — browsers optimize custom property resolution |
