# BIZRA Brand Design Standard

**Status:** [ENFORCEMENT: FROZEN]
**Source:** BIZRA Brand Identity v2.0 Elite (December 2025)
**Applies to:** bizra.ai, bizra.info, BIZRA-OS dashboard, all public artifacts

## Color System

### Primary Palette

| Name | Hex | Usage |
|------|-----|-------|
| **Gold 100** | `#F9F1D8` | Highlights, gradient end |
| **Gold 300** | `#E6D5A6` | Secondary accent |
| **Gold 400** | `#D4B875` | Hover states |
| **Gold 500** | `#C9A962` | **Primary brand color** |
| **Gold 600** | `#B08D45` | Pressed states, dim gold |
| **Gold 900** | `#8A6B2E` | Deep gold, gradient start |
| **Navy 800** | `#0A1628` | Secondary background |
| **Navy 900** | `#050B14` | **Primary background** |
| **Charcoal** | `#121212` | Alternative dark surface |
| **Ivory** | `#F8F6F1` | **Primary text** |

### Gold Gradient (Canonical)

```css
background: linear-gradient(135deg, #8A6B2E 0%, #C9A962 50%, #F9F1D8 100%);
```

### State Colors

| State | Hex | Usage |
|-------|-----|-------|
| Success | `#34D399` | Passed, healthy, admitted |
| Warning | `#FBBF24` | Degraded, partial |
| Error | `#F87171` | Failed, rejected |
| Info | `#60A5FA` | Informational |

### CSS Variables (Canonical)

```css
:root {
  /* Brand */
  --bizra-gold: #C9A962;
  --bizra-gold-light: #F9F1D8;
  --bizra-gold-dim: #B08D45;
  --bizra-gold-deep: #8A6B2E;
  --bizra-navy: #050B14;
  --bizra-navy-light: #0A1628;
  --bizra-charcoal: #121212;
  --bizra-ivory: #F8F6F1;

  /* State */
  --bizra-success: #34D399;
  --bizra-warning: #FBBF24;
  --bizra-error: #F87171;
  --bizra-info: #60A5FA;
}
```

## Typography

### Font Stack (Canonical)

| Purpose | Font | Fallback | Weight |
|---------|------|----------|--------|
| **Display/Headings** | Playfair Display | Georgia, serif | 400, 600, 700 |
| **Body/UI** | Inter | system-ui, sans-serif | 200-600 |
| **Arabic** | Amiri | serif | 400, 700 |
| **Code** | JetBrains Mono | Fira Code, monospace | 400, 500 |

### Google Fonts Import

```html
<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@200;300;400;500;600&family=Playfair+Display:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

### CSS Variables

```css
:root {
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body: 'Inter', system-ui, sans-serif;
  --font-arabic: 'Amiri', serif;
  --font-code: 'JetBrains Mono', 'Fira Code', monospace;
}
```

## Logo

### Seed of Life (بذرة)

The logo is a geometric construction based on the Seed of Life pattern:
- 7 interlocking circles (1 center + 6 surrounding)
- Radius ratio: 40:80 (inner:outer)
- Central nuqta (diamond) at intersection
- Gold gradient stroke on navy background

### Construction

```
Central circle: r=40, center=(0,0)
Six circles: r=40, centers at 40px intervals around center
Outer ring: r=80
Petals: Quadratic bezier curves at circle intersections
Nuqta: 6x6 rotated square at center
```

### Clear Space

Minimum clear space = 1x logo width on all sides.
Never place the logo on a background lighter than `#0A1628`.

## Layout Principles

### Grid

- Background grid: 50px squares, `rgba(201, 169, 98, 0.05)` lines
- Radial mask: fade to transparent at edges
- Content max-width: 1200px (6xl)

### Cards

```css
.glass-card {
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
}
```

### Spacing Scale

4px base unit: 1(4px), 2(8px), 3(12px), 4(16px), 6(24px), 8(32px), 12(48px), 16(64px)

## Arabic Integration

- Arabic text uses `Amiri` font
- The word بِذْرَة (seed) appears as tagline: البذرة
- Arabic text is center-aligned when standalone
- Gold 500 (`#C9A962`) at 60% opacity for Arabic taglines

## Websites

### bizra.ai
- **Purpose:** Product/commercial face
- **Design:** Full brand identity with animated logo reveal
- **Tone:** Professional, sovereign, innovative

### bizra.info
- **Purpose:** Technical documentation, evidence bundles, developer resources
- **Design:** Lighter touch — brand colors + code-focused layout
- **Tone:** Precise, evidence-based, technical

### Shared Requirements
- Both must use the canonical color system above
- Both must use the canonical font stack
- Gold gradient must be identical
- Logo placement: top-left or centered
- Navy 900 background on all pages
- No white backgrounds, no light themes

## Misalignment Log (To Fix)

| Component | Issue | Fix |
|-----------|-------|-----|
| BIZRA-OS tokens.css | `--color-bg-primary: #030810` (should be `#050B14`) | Update to canonical navy |
| BIZRA-OS tokens.css | `--font-ui: JetBrains Mono` only | Add Inter for body, Playfair for headings |
| BIZRA-OS tokens.css | `--font-narrative: Crimson Pro` | Replace with Playfair Display |
| Sovereign dashboard | Uses DM Sans | Replace with Inter |
| Previous session outputs | Mixed font stacks (Cormorant Garamond, IBM Plex Mono, etc.) | Align to canonical |

---

**Standing on:** بذرة — every seed carries within it the blueprint of an entire forest.
