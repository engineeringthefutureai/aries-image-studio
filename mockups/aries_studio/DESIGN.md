---
name: Aries Image Studio
colors:
  surface: '#fcf9f2'
  surface-dim: '#dcdad3'
  surface-bright: '#fcf9f2'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3ec'
  surface-container: '#f1eee7'
  surface-container-high: '#ebe8e1'
  surface-container-highest: '#e5e2db'
  on-surface: '#1c1c18'
  on-surface-variant: '#58423a'
  inverse-surface: '#31312c'
  inverse-on-surface: '#f3f0e9'
  outline: '#8c7169'
  outline-variant: '#dfc0b6'
  surface-tint: '#a83907'
  primary: '#a53705'
  on-primary: '#ffffff'
  primary-container: '#c64e1f'
  on-primary-container: '#fffbff'
  inverse-primary: '#ffb59c'
  secondary: '#585f69'
  on-secondary: '#ffffff'
  secondary-container: '#dce3ef'
  on-secondary-container: '#5e656f'
  tertiary: '#7b542b'
  on-tertiary: '#ffffff'
  tertiary-container: '#966c41'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffdbcf'
  primary-fixed-dim: '#ffb59c'
  on-primary-fixed: '#380c00'
  on-primary-fixed-variant: '#822800'
  secondary-fixed: '#dce3ef'
  secondary-fixed-dim: '#c0c7d2'
  on-secondary-fixed: '#151c25'
  on-secondary-fixed-variant: '#404751'
  tertiary-fixed: '#ffdcbd'
  tertiary-fixed-dim: '#f0bd8b'
  on-tertiary-fixed: '#2c1600'
  on-tertiary-fixed-variant: '#623f18'
  background: '#fcf9f2'
  on-background: '#1c1c18'
  surface-variant: '#e5e2db'
typography:
  headline-xl:
    fontFamily: Plus Jakarta Sans
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: -0.03em
  headline-xl-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 30px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.025em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 30px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 26px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  code-tech-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  code-tech-xs:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-tablet: 1.5rem
  margin-desktop: 2.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system delivers a serene, high-precision generative environment tailored for artists, creative directors, and prompt engineers. The aesthetic balances the restraint of an architectural atelier with the technical rigor of an advanced imaging lab. It rejects loud promotional chrome in favor of quiet surfaces, allowing generated visual assets to command absolute visual focus.

The visual language blends modern organic minimalism with frosted tactile glassmorphism. Subtle hairline borders delineate dynamic workspace panes without imposing visual clutter, while diffuse ambient shadows establish soft physical stratification. Every interaction emphasizes deliberation, quiet tactile feedback, and seamless transition between natural language prompt composition, real-time generation arrays, and full-bleed image inspection.

## Colors

The palette establishes an organic, warm stone and bleached linen foundation that relieves eye fatigue across extended studio sessions. 

- **Primary (`#D95C2C` - Cadmium Terracotta):** Reserved for intentional creative momentum—primary generation actions, active prompt submit triggers, progress meters, and key selection rings.
- **Secondary (`#2B323B` - Deep Slate Charcoal):** High-clarity foreground token for headers, primary typography, high-priority icons, and structural divider emphasis.
- **Tertiary (`#D4A373` - Raw Ochre Amber):** Accent for secondary telemetry, warm highlight badges, sampling parameter highlights, and subtle hover states.
- **Neutral (`#F4F1EA` - Warm Alabaster Stone):** The master canvas tone. In light mode, surfaces scale from `#FAF8F5` (canvas background) down through `#EFECE4` (recessed inputs and canvas trays).

In dark mode, the neutral shifts to an earthy volcanic graphite (`#161514` base, `#201E1C` surface container tiers) paired with slate-ash text (`#EDEAE4`), maintaining the same low-fatigue warmth without stark synthetic OLED pitch blacks.

## Typography

The typography system operates on three deliberate vectors:

1. **Plus Jakarta Sans** brings sculpted, contemporary warmth to high-level headers and primary controls without drifting into display eccentricity.
2. **Inter** ensures hyper-legible, neutral reading cadence across generative chat logs, prompts, revisions, and critique sidebars.
3. **JetBrains Mono** supplies strict tabular metrics for metadata-dense properties: seeds, model checkpoint names, aspect ratio tags, CFG scales, generation latencies, and token counters.

Tabular numerals must be enforced across all parameters to eliminate jitter during generation cycles and numeric scrubbing.

## Layout & Spacing

The spatial framework accommodates a continuous three-pane studio layout: contextual chat/parameter sidebar, dynamic masonry artifact feed, and an on-demand modal lightbox viewer.

- **Desktop (1280px+):** Fluid, asymmetric 12-column foundation. The parameter/prompt panel docks at a fixed 380px or 4-column span, while the remaining 8 columns house the multi-tier masonry canvas with variable aspect columns (2 to 4 items wide). Gutter is locked at `1.5rem` (`gutter-desktop`).
- **Tablet (768px - 1279px):** Adapts to an 8-column layout. The parameter panel collapses into a bottom sliding sheet or split side drawer, expanding the masonry feed into a balanced 2-column or 3-column waterfall. Canvas margin reduces to `1.5rem` (`margin-tablet`).
- **Mobile (<768px):** Linear single-column flow with a persistent floating prompt bar. Margins tighten to `1rem` (`margin`).

All internal component gaps and padding lock strictly to the 4px baseline rhythm expressed through the `space-*` tokens.

## Elevation & Depth

Visual hierarchy leverages frosted translucent tiers, hairline borders, and warm ambient diffusion rather than dark heavy drop-shadows.

- **Surface Tiers:**
  - Base canvas: Solid flat neutral stone (`#FAF8F5` in light, `#161514` in dark).
  - Floating panels and dock inputs: Frosted glass layers utilizing `backdrop-filter: blur(20px)` and an 80% opacity surface fill (`rgba(244, 241, 234, 0.82)` light, `rgba(32, 30, 28, 0.82)` dark).
- **Hairline Outlines:**
  - Panels and floating cards utilize a micro-border: `1px solid rgba(43, 50, 59, 0.08)` in light mode, switching to `1px solid rgba(237, 234, 228, 0.10)` in dark mode.
- **Ambient Diffusion:**
  - **Level 1 (Cards, Chips):** `0 2px 8px -2px rgba(43, 50, 59, 0.04), 0 1px 3px 0 rgba(43, 50, 59, 0.02)`
  - **Level 2 (Hover states, Floating Prompt Bar):** `0 12px 32px -8px rgba(43, 50, 59, 0.08), 0 4px 12px -2px rgba(43, 50, 59, 0.03)`
  - **Level 3 (Modal Lightbox, Param Overlays):** `0 24px 64px -12px rgba(22, 21, 20, 0.18), 0 8px 24px -4px rgba(22, 21, 20, 0.06)`

## Shapes

The design system incorporates full pill-shaped radii for active interactive components and generous, soft corners on content containment cards.

- **Pills (`roundedness: 3`):** Buttons, segmented filter pills, status badges, model tags, prompt submit pills, and parameter toggles utilize continuous capsule geometry (`9999px` / `rounded-full`).
- **Canvas Panels & Containers:** Content cards, masonry containers, and side sheets adopt soft structural corners (`1.5rem` / `rounded-2xl` up to `2rem` / `rounded-3xl`), evoking smooth ceramic hardware.
- **Embedded Visual Previews:** Rendered generative outputs feature subtle corner smoothing (`0.75rem` to `1rem`) to keep the imagery feeling integrated with the stone canvas.

## Components

- **Buttons:**
  - *Primary:* Solid Cadmium Terracotta (`#D95C2C`), crisp white text, pill radius, subtle internal top specular highlight (`inset 0 1px 0 rgba(255, 255, 255, 0.2)`).
  - *Secondary / Ghost:* Frosted glass background, subtle hairline slate outline, Slate text, pill radius. On hover, background transitions to a 10% stone tint.
- **Prompt Input Field:**
  - High-comfort text container anchored with full-bleed frosted glass backdrop blur. Expanding multi-line textarea with no hard perimeter focus ring; focus is signaled by an organic terracotta glow (`0 0 0 1px #D95C2C`). Embedded quick-actions (negative prompt toggle, aspect-ratio selector, and generation button) dock inside the input tray.
- **Masonry Gallery Card:**
  - Minimal frame with `0.75rem` inner radius. Image spans edge-to-edge. Metadata overlays (seed, CFG, resolution) remain hidden until hover, transitioning via a subtle bottom-up gradient scrim with monospace labels. Includes a quick-favorite button and terracotta generation-variant pill.
- **Chips & Parameter Tags:**
  - Height of 26px to 32px, pill radius. JetBrains Mono for parameter key/value readouts (`cfg: 7.5`, `1024×1024`, `seed: 89412`). Inactive state is low-contrast stone; active state displays a terracotta hairline border with light terracotta wash.
- **Progress Meters & Sliders:**
  - Sliders feature an ultra-thin 4px neutral-tinted rail with a circular 16px solid porcelain thumb bounded by a terracotta indicator point. Generation progress bars display continuous cadmium-amber filling accompanied by a mono-spaced percentage readout.
- **Checkboxes & Radios:**
  - Micro-scale controls with soft rounding. Checkbox defaults to 4px radius; radios use circular geometry. Checked state fills with solid Cadmium Terracotta displaying a crisp white micro-glyph.
- **Lightbox / Immersive Inspector:**
  - Deep obsidian frosted backdrop (`rgba(14, 13, 12, 0.88)` with `backdrop-filter: blur(28px)`). Visual image is displayed unconstrained, flanked by an ultra-thin floating glass toolbar for prompt copying, seed cloning, upscaling, and side-by-side variation review.