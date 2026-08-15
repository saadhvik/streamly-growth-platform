"""Presentation layer for the Streamlit app: fonts, density, and interaction polish.

Streamlit's own theming stops at six colours and a font family, which is not
enough to make a data-dense analytics surface feel deliberate. This module adds
the rest as one injected stylesheet.

Design decisions
----------------
* **Overlays, not opaque fills.** Every surface is expressed as an ``rgba``
  overlay on whatever the base theme is, so the same stylesheet reads correctly
  in Streamlit's light *and* dark modes. Hard-coding card backgrounds is the
  usual way a "themed" Streamlit app turns unreadable the moment a viewer flips
  to dark.
* **Tabular figures on metrics.** KPI values sit in columns and must align on
  the decimal; proportional digits make a row of numbers look ragged and are
  genuinely harder to compare down a column.
* **Fira Sans / Fira Code.** A humanist sans for prose with a matching mono for
  figures, so the numeric columns stay legible at small sizes without switching
  typeface families. Loaded from Google Fonts with a system-stack fallback, so
  the app still renders correctly offline or if the CDN is blocked.
* **Motion is capped at 200ms and gated behind ``prefers-reduced-motion``.**
  Transitions here convey state change, never decoration.
* **Focus rings are strengthened, never removed.** Keyboard navigation is the
  accessibility floor, and Streamlit's default ring is easy to lose against a
  tinted surface.

Density is tuned to the dashboard end of the scale: tighter block spacing and
smaller metric labels than Streamlit's defaults, which are set for prose-heavy
apps and waste a lot of vertical space on a page that is mostly tables.
"""
from __future__ import annotations

import streamlit as st

# Design tokens, kept as one dict so the values are auditable in one place and
# can be asserted in tests rather than scattered through an f-string.
TOKENS: dict[str, str] = {
    "primary": "#1E40AF",
    "secondary": "#3B82F6",
    "accent": "#D97706",       # reserved for attention; never a series colour
    "destructive": "#DC2626",
    "good": "#0CA30C",
    "warning": "#FAB219",
    "font_sans": '"Fira Sans", system-ui, -apple-system, "Segoe UI", sans-serif',
    "font_mono": '"Fira Code", ui-monospace, "Cascadia Code", monospace',
    "radius": "8px",
    "space_1": "8px",
    "space_2": "16px",
    "space_3": "24px",
    "space_4": "32px",
    "motion": "180ms",
}

# The primary is tuned for a light surface; on Streamlit's dark background
# (#0E1117) it measures roughly 2:1, under the 3:1 floor for a non-text accent.
# These are the same hues re-stepped for the dark surface -- the approach the
# chart palette in streamly.viz already uses -- not an automatic inversion.
DARK_OVERRIDES: dict[str, str] = {
    "primary": "#3987E5",     # ~4.8:1 on #0E1117
    "accent": "#E0A030",
}


def tokens_for(dark: bool) -> dict[str, str]:
    """Design tokens for the active mode."""
    return {**TOKENS, **(DARK_OVERRIDES if dark else {})}

_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {{
  --sg-primary: {primary};
  --sg-accent: {accent};
  --sg-radius: {radius};
  --sg-motion: {motion};
  /* Surfaces as overlays so light and dark both work from one definition. */
  --sg-surface: rgba(127, 145, 175, 0.07);
  --sg-border: rgba(127, 145, 175, 0.28);
}}

html, body, [class*="st-"], .stMarkdown, button, input, select, textarea {{
  font-family: {font_sans};
}}

/* Streamlit renders icons as ligature glyphs in Material Symbols, and its icon
   spans carry `st-emotion-cache-*` classes -- so the broad selector above
   captures them and the ligature name ("keyboard_arrow_right") renders as
   literal text. Restore the icon font explicitly. */
[data-testid="stIconMaterial"], [data-testid*="Icon"] span, .material-symbols-rounded {{
  font-family: "Material Symbols Rounded", "Material Icons" !important;
}}

/* Density: Streamlit's defaults are tuned for prose, not for a page that is
   mostly tables and KPI rows. */
.block-container {{
  padding-top: {space_3};
  padding-bottom: {space_4};
  max-width: 1500px;
}}
h1 {{ font-size: 1.9rem; font-weight: 600; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.35rem; font-weight: 600; margin-top: {space_2}; }}
h3 {{ font-size: 1.1rem;  font-weight: 600; }}
hr {{ margin: {space_2} 0; opacity: 0.5; }}

/* --- KPI cards ------------------------------------------------------------
   Streamlit renders metrics as bare text. Giving them a surface makes the KPI
   row read as a unit and separates it from the prose below. */
[data-testid="stMetric"] {{
  background: var(--sg-surface);
  border: 1px solid var(--sg-border);
  border-left: 3px solid var(--sg-primary);
  border-radius: var(--sg-radius);
  padding: {space_2} {space_2} 12px {space_2};
  transition: border-color var(--sg-motion) ease, transform var(--sg-motion) ease;
  /* Cards sit in a row; only some carry a delta, so without a floor the row
     ends up ragged. A min-height keeps the baseline consistent. */
  min-height: 104px;
  height: 100%;
}}
[data-testid="stMetric"]:hover {{ border-color: var(--sg-primary); }}
[data-testid="stMetricLabel"] p {{
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  opacity: 0.72;
}}
/* Tabular figures: KPI values sit in columns and must align on the decimal. */
[data-testid="stMetricValue"] {{
  font-family: {font_mono};
  font-variant-numeric: tabular-nums;
  font-size: 1.65rem;
  font-weight: 600;
  line-height: 1.2;
}}
[data-testid="stMetricDelta"] {{ font-variant-numeric: tabular-nums; }}

/* Data tables: monospaced tabular figures so columns of numbers line up. */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--sg-border);
  border-radius: var(--sg-radius);
  overflow: hidden;
}}
[data-testid="stDataFrame"] [role="gridcell"] {{
  font-family: {font_mono};
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
}}

/* Tabs read as navigation, so give the active one a real indicator rather
   than relying on a weight change alone. */
.stTabs [data-baseweb="tab-list"] {{ gap: {space_1}; border-bottom: 1px solid var(--sg-border); }}
.stTabs [data-baseweb="tab"] {{
  font-weight: 500;
  padding: 10px 18px;
  transition: color var(--sg-motion) ease, background var(--sg-motion) ease;
}}
.stTabs [aria-selected="true"] {{ color: var(--sg-primary); }}

/* Every clickable element gets a pointer and a visible transition. */
button, [role="tab"], [data-testid="stExpander"] summary {{ cursor: pointer; }}
.stButton > button, .stDownloadButton > button {{
  border-radius: var(--sg-radius);
  font-weight: 500;
  transition: background var(--sg-motion) ease, border-color var(--sg-motion) ease;
}}

/* Accessibility floor: strengthen the focus ring, never remove it. */
:focus-visible {{
  outline: 2px solid var(--sg-primary) !important;
  outline-offset: 2px !important;
  border-radius: 4px;
}}

/* Captions carry real caveats in this app (ROAS basis, CUPED weakness), so
   they must stay above the 4.5:1 contrast floor rather than fading out. */
[data-testid="stCaptionContainer"] p {{ opacity: 0.82; font-size: 0.82rem; }}

/* Charts inherit the page typeface. */
.vega-embed, .vega-embed text {{ font-family: {font_sans} !important; }}

/* Motion conveys state change only, and never overrides a user's setting. */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }}
}}
</style>
"""


def inject(dark: bool = False) -> None:
    """Apply the stylesheet. Call once, immediately after ``set_page_config``.

    ``dark`` selects the mode-appropriate accent steps. Streamlit's dark mode is
    an app-level setting rather than an OS one, so it cannot be detected from
    ``prefers-color-scheme`` -- the caller has to pass it.
    """
    st.markdown(_STYLES.format(**tokens_for(dark)), unsafe_allow_html=True)


def status_badge(label: str, color: str, caption: str = "") -> str:
    """Markup for a decision badge.

    Colour is deliberately redundant here: the verdict word is always present
    and carries the meaning on its own, so the badge stays readable under any
    colour-vision deficiency and in forced-colours mode.
    """
    sub = (
        f"<div style='font-size:0.82rem;opacity:0.75;margin-top:2px'>{caption}</div>"
        if caption else ""
    )
    return (
        f"<div style='display:inline-block;padding:12px 20px;border-radius:{TOKENS['radius']};"
        f"border:1px solid {color};border-left:4px solid {color};"
        f"background:rgba(127,145,175,0.07)'>"
        f"<div style='font-size:1.3rem;font-weight:600;letter-spacing:-0.01em'>{label}</div>"
        f"{sub}</div>"
    )
