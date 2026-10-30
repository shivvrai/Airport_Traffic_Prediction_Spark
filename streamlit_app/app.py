"""
Airport Traffic Prediction & Network Analytics Platform
Distributed PySpark MLlib Predictive Engine
Author: Rai Yashmit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airport Traffic Forecasting · Eurocontrol",
    page_icon="https://cdn-icons-png.flaticon.com/512/7893/7893979.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Professional SVG Vector Icons (No Emojis) ─────────────────────────────────
SVG_PLANE = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/></svg>'
SVG_DATABASE = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>'
SVG_AIRPORT = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect><line x1="9" y1="22" x2="9" y2="22.01"></line><line x1="15" y1="22" x2="15" y2="22.01"></line><line x1="9" y1="6" x2="9" y2="6.01"></line><line x1="15" y1="6" x2="15" y2="6.01"></line><line x1="9" y1="10" x2="9" y2="10.01"></line><line x1="15" y1="10" x2="15" y2="10.01"></line><line x1="9" y1="14" x2="9" y2="14.01"></line><line x1="15" y1="14" x2="15" y2="14.01"></line><line x1="9" y1="18" x2="9" y2="18.01"></line><line x1="15" y1="18" x2="15" y2="18.01"></line></svg>'
SVG_TARGET = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
SVG_TREND = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
SVG_CYCLE = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>'
SVG_CALENDAR = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>'
SVG_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
SVG_GLOBE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>'
SVG_LAYERS = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'
SVG_PIN = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>'
SVG_CLOCK = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
SVG_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'

# ─── Navigation Header & Theme Switcher ─────────────────────────────────────────
top_left, top_right = st.columns([3.8, 1.2], vertical_alignment="center")

with top_left:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
      <div style="background:linear-gradient(135deg,#3b82f6,#6366f1); color:#fff; width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; box-shadow:0 4px 12px rgba(59,130,246,0.35);">
        {SVG_PLANE}
      </div>
      <div>
        <div style="font-size:1.05rem; font-weight:800; letter-spacing:0.04em; line-height:1.2;">EUROCONTROL AIR TRAFFIC ANALYTICS</div>
        <div style="font-size:0.75rem; color:#64748b; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">Distributed PySpark MLlib Predictive Engine · Enterprise Analytics</div>
      </div>
      <div style="margin-left:14px; display:inline-flex; align-items:center; gap:6px; background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35); padding:3px 10px; border-radius:20px; font-size:0.68rem; font-weight:700; color:#10b981; letter-spacing:0.06em;">
        <span style="width:6px; height:6px; border-radius:50%; background:#10b981; display:inline-block;"></span> CLUSTER ACTIVE
      </div>
    </div>
    """, unsafe_allow_html=True)

with top_right:
    if hasattr(st, "segmented_control"):
        theme_mode = st.segmented_control(
            "Interface Theme",
            options=["Dark Mode", "Light Mode"],
            default=st.session_state.get("theme_mode", "Dark Mode"),
            key="theme_mode",
            label_visibility="collapsed"
        )
    else:
        theme_mode = st.radio(
            "Interface Theme",
            options=["Dark Mode", "Light Mode"],
            index=0 if st.session_state.get("theme_mode", "Dark Mode") == "Dark Mode" else 1,
            horizontal=True,
            key="theme_mode",
            label_visibility="collapsed"
        )

is_dark = (theme_mode == "Dark Mode")

# ─── Dynamic CSS & Design System ───────────────────────────────────────────────
if is_dark:
    # ── DARK MODE (Obsidian Midnight) ──
    BG_APP          = "#080d1a"
    BG_MESH         = "radial-gradient(ellipse at 15% 0%, rgba(59,130,246,0.14) 0%, transparent 60%), radial-gradient(ellipse at 85% 0%, rgba(99,102,241,0.10) 0%, transparent 50%)"
    TEXT_MAIN       = "#f8fafc"
    TEXT_MUTED      = "#94a3b8"
    TEXT_LABEL      = "#64748b"
    CARD_BG         = "#0f172a"
    CARD_BORDER     = "#1e293b"
    CARD_HOVER      = "rgba(99,102,241,0.4)"
    CARD_SHADOW     = "0 8px 24px rgba(0,0,0,0.35)"
    HERO_BG         = "linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.9) 100%)"
    HERO_BORDER     = "rgba(255,255,255,0.1)"
    HERO_TITLE      = "background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    PLOT_BG         = "#0f172a"
    PLOT_GRID       = "#1e293b"
    PLOT_AXIS       = "#334155"
    PLOT_FONT       = "#e2e8f0"
    PLOT_TITLE_FONT = "#f8fafc"
    TABLE_HDR_BG    = "#1e293b"
    TABLE_ROW_ALT   = "#09101f"
    ACCENT_COLOR    = "#6366f1"
    TAB_LIST_BG     = "#0f172a"
    TAB_BORDER      = "#1e293b"
    TAB_INACTIVE_CLR= "#94a3b8"
    TAB_ACTIVE_BG   = "linear-gradient(135deg, #3b82f6, #6366f1)"
    TAB_ACTIVE_CLR  = "#ffffff"
    TAB_ACTIVE_SHADOW= "rgba(59,130,246,0.35)"
    SELECT_BG       = "#0f172a"
    SELECT_BORDER   = "#334155"
    SEGMENTED_BG    = "#0f172a"
    FORECAST_CARD_BG= "#0f172a"
    FORECAST_CARD_BORDER= "#1e293b"
    FORECAST_NUM_CLR= "#a5b4fc"
    ICON_BLUE_BG    = "rgba(59,130,246,0.12)"
    ICON_BLUE_CLR   = "#3b82f6"
    ICON_PURPLE_BG  = "rgba(139,92,246,0.12)"
    ICON_PURPLE_CLR = "#8b5cf6"
    ICON_GREEN_BG   = "rgba(16,185,129,0.12)"
    ICON_GREEN_CLR  = "#10b981"
    ICON_AMBER_BG   = "rgba(245,158,11,0.12)"
    ICON_AMBER_CLR  = "#f59e0b"
    ICON_INDIGO_BG  = "rgba(99,102,241,0.12)"
    ICON_INDIGO_CLR = "#6366f1"
else:
    # ── LIGHT MODE (Enterprise Crisp White & Pure Slate) ──
    BG_APP          = "#f8fafc"
    BG_MESH         = "radial-gradient(ellipse at 15% 0%, rgba(37,99,235,0.06) 0%, transparent 60%), radial-gradient(ellipse at 85% 0%, rgba(99,102,241,0.04) 0%, transparent 50%)"
    TEXT_MAIN       = "#0f172a"
    TEXT_MUTED      = "#334155"
    TEXT_LABEL      = "#475569"
    CARD_BG         = "#ffffff"
    CARD_BORDER     = "#e2e8f0"
    CARD_HOVER      = "rgba(37,99,235,0.35)"
    CARD_SHADOW     = "0 2px 4px -1px rgba(15,23,42,0.04), 0 4px 12px -2px rgba(15,23,42,0.06)"
    HERO_BG         = "#ffffff"
    HERO_BORDER     = "#e2e8f0"
    HERO_TITLE      = "color: #0f172a;"
    PLOT_BG         = "#ffffff"
    PLOT_GRID       = "#f1f5f9"
    PLOT_AXIS       = "#cbd5e1"
    PLOT_FONT       = "#1e293b"
    PLOT_TITLE_FONT = "#0f172a"
    TABLE_HDR_BG    = "#f1f5f9"
    TABLE_ROW_ALT   = "#f8fafc"
    ACCENT_COLOR    = "#2563eb"
    TAB_LIST_BG     = "#f1f5f9"
    TAB_BORDER      = "#e2e8f0"
    TAB_INACTIVE_CLR= "#334155"
    TAB_ACTIVE_BG   = "#ffffff"
    TAB_ACTIVE_CLR  = "#2563eb"
    TAB_ACTIVE_SHADOW= "rgba(15,23,42,0.12)"
    SELECT_BG       = "#ffffff"
    SELECT_BORDER   = "#cbd5e1"
    SEGMENTED_BG    = "#e2e8f0"
    FORECAST_CARD_BG= "#eff6ff"
    FORECAST_CARD_BORDER= "#bfdbfe"
    FORECAST_NUM_CLR= "#1d4ed8"
    ICON_BLUE_BG    = "#eff6ff"
    ICON_BLUE_CLR   = "#2563eb"
    ICON_PURPLE_BG  = "#f5f3ff"
    ICON_PURPLE_CLR = "#7c3aed"
    ICON_GREEN_BG   = "#ecfdf5"
    ICON_GREEN_CLR  = "#059669"
    ICON_AMBER_BG   = "#fffbeb"
    ICON_AMBER_CLR  = "#d97706"
    ICON_INDIGO_BG  = "#eef2ff"
    ICON_INDIGO_CLR = "#4f46e5"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --text-color: {TEXT_MAIN} !important;
    --background-color: {BG_APP} !important;
    --secondary-background-color: {CARD_BG} !important;
    --primary-color: {ACCENT_COLOR} !important;
}}

html, body, [class*="css"], * {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.stApp {{
    background-color: {BG_APP} !important;
    background-image: {BG_MESH} !important;
    color: {TEXT_MAIN} !important;
}}

[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ padding: 1.5rem 2.5rem 3.5rem 2.5rem !important; max-width: 100% !important; }}

/* Force Markdown paragraph and span colors */
[data-testid="stMarkdownContainer"] p, 
[data-testid="stMarkdownContainer"] span {{
    color: {TEXT_MAIN} !important;
}}

/* Keyframes & Smooth Transitions */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.hero {{
    background: {HERO_BG};
    border: 1px solid {HERO_BORDER};
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 30px;
    box-shadow: {CARD_SHADOW};
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    position: relative;
    overflow: hidden;
}}

.hero-tag {{
    display: inline-block;
    background: {'rgba(59,130,246,0.12)' if is_dark else '#eff6ff'};
    border: 1px solid {'rgba(59,130,246,0.3)' if is_dark else '#bfdbfe'};
    color: {ACCENT_COLOR};
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}}

.hero-title {{
    font-size: 2.3rem;
    font-weight: 800;
    {HERO_TITLE}
    line-height: 1.15;
    margin: 0 0 10px 0;
    letter-spacing: -0.02em;
}}

.hero-desc {{
    font-size: 0.88rem;
    color: {TEXT_MUTED} !important;
    margin: 0;
    max-width: 740px;
    line-height: 1.6;
}}

/* Segmented Control & ButtonGroup Theme Overrides (Fix Image 2) */
div[data-testid="stButtonGroup"],
div[data-testid="stSegmentedControl"],
div[role="radiogroup"] {{
    background: {SEGMENTED_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 20px !important;
    padding: 3px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    display: inline-flex !important;
}}

div[data-testid="stButtonGroup"] button,
div[data-testid="stSegmentedControl"] button {{
    border-radius: 16px !important;
    border: none !important;
    padding: 6px 16px !important;
    background: transparent !important;
    background-color: transparent !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}

div[data-testid="stButtonGroup"] button:not([data-selected="true"]),
div[data-testid="stButtonGroup"] button[aria-checked="false"],
div[data-testid="stSegmentedControl"] button:not([aria-checked="true"]),
div[data-testid="stSegmentedControl"] button[aria-checked="false"] {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}

div[data-testid="stButtonGroup"] button:not([data-selected="true"]) *,
div[data-testid="stButtonGroup"] button[aria-checked="false"] *,
div[data-testid="stSegmentedControl"] button:not([aria-checked="true"]) *,
div[data-testid="stSegmentedControl"] button[aria-checked="false"] * {{
    color: {TEXT_MUTED} !important;
    font-weight: 600 !important;
}}

div[data-testid="stButtonGroup"] button[data-selected="true"],
div[data-testid="stButtonGroup"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
    background: {TAB_ACTIVE_BG} !important;
    background-color: {ACCENT_COLOR if is_dark else '#ffffff'} !important;
    box-shadow: 0 2px 8px {TAB_ACTIVE_SHADOW} !important;
    {'border: 1px solid #cbd5e1 !important;' if not is_dark else ''}
}}

div[data-testid="stButtonGroup"] button[data-selected="true"] *,
div[data-testid="stButtonGroup"] button[aria-checked="true"] *,
div[data-testid="stSegmentedControl"] button[aria-checked="true"] * {{
    color: {TAB_ACTIVE_CLR} !important;
    font-weight: 700 !important;
}}

/* Spacious Architectural KPI Cards */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 22px;
    margin: 28px 0 36px 0;
}}

.kpi {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 18px;
    padding: 24px 22px;
    box-shadow: {CARD_SHADOW};
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
    animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
    position: relative;
    overflow: hidden;
}}

.kpi:hover {{
    transform: translateY(-4px);
    border-color: {CARD_HOVER};
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
}}

.kpi.blue   {{ border-top: 4px solid #3b82f6; }}
.kpi.purple {{ border-top: 4px solid #8b5cf6; }}
.kpi.green  {{ border-top: 4px solid #10b981; }}
.kpi.amber  {{ border-top: 4px solid #f59e0b; }}
.kpi.indigo {{ border-top: 4px solid #6366f1; }}

.kpi-icon-wrap {{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
}}

.kpi.blue .kpi-icon-wrap   {{ background: {ICON_BLUE_BG}; color: {ICON_BLUE_CLR}; border: 1px solid rgba(59,130,246,0.2); }}
.kpi.purple .kpi-icon-wrap {{ background: {ICON_PURPLE_BG}; color: {ICON_PURPLE_CLR}; border: 1px solid rgba(139,92,246,0.2); }}
.kpi.green .kpi-icon-wrap  {{ background: {ICON_GREEN_BG}; color: {ICON_GREEN_CLR}; border: 1px solid rgba(16,185,129,0.2); }}
.kpi.amber .kpi-icon-wrap  {{ background: {ICON_AMBER_BG}; color: {ICON_AMBER_CLR}; border: 1px solid rgba(245,158,11,0.2); }}
.kpi.indigo .kpi-icon-wrap {{ background: {ICON_INDIGO_BG}; color: {ICON_INDIGO_CLR}; border: 1px solid rgba(99,102,241,0.2); }}

.kpi-val {{
    font-size: 1.95rem;
    font-weight: 800;
    color: {TEXT_MAIN} !important;
    letter-spacing: -0.03em;
    line-height: 1;
}}

.kpi-label {{
    font-size: 0.70rem;
    color: {TEXT_LABEL} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-top: 10px;
}}

.kpi-sub {{
    font-size: 0.76rem;
    color: {TEXT_MUTED} !important;
    margin-top: 6px;
    font-weight: 500;
}}

/* Universal Streamlit Tab Visibility Overrides (Fix Image 1) */
[data-testid="stTabs"] {{
    margin-top: 14px !important;
    margin-bottom: 26px !important;
}}

[data-testid="stTabs"] div[role="tablist"],
[data-testid="stTabs"] [role="tablist"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {TAB_LIST_BG} !important;
    border: 1px solid {TAB_BORDER} !important;
    border-radius: 14px !important;
    padding: 6px 8px !important;
    gap: 8px !important;
    box-shadow: {CARD_SHADOW} !important;
}}

[data-testid="stTabs"] div[data-testid="stTab"],
[data-testid="stTabs"] div[role="tab"],
[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 9px 22px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}

/* Inactive Tab Label Contrast */
[data-testid="stTabs"] div[data-testid="stTab"]:not([aria-selected="true"]),
[data-testid="stTabs"] div[data-testid="stTab"]:not([aria-selected="true"]) *,
[data-testid="stTabs"] div[role="tab"]:not([aria-selected="true"]),
[data-testid="stTabs"] div[role="tab"]:not([aria-selected="true"]) *,
[data-testid="stTabs"] button[role="tab"]:not([aria-selected="true"]),
[data-testid="stTabs"] button[role="tab"]:not([aria-selected="true"]) *,
[data-testid="stTabs"] button[data-baseweb="tab"]:not([aria-selected="true"]),
[data-testid="stTabs"] button[data-baseweb="tab"]:not([aria-selected="true"]) * {{
    color: {TAB_INACTIVE_CLR} !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}}

/* Active Tab Label Highlight */
[data-testid="stTabs"] div[data-testid="stTab"][aria-selected="true"],
[data-testid="stTabs"] div[data-testid="stTab"][aria-selected="true"] *,
[data-testid="stTabs"] div[role="tab"][aria-selected="true"],
[data-testid="stTabs"] div[role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] * {{
    color: {TAB_ACTIVE_CLR} !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}}

/* Active Selection Underline Bar */
[data-testid="stTabs"] .react-aria-SelectionIndicator {{
    background-color: {ACCENT_COLOR} !important;
    height: 3px !important;
    border-radius: 2px !important;
}}

[data-testid="stTabs"] div[role="tab"]:hover:not([aria-selected="true"]),
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {{
    background: rgba(125, 125, 125, 0.08) !important;
}}
[data-testid="stTabs"] div[role="tab"]:hover:not([aria-selected="true"]) *,
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) * {{
    color: {TEXT_MAIN} !important;
}}

/* Section Headings */
.sec-head {{
    font-size: 0.98rem;
    font-weight: 700;
    color: {TEXT_MAIN} !important;
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: 0.01em;
}}

.sec-head svg {{
    color: {ACCENT_COLOR};
}}

/* Select Box */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p {{
    color: {TEXT_MAIN} !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {SELECT_BG} !important;
    border: 1px solid {SELECT_BORDER} !important;
    color: {TEXT_MAIN} !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}}
[data-testid="stSelectbox"] span, [data-testid="stSelectbox"] div, [data-testid="stSelectbox"] p {{
    color: {TEXT_MAIN} !important;
}}

/* Airport Detail Card */
.apt-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: {CARD_SHADOW};
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}}

.apt-card-name {{
    font-size: 1.35rem;
    font-weight: 800;
    color: {TEXT_MAIN} !important;
}}

.apt-card-icao {{
    display: inline-block;
    background: {'rgba(59,130,246,0.12)' if is_dark else '#eff6ff'};
    border: 1px solid {'rgba(59,130,246,0.3)' if is_dark else '#bfdbfe'};
    color: {ACCENT_COLOR};
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 6px;
    margin-top: 4px;
}}

.apt-card-country {{
    font-size: 0.82rem;
    color: {TEXT_LABEL} !important;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* Forecast Banner */
.forecast-card {{
    background: {FORECAST_CARD_BG};
    border: 1px solid {FORECAST_CARD_BORDER};
    border-radius: 18px;
    padding: 22px 24px;
    text-align: center;
    box-shadow: {CARD_SHADOW};
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}}

.forecast-num {{
    font-size: 2.6rem;
    font-weight: 800;
    color: {FORECAST_NUM_CLR};
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 8px 0;
}}

.forecast-badge {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT_LABEL} !important;
}}

/* Airport Accuracy Card */
.accuracy-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-left: 4px solid #10b981;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 18px 0;
    box-shadow: {CARD_SHADOW};
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}}

/* Custom Table Styles */
.custom-table {{
    width: 100%;
    border-collapse: collapse;
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    overflow: hidden;
    box-shadow: {CARD_SHADOW};
}}

.custom-table th {{
    padding: 11px 14px;
    background: {TABLE_HDR_BG};
    color: {TEXT_LABEL} !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    text-align: left;
    border-bottom: 1px solid {CARD_BORDER};
}}

.custom-table td {{
    padding: 11px 14px;
    color: {TEXT_MAIN} !important;
    font-size: 0.85rem;
    border-bottom: 1px solid {CARD_BORDER};
}}

.custom-table tr:last-child td {{ border-bottom: none; }}
.custom-table tr:hover td {{ background: rgba(125,125,125,0.04); }}

/* Footer */
.footer {{
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid {CARD_BORDER};
    text-align: center;
    color: {TEXT_LABEL} !important;
    font-size: 0.78rem;
    line-height: 1.6;
}}

#MainMenu, footer, header, .stDeployButton {{ visibility: hidden !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Base Layout (Fix Images 3 & 4) ─────────────────────────────────────
def base_layout(h=320, legend=False):
    font_clr       = PLOT_FONT        # High-contrast tick font (#1e293b in light, #e2e8f0 in dark)
    title_font_clr = PLOT_TITLE_FONT  # High-contrast axis title font (#0f172a in light, #f8fafc in dark)
    grid_clr       = PLOT_GRID
    axis_clr       = PLOT_AXIS
    plot_bg        = PLOT_BG

    d = dict(
        template="none",  # Do not inherit washed-out defaults from template
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=font_clr, size=11),
        margin=dict(l=18, r=18, t=36, b=18),
        height=h,
        xaxis=dict(
            gridcolor=grid_clr,
            zerolinecolor=axis_clr,
            linecolor=axis_clr,
            tickfont=dict(color=font_clr, size=10),
            title=dict(font=dict(color=title_font_clr, size=11))
        ),
        yaxis=dict(
            gridcolor=grid_clr,
            zerolinecolor=axis_clr,
            linecolor=axis_clr,
            tickfont=dict(color=font_clr, size=10),
            title=dict(font=dict(color=title_font_clr, size=11))
        ),
        xaxis_tickfont_color=font_clr,
        yaxis_tickfont_color=font_clr,
        xaxis_title_font_color=title_font_clr,
        yaxis_title_font_color=title_font_clr,
        hoverlabel=dict(
            bgcolor=CARD_BG,
            font_color=TEXT_MAIN,
            font_family="Plus Jakarta Sans, sans-serif"
        )
    )
    if legend:
        d["legend"] = dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=font_clr),
            orientation="h",
            y=1.09,
            x=0
        )
    return d

BLUES_SCALE   = [[0, "#1e293b" if is_dark else "#dbeafe"], [0.5, "#3b82f6"], [1, "#93c5fd" if is_dark else "#1d4ed8"]]
PURPLES_SCALE = [[0, "#2e1065" if is_dark else "#ede9fe"], [0.5, "#8b5cf6"], [1, "#c084fc" if is_dark else "#6d28d9"]]
GREENS_SCALE  = [[0, "#064e3b" if is_dark else "#d1fae5"], [0.5, "#10b981"], [1, "#6ee7b7" if is_dark else "#047857"]]

# ─── Resilient File Path Resolution ────────────────────────────────────────────
def resolve_data_path(filename: str) -> Path:
    candidates = [
        Path(filename),
        Path(__file__).parent / filename,
        Path(__file__).parent.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(__file__).parent.parent / filename

DATA_CSV = resolve_data_path("airport_traffic_2025.csv")
FORE_CSV = resolve_data_path("colab_output.csv")
MET_CSV  = resolve_data_path("model_evaluation_metrics.csv")
TEST_CSV = resolve_data_path("test_predictions.csv")

# ─── Cached Data Loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_hist():
    df = pd.read_csv(DATA_CSV, parse_dates=["FLT_DATE"])
    df["FLT_TOT_1"] = pd.to_numeric(df["FLT_TOT_1"], errors="coerce")
    df = df.dropna(subset=["FLT_TOT_1", "FLT_DATE"])
    df["MONTH_NUM"] = df["FLT_DATE"].dt.month
    df["MONTH_ABB"] = df["FLT_DATE"].dt.strftime("%b")
    df["DOW"]       = df["FLT_DATE"].dt.day_name()
    df["DOW_NUM"]   = df["FLT_DATE"].dt.dayofweek
    df["QUARTER"]   = "Q" + df["FLT_DATE"].dt.quarter.astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_forecast():
    df = pd.read_csv(FORE_CSV, parse_dates=["date"])
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").round(1)
    return df

@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(MET_CSV)

@st.cache_data(show_spinner=False)
def load_test_predictions():
    if TEST_CSV.exists():
        df = pd.read_csv(TEST_CSV, parse_dates=["date"])
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        df["abs_error"] = (df["target"] - df["prediction"]).abs()
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def build_profiles(hist_df):
    p = hist_df.groupby(["APT_ICAO", "APT_NAME", "STATE_NAME"]).agg(
        total   =("FLT_TOT_1", "sum"),
        avg_day =("FLT_TOT_1", "mean"),
        peak    =("FLT_TOT_1", "max"),
        variab  =("FLT_TOT_1", "std"),
        days    =("FLT_DATE", "nunique"),
    ).reset_index()
    p["avg_day"] = p["avg_day"].round(1)
    p["peak"]    = p["peak"].round(0).astype(int)
    p["variab"]  = p["variab"].fillna(0).round(1)
    return p

@st.cache_data(show_spinner=False)
def cluster_airports(profile_df):
    feats = profile_df[["avg_day", "variab", "peak"]].fillna(0)
    X = StandardScaler().fit_transform(feats)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    res = profile_df.copy()
    res["_c"] = labels
    means = res.groupby("_c")["avg_day"].mean().sort_values()
    tier_map  = {i: t for i, t in zip(means.index, ["Small / Regional", "Medium Hub", "Major Hub"])}
    color_map = {i: c for i, c in zip(means.index, ["#10b981", "#f59e0b", "#6366f1"])}
    res["tier"]  = res["_c"].map(tier_map)
    res["color"] = res["_c"].map(color_map)
    return res

hist     = load_hist()
forecast = load_forecast()
metrics  = load_metrics()
test_df  = load_test_predictions()
profile  = cluster_airports(build_profiles(hist))

# ─── Hero Overview Card ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">Aviation Infrastructure · Distributed PySpark MLlib</div>
  <h1 class="hero-title">Airport Traffic Forecasting &amp; Network Analytics</h1>
  <p class="hero-desc">
    Scalable distributed processing, time-series feature engineering, and next-day forecasting of IFR flight movements across
    <strong style="color:{ACCENT_COLOR}">331 European airports in 42 sovereign states</strong> using Apache Spark MLlib 
    (Linear Regression · Random Forest · Gradient-Boosted Trees).
  </p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Metric Cards (Spacious Architectural Design) ───────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi blue">
    <div class="kpi-icon-wrap">{SVG_DATABASE}</div>
    <div class="kpi-val">116,020</div>
    <div class="kpi-label">Records Processed</div>
    <div class="kpi-sub">365 calendar days · 2025</div>
  </div>
  <div class="kpi purple">
    <div class="kpi-icon-wrap">{SVG_AIRPORT}</div>
    <div class="kpi-val">331</div>
    <div class="kpi-label">Monitored Airports</div>
    <div class="kpi-sub">across 42 European states</div>
  </div>
  <div class="kpi green">
    <div class="kpi-icon-wrap">{SVG_TARGET}</div>
    <div class="kpi-val">0.9925</div>
    <div class="kpi-label">Test Set R² Score</div>
    <div class="kpi-sub">Linear Regression · Best Model</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-icon-wrap">{SVG_TREND}</div>
    <div class="kpi-val">40.5%</div>
    <div class="kpi-label">MAE Improvement</div>
    <div class="kpi-sub">17.43 → 10.37 flights/day</div>
  </div>
  <div class="kpi indigo">
    <div class="kpi-icon-wrap">{SVG_CYCLE}</div>
    <div class="kpi-val">70.0%</div>
    <div class="kpi-label">Weekly Cycle Weight</div>
    <div class="kpi-sub">β = 0.700 · 7-Day Timetable</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Navigation Tabs ───────────────────────────────────────────────────────────
T1, T2, T3, T4 = st.tabs([
    "Traffic Overview",
    "Airport Explorer",
    "Model Benchmarks",
    "Cluster Segmentation"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TRAFFIC OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with T1:
    r1c1, r1c2 = st.columns([3, 2], gap="large")

    with r1c1:
        st.markdown(f'<div class="sec-head">{SVG_CALENDAR} Monthly Flight Volume (Pan-European Aggregate)</div>', unsafe_allow_html=True)
        monthly = (hist.groupby(["MONTH_NUM", "MONTH_ABB"])["FLT_TOT_1"]
                   .sum().reset_index().sort_values("MONTH_NUM"))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["MONTH_ABB"], y=monthly["FLT_TOT_1"],
            marker=dict(
                color=monthly["FLT_TOT_1"],
                colorscale=BLUES_SCALE,
                line=dict(width=0)
            ),
            text=monthly["FLT_TOT_1"].apply(lambda x: f"{x/1e6:.2f}M"),
            textposition="outside",
            textfont=dict(color=TEXT_MUTED, size=10),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} flights<extra></extra>",
        ))
        fig.update_layout(**base_layout(h=310), yaxis_title="Total Flights")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown(f'<div class="sec-head">{SVG_CHART} Top 10 Busiest European Airports</div>', unsafe_allow_html=True)
        top10 = profile.nlargest(10, "total")[["APT_NAME", "total", "avg_day"]].sort_values("total")
        top10["name_short"] = top10["APT_NAME"].str.replace(" - ", " ").str[:22]
        top10["total_k"] = (top10["total"] / 1000).round(0).astype(int)

        fig = go.Figure(go.Bar(
            x=top10["total_k"], y=top10["name_short"],
            orientation="h",
            marker=dict(
                color=top10["total_k"],
                colorscale=PURPLES_SCALE,
                line=dict(width=0)
            ),
            text=top10["total_k"].apply(lambda x: f"{x:,}K"),
            textposition="outside",
            textfont=dict(color=ACCENT_COLOR, size=10),
            hovertemplate="<b>%{y}</b><br>%{x:,}K flights<extra></extra>",
        ))
        fig.update_layout(**base_layout(h=310))
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2, gap="large")

    with r2c1:
        st.markdown(f'<div class="sec-head">{SVG_CYCLE} Total Flight Volume by Day of Week</div>', unsafe_allow_html=True)
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = hist.groupby("DOW")["FLT_TOT_1"].sum().reindex(dow_order).reset_index()
        dow.columns = ["Day", "Total"]
        dow["TotalM"] = (dow["Total"] / 1e6).round(2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dow["Day"], y=dow["TotalM"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.12)" if is_dark else "rgba(37,99,235,0.08)",
            line=dict(color=ACCENT_COLOR, width=3),
            marker=dict(color="#a5b4fc" if is_dark else "#2563eb", size=8, line=dict(color="#312e81" if is_dark else "#1d4ed8", width=2)),
            text=dow["TotalM"].apply(lambda x: f"{x:.2f}M"),
            textposition="top center",
            textfont=dict(color=TEXT_MUTED, size=9),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}M total flights across Europe<extra></extra>",
        ))
        l_dow = base_layout(h=260)
        l_dow["yaxis"]["title"] = dict(text="Millions of Flights", font=dict(color=PLOT_TITLE_FONT, size=11))
        fig.update_layout(**l_dow)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown(f'<div class="sec-head">{SVG_GLOBE} Top 12 Sovereign States by Flight Volume</div>', unsafe_allow_html=True)
        ctry = (hist.groupby("STATE_NAME")["FLT_TOT_1"].sum()
                .nlargest(12).reset_index().sort_values("FLT_TOT_1"))
        ctry.columns = ["Country", "Total"]
        ctry["TotalM"] = (ctry["Total"] / 1e6).round(2)

        fig = go.Figure(go.Bar(
            x=ctry["TotalM"], y=ctry["Country"],
            orientation="h",
            marker=dict(color=ctry["TotalM"], colorscale=GREENS_SCALE, line=dict(width=0)),
            text=ctry["TotalM"].apply(lambda x: f"{x:.1f}M"),
            textposition="outside",
            textfont=dict(color="#10b981" if is_dark else "#059669", size=9),
            hovertemplate="<b>%{y}</b><br>%{x:.2f}M flights<extra></extra>",
        ))
        fig.update_layout(**base_layout(h=260))
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Additional Analytical Charts ──────────────────────────────────────────
    r3c1, r3c2 = st.columns(2, gap="large")

    with r3c1:
        st.markdown(f'<div class="sec-head">{SVG_CHART} Quarterly Seasonal Progression &amp; Summer Expansion</div>', unsafe_allow_html=True)
        q_order = ["Q1", "Q2", "Q3", "Q4"]
        q_data = hist.groupby("QUARTER")["FLT_TOT_1"].agg(total="sum", avg="mean").reindex(q_order).reset_index()
        q_data["total_M"] = (q_data["total"] / 1e6).round(2)
        q_names = ["Q1 · Winter Trough", "Q2 · Spring Recovery", "Q3 · Summer Peak (+52%)", "Q4 · Autumn/Winter"]

        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=q_names, y=q_data["total_M"],
            marker=dict(color=["#64748b", "#3b82f6", "#f59e0b", "#8b5cf6"] if is_dark else ["#64748b", "#2563eb", "#d97706", "#7c3aed"], line=dict(width=0)),
            text=q_data["total_M"].apply(lambda x: f"{x:.2f}M"),
            textposition="outside",
            textfont=dict(color=TEXT_MUTED, size=10),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}M Flights<extra></extra>",
        ))
        fig_q.update_layout(**base_layout(h=260), yaxis_title="Total Flights (Millions)")
        st.plotly_chart(fig_q, use_container_width=True)

    with r3c2:
        st.markdown(f'<div class="sec-head">{SVG_CYCLE} Weekday vs. Weekend Airspace Load Ratio</div>', unsafe_allow_html=True)
        hist["is_weekend"] = hist["DOW_NUM"].apply(lambda x: "Weekend (Sat-Sun)" if x >= 5 else "Weekday (Mon-Fri)")
        ww = hist.groupby("is_weekend")["FLT_TOT_1"].agg(total="sum", daily_avg="mean").reset_index()
        ww["total_M"] = (ww["total"] / 1e6).round(2)

        fig_ww = go.Figure(go.Pie(
            labels=ww["is_weekend"], values=ww["total_M"], hole=0.55,
            marker=dict(colors=["#2563eb", "#10b981"] if not is_dark else ["#3b82f6", "#10b981"], line=dict(color=BG_APP, width=2)),
            textfont=dict(size=11, color=TEXT_MAIN),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:.2f}M Flights<extra></extra>",
        ))
        fig_ww.update_layout(**base_layout(h=260), showlegend=False)
        st.plotly_chart(fig_ww, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — AIRPORT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with T2:
    airports_sorted = sorted(hist["APT_NAME"].dropna().unique())
    default_idx = airports_sorted.index("London - Heathrow") if "London - Heathrow" in airports_sorted else 0

    col_sel, col_empty = st.columns([2.5, 2.5])
    with col_sel:
        sel_name = st.selectbox(
            "Select Airport by Station Name",
            airports_sorted,
            index=default_idx,
            key="airport_selector",
        )

    sel_icao    = hist[hist["APT_NAME"] == sel_name]["APT_ICAO"].iloc[0]
    sel_country = hist[hist["APT_ICAO"] == sel_icao]["STATE_NAME"].iloc[0]
    apt_h       = hist[hist["APT_ICAO"] == sel_icao].sort_values("FLT_DATE").copy()
    apt_p       = profile[profile["APT_ICAO"] == sel_icao]
    apt_f       = forecast[forecast["APT_ICAO"] == sel_icao]
    
    apt_test = test_df[test_df["APT_ICAO"] == sel_icao].sort_values("date") if not test_df.empty else pd.DataFrame()
    has_test_metrics = not apt_test.empty and len(apt_test) >= 5

    ac1, ac2 = st.columns([3, 1.2], gap="large")

    with ac1:
        tier_color = apt_p["color"].iloc[0] if not apt_p.empty else "#6366f1"
        tier_name  = apt_p["tier"].iloc[0]  if not apt_p.empty else "—"
        total_f    = int(apt_p["total"].iloc[0]) if not apt_p.empty else 0
        avg_f      = apt_p["avg_day"].iloc[0]    if not apt_p.empty else 0
        peak_f     = int(apt_p["peak"].iloc[0])  if not apt_p.empty else 0

        st.markdown(f"""
        <div class="apt-card">
          <div style="display:flex; align-items:flex-start; justify-content:space-between;">
            <div>
              <div class="apt-card-name">{sel_name}</div>
              <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
                <span class="apt-card-icao">{sel_icao}</span>
                <span class="apt-card-country">{SVG_PIN} {sel_country}</span>
              </div>
            </div>
            <div>
              <span style="color:{tier_color}; font-size:0.75rem; font-weight:700;
                           background:{'rgba(125,125,125,0.08)' if is_dark else '#f1f5f9'}; padding:5px 14px;
                           border-radius:20px; border:1px solid {tier_color}44; display:inline-flex; align-items:center; gap:6px;">
                <span style="width:6px; height:6px; border-radius:50%; background:{tier_color}; display:inline-block;"></span>
                {tier_name}
              </span>
            </div>
          </div>
          <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:22px; padding-top:16px; border-top:1px solid {CARD_BORDER}">
            <div style="text-align:center">
              <div style="font-size:1.65rem; font-weight:800; color:{TEXT_MAIN}">{total_f:,}</div>
              <div style="font-size:0.65rem; color:{TEXT_LABEL}; text-transform:uppercase; letter-spacing:.08em; margin-top:4px">Annual Volume 2025</div>
            </div>
            <div style="text-align:center; border-left:1px solid {CARD_BORDER}">
              <div style="font-size:1.65rem; font-weight:800; color:{TEXT_MAIN}">{avg_f:,.1f}</div>
              <div style="font-size:0.65rem; color:{TEXT_LABEL}; text-transform:uppercase; letter-spacing:.08em; margin-top:4px">Daily Average</div>
            </div>
            <div style="text-align:center; border-left:1px solid {CARD_BORDER}">
              <div style="font-size:1.65rem; font-weight:800; color:{TEXT_MAIN}">{peak_f:,}</div>
              <div style="font-size:0.65rem; color:{TEXT_LABEL}; text-transform:uppercase; letter-spacing:.08em; margin-top:4px">Peak Day Record</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ac2:
        if not apt_f.empty:
            row  = apt_f.iloc[0]
            pred = row["prediction"]
            prev = row["yesterday"]
            chg  = ((pred - prev) / prev * 100) if prev > 0 else 0
            chg_color = "#10b981" if chg >= 0 else "#f87171"
            chg_symbol = "+" if chg >= 0 else ""
            st.markdown(f"""
            <div class="forecast-card">
              <div class="forecast-badge">{SVG_CLOCK} NEXT-DAY BATCH FORECAST</div>
              <div class="forecast-num">{pred:,.0f}</div>
              <div style="font-size:0.75rem; color:{TEXT_MUTED}; font-weight:600;">Predicted Flight Movements</div>
              <div style="font-size:0.80rem; color:{ACCENT_COLOR}; font-weight:700; margin-top:8px;">Target Date: 2026-01-01</div>
              <div style="margin-top:12px; font-size:0.78rem; color:{chg_color}; font-weight:700; background:{'rgba(125,125,125,0.06)' if is_dark else '#ffffff'}; padding:6px 10px; border-radius:8px; border:1px solid {CARD_BORDER}">
                {chg_symbol}{chg:.1f}% vs previous day ({int(prev):,} flights)
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No forecast data available for this station.")

    # ── Station-Specific Accuracy Callout ──────────────────────────────────────
    if has_test_metrics:
        apt_mae = apt_test["abs_error"].mean()
        apt_target_sum = apt_test["target"].sum()
        apt_wmape = (apt_test["abs_error"].sum() / apt_target_sum * 100) if apt_target_sum > 0 else 0
        apt_acc = max(0.0, 100.0 - apt_wmape)
        apt_rmse = np.sqrt(((apt_test["target"] - apt_test["prediction"])**2).mean())

        acc_badge_clr = "#10b981" if apt_acc >= 90 else "#f59e0b"

        st.markdown(f"""
        <div class="accuracy-card">
          <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
            <div style="display:flex; align-items:center; gap:12px;">
              <div style="background:{'rgba(16,185,129,0.12)' if is_dark else '#ecfdf5'}; color:{'#10b981' if is_dark else '#059669'}; border:1px solid {'rgba(16,185,129,0.25)' if is_dark else '#a7f3d0'}; width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;">
                {SVG_SHIELD}
              </div>
              <div>
                <div style="font-size:0.72rem; font-weight:700; color:{TEXT_LABEL}; text-transform:uppercase; letter-spacing:0.08em;">
                  Station-Specific Model Accuracy ({sel_name})
                </div>
                <div style="font-size:0.85rem; font-weight:600; color:{TEXT_MUTED};">
                  Evaluated across {len(apt_test)} chronological test days (October 1 – December 31, 2025)
                </div>
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:24px;">
              <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:{acc_badge_clr};">{apt_acc:.1f}%</div>
                <div style="font-size:0.65rem; color:{TEXT_LABEL}; text-transform:uppercase; font-weight:700;">Prediction Accuracy</div>
              </div>
              <div style="text-align:center; border-left:1px solid {CARD_BORDER}; padding-left:20px;">
                <div style="font-size:1.25rem; font-weight:800; color:{TEXT_MAIN};">{apt_mae:.1f}</div>
                <div style="font-size:0.65rem; color:{TEXT_LABEL}; text-transform:uppercase; font-weight:700;">MAE (Flights/Day)</div>
              </div>
              <div style="text-align:center; border-left:1px solid {CARD_BORDER}; padding-left:20px;">
                <div style="font-size:1.25rem; font-weight:800; color:{TEXT_MAIN};">{apt_rmse:.1f}</div>
                <div style="font-size:0.65rem; color:{TEXT_LABEL}; text-transform:uppercase; font-weight:700;">RMSE Error</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Station Test Set Overlap Chart (Actual vs. Predicted)
        st.markdown(f'<div class="sec-head">{SVG_TARGET} Out-of-Time Test Set Tracking (Observed vs. Model Prediction: Oct–Dec 2025)</div>', unsafe_allow_html=True)
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(
            x=apt_test["date"], y=apt_test["target"],
            name="Observed Actual Flights",
            line=dict(color="#2563eb", width=2.5),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate="<b>%{x|%b %d}</b><br>Actual: %{y:.0f} flights<extra></extra>",
        ))
        fig_test.add_trace(go.Scatter(
            x=apt_test["date"], y=apt_test["prediction"],
            name="MLlib Predicted Volume",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            mode="lines",
            hovertemplate="Predicted: %{y:.1f} flights<extra></extra>",
        ))
        l_test = base_layout(h=280, legend=True)
        fig_test.update_layout(**l_test, yaxis_title="Daily Flight Volume")
        st.plotly_chart(fig_test, use_container_width=True)

    # Full Year Traffic History
    st.markdown(f'<div class="sec-head">{SVG_CHART} Daily Flight Movements History (Full Year 2025)</div>', unsafe_allow_html=True)
    apt_h["roll7"] = apt_h["FLT_TOT_1"].rolling(7, center=True, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=apt_h["FLT_DATE"], y=apt_h["FLT_TOT_1"],
        name="Observed Flights",
        marker=dict(color="rgba(59,130,246,0.35)" if is_dark else "rgba(37,99,235,0.25)", line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d}</b><br>%{y:,} flights<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=apt_h["FLT_DATE"], y=apt_h["roll7"],
        name="7-Day Centered Average",
        line=dict(color=ACCENT_COLOR, width=2.5),
        mode="lines",
        hovertemplate="7-Day Avg: %{y:,.0f}<extra></extra>",
    ))
    l = base_layout(h=300, legend=True)
    fig.update_layout(**l, xaxis_title="", yaxis_title="Flights / Day")
    st.plotly_chart(fig, use_container_width=True)

    hc1, hc2 = st.columns(2, gap="large")

    with hc1:
        st.markdown(f'<div class="sec-head">{SVG_CALENDAR} Monthly Distribution Profile</div>', unsafe_allow_html=True)
        am = (apt_h.groupby(["MONTH_NUM", "MONTH_ABB"])["FLT_TOT_1"]
              .agg(total="sum", avg="mean").reset_index().sort_values("MONTH_NUM"))

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(
            x=am["MONTH_ABB"], y=am["total"],
            name="Total Movements",
            marker=dict(color="rgba(99,102,241,0.65)" if is_dark else "rgba(37,99,235,0.65)", line=dict(width=0)),
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=am["MONTH_ABB"], y=am["avg"],
            name="Daily Average",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(color="#f59e0b", size=6),
        ), secondary_y=True)
        l2 = base_layout(h=270, legend=True)
        fig2.update_layout(**l2)
        fig2.update_yaxes(secondary_y=False, gridcolor=PLOT_GRID, title_text="Total Flights")
        fig2.update_yaxes(secondary_y=True, gridcolor="rgba(0,0,0,0)", title_text="Daily Average")
        st.plotly_chart(fig2, use_container_width=True)

    with hc2:
        st.markdown(f'<div class="sec-head">{SVG_CYCLE} Day-of-Week Operational Pattern</div>', unsafe_allow_html=True)
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_apt = apt_h.groupby("DOW")["FLT_TOT_1"].mean().reindex(dow_order).reset_index()
        dow_apt.columns = ["Day", "Avg"]

        fig3 = go.Figure(go.Bar(
            x=dow_apt["Day"], y=dow_apt["Avg"],
            marker=dict(
                color=dow_apt["Avg"],
                colorscale=GREENS_SCALE,
                line=dict(width=0)
            ),
            text=dow_apt["Avg"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="#10b981" if is_dark else "#059669", size=9),
            hovertemplate="<b>%{x}</b><br>Avg %{y:,.0f} flights<extra></extra>",
        ))
        fig3.update_layout(**base_layout(h=270))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL BENCHMARKS (Fix Image 2 with Custom HTML Table)
# ══════════════════════════════════════════════════════════════════════════════
with T3:
    st.markdown(f'<div class="sec-head">{SVG_TARGET} Model Evaluation Matrix — Out-of-Time Test Set (28,402 records · Oct–Dec 2025)</div>', unsafe_allow_html=True)
    
    # Custom HTML Table for Model Evaluation Matrix (Fix Image 2 - Zero black box in light mode)
    badge_html = ('<span style="background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.35); padding:2px 8px; border-radius:12px; font-size:0.68rem; font-weight:700; margin-left:8px;">BEST MODEL</span>'
                  if is_dark else
                  '<span style="background:#dcfce7; color:#15803d; border:1px solid #86efac; padding:2px 8px; border-radius:12px; font-size:0.68rem; font-weight:700; margin-left:8px;">BEST MODEL</span>')

    matrix_rows = [
        {"model": "Previous-Day Baseline", "mae": "17.43", "rmse": "33.75", "wmape": "12.09%", "r2": "0.9809", "imp": "0.0% (Reference)", "badge": ""},
        {"model": "Linear Regression", "mae": "10.37", "rmse": "21.17", "wmape": "7.20%", "r2": "0.9925", "imp": "+40.5%", "badge": badge_html},
        {"model": "Gradient-Boosted Trees (GBT)", "mae": "12.99", "rmse": "25.44", "wmape": "9.01%", "r2": "0.9891", "imp": "+25.5%", "badge": ""},
        {"model": "Random Forest Regressor", "mae": "13.85", "rmse": "26.80", "wmape": "9.62%", "r2": "0.9879", "imp": "+20.5%", "badge": ""},
    ]

    table_rows_html = "".join([
        f'<tr style="background:{("rgba(16,185,129,0.08)" if is_dark else "#f0fdf4") if "Linear Regression" in r["model"] else "transparent"}; font-weight:{"700" if "Linear Regression" in r["model"] else "500"};">'
        f'<td style="font-weight:700; color:{("#10b981" if is_dark else "#15803d") if "Linear Regression" in r["model"] else TEXT_MAIN};">{r["model"]} {r["badge"]}</td>'
        f'<td style="text-align:center; color:{("#10b981" if is_dark else "#15803d") if "Linear Regression" in r["model"] else TEXT_MAIN}; font-weight:{"700" if "Linear Regression" in r["model"] else "500"};">{r["mae"]}</td>'
        f'<td style="text-align:center; color:{("#10b981" if is_dark else "#15803d") if "Linear Regression" in r["model"] else TEXT_MAIN}; font-weight:{"700" if "Linear Regression" in r["model"] else "500"};">{r["rmse"]}</td>'
        f'<td style="text-align:center; color:{("#10b981" if is_dark else "#15803d") if "Linear Regression" in r["model"] else TEXT_MAIN}; font-weight:{"700" if "Linear Regression" in r["model"] else "500"};">{r["wmape"]}</td>'
        f'<td style="text-align:center; color:{("#10b981" if is_dark else "#15803d") if "Linear Regression" in r["model"] else TEXT_MAIN}; font-weight:700;">{r["r2"]}</td>'
        f'<td style="text-align:center; color:{("#10b981" if is_dark else "#15803d") if "Linear Regression" in r["model"] else ACCENT_COLOR}; font-weight:700;">{r["imp"]}</td>'
        f'</tr>'
        for r in matrix_rows
    ])

    matrix_table_html = (
        f'<table class="custom-table" style="margin-bottom:24px;">'
        f'<thead><tr>'
        f'<th>Model Specification</th>'
        f'<th style="text-align:center;">MAE (Flights/Day) ↓</th>'
        f'<th style="text-align:center;">RMSE (Flights/Day) ↓</th>'
        f'<th style="text-align:center;">WMAPE (%) ↓</th>'
        f'<th style="text-align:center;">R² Score ↑</th>'
        f'<th style="text-align:center;">MAE Improvement vs Baseline</th>'
        f'</tr></thead>'
        f'<tbody>{table_rows_html}</tbody>'
        f'</table>'
    )
    st.markdown(matrix_table_html, unsafe_allow_html=True)

    mc1, mc2 = st.columns(2, gap="large")

    with mc1:
        st.markdown(f'<div class="sec-head">{SVG_CHART} Loss Metric Comparison (MAE &amp; RMSE)</div>', unsafe_allow_html=True)
        m_names = ["Baseline", "Linear Regression", "GBT Regressor", "Random Forest"]
        mae_v   = [17.43, 10.37, 12.99, 13.85]
        rmse_v  = [33.75, 21.17, 25.44, 26.80]
        bar_clr = ["#475569", "#3b82f6", "#8b5cf6", "#a855f7"] if is_dark else ["#64748b", "#2563eb", "#7c3aed", "#9333ea"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="MAE (Mean Absolute Error)", x=m_names, y=mae_v,
            marker_color=bar_clr, marker_line_width=0,
            text=[f"{v:.2f}" for v in mae_v],
            textposition="outside", textfont=dict(color=TEXT_MUTED, size=11),
            hovertemplate="MAE: %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="RMSE (Root Mean Squared Error)", x=m_names, y=rmse_v,
            marker_color=[c + "88" for c in bar_clr], marker_line_width=0,
            text=[f"{v:.2f}" for v in rmse_v],
            textposition="outside", textfont=dict(color=TEXT_MUTED, size=11),
            hovertemplate="RMSE: %{y:.2f}<extra></extra>",
        ))
        l3 = base_layout(h=300, legend=True)
        fig.update_layout(**l3, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with mc2:
        st.markdown(f'<div class="sec-head">{SVG_LAYERS} Feature Attribution (Gradient-Boosted Trees)</div>', unsafe_allow_html=True)
        fi_labels = ["Lag-7 (Same Day Last Week)", "7-Day Rolling Mean", "Lag-1 (Previous Day)", "Month (Seasonality)", "Airport Identifier (One-Hot)", "Day of Week", "Residual Features"]
        fi_vals   = [95.52, 1.74, 0.69, 0.76, 0.79, 0.23, 0.27]
        fi_colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#64748b"] if is_dark else ["#2563eb", "#4f46e5", "#7c3aed", "#db2777", "#d97706", "#059669", "#64748b"]

        fig = go.Figure(go.Bar(
            x=fi_vals,
            y=fi_labels,
            orientation="h",
            marker=dict(color=fi_colors, line=dict(width=0)),
            text=[f"{v:.2f}%" for v in fi_vals],
            textposition="outside",
            textfont=dict(color=TEXT_MUTED, size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
        ))
        lfi = base_layout(h=300)
        lfi["xaxis"]["title"] = dict(text="Predictive Contribution (%)", font=dict(color=PLOT_TITLE_FONT, size=11))
        lfi["xaxis"]["range"] = [0, 110]
        fig.update_layout(**lfi)
        st.plotly_chart(fig, use_container_width=True)

    r4c1, r4c2 = st.columns(2, gap="large")

    with r4c1:
        st.markdown(f'<div class="sec-head">{SVG_TARGET} Coefficient of Determination (R² Score Comparison)</div>', unsafe_allow_html=True)
        r2_m = ["Naive Baseline", "Random Forest", "Gradient-Boosted Trees", "Linear Regression (Production)"]
        r2_v = [0.9809, 0.9879, 0.9891, 0.9925]
        r2_c = ["#475569", "#8b5cf6", "#6366f1", "#10b981"] if is_dark else ["#64748b", "#7c3aed", "#4f46e5", "#059669"]

        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=r2_m, y=r2_v,
            marker=dict(color=r2_c, line=dict(width=0)),
            text=[f"{v:.4f}" for v in r2_v],
            textposition="outside",
            textfont=dict(color=TEXT_MAIN, size=11),
            hovertemplate="<b>%{x}</b><br>R² = %{y:.4f}<extra></extra>",
        ))
        fig_r2.add_shape(type="line", x0=-0.5, x1=3.5, y0=0.9809, y1=0.9809,
                         line=dict(color="#f87171" if is_dark else "#dc2626", dash="dot", width=2))
        fig_r2.add_annotation(x=3.5, y=0.981, text="  Baseline R² (0.9809)",
                              showarrow=False, font=dict(color="#f87171" if is_dark else "#dc2626", size=10), xanchor="left")
        l4 = base_layout(h=290)
        l4["yaxis"]["range"] = [0.975, 0.997]
        fig_r2.update_layout(**l4)
        st.plotly_chart(fig_r2, use_container_width=True)

    with r4c2:
        # Prediction Residual Error Distribution (Fix Images 3 & 4)
        st.markdown(f'<div class="sec-head">{SVG_CHART} Prediction Residual Error Distribution (Test Set)</div>', unsafe_allow_html=True)
        if not test_df.empty:
            errors = (test_df["target"] - test_df["prediction"]).clip(-50, 50)
            fig_err = go.Figure(go.Histogram(
                x=errors,
                nbinsx=40,
                marker=dict(color="#3b82f6" if is_dark else "#2563eb", line=dict(color=BG_APP, width=1)),
                hovertemplate="Error: %{x:.1f}<br>Count: %{y:,} flights<extra></extra>",
            ))
            l_err = base_layout(h=290)
            l_err["xaxis"]["title"] = dict(text="Prediction Error (Actual - Predicted Flights)", font=dict(color=PLOT_TITLE_FONT, size=11))
            l_err["yaxis"]["title"] = dict(text="Frequency", font=dict(color=PLOT_TITLE_FONT, size=11))
            fig_err.update_layout(**l_err)
            fig_err.add_vline(x=0, line_dash="dash", line_color="#10b981" if is_dark else "#059669", line_width=2)
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.info("Residual data unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CLUSTER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with T4:
    st.markdown(f'<div class="sec-head">{SVG_LAYERS} K-Means Airport Segmentation — 3 Operational Tiers</div>', unsafe_allow_html=True)

    tier_colors = {
        "Major Hub":        "#6366f1" if is_dark else "#4f46e5",
        "Medium Hub":       "#f59e0b" if is_dark else "#d97706",
        "Small / Regional": "#10b981" if is_dark else "#059669",
    }
    fig = px.scatter(
        profile, x="avg_day", y="variab",
        color="tier",
        size="total",
        hover_name="APT_NAME",
        hover_data={
            "STATE_NAME": True,
            "avg_day":    ":.0f",
            "peak":       ":.0f",
            "tier":       False,
            "color":      False,
            "total":      False,
            "variab":     False,
        },
        color_discrete_map=tier_colors,
        labels={"avg_day": "Average Daily Flights", "variab": "Daily Variability (Std Dev)", "tier": "Operational Tier"},
        size_max=32,
    )
    fig.update_traces(
        marker=dict(line=dict(width=1.5, color=BG_APP)),
        hovertemplate="<b>%{hovertext}</b><br>State: %{customdata[0]}<br>Daily Avg: %{customdata[1]:.0f}<br>Peak Record: %{customdata[2]:.0f}<extra></extra>",
    )
    l5 = base_layout(h=420, legend=True)
    l5["legend"]["y"] = 1.06
    fig.update_layout(**l5)
    st.plotly_chart(fig, use_container_width=True)

    cc1, cc2 = st.columns([1, 1.8], gap="large")

    with cc1:
        st.markdown(f'<div class="sec-head">{SVG_CHART} Airport Tier Distribution</div>', unsafe_allow_html=True)
        tc = profile["tier"].value_counts().reset_index()
        tc.columns = ["Tier", "Count"]
        fig = go.Figure(go.Pie(
            labels=tc["Tier"], values=tc["Count"], hole=0.62,
            marker=dict(colors=[tier_colors[t] for t in tc["Tier"]],
                        line=dict(color=BG_APP, width=2.5)),
            textfont=dict(size=11, color=TEXT_MAIN),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>%{value} airports<extra></extra>",
        ))
        fig.add_annotation(text="331<br>Airports", x=0.5, y=0.5,
                           font=dict(size=13, color=TEXT_MUTED, family="Plus Jakarta Sans"), showarrow=False)
        fig.update_layout(**base_layout(h=290), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Total Volume Share per Tier
        st.markdown(f'<div class="sec-head">{SVG_DATABASE} Cumulative Volume Share by Tier</div>', unsafe_allow_html=True)
        tv = profile.groupby("tier")["total"].sum().reset_index()
        tv["total_M"] = (tv["total"] / 1e6).round(2)
        fig_tv = go.Figure(go.Bar(
            x=tv["tier"], y=tv["total_M"],
            marker=dict(color=[tier_colors[t] for t in tv["tier"]], line=dict(width=0)),
            text=tv["total_M"].apply(lambda x: f"{x:.2f}M"),
            textposition="outside",
            textfont=dict(color=TEXT_MUTED, size=10),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}M Total Flights<extra></extra>",
        ))
        fig_tv.update_layout(**base_layout(h=220), yaxis_title="Total Movements (M)")
        st.plotly_chart(fig_tv, use_container_width=True)

    with cc2:
        st.markdown(f'<div class="sec-head">{SVG_AIRPORT} Leading Airports by Operational Tier</div>', unsafe_allow_html=True)
        tier_colors_map = tier_colors
        
        for tier_name in ["Major Hub", "Medium Hub", "Small / Regional"]:
            grp = (profile[profile["tier"] == tier_name]
                   .nlargest(4, "avg_day")
                   [["APT_NAME", "STATE_NAME", "avg_day", "peak", "total"]]
                   .reset_index(drop=True))
            tc = tier_colors_map[tier_name]
            
            rows_html = "".join([
                f'<tr>'
                f'<td style="font-weight:600; color:{TEXT_MAIN};">{row["APT_NAME"]}</td>'
                f'<td style="color:{TEXT_MUTED};">{row["STATE_NAME"]}</td>'
                f'<td style="color:{tc}; font-weight:700; text-align:center;">{row["avg_day"]:,.0f}</td>'
                f'<td style="text-align:center; color:{TEXT_MAIN}; font-weight:600;">{int(row["peak"]):,}</td>'
                f'<td style="color:{TEXT_MUTED}; text-align:center;">{int(row["total"]):,}</td>'
                f'</tr>'
                for _, row in grp.iterrows()
            ])

            tier_table_html = (
                f'<div style="margin-bottom:16px;">'
                f'<div style="font-size:0.80rem; font-weight:700; color:{tc}; margin-bottom:6px; display:flex; align-items:center; gap:6px;">'
                f'<span style="width:6px; height:6px; border-radius:50%; background:{tc}; display:inline-block;"></span> {tier_name}'
                f'</div>'
                f'<table class="custom-table">'
                f'<thead><tr>'
                f'<th>Airport</th>'
                f'<th>Country</th>'
                f'<th style="text-align:center;">Avg / Day</th>'
                f'<th style="text-align:center;">Peak Day</th>'
                f'<th style="text-align:center;">Annual Total</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>'
                f'</div>'
            )
            st.markdown(tier_table_html, unsafe_allow_html=True)

# ─── Clean Production Footer ───────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  Eurocontrol Air Traffic Analytics Platform · Distributed PySpark MLlib Predictive Engine<br>
  Author: <strong>Rai Yashmit</strong><br>
  Data Source: Official EUROCONTROL Historical Trajectory Records (116,020 records across 331 airports)
</div>
""", unsafe_allow_html=True)
