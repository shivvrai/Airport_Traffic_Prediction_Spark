"""
✈️ Airport Traffic Prediction Dashboard  –  INT315 Cluster Computing
Author: Rai Yashmit Ashok | Reg: 12313527
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airport Traffic Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, * { font-family: 'Inter', sans-serif !important; }

/* Background */
.stApp {
    background: #050816;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(168,85,247,0.12) 0%, transparent 50%);
}

/* Sidebar collapse (we don't use it) */
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* Main content padding */
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 100% !important; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.2) 0%,
        rgba(168,85,247,0.15) 50%,
        rgba(59,130,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; top:-60px; right:-60px;
    width:250px; height:250px;
    background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 70%);
    pointer-events:none;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    border: 1px solid rgba(99,102,241,0.5);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 10px 0;
}
.hero-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
    max-width: 680px;
    line-height: 1.6;
}
.hero-right {
    text-align: right;
}
.hero-stat {
    font-size: 2.8rem;
    font-weight: 900;
    color: #c084fc;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── KPI CARDS ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.kpi {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    cursor: default;
}
.kpi:hover { transform: translateY(-2px); }
.kpi.blue  { border-top: 3px solid #6366f1; }
.kpi.purple{ border-top: 3px solid #a855f7; }
.kpi.green { border-top: 3px solid #10b981; }
.kpi.amber { border-top: 3px solid #f59e0b; }
.kpi.pink  { border-top: 3px solid #ec4899; }
.kpi-emoji { font-size: 1.6rem; margin-bottom: 10px; }
.kpi-val {
    font-size: 2rem;
    font-weight: 900;
    color: #f1f5f9;
    letter-spacing: -0.03em;
    line-height: 1;
}
.kpi-label {
    font-size: 0.67rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-top: 8px;
}
.kpi-sub {
    font-size: 0.76rem;
    margin-top: 6px;
    font-weight: 600;
}
.kpi.blue  .kpi-sub { color: #818cf8; }
.kpi.purple .kpi-sub { color: #c084fc; }
.kpi.green .kpi-sub { color: #34d399; }
.kpi.amber .kpi-sub { color: #fbbf24; }
.kpi.pink  .kpi-sub { color: #f472b6; }

/* ── SECTION TITLE ── */
.sec-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e2433, transparent);
}

/* ── TAB PILLS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 6px;
    gap: 6px;
}
button[role="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 9px 22px !important;
    border: none !important;
    transition: all 0.15s !important;
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}
button[role="tab"]:hover:not([aria-selected="true"]) {
    background: #1e2433 !important;
    color: #94a3b8 !important;
}

/* ── METRIC BOXES ── */
[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 14px;
    padding: 16px 20px !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] { color: #34d399 !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── SELECT BOX ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
[data-testid="stSelectbox"] label {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] iframe { border-radius: 12px; }
.dvn-scroller { background: #0d1117 !important; }

/* ── AIRPORT CARD ── */
.apt-card {
    background: linear-gradient(135deg, #0d1117, #111827);
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.apt-card-name {
    font-size: 1.3rem;
    font-weight: 800;
    color: #f1f5f9;
}
.apt-card-icao {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 6px;
    margin-top: 4px;
}
.apt-card-country {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
}

/* ── FORECAST BANNER ── */
.forecast-banner {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.15), rgba(168,85,247,0.1));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    margin-top: 12px;
}
.forecast-num {
    font-size: 2.8rem;
    font-weight: 900;
    color: #a5b4fc;
    letter-spacing: -0.04em;
    line-height: 1;
}
.forecast-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
    font-weight: 700;
}
.forecast-date {
    font-size: 0.85rem;
    color: #34d399;
    font-weight: 600;
    margin-top: 8px;
}

/* ── FOOTER ── */
.footer {
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #1e2433;
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ─── Chart theme ──────────────────────────────────────────────────────────────
PLOT_BG   = "#0d1117"
PAPER_BG  = "rgba(0,0,0,0)"
FONT_CLR  = "#94a3b8"
GRID_CLR  = "#1e2433"
AXIS_CLR  = "#2d3748"

def base_layout(h=320, legend=False):
    d = dict(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=FONT_CLR, size=11),
        margin=dict(l=8, r=8, t=28, b=8),
        height=h,
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=AXIS_CLR, linecolor=AXIS_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=AXIS_CLR, linecolor=AXIS_CLR),
    )
    if legend:
        d["legend"] = dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor=GRID_CLR, borderwidth=1,
            font=dict(size=11), orientation="h", y=1.08
        )
    return d

ACCENT   = ["#6366f1","#a855f7","#10b981","#f59e0b","#ec4899","#06b6d4"]
BLUES    = [[0,"#1e1b4b"],[0.5,"#6366f1"],[1,"#a5b4fc"]]
PURPLES  = [[0,"#2e1065"],[0.5,"#a855f7"],[1,"#e9d5ff"]]
GREENS   = [[0,"#022c22"],[0.5,"#10b981"],[1,"#6ee7b7"]]

# ─── Paths ────────────────────────────────────────────────────────────────────
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

# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_hist():
    df = pd.read_csv(DATA_CSV, parse_dates=["FLT_DATE"])
    df["FLT_TOT_1"] = pd.to_numeric(df["FLT_TOT_1"], errors="coerce")
    df = df.dropna(subset=["FLT_TOT_1","FLT_DATE"])
    df["MONTH_NUM"]  = df["FLT_DATE"].dt.month
    df["MONTH_ABB"]  = df["FLT_DATE"].dt.strftime("%b")
    df["DOW"]        = df["FLT_DATE"].dt.day_name()
    df["DOW_NUM"]    = df["FLT_DATE"].dt.dayofweek
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
def build_profiles(hist):
    p = hist.groupby(["APT_ICAO","APT_NAME","STATE_NAME"]).agg(
        total   =("FLT_TOT_1","sum"),
        avg_day =("FLT_TOT_1","mean"),
        peak    =("FLT_TOT_1","max"),
        variab  =("FLT_TOT_1","std"),
        days    =("FLT_DATE","nunique"),
    ).reset_index()
    p["avg_day"] = p["avg_day"].round(1)
    p["peak"]    = p["peak"].round(0).astype(int)
    p["variab"]  = p["variab"].fillna(0).round(1)
    return p

@st.cache_data(show_spinner=False)
def cluster_airports(profile):
    feats = profile[["avg_day","variab","peak"]].fillna(0)
    X = StandardScaler().fit_transform(feats)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    profile = profile.copy()
    profile["_c"] = labels
    means = profile.groupby("_c")["avg_day"].mean().sort_values()
    tier_map  = {i:t for i,t in zip(means.index,["Small / Regional","Medium Hub","Major Hub"])}
    color_map = {i:c for i,c in zip(means.index,["#10b981","#f59e0b","#6366f1"])}
    profile["tier"]  = profile["_c"].map(tier_map)
    profile["color"] = profile["_c"].map(color_map)
    return profile

# ─── Load ─────────────────────────────────────────────────────────────────────
hist     = load_hist()
forecast = load_forecast()
metrics  = load_metrics()
profile  = cluster_airports(build_profiles(hist))

# ─── HERO ─────────────────────────────────────────────────────────────────────
c_left, c_right = st.columns([3,1])
with c_left:
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">INT315 · Cluster Computing · PySpark MLlib</div>
      <h1 class="hero-title">✈️ Airport Traffic<br>Prediction Dashboard</h1>
      <p class="hero-desc">
        Distributed analysis &amp; next-day forecasting of IFR flight movements across
        <strong style="color:#a5b4fc">331 European airports in 42 countries</strong>
        using Apache Spark MLlib — Linear Regression · Random Forest · Gradient-Boosted Trees.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─── KPI CARDS ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
  <div class="kpi blue">
    <div class="kpi-emoji">🗃️</div>
    <div class="kpi-val">116,020</div>
    <div class="kpi-label">Records Processed</div>
    <div class="kpi-sub">365 calendar days · 2025</div>
  </div>
  <div class="kpi purple">
    <div class="kpi-emoji">🏭</div>
    <div class="kpi-val">331</div>
    <div class="kpi-label">European Airports</div>
    <div class="kpi-sub">across 42 countries</div>
  </div>
  <div class="kpi green">
    <div class="kpi-emoji">🎯</div>
    <div class="kpi-val">0.9925</div>
    <div class="kpi-label">R² Score (Test Set)</div>
    <div class="kpi-sub">Linear Regression · Best</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-emoji">📉</div>
    <div class="kpi-val">40.5%</div>
    <div class="kpi-label">MAE Improvement</div>
    <div class="kpi-sub">17.43 → 10.37 flights/day</div>
  </div>
  <div class="kpi pink">
    <div class="kpi-emoji">🔁</div>
    <div class="kpi-val">95.5%</div>
    <div class="kpi-label">Weekly Lag Importance</div>
    <div class="kpi-sub">dominant signal · lag-7</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
T1, T2, T3, T4 = st.tabs([
    "📊  Traffic Overview",
    "🔍  Airport Explorer",
    "🤖  Model Performance",
    "🗂️  Airport Clustering",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TRAFFIC OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with T1:
    r1c1, r1c2 = st.columns([3,2], gap="large")

    with r1c1:
        st.markdown('<p class="sec-title">📅 Monthly Flight Totals across Europe (2025)</p>', unsafe_allow_html=True)
        monthly = (hist.groupby(["MONTH_NUM","MONTH_ABB"])["FLT_TOT_1"]
                   .sum().reset_index().sort_values("MONTH_NUM"))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["MONTH_ABB"], y=monthly["FLT_TOT_1"],
            marker=dict(
                color=monthly["FLT_TOT_1"],
                colorscale=BLUES,
                line=dict(width=0)
            ),
            text=monthly["FLT_TOT_1"].apply(lambda x: f"{x/1e6:.2f}M"),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} flights<extra></extra>",
        ))
        fig.update_layout(**base_layout(h=310), yaxis_title="Total Flights")
        fig.update_xaxes(tickfont=dict(color="#e2e8f0", size=11))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<p class="sec-title">🏆 Top 10 Busiest Airports</p>', unsafe_allow_html=True)
        top10 = profile.nlargest(10,"total")[["APT_NAME","total","avg_day"]].sort_values("total")
        top10["name_short"] = top10["APT_NAME"].str.replace(" - "," ").str[:22]
        top10["total_k"] = (top10["total"]/1000).round(0).astype(int)

        fig = go.Figure(go.Bar(
            x=top10["total_k"], y=top10["name_short"],
            orientation="h",
            marker=dict(
                color=top10["total_k"],
                colorscale=PURPLES,
                line=dict(width=0)
            ),
            text=top10["total_k"].apply(lambda x: f"{x:,}K"),
            textposition="outside",
            textfont=dict(color="#c084fc", size=10, family="Inter"),
            hovertemplate="<b>%{y}</b><br>%{x:,}K flights<extra></extra>",
        ))
        fig.update_layout(**base_layout(h=310))
        fig.update_yaxes(tickfont=dict(color="#e2e8f0", size=10))
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2, gap="large")

    with r2c1:
        st.markdown('<p class="sec-title">📆 Total Flights by Day of Week (All Europe)</p>', unsafe_allow_html=True)
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        # SUM across all airports — shows total European airspace load per day of week
        dow = hist.groupby("DOW")["FLT_TOT_1"].sum().reindex(dow_order).reset_index()
        dow.columns = ["Day","Total"]
        dow["TotalM"] = (dow["Total"] / 1e6).round(2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dow["Day"], y=dow["TotalM"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.12)",
            line=dict(color="#6366f1", width=3),
            marker=dict(color="#a5b4fc", size=9,
                        line=dict(color="#312e81", width=2)),
            text=dow["TotalM"].apply(lambda x: f"{x:.2f}M"),
            textposition="top center",
            textfont=dict(color="#a5b4fc", size=9),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}M total flights across all airports<extra></extra>",
        ))
        l_dow = base_layout(h=260)
        l_dow["yaxis"]["title"] = "Millions of Flights"
        fig.update_layout(**l_dow)
        fig.update_xaxes(tickfont=dict(color="#e2e8f0", size=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown('<p class="sec-title">🌍 Top 12 Countries by Total Volume</p>', unsafe_allow_html=True)
        ctry = (hist.groupby("STATE_NAME")["FLT_TOT_1"].sum()
                .nlargest(12).reset_index().sort_values("FLT_TOT_1"))
        ctry.columns = ["Country","Total"]
        ctry["TotalM"] = (ctry["Total"]/1e6).round(2)

        fig = go.Figure(go.Bar(
            x=ctry["TotalM"], y=ctry["Country"],
            orientation="h",
            marker=dict(color=ctry["TotalM"], colorscale=GREENS, line=dict(width=0)),
            text=ctry["TotalM"].apply(lambda x: f"{x:.1f}M"),
            textposition="outside",
            textfont=dict(color="#34d399", size=9),
            hovertemplate="<b>%{y}</b><br>%{x:.2f}M flights<extra></extra>",
        ))
        fig.update_layout(**base_layout(h=260))
        fig.update_yaxes(tickfont=dict(color="#e2e8f0", size=9))
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — AIRPORT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with T2:
    # Airport selector INSIDE this tab — visible and obvious
    airports_sorted = sorted(hist["APT_NAME"].dropna().unique())
    default_idx = airports_sorted.index("London - Heathrow") if "London - Heathrow" in airports_sorted else 0

    col_sel, col_gap = st.columns([2, 3])
    with col_sel:
        sel_name = st.selectbox(
            "Select an Airport to Explore",
            airports_sorted,
            index=default_idx,
            key="airport_sel",
        )

    # Filter data
    sel_icao = hist[hist["APT_NAME"] == sel_name]["APT_ICAO"].iloc[0]
    sel_country = hist[hist["APT_ICAO"] == sel_icao]["STATE_NAME"].iloc[0]
    apt_h = hist[hist["APT_ICAO"] == sel_icao].sort_values("FLT_DATE").copy()
    apt_p = profile[profile["APT_ICAO"] == sel_icao]
    apt_f = forecast[forecast["APT_ICAO"] == sel_icao]

    # Airport header card + forecast in same row
    ac1, ac2 = st.columns([3, 1], gap="large")

    with ac1:
        tier_color  = apt_p["color"].iloc[0] if not apt_p.empty else "#6366f1"
        tier_name   = apt_p["tier"].iloc[0]  if not apt_p.empty else "—"
        total_f     = int(apt_p["total"].iloc[0]) if not apt_p.empty else 0
        avg_f       = apt_p["avg_day"].iloc[0]    if not apt_p.empty else 0
        peak_f      = int(apt_p["peak"].iloc[0])  if not apt_p.empty else 0

        st.markdown(f"""
        <div class="apt-card">
          <div style="display:flex; align-items:flex-start; gap:16px">
            <div style="font-size:2.4rem">🏢</div>
            <div>
              <div class="apt-card-name">{sel_name}</div>
              <span class="apt-card-icao">{sel_icao}</span>
              <div class="apt-card-country">📍 {sel_country}</div>
            </div>
            <div style="margin-left:auto; text-align:right">
              <div style="color:{tier_color}; font-size:0.75rem; font-weight:700;
                           background:rgba(255,255,255,0.05); padding:4px 12px;
                           border-radius:20px; border:1px solid {tier_color}44">
                ● {tier_name}
              </div>
            </div>
          </div>
          <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:18px">
            <div style="text-align:center">
              <div style="font-size:1.6rem; font-weight:900; color:#f1f5f9">{total_f:,}</div>
              <div style="font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin-top:4px">Total 2025 Flights</div>
            </div>
            <div style="text-align:center; border-left:1px solid #1e2433">
              <div style="font-size:1.6rem; font-weight:900; color:#f1f5f9">{avg_f:,.1f}</div>
              <div style="font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin-top:4px">Avg Flights / Day</div>
            </div>
            <div style="text-align:center; border-left:1px solid #1e2433">
              <div style="font-size:1.6rem; font-weight:900; color:#f1f5f9">{peak_f:,}</div>
              <div style="font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin-top:4px">Peak Single Day</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ac2:
        if not apt_f.empty:
            row  = apt_f.iloc[0]
            pred = row["prediction"]; prev = row["yesterday"]
            chg  = ((pred - prev) / prev * 100) if prev > 0 else 0
            chg_color = "#34d399" if chg >= 0 else "#f87171"
            chg_arrow = "▲" if chg >= 0 else "▼"
            st.markdown(f"""
            <div class="forecast-banner">
              <div class="forecast-label">🔮 Next-Day Forecast</div>
              <div class="forecast-num">{pred:,.0f}</div>
              <div class="forecast-label">Predicted Flights</div>
              <div class="forecast-date">2026-01-01</div>
              <div style="margin-top:12px; font-size:0.8rem; color:{chg_color}; font-weight:700">
                {chg_arrow} {abs(chg):.1f}% vs prev day ({int(prev):,} flights)
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No forecast data for this airport.")

    # Daily traffic history
    st.markdown('<p class="sec-title">📈 Daily Traffic History — Full Year 2025</p>', unsafe_allow_html=True)
    apt_h["roll7"] = apt_h["FLT_TOT_1"].rolling(7, center=True, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=apt_h["FLT_DATE"], y=apt_h["FLT_TOT_1"],
        name="Daily Flights",
        marker=dict(color="rgba(99,102,241,0.35)", line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d}</b><br>%{y:,} flights<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=apt_h["FLT_DATE"], y=apt_h["roll7"],
        name="7-Day Rolling Avg",
        line=dict(color="#a5b4fc", width=2.5),
        mode="lines",
        hovertemplate="7-day avg: %{y:,.0f}<extra></extra>",
    ))
    l = base_layout(h=300, legend=True)
    fig.update_layout(**l, xaxis_title="", yaxis_title="Flights per Day")
    st.plotly_chart(fig, use_container_width=True)

    hc1, hc2 = st.columns(2, gap="large")

    with hc1:
        st.markdown('<p class="sec-title">📅 Monthly Breakdown</p>', unsafe_allow_html=True)
        am = (apt_h.groupby(["MONTH_NUM","MONTH_ABB"])["FLT_TOT_1"]
              .agg(total="sum", avg="mean").reset_index().sort_values("MONTH_NUM"))

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(
            x=am["MONTH_ABB"], y=am["total"],
            name="Total Flights",
            marker=dict(color="rgba(99,102,241,0.7)", line=dict(width=0)),
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=am["MONTH_ABB"], y=am["avg"],
            name="Daily Avg",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(color="#fde68a", size=7),
        ), secondary_y=True)
        l2 = base_layout(h=270, legend=True)
        fig2.update_layout(**l2)
        fig2.update_yaxes(secondary_y=False,  gridcolor=GRID_CLR, title_text="Total")
        fig2.update_yaxes(secondary_y=True,   gridcolor="rgba(0,0,0,0)", title_text="Daily Avg")
        st.plotly_chart(fig2, use_container_width=True)

    with hc2:
        st.markdown('<p class="sec-title">📆 Day-of-Week Traffic Pattern</p>', unsafe_allow_html=True)
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_apt = apt_h.groupby("DOW")["FLT_TOT_1"].mean().reindex(dow_order).reset_index()
        dow_apt.columns = ["Day","Avg"]

        fig3 = go.Figure(go.Bar(
            x=dow_apt["Day"], y=dow_apt["Avg"],
            marker=dict(
                color=dow_apt["Avg"],
                colorscale=[[0,"#134e4a"],[0.5,"#10b981"],[1,"#6ee7b7"]],
                line=dict(width=0)
            ),
            text=dow_apt["Avg"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="#34d399", size=9),
            hovertemplate="<b>%{x}</b><br>Avg %{y:,.0f} flights<extra></extra>",
        ))
        fig3.update_layout(**base_layout(h=270))
        fig3.update_xaxes(tickfont=dict(color="#e2e8f0", size=10))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with T3:
    st.markdown('<p class="sec-title">📋 Model Benchmark — Out-of-Time Test Set (28,402 records · Oct–Dec 2025)</p>', unsafe_allow_html=True)
    st.dataframe(
        metrics, use_container_width=True, hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "MAE (Flights/Day)": st.column_config.NumberColumn("MAE ↓", format="%.2f"),
            "RMSE (Flights/Day)": st.column_config.NumberColumn("RMSE ↓", format="%.2f"),
            "WMAPE (%)": st.column_config.TextColumn("WMAPE ↓"),
            "R2 Score": st.column_config.TextColumn("R² ↑"),
            "Improvement vs Baseline (MAE)": st.column_config.TextColumn("MAE Improvement"),
        }
    )

    mc1, mc2 = st.columns(2, gap="large")

    with mc1:
        st.markdown('<p class="sec-title">📊 MAE & RMSE Comparison</p>', unsafe_allow_html=True)
        m_names = ["Baseline","Linear Reg","GBT","Random Forest"]
        mae_v   = [17.43, 10.37, 12.99, 13.85]
        rmse_v  = [33.75, 21.17, 25.44, 26.80]
        bar_clr = ["#334155","#6366f1","#8b5cf6","#a855f7"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="MAE", x=m_names, y=mae_v,
            marker_color=bar_clr, marker_line_width=0,
            text=[f"{v:.2f}" for v in mae_v],
            textposition="outside", textfont=dict(color="#a5b4fc", size=11),
            hovertemplate="MAE: %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="RMSE", x=m_names, y=rmse_v,
            marker_color=[c+"88" for c in bar_clr], marker_line_width=0,
            text=[f"{v:.2f}" for v in rmse_v],
            textposition="outside", textfont=dict(color="#c084fc", size=11),
            hovertemplate="RMSE: %{y:.2f}<extra></extra>",
        ))
        l3 = base_layout(h=300, legend=True)
        fig.update_layout(**l3, barmode="group")
        fig.update_xaxes(tickfont=dict(color="#e2e8f0", size=11))
        st.plotly_chart(fig, use_container_width=True)

    with mc2:
        st.markdown('<p class="sec-title">📊 Model Feature Importance (Gradient-Boosted Trees)</p>', unsafe_allow_html=True)
        st.caption("Which input features matter most for predicting flight volumes?")
        fi_labels = ["Lag-7 — flights 7 days ago","7-Day Rolling Avg","Lag-1 — yesterday's flights","Month (seasonality)","Airport identity","Day of week","Other"]
        fi_vals   = [95.52, 1.74, 0.69, 0.76, 0.79, 0.23, 0.27]
        fi_colors = ["#6366f1","#8b5cf6","#a855f7","#ec4899","#f59e0b","#10b981","#334155"]

        fig = go.Figure(go.Bar(
            x=fi_vals,
            y=fi_labels,
            orientation="h",
            marker=dict(color=fi_colors, line=dict(width=0)),
            text=[f"{v:.2f}%" for v in fi_vals],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
        ))
        lfi = base_layout(h=300)
        lfi["xaxis"]["title"] = "Importance (%)"
        lfi["xaxis"]["range"] = [0, 110]
        fig.update_layout(**lfi)
        fig.update_yaxes(tickfont=dict(color="#e2e8f0", size=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-title">📈 R² Score — All Models vs Baseline</p>', unsafe_allow_html=True)
    r2_m = ["Baseline","Random Forest","Gradient-Boosted Trees","⭐ Linear Regression"]
    r2_v = [0.9809, 0.9879, 0.9891, 0.9925]
    r2_c = ["#334155","#8b5cf6","#6366f1","#10b981"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=r2_m, y=r2_v,
        marker=dict(color=r2_c, line=dict(width=0)),
        text=[f"{v:.4f}" for v in r2_v],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="Inter"),
        hovertemplate="<b>%{x}</b><br>R² = %{y:.4f}<extra></extra>",
    ))
    fig.add_shape(type="line", x0=-0.5, x1=3.5, y0=0.9809, y1=0.9809,
                  line=dict(color="#f87171", dash="dot", width=2))
    fig.add_annotation(x=3.5, y=0.981, text="  Baseline R²",
                       showarrow=False, font=dict(color="#f87171", size=10), xanchor="left")
    l4 = base_layout(h=290)
    l4["yaxis"]["range"] = [0.975, 0.997]
    fig.update_layout(**l4)
    fig.update_xaxes(tickfont=dict(color="#e2e8f0", size=11))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with T4:
    st.markdown('<p class="sec-title">🗂️ K-Means Airport Segmentation — 3 Operational Tiers</p>', unsafe_allow_html=True)
    st.caption("Scikit-Learn KMeans · Features: avg daily flights · daily variability (std) · peak day capacity · Bubble size = total annual volume")

    tier_colors = {
        "Major Hub":      "#6366f1",
        "Medium Hub":     "#f59e0b",
        "Small / Regional": "#10b981",
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
        labels={"avg_day":"Average Daily Flights","variab":"Daily Variability (Std Dev)","tier":"Tier"},
        size_max=35,
    )
    fig.update_traces(
        marker=dict(line=dict(width=1.5, color="#050816")),
        hovertemplate="<b>%{hovertext}</b><br>Country: %{customdata[0]}<br>Avg/day: %{customdata[1]:.0f}<br>Peak day: %{customdata[2]:.0f}<extra></extra>",
    )
    l5 = base_layout(h=430, legend=True)
    l5["legend"]["y"] = 1.06
    l5["legend"]["title"] = dict(text="Tier")
    fig.update_layout(**l5)
    st.plotly_chart(fig, use_container_width=True)

    cc1, cc2 = st.columns([1,2], gap="large")

    with cc1:
        st.markdown('<p class="sec-title">🥧 Distribution</p>', unsafe_allow_html=True)
        tc = profile["tier"].value_counts().reset_index()
        tc.columns = ["Tier","Count"]
        fig = go.Figure(go.Pie(
            labels=tc["Tier"], values=tc["Count"], hole=0.58,
            marker=dict(colors=[tier_colors[t] for t in tc["Tier"]],
                        line=dict(color="#050816", width=3)),
            textfont=dict(size=11, color="#f1f5f9"),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>%{value} airports<extra></extra>",
        ))
        fig.add_annotation(text="331<br>Airports", x=0.5, y=0.5,
                           font=dict(size=14, color="#94a3b8"), showarrow=False)
        fig.update_layout(**base_layout(h=290), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        st.markdown('<p class="sec-title">📋 Top Airports per Tier</p>', unsafe_allow_html=True)
        tier_colors_map = {"Major Hub":"#6366f1","Medium Hub":"#f59e0b","Small / Regional":"#10b981"}
        tier_emoji_map  = {"Major Hub":"🔵","Medium Hub":"🟡","Small / Regional":"🟢"}
        for tier_name in ["Major Hub","Medium Hub","Small / Regional"]:
            grp = (profile[profile["tier"]==tier_name]
                   .nlargest(5, "avg_day")
                   [["APT_NAME","STATE_NAME","avg_day","peak","total"]]
                   .reset_index(drop=True))
            tc  = tier_colors_map[tier_name]
            em  = tier_emoji_map[tier_name]
            rows_html = ""
            for _, row in grp.iterrows():
                rows_html += f"""
                <tr>
                  <td style='padding:8px 12px; color:#e2e8f0; font-size:.85rem; font-weight:600'>{row['APT_NAME']}</td>
                  <td style='padding:8px 12px; color:#94a3b8; font-size:.82rem'>{row['STATE_NAME']}</td>
                  <td style='padding:8px 12px; color:{tc}; font-weight:700; text-align:center'>{row['avg_day']:,.0f}</td>
                  <td style='padding:8px 12px; color:#f1f5f9; text-align:center'>{int(row['peak']):,}</td>
                  <td style='padding:8px 12px; color:#94a3b8; text-align:center'>{int(row['total']):,}</td>
                </tr>"""
            st.markdown(f"""
            <div style='margin-bottom:18px'>
              <div style='font-size:.88rem; font-weight:700; color:{tc};
                          margin-bottom:8px; display:flex; align-items:center; gap:6px'>
                {em} {tier_name}
              </div>
              <table style='width:100%; border-collapse:collapse;
                            background:#0d1117; border:1px solid #1e2433;
                            border-radius:10px; overflow:hidden'>
                <thead>
                  <tr style='background:#111827'>
                    <th style='padding:8px 12px; color:#475569; font-size:.72rem;
                               text-transform:uppercase; letter-spacing:.06em; text-align:left'>Airport</th>
                    <th style='padding:8px 12px; color:#475569; font-size:.72rem;
                               text-transform:uppercase; letter-spacing:.06em; text-align:left'>Country</th>
                    <th style='padding:8px 12px; color:#475569; font-size:.72rem;
                               text-transform:uppercase; letter-spacing:.06em; text-align:center'>Avg/Day</th>
                    <th style='padding:8px 12px; color:#475569; font-size:.72rem;
                               text-transform:uppercase; letter-spacing:.06em; text-align:center'>Peak Day</th>
                    <th style='padding:8px 12px; color:#475569; font-size:.72rem;
                               text-transform:uppercase; letter-spacing:.06em; text-align:center'>Total 2025</th>
                  </tr>
                </thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  ✈️ Airport Traffic Prediction · INT315 Cluster Computing ·
  <strong style="color:#6366f1">Rai Yashmit Ashok</strong> ·
  Reg: 12313527 · RK4M2310A ·
  Data: EUROCONTROL 2025 (116,020 records)
</div>
""", unsafe_allow_html=True)
