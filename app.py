import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_COLUMNS = [
    "Date", "Daily_Attacks", "Fatalities", "Inflation_Pressure_Idx",
    "Estimated_CO2_Tonnes", "Conflict_Intensity_Index", "Oil_Price", "Gold", "Stock_Index",
]

EVENTS = [
    {"date": "2026-03-06", "label": "Airstrikes",           "color": "#f87171"},
    {"date": "2026-03-18", "label": "Oil Facility Attacks", "color": "#fbbf24"},
    {"date": "2026-03-30", "label": "Strait Closure",       "color": "#38bdf8"},
]

# ─── DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("final_master_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def prep_series(s, win, norm):
    r = s.astype(float)
    if win > 1: r = r.rolling(win, min_periods=1).mean()
    if norm:
        std = r.std(ddof=0)
        if std > 0: r = (r - r.mean()) / std
    return r

def lag_corr_frame(df, src, tgt, max_lag):
    return pd.DataFrame([{"Lag_Days": l, "Correlation": df[src].shift(l).corr(df[tgt])} for l in range(1, max_lag+1)])

def event_impact_table(df, events, window):
    rows = []
    for ev in events:
        d = pd.Timestamp(ev["date"])
        pre  = df[(df["Date"] < d) & (df["Date"] >= d - pd.Timedelta(days=window))]
        post = df[(df["Date"] > d) & (df["Date"] <= d + pd.Timedelta(days=window))]
        if pre.empty or post.empty: continue
        rows.append({
            "Event": ev["label"], "Date": d.date().isoformat(),
            "Oil Delta %":       round(((post["Oil_Price"].mean() - pre["Oil_Price"].mean()) / pre["Oil_Price"].mean()) * 100, 2),
            "Stock Delta %":     round(((post["Stock_Index"].mean() - pre["Stock_Index"].mean()) / pre["Stock_Index"].mean()) * 100, 2),
            "Inflation Delta %": round(((post["Inflation_Pressure_Idx"].mean() - pre["Inflation_Pressure_Idx"].mean()) / pre["Inflation_Pressure_Idx"].mean()) * 100, 2),
        })
    return pd.DataFrame(rows)

def safe_corr(a, b):
    v = a.corr(b); return 0.0 if pd.isna(v) else float(v)

def pct_chg(a, b):
    return 0.0 if (pd.isna(a) or pd.isna(b) or a == 0) else ((b - a) / a) * 100

def top_corr_pairs(df, cols, n=3):
    pairs = [(l, r, float(df[l].corr(df[r]))) for i, l in enumerate(cols) for r in cols[i+1:] if pd.notna(df[l].corr(df[r]))]
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:n]

# ─── STYLE ─────────────────────────────────────────────────────
def inject_style():
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* Base */
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:#050d1a !important;}
.stApp{
    background:
        radial-gradient(ellipse 1000px 650px at 10% -5%, rgba(56,189,248,0.07) 0%,transparent 55%),
        radial-gradient(ellipse 750px 550px  at 95% 5%,  rgba(45,212,191,0.055) 0%,transparent 50%),
        repeating-linear-gradient(0deg,  transparent,transparent 59px,rgba(56,189,248,0.018) 60px),
        repeating-linear-gradient(90deg, transparent,transparent 59px,rgba(56,189,248,0.012) 60px),
        #050d1a !important;
    font-family:'Space Grotesk',sans-serif;
    color:#d0e4f7;
}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060e1c 0%,#04090f 100%) !important;border-right:1px solid rgba(56,189,248,0.1) !important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] div{color:#7a9ab5 !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#d0e4f7 !important;font-family:'Syne',sans-serif !important;}
/* Typography */
h1,h2,h3,h4{font-family:'Syne',sans-serif !important;color:#e8f4ff !important;letter-spacing:-0.015em;}
p,.stMarkdown p,label{font-family:'Space Grotesk',sans-serif !important;color:#7a9ab5 !important;}
[data-testid="stCaptionContainer"] p{color:#4a6478 !important;font-size:0.78rem !important;}
/* Tabs */
[data-baseweb="tab-list"]{gap:0.4rem;border-bottom:1px solid rgba(56,189,248,0.1) !important;background:transparent !important;padding-bottom:0.3rem;}
[data-baseweb="tab"]{background:rgba(10,22,42,0.8) !important;border:1px solid rgba(56,189,248,0.12) !important;border-radius:9px !important;padding:0.38rem 0.85rem !important;font-family:'Space Grotesk',sans-serif !important;font-weight:500 !important;font-size:0.82rem !important;color:#5a7a95 !important;transition:all 0.2s;}
[data-baseweb="tab"]:hover{background:rgba(20,40,65,0.9) !important;color:#a0c4d8 !important;border-color:rgba(45,212,191,0.25) !important;}
[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,rgba(45,212,191,0.18),rgba(56,189,248,0.14)) !important;border-color:rgba(45,212,191,0.45) !important;color:#2dd4bf !important;box-shadow:0 0 16px rgba(45,212,191,0.12) !important;}
/* Charts */
[data-testid="stPlotlyChart"]{background:rgba(8,18,36,0.9) !important;border:1px solid rgba(56,189,248,0.1) !important;border-radius:14px !important;box-shadow:0 8px 32px rgba(0,0,0,0.5) !important;padding:0.3rem !important;}
[data-testid="stDataFrame"]{background:rgba(8,18,36,0.9) !important;border:1px solid rgba(56,189,248,0.1) !important;border-radius:12px !important;}
/* Expanders */
[data-testid="stExpander"]{background:rgba(8,18,36,0.85) !important;border:1px solid rgba(56,189,248,0.1) !important;border-radius:12px !important;}
[data-testid="stExpander"] summary{font-family:'Syne',sans-serif !important;color:#a0c4d8 !important;font-weight:600 !important;}
[data-testid="stExpander"] li{color:#7a9ab5 !important;font-size:0.88rem !important;}
/* Download button */
[data-testid="stDownloadButton"] button{background:linear-gradient(135deg,rgba(45,212,191,0.15),rgba(56,189,248,0.12)) !important;border:1px solid rgba(45,212,191,0.4) !important;color:#2dd4bf !important;font-family:'Space Grotesk',sans-serif !important;font-weight:600 !important;border-radius:9px !important;transition:all 0.2s;}
[data-testid="stDownloadButton"] button:hover{background:linear-gradient(135deg,rgba(45,212,191,0.28),rgba(56,189,248,0.22)) !important;box-shadow:0 0 18px rgba(45,212,191,0.25) !important;}
/* Select / inputs */
[data-baseweb="select"]>div,.stDateInput input{background:rgba(10,22,42,0.9) !important;border-color:rgba(56,189,248,0.18) !important;color:#d0e4f7 !important;border-radius:8px !important;}
/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#040a14;}
::-webkit-scrollbar-thumb{background:rgba(45,212,191,0.28);border-radius:999px;}
/* KPI card */
.kpi-box{background:rgba(8,18,36,0.92);border:1px solid rgba(56,189,248,0.12);border-radius:14px;padding:1.1rem 1.05rem 0.95rem;position:relative;overflow:hidden;transition:border-color 0.25s,transform 0.2s,box-shadow 0.25s;min-height:118px;}
.kpi-box:hover{border-color:rgba(45,212,191,0.38);transform:translateY(-2px);box-shadow:0 12px 28px rgba(0,0,0,0.45);}
.kpi-box::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--ac,#2dd4bf),transparent);}
.c-red   {--ac:#f87171;} .c-amber{--ac:#fbbf24;} .c-teal{--ac:#2dd4bf;} .c-sky{--ac:#38bdf8;} .c-violet{--ac:#a78bfa;}
.kpi-icon{font-size:1.1rem;margin-bottom:0.38rem;display:block;}
.kpi-lbl {font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#3d5c72;margin-bottom:0.28rem;}
.kpi-val {font-family:'JetBrains Mono',monospace;font-size:1.35rem;font-weight:600;color:#e8f4ff;line-height:1.1;margin-bottom:0.2rem;}
.kpi-note{font-family:'Space Grotesk',sans-serif;font-size:0.67rem;color:#3d5c72;}
/* Hero */
.hero{background:linear-gradient(130deg,rgba(4,12,28,0.98) 0%,rgba(6,20,42,0.97) 55%,rgba(4,14,30,0.98) 100%);border:1px solid rgba(45,212,191,0.2);border-radius:20px;padding:2rem 2.2rem 1.7rem;margin-bottom:1rem;position:relative;overflow:hidden;box-shadow:0 20px 55px rgba(0,0,0,0.55);}
.hero::before{content:"";position:absolute;inset:0;background:radial-gradient(ellipse 700px 350px at 85% 50%,rgba(45,212,191,0.06),transparent),radial-gradient(ellipse 500px 250px at 5% 20%,rgba(56,189,248,0.055),transparent);pointer-events:none;}
.hero::after{content:"CLASSIFIED  \00B7  INTEL DIVISION  \00B7  RESTRICTED ACCESS  \00B7  LIVE MONITORING  \00B7  ANALYTICAL CONSOLE";position:absolute;bottom:0.55rem;left:0;right:0;text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.57rem;letter-spacing:0.2em;color:rgba(45,212,191,0.17);pointer-events:none;white-space:nowrap;overflow:hidden;}
.hero-eye{font-family:'JetBrains Mono',monospace;font-size:0.66rem;letter-spacing:0.18em;color:#2dd4bf;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;}
.hero-eye::before{content:"";width:24px;height:2px;background:#2dd4bf;box-shadow:0 0 6px #2dd4bf;}
.live-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#4ade80;box-shadow:0 0 7px #4ade80;margin-right:0.3rem;animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 6px #4ade80}50%{opacity:0.4;box-shadow:0 0 12px #4ade80}}
.hero-title{font-family:'Syne',sans-serif;font-size:2.25rem;font-weight:800;color:#e8f4ff;line-height:1.06;margin-bottom:0.42rem;letter-spacing:-0.022em;}
.hero-title span{background:linear-gradient(90deg,#2dd4bf,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-family:'Space Grotesk',sans-serif;font-size:0.89rem;color:#5a7a95;line-height:1.6;margin-bottom:0.95rem;max-width:580px;}
.hero-chips{display:flex;flex-wrap:wrap;gap:0.42rem;}
.hchip{font-family:'JetBrains Mono',monospace;font-size:0.67rem;padding:0.25rem 0.58rem;border:1px solid rgba(56,189,248,0.22);border-radius:999px;background:rgba(56,189,248,0.07);color:#38bdf8;letter-spacing:0.05em;}
/* Section banner */
.sbanner{display:flex;align-items:stretch;gap:0.8rem;background:rgba(6,16,32,0.9);border:1px solid rgba(56,189,248,0.1);border-radius:13px;padding:0.78rem 1rem;margin:1rem 0 0.6rem;box-shadow:0 4px 16px rgba(0,0,0,0.3);}
.sbanner-bar{width:3px;border-radius:999px;flex-shrink:0;}
.sbanner.copper .sbanner-bar{background:linear-gradient(180deg,#fbbf24,#f97316);box-shadow:0 0 8px rgba(251,191,36,0.45);}
.sbanner.teal   .sbanner-bar{background:linear-gradient(180deg,#2dd4bf,#0891b2);box-shadow:0 0 8px rgba(45,212,191,0.45);}
.sbanner.sky    .sbanner-bar{background:linear-gradient(180deg,#38bdf8,#6366f1);box-shadow:0 0 8px rgba(56,189,248,0.45);}
.sbanner.violet .sbanner-bar{background:linear-gradient(180deg,#a78bfa,#ec4899);box-shadow:0 0 8px rgba(167,139,250,0.45);}
.sbanner-kicker{font-family:'JetBrains Mono',monospace;font-size:0.61rem;letter-spacing:0.14em;text-transform:uppercase;color:#3d5c72;margin-bottom:0.14rem;}
.sbanner-title {font-family:'Syne',sans-serif;font-size:1.02rem;font-weight:700;color:#d0e4f7;}
/* Pill strip */
.pill-strip{display:flex;flex-wrap:wrap;gap:0.42rem;margin-bottom:0.9rem;}
.spill{display:inline-flex;align-items:center;gap:0.38rem;background:rgba(6,16,32,0.88);border:1px solid rgba(56,189,248,0.12);border-radius:9px;padding:0.33rem 0.6rem;font-family:'Space Grotesk',sans-serif;font-size:0.77rem;color:#5a7a95;}
.spill-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.spill-date{font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#3d5c72;}
/* Sidebar terminal */
.term-card{margin-top:1.8rem;padding:0.72rem;background:rgba(45,212,191,0.04);border:1px solid rgba(45,212,191,0.12);border-radius:9px;font-family:'JetBrains Mono',monospace;font-size:0.61rem;color:rgba(45,212,191,0.42);letter-spacing:0.07em;line-height:1.9;}
.term-ok{color:rgba(74,222,128,0.6);}
@media(max-width:900px){.hero-title{font-size:1.5rem}.kpi-val{font-size:1.05rem}}
</style>
""", unsafe_allow_html=True)


def render_hero(start, end, n):
    st.markdown(f"""
<div class="hero">
    <div class="hero-eye"><span class="live-dot"></span>INTEL CONSOLE — OPERATIONAL</div>
    <div class="hero-title">Iran Conflict<br><span>Intelligence Console</span></div>
    <p class="hero-sub">A narrative cockpit for conflict shocks, commodity stress, market fragility, and environmental spillover.</p>
    <div class="hero-chips">
        <span class="hchip">&#x23F1; {start} &#x2192; {end}</span>
        <span class="hchip">&#x25C8; {n} DATA POINTS</span>
        <span class="hchip">&#x25C9; LIVE ANALYTICAL MODE</span>
        <span class="hchip">&#x25A3; MULTI-DOMAIN ANALYSIS</span>
    </div>
</div>""", unsafe_allow_html=True)


def render_kpi(label, value, note, color="c-teal", icon=""):
    st.markdown(f"""
<div class="kpi-box {color}">
    <span class="kpi-icon">{icon}</span>
    <div class="kpi-lbl">{label}</div>
    <div class="kpi-val">{value}</div>
    <div class="kpi-note">{note}</div>
</div>""", unsafe_allow_html=True)


def render_story_strip(events):
    dot_map = {"Airstrikes": "#f87171", "Oil Facility Attacks": "#fbbf24", "Strait Closure": "#38bdf8"}
    pills = "".join(
        f'<div class="spill"><span class="spill-dot" style="background:{dot_map.get(e["label"],"#2dd4bf")};'
        f'box-shadow:0 0 5px {dot_map.get(e["label"],"#2dd4bf")}"></span>'
        f'<span class="spill-date">{e["date"]}</span>{e["label"]}</div>'
        for e in events
    )
    st.markdown(f'<div class="pill-strip">{pills}</div>', unsafe_allow_html=True)


def section_banner(title, sub, tone="copper"):
    st.markdown(f"""
<div class="sbanner {tone}">
    <div class="sbanner-bar"></div>
    <div><div class="sbanner-kicker">{sub}</div><div class="sbanner-title">{title}</div></div>
</div>""", unsafe_allow_html=True)


CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.022)",
    font=dict(family="Space Grotesk, sans-serif", size=12, color="#7a9ab5"),
    title=dict(font=dict(size=16, color="#c8dff2", family="Syne, sans-serif"), x=0.01, xanchor="left"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor="rgba(4,12,26,0.7)", bordercolor="rgba(56,189,248,0.2)", borderwidth=1,
                font=dict(color="#7a9ab5", size=11)),
    margin=dict(l=14, r=14, t=64, b=14),
    hoverlabel=dict(bgcolor="rgba(4,12,26,0.95)", font=dict(color="#d0e4f7"), bordercolor="rgba(56,189,248,0.35)"),
)
AXIS_STYLE = dict(
    showgrid=True, gridcolor="rgba(56,189,248,0.06)", zeroline=False,
    linecolor="rgba(56,189,248,0.12)",
    tickfont=dict(color="#3d5c72", family="JetBrains Mono, monospace", size=10),
)

def sfig(fig, h=470, geo=False):
    fig.update_layout(height=h, **CHART_BASE)
    if not geo:
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)


# ═══════════════════════════════════════════════════════════════
#  APP START
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Iran Conflict Intelligence Console", layout="wide", page_icon="⚡")
inject_style()

df = load_data()
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
color:#c8dff2;margin-bottom:1rem;padding-bottom:0.5rem;
border-bottom:1px solid rgba(56,189,248,0.12);">&#9881; Control Panel</div>""", unsafe_allow_html=True)

raw = st.sidebar.date_input("Select timeline",
    value=(df["Date"].min().date(), df["Date"].max().date()),
    min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
start_date, end_date = (raw if isinstance(raw, tuple) and len(raw) == 2 else (raw, raw))

rolling   = st.sidebar.slider("Smoothing window (days)", 1, 10, 1)
normalize = st.sidebar.checkbox("Normalize (z-score)", False)
ev_win    = st.sidebar.slider("Event impact window (±days)", 2, 10, 3)

mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
fdf  = df.loc[mask].copy()
if fdf.empty:
    st.warning("No records in date range.")
    st.stop()

snap_opts  = fdf["Date"].dt.strftime("%Y-%m-%d").tolist()
snap_label = st.sidebar.selectbox("Map snapshot date", snap_opts, index=len(snap_opts)-1)
snap_row   = fdf.loc[fdf["Date"] == pd.Timestamp(snap_label)].iloc[0]

st.sidebar.markdown("""<div class="term-card">
IRAN CONFLICT<br>INTELLIGENCE CONSOLE<br>v2.0 // ANALYTICAL MODE<br>
<span class="term-ok">&#x25CF; SYSTEM NOMINAL</span>
</div>""", unsafe_allow_html=True)

# ── Computed stats ───────────────────────────────────────────
oil_stk_r = safe_corr(fdf["Oil_Price"],               fdf["Stock_Index"])
con_oil_r = safe_corr(fdf["Conflict_Intensity_Index"], fdf["Oil_Price"])
atk_co2_r = safe_corr(fdf["Daily_Attacks"],           fdf["Estimated_CO2_Tonnes"])
con_inf_r = safe_corr(fdf["Conflict_Intensity_Index"], fdf["Inflation_Pressure_Idx"])
oil_inf_r = safe_corr(fdf["Oil_Price"],               fdf["Inflation_Pressure_Idx"])
gld_oil_r = safe_corr(fdf["Gold"],                    fdf["Oil_Price"])
r0, rl    = fdf.iloc[0], fdf.iloc[-1]
oil_pct   = pct_chg(r0["Oil_Price"],             rl["Oil_Price"])
stk_pct   = pct_chg(r0["Stock_Index"],           rl["Stock_Index"])
inf_pct   = pct_chg(r0["Inflation_Pressure_Idx"], rl["Inflation_Pressure_Idx"])
co2_pct   = pct_chg(r0["Estimated_CO2_Tonnes"],  rl["Estimated_CO2_Tonnes"])
pk_idx    = fdf["Estimated_CO2_Tonnes"].idxmax()
pk_co2_d  = fdf.loc[pk_idx, "Date"].date().isoformat()
pk_co2_v  = int(fdf.loc[pk_idx, "Estimated_CO2_Tonnes"])
imp_df    = event_impact_table(fdf, EVENTS, ev_win)
top_pairs = top_corr_pairs(fdf, ["Oil_Price","Stock_Index","Gold","Inflation_Pressure_Idx","Estimated_CO2_Tonnes"])

# ── Hero + KPI row ───────────────────────────────────────────
st.caption("&#9658; Data source: final_master_dataset.csv")
render_hero(start_date, end_date, len(fdf))

k1, k2, k3, k4, k5 = st.columns(5)
with k1: render_kpi("Peak Oil Price",      f"${fdf['Oil_Price'].max():.2f}",                                "Energy shock ceiling",  "c-red",   "&#9981;")
with k2: render_kpi("Stock Drawdown",      f"{fdf['Stock_Index'].min()-fdf['Stock_Index'].max():.0f} pts", "Market compression",    "c-amber", "&#128201;")
with k3: render_kpi("Peak Conflict Index", f"{fdf['Conflict_Intensity_Index'].max():.1f}",                 "Highest stress signal", "c-teal",  "&#9888;")
with k4: render_kpi("Total CO&#x2082; Emitted", f"{int(fdf['Estimated_CO2_Tonnes'].sum()):,} t",          "Cumulative emissions",  "c-sky",   "&#127787;")
with k5: render_kpi("Avg Inflation Index", f"{fdf['Inflation_Pressure_Idx'].mean():.1f}",                 "Price pressure base",   "c-violet","&#128202;")

st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
render_story_strip(EVENTS)

# ═══════════════════════════════════════════════════════════════
tab_dash, tab_report = st.tabs(["&#9672;  Dashboard", "&#9673;  Analytical Report"])

with tab_dash:

    # A: Time-Series ─────────────────────────────────────────
    section_banner("Time-Series Correlation Analysis", "A  ·  Market & conflict pulse", "copper")
    ta1, ta2, ta3 = st.tabs(["Oil vs Stock", "Conflict vs Oil", "CO\u2082 vs Attacks"])

    with ta1:
        ots = prep_series(fdf["Oil_Price"],   rolling, normalize)
        sts = prep_series(fdf["Stock_Index"], rolling, normalize)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=fdf["Date"], y=ots, name="Oil Price",
            line=dict(color="#f87171",width=2.5), fill="tozeroy", fillcolor="rgba(248,113,113,0.055)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=fdf["Date"], y=sts, name="Stock Index",
            line=dict(color="#38bdf8",width=2.5), fill="tozeroy", fillcolor="rgba(56,189,248,0.04)"),  secondary_y=True)
        fig.update_layout(title="Oil Price vs Stock Index", hovermode="x unified")
        fig.update_yaxes(title_text="Oil"+(" (z)" if normalize else " $"),  secondary_y=False, title_font=dict(color="#f87171"))
        fig.update_yaxes(title_text="Stock"+(" (z)" if normalize else ""),  secondary_y=True,  title_font=dict(color="#38bdf8"))
        sfig(fig); st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Pearson r (Oil, Stock) = {ots.corr(sts):.3f}")

    with ta2:
        cts = prep_series(fdf["Conflict_Intensity_Index"], rolling, normalize)
        ot2 = prep_series(fdf["Oil_Price"],                rolling, normalize)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=fdf["Date"], y=cts, name="Conflict Intensity",
            line=dict(color="#a78bfa",width=2.5), fill="tozeroy", fillcolor="rgba(167,139,250,0.055)"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=fdf["Date"], y=ot2, name="Oil Price",
            line=dict(color="#fb923c",width=2.5)), secondary_y=True)
        fig2.update_layout(title="Conflict Intensity vs Oil Price", hovermode="x unified")
        fig2.update_yaxes(title_text="Conflict"+(" (z)" if normalize else ""), secondary_y=False, title_font=dict(color="#a78bfa"))
        fig2.update_yaxes(title_text="Oil"+(" (z)" if normalize else " $"),    secondary_y=True,  title_font=dict(color="#fb923c"))
        sfig(fig2); st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Pearson r (Conflict, Oil) = {cts.corr(ot2):.3f}")

    with ta3:
        ats = prep_series(fdf["Daily_Attacks"],          rolling, normalize)
        c2s = prep_series(fdf["Estimated_CO2_Tonnes"],  rolling, normalize)
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=fdf["Date"], y=ats, name="Daily Attacks",
            marker_color="rgba(248,113,113,0.45)", marker_line_color="#f87171", marker_line_width=0.4), secondary_y=False)
        fig3.add_trace(go.Scatter(x=fdf["Date"], y=c2s, name="CO\u2082 Emissions",
            line=dict(color="#4ade80",width=2.5)), secondary_y=True)
        fig3.update_layout(title="CO\u2082 Emissions vs Daily Attacks", hovermode="x unified")
        fig3.update_yaxes(title_text="Attacks"+(" (z)" if normalize else ""), secondary_y=False, title_font=dict(color="#f87171"))
        fig3.update_yaxes(title_text="CO\u2082"+(" (z)" if normalize else " t"), secondary_y=True, title_font=dict(color="#4ade80"))
        sfig(fig3); st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"Pearson r (Attacks, CO\u2082) = {ats.corr(c2s):.3f}")

    # B: Multi-axis ──────────────────────────────────────────
    section_banner("Multi-Axis Pressure Monitor", "B  ·  Three-domain overlay", "teal")
    om = prep_series(fdf["Oil_Price"],           rolling, normalize)
    sm = prep_series(fdf["Stock_Index"],         rolling, normalize)
    cm = prep_series(fdf["Estimated_CO2_Tonnes"],rolling, normalize)
    figB = go.Figure()
    figB.add_trace(go.Scatter(x=fdf["Date"], y=om, name="Oil Price",   line=dict(color="#f87171",width=2.2), yaxis="y"))
    figB.add_trace(go.Scatter(x=fdf["Date"], y=sm, name="Stock Index", line=dict(color="#38bdf8",width=2.2), yaxis="y2"))
    figB.add_trace(go.Scatter(x=fdf["Date"], y=cm, name="CO\u2082",   line=dict(color="#4ade80",width=2.2), yaxis="y3"))
    figB.update_layout(
        title="Oil \u00B7 Stock \u00B7 CO\u2082 \u2014 Triple-Axis Overlay", xaxis=dict(title="Date"),
        yaxis =dict(title=dict(text="Oil"+(" (z)" if normalize else " $"),   font=dict(color="#f87171")), tickfont=dict(color="#f87171",family="JetBrains Mono",size=10)),
        yaxis2=dict(title=dict(text="Stock"+(" (z)" if normalize else ""),   font=dict(color="#38bdf8")), tickfont=dict(color="#38bdf8",family="JetBrains Mono",size=10), overlaying="y", side="right"),
        yaxis3=dict(title=dict(text="CO\u2082"+(" (z)" if normalize else " t"), font=dict(color="#4ade80")), tickfont=dict(color="#4ade80",family="JetBrains Mono",size=10), anchor="free", overlaying="y", side="right", position=0.93),
        hovermode="x unified",
    )
    sfig(figB, 500); st.plotly_chart(figB, use_container_width=True)

    # C: Geo ─────────────────────────────────────────────────
    section_banner("Geospatial Risk Map", "C  ·  Regional conflict & pollution zones", "sky")
    st.caption("Red = Conflict zones  \u00B7  Amber = Oil route (Strait of Hormuz)  \u00B7  Green = Pollution hotspots")
    conf_pts = pd.DataFrame({"Location":["Tehran","Isfahan","Khuzestan","Bushehr","Bandar Abbas"],
        "Lat":[35.6892,32.6546,31.3183,28.9234,27.1832],"Lon":[51.389,51.668,48.6706,50.8203,56.2666],"W":[1.0,0.75,1.1,0.85,1.2]})
    conf_pts["I"] = snap_row["Conflict_Intensity_Index"] * conf_pts["W"]
    poll_pts = pd.DataFrame({"Location":["Asaluyeh","Abadan","Bandar Abbas","Kharg Island"],
        "Lat":[27.4762,30.3473,27.1832,29.2614],"Lon":[52.6096,48.2934,56.2666,50.33],"W":[1.2,1.0,1.1,1.25]})
    poll_pts["C"] = snap_row["Estimated_CO2_Tonnes"] * poll_pts["W"]
    figC = go.Figure()
    figC.add_trace(go.Scattergeo(lon=[50.33,52.61,56.27,56.45,58.3],lat=[29.26,27.48,27.18,26.4,25.6],
        mode="lines+markers",line=dict(width=3,color="#fbbf24"),marker=dict(size=6,color="#fbbf24"),
        name="Oil Route",hovertemplate="Oil route<extra></extra>"))
    figC.add_trace(go.Scattergeo(lon=conf_pts["Lon"],lat=conf_pts["Lat"],text=conf_pts["Location"],
        mode="markers",marker=dict(size=10+(conf_pts["I"]/max(conf_pts["I"].max(),1))*18,
        color="#f87171",opacity=0.88,line=dict(color="#fca5a5",width=1.5)),name="Conflict Zones",
        hovertemplate="%{text}<extra></extra>"))
    figC.add_trace(go.Scattergeo(lon=poll_pts["Lon"],lat=poll_pts["Lat"],text=poll_pts["Location"],
        mode="markers",marker=dict(size=8+(poll_pts["C"]/max(poll_pts["C"].max(),1))*20,
        color="#4ade80",opacity=0.78,line=dict(color="#86efac",width=1)),name="Pollution Hotspots",
        hovertemplate="%{text}<extra></extra>"))
    figC.update_layout(
        title=f"Iran Region Risk Map \u2014 {snap_label}",
        geo=dict(scope="asia",projection_type="natural earth",showland=True,landcolor="rgb(16,28,44)",
                 showcountries=True,countrycolor="rgb(50,72,94)",showocean=True,oceancolor="rgb(6,16,34)",
                 showlakes=True,lakecolor="rgb(8,18,38)",showrivers=True,rivercolor="rgb(10,22,42)",
                 lataxis=dict(range=[23,38]),lonaxis=dict(range=[44,62]),bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8,r=8,t=56,b=8),
    )
    sfig(figC, 640, geo=True); st.plotly_chart(figC, use_container_width=True)

    # D: Heatmap ─────────────────────────────────────────────
    section_banner("Correlation Heatmap", "D  ·  Cross-variable interaction grid", "copper")
    hv    = ["Oil_Price","Stock_Index","Gold","Inflation_Pressure_Idx","Estimated_CO2_Tonnes"]
    cm_df = fdf[hv].corr().round(2)
    figD  = px.imshow(cm_df, text_auto=True,
        color_continuous_scale=[[0,"#0d2a4a"],[0.5,"#080f1e"],[1,"#5e1a1a"]],
        zmin=-1, zmax=1, title="Correlation Matrix: Oil \u00B7 Stock \u00B7 Gold \u00B7 Inflation \u00B7 CO\u2082")
    figD.update_traces(textfont=dict(family="JetBrains Mono",size=13,color="#c8dff2"))
    sfig(figD, 500); st.plotly_chart(figD, use_container_width=True)

    # E: Events ──────────────────────────────────────────────
    section_banner("Event Impact & Annotations", "E  ·  Disruption timeline", "teal")
    figE = make_subplots(specs=[[{"secondary_y": True}]])
    figE.add_trace(go.Scatter(x=fdf["Date"],y=fdf["Oil_Price"],  name="Oil Price",  line=dict(color="#f87171",width=2.5)), secondary_y=False)
    figE.add_trace(go.Scatter(x=fdf["Date"],y=fdf["Stock_Index"],name="Stock Index",line=dict(color="#38bdf8",width=2.5)), secondary_y=True)
    for ev in EVENTS:
        et = pd.Timestamp(ev["date"])
        if fdf["Date"].min() <= et <= fdf["Date"].max():
            figE.add_vline(x=et, line_dash="dot", line_width=1.5, line_color=ev["color"])
            figE.add_annotation(x=et,y=1.07,yref="paper",text=ev["label"],showarrow=False,
                bgcolor="rgba(4,12,26,0.88)",font=dict(color=ev["color"],size=11),
                bordercolor=ev["color"],borderwidth=1,borderpad=4)
    figE.update_layout(title="Oil & Stock \u2014 Annotated Event Timeline", hovermode="x unified")
    figE.update_yaxes(title_text="Oil Price ($)", secondary_y=False)
    figE.update_yaxes(title_text="Stock Index",   secondary_y=True)
    sfig(figE, 500); st.plotly_chart(figE, use_container_width=True)
    if imp_df.empty: st.caption("\u26A0 Insufficient data around events in current filter.")
    else:            st.dataframe(imp_df, use_container_width=True, hide_index=True)

    # F: Lag ─────────────────────────────────────────────────
    section_banner("Causal Lag Analysis", "F  ·  Lead-lag signal detection", "violet")
    lc1, lc2 = st.columns(2)
    oil_lag = lc1.slider("Oil \u2192 Stock lag (days)",         3, 5,  4)
    con_lag = lc2.slider("Conflict \u2192 Inflation lag (days)", 7, 30, 14)

    los = fdf[["Date","Oil_Price","Stock_Index"]].copy()
    los[f"OL{oil_lag}"] = los["Oil_Price"].shift(oil_lag)
    los_p = los.dropna(subset=[f"OL{oil_lag}","Stock_Index"])
    lci = fdf[["Date","Conflict_Intensity_Index","Inflation_Pressure_Idx"]].copy()
    lci[f"CL{con_lag}"] = lci["Conflict_Intensity_Index"].shift(con_lag)
    lci_p = lci.dropna(subset=[f"CL{con_lag}","Inflation_Pressure_Idx"])

    pc1, pc2 = st.columns(2)
    with pc1:
        f1 = px.scatter(los_p, x=f"OL{oil_lag}", y="Stock_Index",
            title=f"Oil (t\u2212{oil_lag}) \u2192 Stock (t)", opacity=0.78,
            trendline="ols", trendline_color_override="#a78bfa")
        f1.update_traces(marker=dict(color="#7c3aed",size=9,line=dict(color="#a78bfa",width=1)))
        sfig(f1, 430); st.plotly_chart(f1, use_container_width=True)
    with pc2:
        f2 = px.scatter(lci_p, x=f"CL{con_lag}", y="Inflation_Pressure_Idx",
            title=f"Conflict (t\u2212{con_lag}) \u2192 Inflation (t)", opacity=0.78,
            trendline="ols", trendline_color_override="#fbbf24")
        f2.update_traces(marker=dict(color="#0891b2",size=9,line=dict(color="#38bdf8",width=1)))
        sfig(f2, 430); st.plotly_chart(f2, use_container_width=True)

    lc_os = lag_corr_frame(fdf, "Oil_Price", "Stock_Index", 10)
    lc_ci = lag_corr_frame(fdf, "Conflict_Intensity_Index", "Inflation_Pressure_Idx", 30)
    f3 = make_subplots(rows=1,cols=2,subplot_titles=["Oil \u2192 Stock Lag Profile","Conflict \u2192 Inflation Lag Profile"])
    f3.add_trace(go.Scatter(x=lc_os["Lag_Days"],y=lc_os["Correlation"],mode="lines+markers",
        line=dict(color="#a78bfa",width=2.5),marker=dict(size=5),name="Oil\u2192Stock"),row=1,col=1)
    f3.add_trace(go.Scatter(x=lc_ci["Lag_Days"],y=lc_ci["Correlation"],mode="lines+markers",
        line=dict(color="#38bdf8",width=2.5),marker=dict(size=5),name="Conflict\u2192Inflation"),row=1,col=2)
    f3.add_vline(x=oil_lag,line_dash="dot",line_color="#a78bfa",opacity=0.6,row=1,col=1)
    f3.add_vline(x=con_lag,line_dash="dot",line_color="#38bdf8",opacity=0.6,row=1,col=2)
    f3.update_xaxes(title_text="Lag (days)")
    f3.update_yaxes(title_text="Correlation",row=1,col=1)
    f3.update_layout(showlegend=False)
    sfig(f3, 450); st.plotly_chart(f3, use_container_width=True)
    r_os = fdf["Oil_Price"].shift(oil_lag).corr(fdf["Stock_Index"])
    r_ci = fdf["Conflict_Intensity_Index"].shift(con_lag).corr(fdf["Inflation_Pressure_Idx"])
    st.caption(f"Oil(t\u2212{oil_lag}) \u2192 Stock(t): r = {r_os:.3f}  |  Conflict(t\u2212{con_lag}) \u2192 Inflation(t): r = {r_ci:.3f}")


with tab_report:
    section_banner("Analytical Report", "Auto-generated strategic summary", "copper")
    st.caption("Derived from the active timeline filter.")

    with st.expander("\u25B8  Key Patterns", expanded=True):
        for cl, cr, cv in top_pairs:
            st.markdown(f"- Strong relationship: **{cl}** vs **{cr}** (r = {cv:.2f})")
        st.markdown(f"- Oil moved **{oil_pct:+.2f}%** across the timeline.")
        st.markdown(f"- Stock index moved **{stk_pct:+.2f}%** across the timeline.")
        st.markdown(f"- Conflict vs Oil: **r = {con_oil_r:.2f}**")
        st.markdown(f"- Attacks vs CO\u2082: **r = {atk_co2_r:.2f}**")

    with st.expander("\u25B8  Economic Insights", expanded=True):
        st.markdown(f"- Oil & stock: **{'inverse' if oil_stk_r<0 else 'positive'} linkage** (r = {oil_stk_r:.2f}).")
        st.markdown(f"- Conflict & inflation: **{'positive' if con_inf_r>0 else 'inverse'} linkage** (r = {con_inf_r:.2f}).")
        st.markdown(f"- Oil vs inflation: **r = {oil_inf_r:.2f}**, inflation shifted **{inf_pct:+.2f}%** over the period.")
        st.markdown(f"- Gold vs oil: **r = {gld_oil_r:.2f}**.")
        if not imp_df.empty:
            st.markdown(f"- Avg post-event oil: **{imp_df['Oil Delta %'].mean():+.2f}%**; stock: **{imp_df['Stock Delta %'].mean():+.2f}%**.")

    with st.expander("\u25B8  Environmental Insights", expanded=True):
        total = int(fdf["Estimated_CO2_Tonnes"].sum())
        st.markdown(f"- Total estimated emissions: **{total:,} tonnes CO\u2082**.")
        st.markdown(f"- Emissions **{'rise' if atk_co2_r>0 else 'fall'}** with attack intensity (r = {atk_co2_r:.2f}).")
        st.markdown(f"- Peak CO\u2082 day: **{pk_co2_d}** at **{pk_co2_v:,} tonnes**.")
        st.markdown(f"- CO\u2082 changed **{co2_pct:+.2f}%** from start to end of window.")

    lines = [
        "Analytical Report \u2014 Iran Conflict Intelligence Console",
        f"Window: {start_date} \u2192 {end_date}", "",
        "KEY PATTERNS:",
        f"  Oil vs Stock:    r = {oil_stk_r:.3f}",
        f"  Conflict vs Oil: r = {con_oil_r:.3f}",
        f"  Attacks vs CO2:  r = {atk_co2_r:.3f}",
        f"  Oil delta:   {oil_pct:+.2f}%",
        f"  Stock delta: {stk_pct:+.2f}%", "",
        "ECONOMIC:",
        f"  Conflict vs Inflation: r = {con_inf_r:.3f}",
        f"  Oil vs Inflation:      r = {oil_inf_r:.3f}",
        f"  Gold vs Oil:           r = {gld_oil_r:.3f}",
        f"  Inflation delta: {inf_pct:+.2f}%", "",
        "ENVIRONMENTAL:",
        f"  Total CO2: {total:,} t",
        f"  CO2 delta: {co2_pct:+.2f}%",
        f"  Peak CO2:  {pk_co2_d} ({pk_co2_v:,} t)",
    ]
    st.download_button("\u2B07  Download Report (.txt)", "\n".join(lines),
        file_name=f"report_{start_date}_{end_date}.txt", mime="text/plain")