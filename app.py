"""
app.py — Phase 4: DecisionLens Interactive BI Dashboard
A production-grade Streamlit analytics product for bank branch efficiency.

Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── import Phase 3 visualization engine ─────────────────────────────────────
from visualization import (
    load_data,
    create_efficiency_leaderboard,
    create_efficiency_distribution,
    create_frontier_plot,
    create_radar_comparison,
    create_branch_map,
    generate_summary_table,
    EFFICIENT_COLOR,
    MODERATE_COLOR,
    INEFFICIENT_COLOR,
    ACCENT_COLOR,
    DARK_BG,
    CARD_BG,
    TEXT_COLOR,
    EFFICIENCY_THRESHOLD,
)

# ── import Phase 5 simulation engine ────────────────────────────────────────
from simulation import (
    compute_slack,
    generate_recommendations,
    simulate_branch,
    target_efficiency_mode,
    ScenarioStore,
    create_before_after_chart,
    create_improvement_gap_chart,
    create_scenario_comparison_chart,
    SIMULATED_COLOR,
    INPUT_COLS,
)

# ── import Phase 6 AI Consultant engine ─────────────────────────────────────
from intelligence import (
    compute_benchmarks,
    root_cause_analysis,
    smart_recommendations,
    run_sensitivity_analysis,
    compute_confidence,
    generate_insight_summary,
    create_benchmark_radar_chart,
    create_sensitivity_chart,
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DecisionLens — Bank Branch Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (dark theme, custom KPI cards, metric colours)
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ── root & body ─────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f0f1a;
        color: #edf2f4;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a 0%, #1a1a3a 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    /* ── tab styling ─────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #16213e;
        border-radius: 12px;
        padding: 6px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8592a3;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: #ffffff !important;
    }
    /* ── KPI cards ───────────────────────────────────────────────────── */
    .kpi-card {
        background: linear-gradient(135deg, #1e2a45 0%, #16213e 100%);
        border: 1px solid rgba(52,152,219,0.25);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(52,152,219,0.2);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8592a3;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-sub {
        font-size: 11px;
        color: #636e72;
        margin-top: 6px;
    }
    /* ── branch card ─────────────────────────────────────────────────── */
    .branch-card {
        background: linear-gradient(135deg, #1a2744 0%, #16213e 100%);
        border-left: 4px solid #3498db;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .branch-card.efficient {
        border-left-color: #2ecc71;
        background: linear-gradient(135deg, #1a3028 0%, #16213e 100%);
    }
    .branch-card.inefficient {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #2d1a1e 0%, #16213e 100%);
    }
    /* ── status badge ────────────────────────────────────────────────── */
    .badge-efficient {
        background: #2ecc71;
        color: #0a1a10;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-inefficient {
        background: #e74c3c;
        color: #fff;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    /* ── section headers ─────────────────────────────────────────────── */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #edf2f4;
        border-bottom: 2px solid #2d3561;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }
    /* ── sidebar elements ────────────────────────────────────────────── */
    .sidebar-logo {
        text-align: center;
        padding: 16px 0 24px 0;
    }
    .sidebar-logo h1 {
        font-size: 26px;
        font-weight: 900;
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sidebar-logo p {
        font-size: 11px;
        color: #636e72;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 4px 0 0 0;
    }
    /* ── info box ────────────────────────────────────────────────────── */
    .info-box {
        background: rgba(52,152,219,0.1);
        border: 1px solid rgba(52,152,219,0.3);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 13px;
        color: #9bb5d6;
        margin: 10px 0;
    }
    /* ── rank pill ───────────────────────────────────────────────────── */
    .rank-pill {
        display: inline-block;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
    }
    /* ── delta metrics ───────────────────────────────────────────────── */
    [data-testid="stMetricDelta"] { font-size: 13px; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 800; }
    /* ── hide streamlit branding ─────────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached — loads only once per session)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading branch data…")
def get_data() -> pd.DataFrame:
    df = load_data()
    # ── derive Rank (1 = most efficient) ────────────────────────────────
    df["Rank"] = df["Efficiency_Score"].rank(ascending=False, method="min").astype(int)
    # ── extract city hint from branch name (last word before "Branch") ──
    df["City_Hint"] = df["Branch"].str.extract(r"^([\w\s]+?)(?:\s+Branch| FSC|$)")[0]
    return df

df = get_data()

# ── pre-compute slack once (cached) ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_slack(_df) -> pd.DataFrame:
    return compute_slack(_df)

slack_df = get_slack(df)

# ── session state for scenario store ─────────────────────────────────────────
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []

# ── derived globals ──────────────────────────────────────────────────────────
total_branches  = len(df)
avg_efficiency  = df["Efficiency_Score"].mean()
n_efficient     = (df["Status"] == "Efficient").sum()
n_inefficient   = (df["Status"] == "Inefficient").sum()
top_branch      = df.loc[df["Efficiency_Score"].idxmax(), "Branch"]
bottom_branch   = df.loc[df["Efficiency_Score"].idxmin(), "Branch"]
sorted_branches = df.sort_values("Branch")["Branch"].tolist()

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── logo ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-logo">
            <h1>🏦 DecisionLens</h1>
            <p>Branch Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── branch selector ───────────────────────────────────────────────────
    st.markdown("**🔍 Branch Selection**")
    selected_branch = st.selectbox(
        "Select a branch",
        options=sorted_branches,
        index=sorted_branches.index(bottom_branch),
        label_visibility="collapsed",
    )

    st.divider()

    # ── performance filters ───────────────────────────────────────────────
    st.markdown("**⚙️ Performance Filters**")

    eff_min, eff_max = float(df["Efficiency_Score"].min()), float(df["Efficiency_Score"].max())
    eff_range = st.slider(
        "Efficiency Score Range",
        min_value=round(eff_min, 4),
        max_value=round(eff_max, 4),
        value=(round(eff_min, 4), round(eff_max, 4)),
        step=0.0001,
        format="%.4f",
    )

    status_filter = st.multiselect(
        "Status Filter",
        options=["Efficient", "Inefficient"],
        default=["Efficient", "Inefficient"],
    )

    # ── branch search ─────────────────────────────────────────────────────
    st.markdown("**🔎 Branch Search**")
    search_term = st.text_input("Search branches…", placeholder="e.g. Chicago, Kroger", label_visibility="collapsed")

    st.divider()

    # ── top-N leaderboard control ─────────────────────────────────────────
    st.markdown("**📊 Leaderboard Size**")
    top_n = st.slider("Show top N branches", min_value=5, max_value=50, value=20, step=5)

    st.divider()

    # ── sidebar stats ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="info-box">
            <b>📡 Dataset</b><br>
            {total_branches:,} branches loaded<br>
            {n_efficient:,} efficient &nbsp;|&nbsp; {n_inefficient:,} inefficient
        </div>
        """,
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# FILTERED DATAFRAME  (responds to sidebar controls)
# ════════════════════════════════════════════════════════════════════════════

filtered_df = df[
    (df["Efficiency_Score"] >= eff_range[0]) &
    (df["Efficiency_Score"] <= eff_range[1]) &
    (df["Status"].isin(status_filter))
].copy()

if search_term.strip():
    filtered_df = filtered_df[
        filtered_df["Branch"].str.contains(search_term.strip(), case=False, na=False)
    ]

# ── selected branch row ──────────────────────────────────────────────────────
sel = df[df["Branch"] == selected_branch].iloc[0]
sel_score  = sel["Efficiency_Score"]
sel_status = sel["Status"]
sel_rank   = int(sel["Rank"])
sel_pct    = sel_rank / total_branches * 100

# ── top efficient branch ───────────────────────────────────────────────────
top_eff_row = df[df["Status"] == "Efficient"].sort_values("Efficiency_Score", ascending=False).iloc[0]

# ════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════════════════

badge_class = "badge-efficient" if sel_status == "Efficient" else "badge-inefficient"

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:10px;">
        <div>
            <h2 style="margin:0; font-size:28px; font-weight:900; color:#edf2f4;">
                🏦 DecisionLens Analytics
            </h2>
            <p style="margin:0; color:#8592a3; font-size:14px;">
                Selected: <b style="color:#74b9ff;">{selected_branch}</b>
                &nbsp; <span class="{badge_class}">{sel_status}</span>
                &nbsp; <span class="rank-pill">Rank #{sel_rank:,} / {total_branches:,}</span>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL KPI ROW  (always visible)
# ════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5 = st.columns(5)

def kpi_card(col, label, value, sub=""):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_card(k1, "Total Branches", f"{total_branches:,}", f"{len(filtered_df):,} in filter")
kpi_card(k2, "Efficient", f"{n_efficient:,}", f"{n_efficient/total_branches*100:.1f}% of fleet")
kpi_card(k3, "Avg Efficiency", f"{avg_efficiency:.5f}", "across all branches")
kpi_card(k4, "Selected Score", f"{sel_score:.5f}", f"Rank #{sel_rank:,}")
kpi_card(k5, "Percentile", f"Top {100-sel_pct:.1f}%", f"better than {sel_pct:.1f}% of branches")

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab_overview, tab_branch, tab_map, tab_explorer, tab_sim, tab_scenarios = st.tabs([
    "📊 Overview",
    "🔬 Branch Analysis",
    "🗺️ Geographic Map",
    "📋 Data Explorer",
    "🔮 What-If Simulator",
    "📐 Scenario Comparison",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ────────────────────────────────────────────────────────────────────────────

with tab_overview:

    st.markdown('<div class="section-header">📈 Efficiency Leaderboard</div>', unsafe_allow_html=True)

    col_lead, col_dist = st.columns([1.3, 1])

    with col_lead:
        st.caption(f"Top {top_n} branches by DEA efficiency score — filtered view ({len(filtered_df):,} branches)")
        # pass filtered df to leaderboard, highlight selected
        lead_fig = create_efficiency_leaderboard(filtered_df, top_n=top_n)
        # mark selected branch if it appears in the top N
        top_n_df = filtered_df.sort_values("Efficiency_Score", ascending=False).head(top_n)
        if selected_branch in top_n_df["Branch"].values:
            # add annotation arrow
            sel_score_top = sel_score
            lead_fig.add_annotation(
                y=selected_branch,
                x=sel_score_top,
                text="◀ Selected",
                showarrow=False,
                xanchor="left",
                xshift=5,
                font=dict(color=ACCENT_COLOR, size=10),
            )
        st.plotly_chart(lead_fig, use_container_width=True, config={"displayModeBar": False})

    with col_dist:
        st.caption(f"Efficiency score distribution — {len(filtered_df):,} branches")
        dist_fig = create_efficiency_distribution(filtered_df)
        # mark selected branch score
        dist_fig.add_vline(
            x=sel_score,
            line_dash="solid",
            line_color=ACCENT_COLOR,
            line_width=2,
            annotation_text="Selected",
            annotation_font_color=ACCENT_COLOR,
            annotation_position="top right",
        )
        st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})

    # ── fleet health summary ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏥 Fleet Health Summary</div>', unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)

    with h1:
        # gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_efficiency * 100,
            title=dict(text="Fleet Average Efficiency (%)", font=dict(color=TEXT_COLOR, size=14)),
            number=dict(suffix="%", font=dict(color=TEXT_COLOR, size=28)),
            delta=dict(reference=99.95, relative=False, valueformat=".4f"),
            gauge=dict(
                axis=dict(range=[99.8, 100], tickfont=dict(color=TEXT_COLOR), tickformat=".2f"),
                bar=dict(color=ACCENT_COLOR),
                bgcolor=CARD_BG,
                bordercolor="rgba(255,255,255,0.1)",
                steps=[
                    dict(range=[99.8, 99.9],  color="rgba(231,76,60,0.18)"),
                    dict(range=[99.9, 99.99], color="rgba(241,196,15,0.18)"),
                    dict(range=[99.99, 100],  color="rgba(46,204,113,0.18)"),
                ],
                threshold=dict(
                    line=dict(color=EFFICIENT_COLOR, width=3),
                    thickness=0.75,
                    value=EFFICIENCY_THRESHOLD * 100,
                ),
            ),
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    with h2:
        # status donut
        donut = go.Figure(go.Pie(
            labels=["Efficient", "Inefficient"],
            values=[n_efficient, n_inefficient],
            hole=0.6,
            marker=dict(colors=[EFFICIENT_COLOR, INEFFICIENT_COLOR],
                        line=dict(color=DARK_BG, width=2)),
            textinfo="percent",
            textfont=dict(color=TEXT_COLOR, size=12),
            hovertemplate="%{label}: %{value:,} branches (%{percent})<extra></extra>",
        ))
        donut.update_layout(
            title=dict(text="Status Split", font=dict(color=TEXT_COLOR, size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color=TEXT_COLOR), bgcolor="rgba(0,0,0,0)"),
            height=240,
            margin=dict(l=10, r=10, t=50, b=10),
            annotations=[dict(
                text=f"{n_efficient/total_branches*100:.1f}%<br>Efficient",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font=dict(color=TEXT_COLOR),
            )],
        )
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

    with h3:
        # score percentile histogram (selected branch highlighted)
        pct_df = df["Efficiency_Score"].sort_values()
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Histogram(
            x=pct_df,
            nbinsx=50,
            marker_color=CARD_BG,
            marker_line_color=ACCENT_COLOR,
            marker_line_width=0.5,
            opacity=0.9,
            name="All Branches",
        ))
        fig_pct.add_vline(
            x=sel_score, line_dash="solid",
            line_color=ACCENT_COLOR, line_width=2.5,
            annotation_text=f"Selected ({sel_score:.4f})",
            annotation_font_color=ACCENT_COLOR,
            annotation_position="top left",
        )
        fig_pct.update_layout(
            title=dict(text="Score Distribution (selected ▲)", font=dict(color=TEXT_COLOR, size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color=TEXT_COLOR), gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(tickfont=dict(color=TEXT_COLOR), gridcolor="rgba(255,255,255,0.06)"),
            height=240,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_pct, use_container_width=True, config={"displayModeBar": False})

    # ── efficiency frontier scatter ─────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Efficiency Frontier</div>', unsafe_allow_html=True)
    st.caption("Staff vs Deposits 2016 — brighter = more efficient. Dot size reflects Staff count.")
    front_fig = create_frontier_plot(filtered_df)
    # add selected branch star
    if sel["Staff"] and sel["Deposits_2016"]:
        front_fig.add_trace(
            go.Scatter(
                x=[sel["Staff"]],
                y=[sel["Deposits_2016"]],
                mode="markers+text",
                marker=dict(symbol="star", size=18, color=ACCENT_COLOR,
                            line=dict(color="white", width=1.5)),
                text=[selected_branch[:22] + "…" if len(selected_branch) > 22 else selected_branch],
                textposition="top center",
                textfont=dict(color=ACCENT_COLOR, size=10),
                name="Selected",
                hoverinfo="skip",
            )
        )
    st.plotly_chart(front_fig, use_container_width=True, config={"displayModeBar": False})


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — BRANCH ANALYSIS
# ────────────────────────────────────────────────────────────────────────────

with tab_branch:

    # ── selected branch hero card ───────────────────────────────────────────
    card_cls = "branch-card efficient" if sel_status == "Efficient" else "branch-card inefficient"
    score_color = EFFICIENT_COLOR if sel_status == "Efficient" else INEFFICIENT_COLOR
    gap_to_top  = top_eff_row["Efficiency_Score"] - sel_score
    gap_to_avg  = sel_score - avg_efficiency

    st.markdown(
        f"""
        <div class="{card_cls}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:22px; font-weight:800; color:#edf2f4; margin-bottom:6px;">
                        {selected_branch}
                    </div>
                    <span class="{'badge-efficient' if sel_status=='Efficient' else 'badge-inefficient'}">{sel_status}</span>
                    &nbsp;
                    <span class="rank-pill">Rank #{sel_rank:,} of {total_branches:,}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:40px; font-weight:900; color:{score_color}; font-family:monospace;">
                        {sel_score:.6f}
                    </div>
                    <div style="font-size:12px; color:#8592a3;">DEA Efficiency Score</div>
                </div>
            </div>
            <div style="margin-top:14px; display:flex; gap:32px; font-size:13px; color:#8592a3;">
                <span>📉 Gap to top: <b style="color:#e74c3c;">{gap_to_top:.6f}</b></span>
                <span>📊 vs average: <b style="color:{'#2ecc71' if gap_to_avg>=0 else '#e74c3c'};">{gap_to_avg:+.6f}</b></span>
                <span>👥 Staff: <b style="color:#edf2f4;">{sel['Staff']:.3f}</b></span>
                <span>💰 Deposits 2016: <b style="color:#edf2f4;">${sel['Deposits_2016']:,.0f}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI metrics row ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Efficiency Score", f"{sel_score:.6f}",
              delta=f"{gap_to_avg:+.6f} vs avg", delta_color="normal")
    m2.metric("Staff Count",      f"{sel['Staff']:.3f}",
              delta=f"{sel['Staff'] - df['Staff'].mean():+.3f} vs avg")
    m3.metric("Deposits 2016",    f"${sel['Deposits_2016']:,.0f}",
              delta=f"${sel['Deposits_2016']-df['Deposits_2016'].mean():+,.0f} vs avg")
    m4.metric("Avg Deposits",     f"${sel['Avg_Deposits']:,.0f}",
              delta=f"${sel['Avg_Deposits']-df['Avg_Deposits'].mean():+,.0f} vs avg")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── radar + positional scatter ──────────────────────────────────────────
    col_radar, col_pos = st.columns([1, 1])

    with col_radar:
        st.markdown('<div class="section-header">🕸️ Radar Comparison</div>', unsafe_allow_html=True)
        st.caption(f"**{selected_branch[:40]}** vs top efficient branch: **{top_eff_row['Branch'][:40]}**")
        radar_fig = create_radar_comparison(df, selected_branch=selected_branch)
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

    with col_pos:
        st.markdown('<div class="section-header">📍 Rank Position</div>', unsafe_allow_html=True)
        st.caption("Where does this branch sit in the efficiency spectrum?")

        # build waterfall-like rank chart
        rank_df = df.sort_values("Efficiency_Score", ascending=False).reset_index(drop=True)
        rank_df["X_idx"] = range(len(rank_df))
        colors_rank = [
            EFFICIENT_COLOR if s == "Efficient" else INEFFICIENT_COLOR
            for s in rank_df["Status"]
        ]
        # highlight selected
        highlight_idx = rank_df[rank_df["Branch"] == selected_branch].index[0]

        fig_rank = go.Figure()
        fig_rank.add_trace(go.Scatter(
            x=rank_df["X_idx"],
            y=rank_df["Efficiency_Score"],
            mode="lines",
            line=dict(color="rgba(52,152,219,0.4)", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.07)",
            name="All Branches",
            hovertemplate="%{customdata}: %{y:.6f}<extra></extra>",
            customdata=rank_df["Branch"],
        ))
        fig_rank.add_trace(go.Scatter(
            x=[highlight_idx],
            y=[sel_score],
            mode="markers+text",
            marker=dict(size=14, color=ACCENT_COLOR,
                        symbol="diamond", line=dict(color="white", width=2)),
            text=[f"Rank #{sel_rank}"],
            textposition="top center",
            textfont=dict(color=ACCENT_COLOR, size=11),
            name="Selected Branch",
            hovertemplate=f"{selected_branch}: {sel_score:.6f}<extra></extra>",
        ))
        fig_rank.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CARD_BG,
            xaxis=dict(title="Rank Position", tickfont=dict(color=TEXT_COLOR),
                       gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Efficiency Score", tickfont=dict(color=TEXT_COLOR),
                       gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(font=dict(color=TEXT_COLOR), bgcolor="rgba(0,0,0,0.3)"),
            height=380,
            margin=dict(l=60, r=20, t=30, b=50),
        )
        st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})

    # ── input vs output comparison ──────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Input vs Output Comparison</div>', unsafe_allow_html=True)
    st.caption("Comparing selected branch (blue) against fleet average (grey) and top efficient branch (green)")

    vars5 = ["Staff", "Operating_Cost", "Deposits_2016", "Deposit_Growth", "Avg_Deposits"]
    pretty = ["Staff", "Op. Cost ($)", "Deposits 2016 ($)", "Deposit Growth ($)", "Avg Deposits ($)"]

    # normalise all values to percentages of the max for barplot
    maxvals = df[vars5].max()
    sel_norm  = [(sel[v] / maxvals[v] * 100) if maxvals[v] != 0 else 0 for v in vars5]
    avg_norm  = [(df[v].mean() / maxvals[v] * 100) if maxvals[v] != 0 else 0 for v in vars5]
    top_norm  = [(top_eff_row[v] / maxvals[v] * 100) if maxvals[v] != 0 else 0 for v in vars5]

    fig_cmp = go.Figure()
    for name, vals, color in [
        ("Fleet Average", avg_norm, "rgba(127,140,141,0.7)"),
        (f"Top: {top_eff_row['Branch'][:25]}", top_norm, EFFICIENT_COLOR),
        (f"Selected: {selected_branch[:25]}", sel_norm, ACCENT_COLOR),
    ]:
        fig_cmp.add_trace(go.Bar(
            name=name,
            x=pretty,
            y=vals,
            marker_color=color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.2f}}% of max<extra></extra>",
        ))

    fig_cmp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        xaxis=dict(tickfont=dict(color=TEXT_COLOR), gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="% of fleet maximum", tickfont=dict(color=TEXT_COLOR),
                   gridcolor="rgba(255,255,255,0.06)", ticksuffix="%"),
        legend=dict(font=dict(color=TEXT_COLOR, size=11), bgcolor="rgba(0,0,0,0.3)"),
        height=360,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    # ── why inefficient analysis box ────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Performance Diagnosis</div>', unsafe_allow_html=True)

    diag_cols = st.columns(3)
    insights = []

    # deposit-to-staff ratio
    if sel["Staff"] > 0:
        dep_per_staff_sel = sel["Deposits_2016"] / sel["Staff"]
        dep_per_staff_avg = (df["Deposits_2016"] / df["Staff"].replace(0, np.nan)).mean()
        ratio_vs_avg = dep_per_staff_sel / dep_per_staff_avg if dep_per_staff_avg else 1
        insights.append((
            "💼 Deposit-per-Staff Ratio",
            f"${dep_per_staff_sel:,.0f} per staff unit",
            f"Fleet avg: ${dep_per_staff_avg:,.0f}",
            ratio_vs_avg >= 1,
        ))

    # operating cost vs deposits
    if sel["Deposits_2016"] > 0:
        cost_ratio_sel = sel["Operating_Cost"] / sel["Deposits_2016"]
        cost_ratio_avg = (df["Operating_Cost"] / df["Deposits_2016"].replace(0, np.nan)).mean()
        cr_vs_avg = cost_ratio_sel / cost_ratio_avg if cost_ratio_avg else 1
        insights.append((
            "⚙️ Cost-to-Deposit Ratio",
            f"{cost_ratio_sel:.4f} ($cost per $1 deposit)",
            f"Fleet avg: {cost_ratio_avg:.4f}",
            cr_vs_avg <= 1,
        ))

    # deposit growth
    growth_avg = df["Deposit_Growth"].mean()
    growth_vs_avg = sel["Deposit_Growth"] / growth_avg if growth_avg else 1
    insights.append((
        "📈 Deposit Growth",
        f"${sel['Deposit_Growth']:,.0f}",
        f"Fleet avg: ${growth_avg:,.0f}",
        growth_vs_avg >= 1,
    ))

    for i, (title, val, sub, is_good) in enumerate(insights):
        icon = "✅" if is_good else "⚠️"
        col_bg = "rgba(46,204,113,0.08)" if is_good else "rgba(231,76,60,0.08)"
        border = EFFICIENT_COLOR if is_good else INEFFICIENT_COLOR
        diag_cols[i].markdown(
            f"""
            <div style="background:{col_bg}; border:1px solid {border}33;
                border-radius:10px; padding:16px 18px; height:100%;">
                <div style="font-size:13px; font-weight:700; color:#edf2f4; margin-bottom:6px;">
                    {icon} {title}
                </div>
                <div style="font-size:18px; font-weight:800; color:{'#2ecc71' if is_good else '#e74c3c'};">
                    {val}
                </div>
                <div style="font-size:11px; color:#8592a3; margin-top:4px;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── neighbours (nearest ranked) ─────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Nearby Ranked Branches</div>', unsafe_allow_html=True)
    st.caption("5 branches ranked just above and below the selected branch")
    margin_df = df.sort_values("Efficiency_Score", ascending=False).reset_index(drop=True)
    sel_idx   = margin_df[margin_df["Branch"] == selected_branch].index[0]
    lo = max(0, sel_idx - 5)
    hi = min(len(margin_df), sel_idx + 6)
    nbr_df = margin_df.iloc[lo:hi][
        ["Branch", "Efficiency_Score", "Status", "Staff", "Deposits_2016"]
    ].copy()
    nbr_df["Rank"] = range(lo + 1, hi + 1)

    def highlight_sel(row):
        if row["Branch"] == selected_branch:
            return ["background-color: rgba(52,152,219,0.2)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        nbr_df.style.apply(highlight_sel, axis=1).format(
            {"Efficiency_Score": "{:.6f}", "Staff": "{:.3f}",
             "Deposits_2016": "${:,.0f}"}
        ),
        use_container_width=True,
        height=280,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 6 — AI DECISION CONSULTANT
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🧠 AI Decision Consultant</div>', unsafe_allow_html=True)
    
    slack_row = slack_df[slack_df["Branch"] == selected_branch]
    if len(slack_row) == 0:
        st.info("No slack data available for the selected branch.")
    else:
        slack_row = slack_row.iloc[0]
        
        # 1. Compute all AI elements
        top_peers    = compute_benchmarks(df, selected_branch)
        root_cause   = root_cause_analysis(df, selected_branch, top_peers)
        recs         = smart_recommendations(slack_row, root_cause)
        sens_results = run_sensitivity_analysis(df, selected_branch)
        confidence   = compute_confidence(df, selected_branch, top_peers)
        narrative    = generate_insight_summary(selected_branch, sel_score, top_peers, recs, sens_results)
        
        # 2. Render Insight Summary
        conf_color = EFFICIENT_COLOR if confidence > 80 else MODERATE_COLOR if confidence > 60 else INEFFICIENT_COLOR
        st.markdown(
            f"""
            <div style="background:rgba(52,152,219,0.1); border-left:4px solid {ACCENT_COLOR}; 
                 border-radius:8px; padding:16px; margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <span style="font-size:15px; color:{TEXT_COLOR}; line-height:1.5; padding-right:20px;">
                        {narrative}
                    </span>
                    <div style="text-align:right; min-width:120px;">
                        <span style="font-size:11px; color:#bdc3c7; text-transform:uppercase; letter-spacing:1px;">Confidence</span><br>
                        <span style="font-size:24px; font-weight:700; color:{conf_color};">{confidence}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        if sel_score < EFFICIENCY_THRESHOLD:
            
            c1, c2 = st.columns(2)
            
            # 3. Root Cause Analysis
            with c1:
                st.markdown('<div style="font-size:14px; font-weight:600; margin-bottom:12px;">🔍 Root Cause Analysis (% Deviation)</div>', unsafe_allow_html=True)
                
                # Build table for root cause
                rc_data = []
                for var, data in root_cause.items():
                    # Format deviation with color
                    dev_p = data['dev_peer']
                    dev_t = data['dev_top']
                    
                    if var in INPUT_COLS: # Positive is bad (excess)
                        color_p = "#e74c3c" if dev_p > 0 else "#2ecc71"
                        color_t = "#e74c3c" if dev_t > 0 else "#2ecc71"
                    else: # Outputs: Negative is bad (shortfall)
                        color_p = "#e74c3c" if dev_p < 0 else "#2ecc71"
                        color_t = "#e74c3c" if dev_t < 0 else "#2ecc71"
                        
                    rc_data.append({
                        "Metric": var.replace("_", " "),
                        "Actual": f"{data['value']:,.1f}",
                        "vs Peer Avg": f"<span style='color:{color_p}'>{dev_p:+.1f}%</span>",
                        "vs Top Rank": f"<span style='color:{color_t}'>{dev_t:+.1f}%</span>",
                    })
                    
                st.markdown(
                    pd.DataFrame(rc_data).to_html(escape=False, index=False, 
                    classes=["table", "table-striped", "table-dark", "table-sm"],
                    border=0).replace('border="1"', 'style="width:100%; text-align:right; font-size:13px;"')
                    .replace('<th>', '<th style="text-align:right; border-bottom:1px solid #333; padding:8px;">')
                    .replace('<td>', '<td style="padding:8px; border-bottom:1px solid #1a1a2e;">'),
                    unsafe_allow_html=True
                )
                
            # 4. Benchmark Peers Radar
            with c2:
                st.markdown('<div style="font-size:14px; font-weight:600; margin-bottom:12px;">🧑🤝🧑 Benchmark Peers</div>', unsafe_allow_html=True)
                if len(top_peers) > 0:
                    peer_names = ", ".join(top_peers["Branch"].iloc[:3].tolist())
                    st.caption(f"Closest Efficient Peers: **{peer_names}**")
                    radar_fig2 = create_benchmark_radar_chart(df, selected_branch, top_peers)
                    st.plotly_chart(radar_fig2, use_container_width=True, config={"displayModeBar": False}, key="radar_benchmark")
                else:
                    st.info("No efficient peers found for benchmarking.")

            st.markdown("---")
            
            # 5. Smart Recommendations & Sensitivity
            c3, c4 = st.columns([1.3, 1])
            
            with c3:
                st.markdown('<div style="font-size:14px; font-weight:600; margin-bottom:12px;">📌 Priority Action Plan</div>', unsafe_allow_html=True)
                priority_colors = {
                    "High":   ("#e74c3c", "rgba(231,76,60,0.1)"),
                    "Medium": ("#f1c40f", "rgba(241,196,15,0.1)"),
                    "Low":    ("#2ecc71", "rgba(46,204,113,0.1)"),
                }
                for r in recs:
                    pc, bg = priority_colors.get(r["priority"], ("#3498db", "rgba(52,152,219,0.1)"))
                    st.markdown(
                        f"""
                        <div style="background:{bg}; border-left:4px solid {pc};
                            border-radius:8px; padding:12px 16px; margin:8px 0;">
                            <span style="font-size:14px; color:#edf2f4;">
                                {r['icon']} <b>{r['action']} {r['variable']}</b> 
                                by <span style="color:{pc}; font-weight:700;">{r['slack_pct']:.1f}%</span>
                            </span><br>
                            <span style="font-size:12px; color:#bdc3c7; display:block; margin-top:4px;">
                                {r['explanation']}
                            </span>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    
            with c4:
                st.markdown('<div style="font-size:14px; font-weight:600; margin-bottom:12px;">📊 Sensitivity Analysis</div>', unsafe_allow_html=True)
                st.caption("Marginal efficiency gain from a 5% isolated improvement.")
                sens_fig = create_sensitivity_chart(sens_results)
                st.plotly_chart(sens_fig, use_container_width=True, config={"displayModeBar": False}, key="sens_tornado")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — MAP
# ────────────────────────────────────────────────────────────────────────────

with tab_map:

    st.markdown('<div class="section-header">🗺️ Branch Geographic Distribution</div>', unsafe_allow_html=True)

    map_col, ctrl_col = st.columns([4, 1])

    with ctrl_col:
        st.markdown("**Map Controls**")
        show_selected_only = st.checkbox("Highlight selected branch", value=True)
        map_color_by = st.radio(
            "Colour by",
            options=["Efficiency_Score", "Status"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.metric("Mapped Branches",
                  f"{df[df['Latitude'].notna() & df['Longitude'].notna()].shape[0]:,}",
                  help="Branches with valid lat/lon")

    with map_col:
        map_df = filtered_df[
            filtered_df["Latitude"].notna() & filtered_df["Longitude"].notna()
        ].copy()
        map_df = map_df[
            (map_df["Latitude"].between(18, 72)) &
            (map_df["Longitude"].between(-170, -65))
        ]

        if map_color_by == "Efficiency_Score":
            map_fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Efficiency_Score",
                color_continuous_scale=[
                    [0.0, INEFFICIENT_COLOR],
                    [0.5, MODERATE_COLOR],
                    [1.0, EFFICIENT_COLOR],
                ],
                size_max=10,
                zoom=3.5,
                center=dict(lat=39.5, lon=-98.35),
                hover_name="Branch",
                hover_data={
                    "Efficiency_Score": ":.6f",
                    "Status": True,
                    "Staff": ":.2f",
                    "Deposits_2016": ":,.0f",
                    "Latitude": False,
                    "Longitude": False,
                },
                mapbox_style="carto-darkmatter",
                opacity=0.85,
                title="",
            )
        else:
            map_fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Status",
                color_discrete_map={"Efficient": EFFICIENT_COLOR, "Inefficient": INEFFICIENT_COLOR},
                size_max=10,
                zoom=3.5,
                center=dict(lat=39.5, lon=-98.35),
                hover_name="Branch",
                hover_data={
                    "Efficiency_Score": ":.6f",
                    "Status": True,
                    "Latitude": False,
                    "Longitude": False,
                },
                mapbox_style="carto-darkmatter",
                opacity=0.85,
                title="",
            )

        # ── overlay selected branch ──────────────────────────────────────
        if show_selected_only and pd.notna(sel.get("Latitude")) and pd.notna(sel.get("Longitude")):
            map_fig.add_trace(go.Scattermapbox(
                lat=[sel["Latitude"]],
                lon=[sel["Longitude"]],
                mode="markers+text",
                marker=dict(size=18, color=ACCENT_COLOR,
                            symbol="star"),
                text=[selected_branch[:20] + "…" if len(selected_branch) > 20 else selected_branch],
                textposition="top right",
                textfont=dict(color="white", size=11),
                name="Selected Branch",
                hovertemplate=f"<b>{selected_branch}</b><br>Score: {sel_score:.6f}<br>Status: {sel_status}<extra></extra>",
            ))

        map_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(font=dict(color=TEXT_COLOR), bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

    # ── regional breakdown ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌎 Regional Efficiency Breakdown</div>', unsafe_allow_html=True)

    # bin lat into rough US regions
    def lat_bin(lat):
        if pd.isna(lat):
            return "Unknown"
        if lat > 44:
            return "North (>44°N)"
        elif lat > 38:
            return "Central (38–44°N)"
        elif lat > 31:
            return "South (31–38°N)"
        else:
            return "Deep South (<31°N)"

    map_df["Region"] = map_df["Latitude"].apply(lat_bin)
    region_stats = (
        map_df.groupby("Region")["Efficiency_Score"]
        .agg(["mean", "count", "min", "max"])
        .round(6)
        .reset_index()
        .rename(columns={"mean": "Avg Score", "count": "Branches",
                          "min": "Min Score", "max": "Max Score"})
        .sort_values("Avg Score", ascending=False)
    )
    st.dataframe(region_stats, use_container_width=True, height=180)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — DATA EXPLORER
# ────────────────────────────────────────────────────────────────────────────

with tab_explorer:

    ex_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("**Explorer Controls**")
        sort_col = st.selectbox(
            "Sort by",
            ["Efficiency_Score", "Staff", "Deposits_2016", "Deposit_Growth", "Avg_Deposits"],
        )
        sort_asc = st.radio("Order", ["Descending", "Ascending"], index=0) == "Ascending"
        st.markdown("---")
        st.caption(f"**{len(filtered_df):,}** branches in current filter")

        # ── download ──────────────────────────────────────────────────────
        @st.cache_data
        def convert_csv(d):
            return d.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download filtered CSV",
            data=convert_csv(filtered_df),
            file_name="decisionlens_filtered.csv",
            mime="text/csv",
        )

    with ex_col:
        st.markdown('<div class="section-header">📋 Branch Data Explorer</div>', unsafe_allow_html=True)

        show_cols = ["Branch", "Efficiency_Score", "Status", "Rank",
                     "Staff", "Operating_Cost", "Deposits_2016",
                     "Deposit_Growth", "Avg_Deposits"]

        display_df = (
            filtered_df[show_cols]
            .sort_values(sort_col, ascending=sort_asc)
            .reset_index(drop=True)
        )

        st.dataframe(
            display_df.style
            .background_gradient(subset=["Efficiency_Score"], cmap="RdYlGn")
            .format({
                "Efficiency_Score": "{:.6f}",
                "Staff": "{:.3f}",
                "Operating_Cost": "${:,.0f}",
                "Deposits_2016": "${:,.0f}",
                "Deposit_Growth": "${:,.0f}",
                "Avg_Deposits": "${:,.0f}",
            }),
            use_container_width=True,
            height=520,
        )

    # ── full interactive Plotly table ───────────────────────────────────────
    st.markdown('<div class="section-header">📊 Interactive Table (Plotly)</div>', unsafe_allow_html=True)
    tbl_fig = generate_summary_table(filtered_df, sort_by=sort_col, ascending=sort_asc)
    st.plotly_chart(tbl_fig, use_container_width=True, config={"displayModeBar": False})



# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — WHAT-IF SIMULATOR
# ────────────────────────────────────────────────────────────────────────────

with tab_sim:
    st.markdown('<div class="section-header">🔮 What-If Simulation Engine</div>', unsafe_allow_html=True)
    st.caption(f"Simulating: **{selected_branch}** | Original Score: **{sel_score:.6f}** | Status: **{sel_status}**")

    sim_col, ctrl_col2 = st.columns([2, 1])

    with ctrl_col2:
        st.markdown("**⚙️ Simulation Controls**")

        # ── Target Efficiency Mode toggle ─────────────────────────────────
        target_mode = st.toggle("🎯 Target Efficiency Mode", value=False,
                                help="Automatically compute minimum input reduction to hit your target score")

        if target_mode:
            target_score = st.slider("Target Efficiency Score",
                                     min_value=round(float(sel_score) + 0.0001, 4),
                                     max_value=1.0,
                                     value=min(round(float(sel_score) + 0.001, 4), 1.0),
                                     step=0.0001, format="%.4f")
            st.markdown("---")
            with st.spinner("Computing minimum changes…"):
                target_result = target_efficiency_mode(df, selected_branch, target_score)

            if target_result.get("converged"):
                alpha = target_result["pct_reduction"]
                st.markdown(
                    f"""
                    <div class="kpi-card" style="margin-bottom:12px;">
                        <div class="kpi-label">Min Input Reduction</div>
                        <div class="kpi-value" style="font-size:26px;">{alpha:.2f}%</div>
                        <div class="kpi-sub">on Staff & Operating Cost</div>
                    </div>
                    <div class="info-box">
                        New Staff: <b>{target_result['new_staff']:,.3f}</b><br>
                        New Op. Cost: <b>${target_result['new_opcost']:,.0f}</b><br>
                        Projected Score: <b style="color:#2ecc71">{target_result['simulated_score']:.6f}</b>
                    </div>
                    """, unsafe_allow_html=True
                )
                # use target values for sliders default
                target_adj = 1.0 - target_result["alpha"]
                staff_m   = target_adj
                opcost_m  = target_adj
                dep_m = grw_m = avg_m = 1.0
            else:
                st.warning("Convergence not reached. Try a lower target score.")
                staff_m = opcost_m = dep_m = grw_m = avg_m = 1.0
        else:
            target_score = None
            st.markdown("**Input Adjustments** (reduce → improve efficiency)")
            staff_pct  = st.slider("Staff (%)",          -50, 50, 0, 1, format="%+d%%")
            opcost_pct = st.slider("Operating Cost (%)", -50, 50, 0, 1, format="%+d%%")
            st.markdown("**Output Adjustments** (increase → improve efficiency)")
            dep_pct   = st.slider("Deposits 2016 (%)",   -50, 50, 0, 1, format="%+d%%")
            grw_pct   = st.slider("Deposit Growth (%)",  -50, 50, 0, 1, format="%+d%%")
            avg_pct   = st.slider("Avg Deposits (%)",    -50, 50, 0, 1, format="%+d%%")
            staff_m   = 1 + staff_pct  / 100
            opcost_m  = 1 + opcost_pct / 100
            dep_m     = 1 + dep_pct    / 100
            grw_m     = 1 + grw_pct    / 100
            avg_m     = 1 + avg_pct    / 100

        st.markdown("---")
        run_sim = st.button("▶ Run Simulation", use_container_width=True, type="primary")

        # scenario saving
        st.markdown("**💾 Save Scenario**")
        scenario_name = st.text_input("Scenario name",
                                      value=f"{selected_branch[:20]}… #{len(st.session_state['scenarios'])+1}",
                                      label_visibility="collapsed")
        save_btn = st.button("Save Scenario", use_container_width=True)

    with sim_col:
        adjustments = {
            "Staff":          staff_m,
            "Operating_Cost": opcost_m,
            "Deposits_2016":  dep_m,
            "Deposit_Growth": grw_m,
            "Avg_Deposits":   avg_m,
        }

        if run_sim or target_mode:
            with st.spinner("Running single-branch CCR LP…"):
                sim_score = simulate_branch(
                    df, selected_branch,
                    staff_adj=staff_m, opcost_adj=opcost_m,
                    dep2016_adj=dep_m, growth_adj=grw_m, avgdep_adj=avg_m,
                )

            delta_score = sim_score - sel_score
            status_sim  = "Efficient" if sim_score >= EFFICIENCY_THRESHOLD else "Inefficient"
            sc_color    = EFFICIENT_COLOR if status_sim == "Efficient" else INEFFICIENT_COLOR

            # ── result KPI cards ─────────────────────────────────────────────
            r1, r2, r3 = st.columns(3)
            r1.metric("Original Score",  f"{sel_score:.6f}")
            r2.metric("Simulated Score", f"{sim_score:.6f}",
                      delta=f"{delta_score:+.6f}", delta_color="normal")
            r3.metric("New Status", status_sim)

            st.markdown(
                f"""
                <div style="background:{'rgba(46,204,113,0.1)' if delta_score >= 0 else 'rgba(231,76,60,0.1)'};
                    border:1px solid {sc_color}33; border-radius:10px; padding:14px 18px; margin:12px 0;">
                    <span style="font-size:14px; color:#edf2f4;">
                        {'✅ Improvement of' if delta_score >= 0 else '⚠️ Regression of'}
                        <b style="color:{sc_color};">&nbsp;{abs(delta_score):.6f}</b>
                        &nbsp;({abs(delta_score/sel_score*100) if sel_score else 0:.4f}%)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── before vs after chart ────────────────────────────────────────
            st.markdown('<div class="section-header">📊 Before vs After Comparison</div>', unsafe_allow_html=True)
            orig_row = df[df["Branch"] == selected_branch].iloc[0]
            ba_fig = create_before_after_chart(orig_row, adjustments)
            st.plotly_chart(ba_fig, use_container_width=True, config={"displayModeBar": False},
                            key="ba_chart_sim")

            # handle save
            if save_btn:
                store = ScenarioStore(st.session_state["scenarios"])
                store.add(
                    name            = scenario_name,
                    branch          = selected_branch,
                    original_score  = sel_score,
                    simulated_score = sim_score,
                    adjustments     = {k: round(v, 4) for k, v in adjustments.items()},
                )
                st.session_state["scenarios"] = store.scenarios
                st.success(f"✅ Scenario '{scenario_name}' saved! Switch to Scenario Comparison tab.")
        else:
            # ── pre-run state: show improvement gap chart + instructions ─────
            slack_row_sim = slack_df[slack_df["Branch"] == selected_branch]
            if len(slack_row_sim) > 0:
                gap_fig2 = create_improvement_gap_chart(slack_row_sim.iloc[0])
                st.plotly_chart(gap_fig2, use_container_width=True, config={"displayModeBar": False},
                                key="gap_fig_sim_pre_run")

            st.markdown(
                """
                <div class="info-box">
                    <b>💡 How to use the simulator</b><br>
                    1. Adjust the sliders on the right to change input/output values.<br>
                    2. Click <b>▶ Run Simulation</b> to compute the new efficiency score via a single CCR LP.<br>
                    3. Enable <b>Target Efficiency Mode</b> to let the system find the <i>minimum</i> change needed to hit your goal.<br>
                    4. Save scenarios and compare them in the <b>Scenario Comparison</b> tab.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — SCENARIO COMPARISON
# ────────────────────────────────────────────────────────────────────────────

with tab_scenarios:
    store = ScenarioStore(st.session_state.get("scenarios", []))

    sc1, sc2 = st.columns([3, 1])
    with sc2:
        if st.button("🗑️ Clear All Scenarios", use_container_width=True):
            st.session_state["scenarios"] = []
            st.rerun()

    with sc1:
        st.markdown('<div class="section-header">📐 Saved Scenario Library</div>', unsafe_allow_html=True)

    if len(store) == 0:
        st.markdown(
            """
            <div class="info-box">
                No scenarios saved yet.<br>
                Go to the <b>🔮 What-If Simulator</b> tab, run a simulation, then click <b>Save Scenario</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        scen_df = store.to_dataframe()

        # ── scenario summary table ───────────────────────────────────────────
        st.dataframe(
            scen_df.style
            .background_gradient(subset=["simulated_score"], cmap="RdYlGn")
            .format({
                "original_score":  "{:.6f}",
                "simulated_score": "{:.6f}",
                "delta":           "{:+.6f}",
                "delta_pct":       "{:+.4f}%",
            }),
            use_container_width=True,
            height=min(300, len(scen_df) * 35 + 60),
        )

        # ── comparison bar chart ─────────────────────────────────────────────
        comp_fig = create_scenario_comparison_chart(scen_df)
        st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})

        # ── delta waterfall ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Efficiency Improvement Summary</div>', unsafe_allow_html=True)
        wf_fig = go.Figure(go.Waterfall(
            name="Delta",
            orientation="v",
            x=scen_df["name"].tolist(),
            y=scen_df["delta"].tolist(),
            text=[f"{v:+.6f}" for v in scen_df["delta"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            increasing=dict(marker_color=EFFICIENT_COLOR),
            decreasing=dict(marker_color=INEFFICIENT_COLOR),
            totals=dict(marker_color=ACCENT_COLOR),
            connector=dict(line=dict(color="rgba(255,255,255,0.2)", width=1)),
        ))
        wf_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CARD_BG,
            xaxis=dict(tickfont=dict(color=TEXT_COLOR, size=10)),
            yaxis=dict(title="Δ Efficiency Score", tickfont=dict(color=TEXT_COLOR),
                       gridcolor="rgba(255,255,255,0.06)"),
            height=340,
            margin=dict(l=60, r=30, t=40, b=100),
        )
        st.plotly_chart(wf_fig, use_container_width=True, config={"displayModeBar": False})

        # ── export saved scenarios ────────────────────────────────────────────
        st.download_button(
            label="⬇️ Export Scenarios CSV",
            data=scen_df.to_csv(index=False).encode("utf-8"),
            file_name="decisionlens_scenarios.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <hr style="border:1px solid #1e2a45; margin:32px 0 16px 0;">
    <div style="text-align:center; color:#636e72; font-size:12px; padding-bottom:20px;">
        🏦 <b style="color:#74b9ff;">DecisionLens</b> · Phase 4–5 Decision Intelligence Platform ·
        {total_branches:,} branches · DEA Efficiency Analysis · What-If Simulation · Recommendation Engine ·
        Built with Streamlit + Plotly + PuLP
    </div>
    """,
    unsafe_allow_html=True,
)
