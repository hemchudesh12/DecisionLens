"""
visualization.py — Phase 3: Efficiency Analytics and Visualization Engine
DecisionLens BI Platform

This module provides all analytical visualizations for bank branch DEA efficiency
results. Each function is self-contained and returns a Plotly/Matplotlib figure,
making it easy to integrate into the Phase 4 Streamlit dashboard.

Dependencies: pandas, plotly, matplotlib, seaborn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── suppress non-critical warnings in large datasets ───────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")          # headless backend (safe for Streamlit / servers)

# ─── colour palette constants ────────────────────────────────────────────────
EFFICIENT_COLOR   = "#2ecc71"   # green
MODERATE_COLOR    = "#f1c40f"   # yellow
INEFFICIENT_COLOR = "#e74c3c"   # red
ACCENT_COLOR      = "#3498db"   # blue (used for highlights / single-series)
DARK_BG           = "#1a1a2e"
CARD_BG           = "#16213e"
TEXT_COLOR        = "#edf2f4"

# Efficiency threshold used in dea_model.py (score ≥ 0.9999 → "Efficient")
EFFICIENCY_THRESHOLD = 0.9999


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(
    dea_path: str = "dea_results.csv",
    bank_path: str = "bank_dea_dataset.csv",
) -> pd.DataFrame:
    """
    Load and merge the DEA results with the bank branch dataset.

    Parameters
    ----------
    dea_path  : path to dea_results.csv
    bank_path : path to bank_dea_dataset.csv

    Returns
    -------
    pd.DataFrame with columns:
        Branch, Staff, Operating_Cost, Deposits_2016, Deposit_Growth,
        Avg_Deposits, Latitude, Longitude, Efficiency_Score, Status
    """
    # ── load ────────────────────────────────────────────────────────────────
    dea   = pd.read_csv(dea_path)
    bank  = pd.read_csv(bank_path)

    # ── normalise column names (trim whitespace) ─────────────────────────────
    dea.columns  = dea.columns.str.strip()
    bank.columns = bank.columns.str.strip()

    # ── merge on Branch ──────────────────────────────────────────────────────
    merged = pd.merge(bank, dea, on="Branch", how="inner")

    # ── coerce numeric columns ───────────────────────────────────────────────
    numeric_cols = [
        "Staff", "Operating_Cost", "Deposits_2016",
        "Deposit_Growth", "Avg_Deposits",
        "Latitude", "Longitude", "Efficiency_Score",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ── add a human-readable efficiency percentage ───────────────────────────
    merged["Efficiency_Pct"] = (merged["Efficiency_Score"] * 100).round(4)

    # ── add colour label for maps / charts ──────────────────────────────────
    merged["Color_Label"] = merged["Efficiency_Score"].apply(_efficiency_color_label)

    print(f"[load_data] Loaded {len(merged):,} branches  |  "
          f"Efficient: {(merged['Status']=='Efficient').sum():,}  |  "
          f"Inefficient: {(merged['Status']=='Inefficient').sum():,}")

    return merged


def _efficiency_color_label(score: float) -> str:
    """Map an efficiency score to 'Efficient', 'Moderate', or 'Inefficient'."""
    if score >= EFFICIENCY_THRESHOLD:
        return "Efficient"
    elif score >= EFFICIENCY_THRESHOLD * 0.999:
        return "Moderate"
    else:
        return "Inefficient"


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: EFFICIENCY LEADERBOARD  (top-20 bar chart)
# ════════════════════════════════════════════════════════════════════════════

def create_efficiency_leaderboard(
    df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """
    Horizontal bar chart ranking the top-N branches by efficiency score.

    Efficient branches are coloured green; all others are coloured red so
    managers can immediately see where the threshold sits.

    Parameters
    ----------
    df    : merged DataFrame from load_data()
    top_n : number of branches to display (default 20)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # ── sort and pick top N ──────────────────────────────────────────────────
    top = (
        df.sort_values("Efficiency_Score", ascending=False)
        .head(top_n)
        .copy()
    )
    top = top.sort_values("Efficiency_Score", ascending=True)   # ascending for hbar

    # ── colour vector ────────────────────────────────────────────────────────
    colors = [
        EFFICIENT_COLOR if s == "Efficient" else INEFFICIENT_COLOR
        for s in top["Status"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=top["Branch"],
            x=top["Efficiency_Score"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            ),
            text=[f"{s:.6f}" for s in top["Efficiency_Score"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Efficiency Score: %{x:.6f}<br>"
                "Status: %{customdata}<extra></extra>"
            ),
            customdata=top["Status"],
        )
    )

    # ── threshold line ───────────────────────────────────────────────────────
    fig.add_vline(
        x=EFFICIENCY_THRESHOLD,
        line_dash="dash",
        line_color=MODERATE_COLOR,
        annotation_text=f"Threshold ({EFFICIENCY_THRESHOLD})",
        annotation_font_color=MODERATE_COLOR,
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=f"🏆 Branch Efficiency Leaderboard — Top {top_n}",
            font=dict(size=20, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Efficiency Score",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(255,255,255,0.08)",
            range=[top["Efficiency_Score"].min() - 0.0001,
                   top["Efficiency_Score"].max() + 0.0002],
        ),
        yaxis=dict(
            title="Branch",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR, size=10),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        height=max(500, top_n * 32),
        margin=dict(l=280, r=120, t=80, b=60),
        legend=dict(
            font=dict(color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0.3)",
        ),
    )

    # ── add manual legend ────────────────────────────────────────────────────
    for label, color in [("Efficient", EFFICIENT_COLOR),
                         ("Inefficient", INEFFICIENT_COLOR)]:
        fig.add_trace(
            go.Bar(
                x=[None], y=[None],
                orientation="h",
                name=label,
                marker_color=color,
            )
        )

    return fig


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: EFFICIENCY DISTRIBUTION  (histogram)
# ════════════════════════════════════════════════════════════════════════════

def create_efficiency_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of efficiency scores across all branches.

    Helps answer:
      • Are most branches efficient?
      • Is there a long tail of under-performers?

    Returns
    -------
    plotly.graph_objects.Figure
    """
    scores  = df["Efficiency_Score"].dropna()
    eff_cnt = (df["Status"] == "Efficient").sum()
    ine_cnt = (df["Status"] == "Inefficient").sum()

    fig = go.Figure()

    # ── inefficient portion ──────────────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=scores[df["Status"] == "Inefficient"],
            nbinsx=60,
            name="Inefficient",
            marker_color=INEFFICIENT_COLOR,
            opacity=0.85,
            hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra></extra>",
        )
    )

    # ── efficient portion ────────────────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=scores[df["Status"] == "Efficient"],
            nbinsx=60,
            name="Efficient",
            marker_color=EFFICIENT_COLOR,
            opacity=0.85,
            hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra></extra>",
        )
    )

    # ── threshold annotation ─────────────────────────────────────────────────
    fig.add_vline(
        x=EFFICIENCY_THRESHOLD,
        line_dash="dot",
        line_color=MODERATE_COLOR,
        line_width=2,
        annotation_text=f"Threshold = {EFFICIENCY_THRESHOLD}",
        annotation_font_color=MODERATE_COLOR,
        annotation_position="top left",
    )

    # ── summary box ─────────────────────────────────────────────────────────
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        text=(
            f"<b>Summary</b><br>"
            f"Total Branches: {len(df):,}<br>"
            f"Efficient: {eff_cnt:,} ({eff_cnt/len(df)*100:.1f}%)<br>"
            f"Inefficient: {ine_cnt:,} ({ine_cnt/len(df)*100:.1f}%)<br>"
            f"Mean Score: {scores.mean():.6f}<br>"
            f"Median Score: {scores.median():.6f}"
        ),
        showarrow=False,
        bgcolor="rgba(0,0,0,0.55)",
        bordercolor=ACCENT_COLOR,
        borderwidth=1,
        font=dict(color=TEXT_COLOR, size=11),
        align="left",
        xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text="📊 Efficiency Score Distribution — All Branches",
            font=dict(size=20, color=TEXT_COLOR),
        ),
        barmode="overlay",
        xaxis=dict(
            title="Efficiency Score",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Number of Branches",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        legend=dict(
            font=dict(color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0.3)",
        ),
        height=500,
        margin=dict(l=70, r=30, t=80, b=70),
    )

    return fig


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: EFFICIENCY FRONTIER PLOT  (scatter — Staff vs Deposits_2016)
# ════════════════════════════════════════════════════════════════════════════

def create_frontier_plot(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: Staff (x) vs Deposits_2016 (y), coloured by Efficiency_Score.

    Efficient branches cluster at the frontier (high output per input).
    Points are capped at the 99th percentile on both axes to prevent extreme
    outliers from collapsing the scale; outliers are shown with a different
    marker shape.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    plot_df = df[["Branch", "Staff", "Deposits_2016",
                  "Efficiency_Score", "Status", "Operating_Cost"]].dropna().copy()

    # ── cap extremes for readability ─────────────────────────────────────────
    x_cap = plot_df["Staff"].quantile(0.99)
    y_cap = plot_df["Deposits_2016"].quantile(0.99)
    inliers  = plot_df[(plot_df["Staff"] <= x_cap) &
                        (plot_df["Deposits_2016"] <= y_cap)]
    outliers = plot_df[~((plot_df["Staff"] <= x_cap) &
                          (plot_df["Deposits_2016"] <= y_cap))]

    fig = go.Figure()

    # ── inlier scatter (colour gradient) ────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=inliers["Staff"],
            y=inliers["Deposits_2016"],
            mode="markers",
            marker=dict(
                size=7,
                color=inliers["Efficiency_Score"],
                colorscale=[
                    [0.0, INEFFICIENT_COLOR],
                    [0.5, MODERATE_COLOR],
                    [1.0, EFFICIENT_COLOR],
                ],
                colorbar=dict(
                    title=dict(text="Efficiency Score",
                               font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    thickness=14,
                ),
                opacity=0.8,
                line=dict(width=0),
            ),
            name="Branches",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Staff: %{x:,.2f}<br>"
                "Deposits 2016: $%{y:,.0f}<br>"
                "Efficiency: %{customdata[1]:.6f}<br>"
                "Status: %{customdata[2]}<extra></extra>"
            ),
            customdata=inliers[["Branch", "Efficiency_Score", "Status"]].values,
        )
    )

    # ── outlier markers ──────────────────────────────────────────────────────
    if len(outliers) > 0:
        fig.add_trace(
            go.Scatter(
                x=outliers["Staff"],
                y=outliers["Deposits_2016"],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color=outliers["Efficiency_Score"],
                    colorscale=[[0, INEFFICIENT_COLOR], [1, EFFICIENT_COLOR]],
                    line=dict(color="white", width=0.5),
                ),
                name="Outlier Branches",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Staff: %{x:,.2f}<br>"
                    "Deposits 2016: $%{y:,.0f}<br>"
                    "Efficiency: %{customdata[1]:.6f}<extra></extra>"
                ),
                customdata=outliers[["Branch", "Efficiency_Score", "Status"]].values,
            )
        )

    fig.update_layout(
        title=dict(
            text="🎯 Efficiency Frontier — Staff vs Deposits 2016",
            font=dict(size=20, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Staff Count",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Deposits 2016 ($)",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        legend=dict(
            font=dict(color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0.3)",
        ),
        height=550,
        margin=dict(l=70, r=30, t=80, b=70),
    )

    return fig


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: RADAR CHART  (selected branch vs top efficient branch)
# ════════════════════════════════════════════════════════════════════════════

def create_radar_comparison(
    df: pd.DataFrame,
    selected_branch: str | None = None,
) -> go.Figure:
    """
    Radar (spider) chart comparing a chosen branch to the top efficient branch.

    Variables normalised to [0, 1] per-column (min-max) so they can share one
    common scale on the radar axes.

    Parameters
    ----------
    df              : merged DataFrame from load_data()
    selected_branch : branch name to compare; if None the most-inefficient
                      branch is chosen automatically

    Returns
    -------
    plotly.graph_objects.Figure
    """
    radar_vars = [
        "Staff", "Operating_Cost", "Deposits_2016",
        "Deposit_Growth", "Avg_Deposits",
    ]
    plot_df = df[["Branch", "Status", "Efficiency_Score"] + radar_vars].dropna()

    # ── pick reference branches ──────────────────────────────────────────────
    top_branch_row = (
        plot_df[plot_df["Status"] == "Efficient"]
        .sort_values("Efficiency_Score", ascending=False)
        .iloc[0]
    )

    if selected_branch and selected_branch in plot_df["Branch"].values:
        sel_row = plot_df[plot_df["Branch"] == selected_branch].iloc[0]
    else:
        # default: worst-performing branch
        sel_row = (
            plot_df.sort_values("Efficiency_Score", ascending=True).iloc[0]
        )
        selected_branch = sel_row["Branch"]

    # ── min-max normalise across the full dataset ────────────────────────────
    mins = plot_df[radar_vars].min()
    maxs = plot_df[radar_vars].max()
    rngs = (maxs - mins).replace(0, 1)  # avoid div-by-zero

    def normalise(row):
        return ((row[radar_vars] - mins) / rngs).tolist()

    top_vals = normalise(top_branch_row)
    sel_vals = normalise(sel_row)

    # ── close the polygon ────────────────────────────────────────────────────
    categories = radar_vars + [radar_vars[0]]
    top_vals   = top_vals   + [top_vals[0]]
    sel_vals   = sel_vals   + [sel_vals[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=top_vals,
            theta=categories,
            fill="toself",
            name=f"Top: {top_branch_row['Branch'][:35]}",
            line=dict(color=EFFICIENT_COLOR, width=2),
            fillcolor=f"rgba(46,204,113,0.20)",
            hovertemplate="<b>%{theta}</b><br>Normalised Value: %{r:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=sel_vals,
            theta=categories,
            fill="toself",
            name=f"Selected: {selected_branch[:35]}",
            line=dict(color=ACCENT_COLOR, width=2),
            fillcolor=f"rgba(52,152,219,0.20)",
            hovertemplate="<b>%{theta}</b><br>Normalised Value: %{r:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="🕸️ Radar Comparison — Selected vs Top Efficient Branch",
            font=dict(size=20, color=TEXT_COLOR),
        ),
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color=TEXT_COLOR),
                gridcolor="rgba(255,255,255,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT_COLOR, size=12),
                gridcolor="rgba(255,255,255,0.15)",
            ),
        ),
        paper_bgcolor=DARK_BG,
        legend=dict(
            font=dict(color=TEXT_COLOR, size=11),
            bgcolor="rgba(0,0,0,0.4)",
            x=0.85, y=1.15,
        ),
        height=520,
        margin=dict(l=80, r=80, t=100, b=80),
    )

    return fig


# ════════════════════════════════════════════════════════════════════════════
# STEP 6: GEOSPATIAL BRANCH MAP
# ════════════════════════════════════════════════════════════════════════════

def create_branch_map(df: pd.DataFrame) -> go.Figure:
    """
    Interactive map plotting all branches coloured by efficiency score.

    Colour scale:
        Red  → Inefficient (low score)
        Yellow → Moderate
        Green → Efficient (score ≈ 1.0)

    Branches without valid lat/lon are silently excluded.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    map_df = df[
        df["Latitude"].notna() & df["Longitude"].notna()
    ].copy()

    # ── clamp lat/lon to reasonable continental-US bounds ───────────────────
    map_df = map_df[
        (map_df["Latitude"].between(18, 72)) &
        (map_df["Longitude"].between(-170, -65))
    ]

    fig = px.scatter_mapbox(
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
        center=dict(lat=39.5, lon=-98.35),   # geographic centre of US
        hover_name="Branch",
        hover_data={
            "Efficiency_Score": ":.6f",
            "Status": True,
            "Staff": ":.2f",
            "Deposits_2016": ":,.0f",
            "Latitude": False,
            "Longitude": False,
        },
        title="🗺️ Branch Efficiency Map",
        mapbox_style="carto-darkmatter",
        opacity=0.85,
    )

    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text="Efficiency Score",
                       font=dict(color=TEXT_COLOR)),
            tickfont=dict(color=TEXT_COLOR),
            thickness=14,
        )
    )

    fig.update_layout(
        title=dict(
            text="🗺️ Branch Efficiency Map — Geographic View",
            font=dict(size=20, color=TEXT_COLOR),
        ),
        paper_bgcolor=DARK_BG,
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


# ════════════════════════════════════════════════════════════════════════════
# STEP 7: BRANCH EFFICIENCY TABLE
# ════════════════════════════════════════════════════════════════════════════

def generate_summary_table(
    df: pd.DataFrame,
    sort_by: str = "Efficiency_Score",
    ascending: bool = False,
    max_rows: int = 5000,
) -> go.Figure:
    """
    Interactive sortable table: Branch | Efficiency_Score | Status |
                                 Staff | Deposits_2016

    The table uses conditional cell colours so efficient branches stand
    out in green and inefficient ones in red.

    Parameters
    ----------
    df        : merged DataFrame from load_data()
    sort_by   : column to sort by (default 'Efficiency_Score')
    ascending : sort direction
    max_rows  : cap for very large datasets (default 5 000)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cols = ["Branch", "Efficiency_Score", "Status", "Staff", "Deposits_2016"]
    tbl  = (
        df[cols]
        .dropna(subset=["Efficiency_Score"])
        .sort_values(sort_by, ascending=ascending)
        .head(max_rows)
        .copy()
    )

    # ── cell colours for Status column ──────────────────────────────────────
    status_colors = [
        EFFICIENT_COLOR if s == "Efficient" else INEFFICIENT_COLOR
        for s in tbl["Status"]
    ]

    # ── format numeric cols for display ─────────────────────────────────────
    eff_fmt      = tbl["Efficiency_Score"].map(lambda x: f"{x:.6f}")
    staff_fmt    = tbl["Staff"].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "—")
    deposits_fmt = tbl["Deposits_2016"].map(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Branch</b>", "<b>Efficiency Score</b>",
                            "<b>Status</b>", "<b>Staff</b>",
                            "<b>Deposits 2016</b>"],
                    fill_color=CARD_BG,
                    font=dict(color=TEXT_COLOR, size=13),
                    align="left",
                    line_color="rgba(255,255,255,0.1)",
                    height=36,
                ),
                cells=dict(
                    values=[
                        tbl["Branch"],
                        eff_fmt,
                        tbl["Status"],
                        staff_fmt,
                        deposits_fmt,
                    ],
                    fill_color=[
                        # alternate row shading for readability
                        ["#1a1a2e" if i % 2 == 0 else "#16213e"
                         for i in range(len(tbl))],
                        ["#1a1a2e" if i % 2 == 0 else "#16213e"
                         for i in range(len(tbl))],
                        status_colors,        # status column uses traffic-light colours
                        ["#1a1a2e" if i % 2 == 0 else "#16213e"
                         for i in range(len(tbl))],
                        ["#1a1a2e" if i % 2 == 0 else "#16213e"
                         for i in range(len(tbl))],
                    ],
                    font=dict(color=[TEXT_COLOR, TEXT_COLOR,
                                     "white", TEXT_COLOR, TEXT_COLOR],
                              size=11),
                    align=["left", "center", "center", "right", "right"],
                    line_color="rgba(255,255,255,0.05)",
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"📋 Branch Efficiency Table  (sorted by {sort_by})",
            font=dict(size=20, color=TEXT_COLOR),
        ),
        paper_bgcolor=DARK_BG,
        height=max(400, min(800, len(tbl) * 30 + 100)),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE RUNNER  (generates & saves all charts as HTML)
# ════════════════════════════════════════════════════════════════════════════

def generate_all_charts(
    dea_path: str = "dea_results.csv",
    bank_path: str = "bank_dea_dataset.csv",
    output_dir: str = "charts",
) -> dict[str, go.Figure]:
    """
    Load data and generate all visualisations in one call.

    Also saves each chart as a stand-alone HTML file to `output_dir`.

    Parameters
    ----------
    dea_path   : path to dea_results.csv
    bank_path  : path to bank_dea_dataset.csv
    output_dir : directory to write HTML files (created if absent)

    Returns
    -------
    dict mapping chart name → Plotly figure
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    df = load_data(dea_path, bank_path)

    # ── generate every chart ─────────────────────────────────────────────────
    charts = {
        "leaderboard":    create_efficiency_leaderboard(df),
        "distribution":   create_efficiency_distribution(df),
        "frontier":       create_frontier_plot(df),
        "radar":          create_radar_comparison(df),   # default: top vs worst
        "map":            create_branch_map(df),
        "table":          generate_summary_table(df),
    }

    # ── persist to HTML for quick browser preview ────────────────────────────
    for name, fig in charts.items():
        out_path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"  ✅  {name:15s} → {out_path}")

    print(f"\n[generate_all_charts] All {len(charts)} charts saved to '{output_dir}/'")
    return charts


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    charts = generate_all_charts()
    print("\nPhase 3 complete. Open any file in the 'charts/' folder to preview.")
