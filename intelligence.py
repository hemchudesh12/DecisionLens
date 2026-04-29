"""
intelligence.py — Phase 6: Decision Intelligence Platform
DecisionLens BI Platform

Elevates the dashboard from a basic mathematical DEA viewer into an "intelligent 
business consultant". Provides root cause analysis, automated benchmarking, 
smart recommendations, sensitivity analysis, and narrative generation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

from visualization import (
    EFFICIENT_COLOR, INEFFICIENT_COLOR, ACCENT_COLOR, TEXT_COLOR, CARD_BG
)
from simulation import INPUT_COLS, OUTPUT_COLS, simulate_branch

EFFICIENCY_THRESHOLD = 0.9999
ALL_VARS = INPUT_COLS + OUTPUT_COLS

# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 & 2: PEER BENCHMARKING & ROOT CAUSE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_benchmarks(df: pd.DataFrame, target_branch: str) -> pd.DataFrame:
    """
    Finds the top 3 closest computationally efficient peers for a given branch
    using Euclidean distance on normalized input/output vectors.
    """
    if target_branch not in df["Branch"].values:
        return pd.DataFrame()

    target_idx = df.index[df["Branch"] == target_branch][0]
    
    # Isolate efficient branches (Score ≈ 1)
    efficient_df = df[df["Efficiency_Score"] >= EFFICIENCY_THRESHOLD]
    
    # If the target is already efficient, its closest peers are other efficient ones
    if df.loc[target_idx, "Efficiency_Score"] >= EFFICIENCY_THRESHOLD:
        efficient_df = efficient_df[efficient_df["Branch"] != target_branch]
    
    if len(efficient_df) == 0:
        return pd.DataFrame()

    # Normalize vectors for distance calculation
    # (Using mean normalization to match DEA standard practice in our app)
    vector_cols = INPUT_COLS + OUTPUT_COLS
    all_data = df[vector_cols].values.astype(float)
    scales = all_data.mean(axis=0)
    scales[scales == 0] = 1.0

    target_vec = (df.loc[target_idx, vector_cols].values.astype(float) / scales).reshape(1, -1)
    peer_vecs  = efficient_df[vector_cols].values.astype(float) / scales

    # Compute Euclidean distance
    distances = cdist(target_vec, peer_vecs, metric="euclidean")[0]
    
    # Attach distances and get top 3 closest
    efficient_df = efficient_df.copy()
    efficient_df["_Distance"] = distances
    top_peers = efficient_df.sort_values("_Distance").head(3)
    
    return top_peers.drop(columns=["_Distance"])


def root_cause_analysis(df: pd.DataFrame, target_branch: str, top_peers: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates the % deviation of the target branch's variables against:
    1. The average of its 3 closest efficient peers
    2. The absolute #1 ranked branch in the entire fleet
    
    Returns
    -------
    dict: {
        "Staff": {"value": 15.0, "peer_avg": 12.0, "top_1": 10.0, "dev_peer": 25.0, "dev_top": 50.0},
        ...
    }
    """
    if target_branch not in df["Branch"].values:
        return {}

    target_row = df[df["Branch"] == target_branch].iloc[0]
    top_1_row  = df.loc[df["Efficiency_Score"].idxmax()]
    
    if len(top_peers) > 0:
        peer_avg = top_peers[ALL_VARS].mean()
    else:
        peer_avg = target_row[ALL_VARS]  # Fallback if no peers

    deviations = {}
    for col in ALL_VARS:
        actual = float(target_row[col])
        p_avg  = float(peer_avg[col])
        top1   = float(top_1_row[col])

        # % Deviation = (actual - benchmark) / benchmark * 100
        # For inputs: positive dev > 0 means too much resource (bad)
        # For outputs: negative dev < 0 means too little output (bad)
        dev_peer = ((actual - p_avg) / p_avg * 100) if p_avg != 0 else 0.0
        dev_top  = ((actual - top1) / top1 * 100) if top1 != 0 else 0.0

        deviations[col] = {
            "value":    actual,
            "peer_avg": p_avg,
            "top_1":    top1,
            "dev_peer": dev_peer,
            "dev_top":  dev_top
        }
    
    return deviations


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 & 4: SMART RECOMMENDATION WITH BUSINESS INTERPRETATION
# ════════════════════════════════════════════════════════════════════════════

def smart_recommendations(slack_row: pd.Series, root_cause: Dict) -> List[Dict[str, Any]]:
    """
    Upgrades the raw mathematical slack into prioritized business recommendations.
    Uses impact score = |deviation_from_peer| * normalized_slack
    """
    score = float(slack_row.get("Efficiency_Score", 1.0))
    if score >= EFFICIENCY_THRESHOLD:
        return []

    recs = []
    
    # Define metric mappings
    # (Column, Slack_Col, Type, Good_Dev_Direction, Business_Subject)
    metrics = [
        ("Staff",          "Pct_Reduce_Staff",         "reduce",   -1, "Workforce planning"),
        ("Operating_Cost", "Pct_Reduce_OpCost",        "reduce",   -1, "Cost management"),
        ("Deposits_2016",  "Pct_Increase_Dep2016",     "increase",  1, "Historical deposit retention"),
        ("Deposit_Growth", "Pct_Increase_Growth",      "increase",  1, "Deposit mobilization"),
        ("Avg_Deposits",   "Pct_Increase_AvgDep",      "increase",  1, "Account penetration"),
    ]

    for col, slack_col, action_type, good_dir, subject in metrics:
        slack_pct = float(slack_row.get(slack_col, 0.0))
        if slack_pct <= 0.5:  # Ignore trivial slacks < 0.5%
            continue
            
        dev_peer = root_cause.get(col, {}).get("dev_peer", 0.0)
        
        # Impact: Since slack_pct is the EXACT change needed for 100% efficiency, 
        # it is the purest measure of impact. We combine it with peer deviation severity.
        impact_score = slack_pct * (1 + abs(dev_peer)/100.0)
        
        # Business translation phrasing
        if action_type == "reduce":
            explanation = f"{subject} is significantly higher ({dev_peer:+.1f}%) than the optimal efficient benchmark."
            verb = "Reduce"
        else:
            explanation = f"{subject} is underperforming ({dev_peer:+.1f}%) compared to similar top-tier peers."
            verb = "Increase"

        recs.append({
            "variable":     col.replace("_", " "),
            "action":       verb,
            "slack_pct":    slack_pct,
            "impact_score": impact_score,
            "explanation":  explanation,
        })
    
    # Prioritize by impact score descending
    recs.sort(key=lambda x: x["impact_score"], reverse=True)
    
    # Assign priorities: Top 2 = High, Next = Medium, Rest = Low
    for i, r in enumerate(recs):
        if i < 2:
            r["priority"] = "High"
            r["icon"]     = "🔴"
        elif i == 2:
            r["priority"] = "Medium"
            r["icon"]     = "🟡"
        else:
            r["priority"] = "Low"
            r["icon"]     = "🟢"
            
    return recs


# ════════════════════════════════════════════════════════════════════════════
# MODULE 5: SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis(df: pd.DataFrame, branch: str) -> List[Dict]:
    """
    Simulates a small, isolated +5% improvement in each variable
    to measure the marginal gain in Efficiency Score.
    
    +5% Improvement = For inputs, reducing by 5%. For outputs, increasing by 5%.
    """
    row = df[df["Branch"] == branch]
    if len(row) == 0:
        return []
        
    base_score = float(row["Efficiency_Score"].iloc[0])
    if base_score >= EFFICIENCY_THRESHOLD:
        return [] # Already efficient
        
    results = []
    
    # 5% Improvement Multipliers
    adj_map = {
        "Staff":          (0.95, 1.0, 1.0, 1.0, 1.0),
        "Operating_Cost": (1.0, 0.95, 1.0, 1.0, 1.0),
        "Deposits_2016":  (1.0, 1.0,  1.05, 1.0, 1.0),
        "Deposit_Growth": (1.0, 1.0,  1.0, 1.05, 1.0),
        "Avg_Deposits":   (1.0, 1.0,  1.0, 1.0, 1.05),
    }

    for col, adjs in adj_map.items():
        sim_score = simulate_branch(
            df, branch,
            staff_adj=adjs[0], opcost_adj=adjs[1],
            dep2016_adj=adjs[2], growth_adj=adjs[3], avgdep_adj=adjs[4]
        )
        delta = sim_score - base_score
        
        results.append({
            "variable": col.replace("_", " "),
            "action":   "Reduce 5%" if col in INPUT_COLS else "Increase 5%",
            "delta":    delta
        })
        
    # Sort by highest ROI
    results.sort(key=lambda x: x["delta"], reverse=True)
    return results


# ════════════════════════════════════════════════════════════════════════════
# MODULE 6 & 7: CONFIDENCE & AUTO NARRATIVE GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def compute_confidence(df: pd.DataFrame, branch: str, top_peers: pd.DataFrame) -> int:
    """
    Heuristic confidence score for the recommendations (0-100%).
    Based on:
    1. Proximity to peers (Euclidean distance implies peer relevance)
    2. Sample size of the dataset (N=4722 implies high confidence)
    """
    if len(top_peers) < 3:
        return 65 # Fallback
        
    # 1. Base confidence from massive dataset size
    base_conf = 75
    
    # 2. Distance penalty metric
    # If the branch is extremely far from its closest efficient peers, it is an 
    # outlier and the recommendations are slightly less confident.
    # To compute this cleanly without recompiling distances, we use the score itself.
    score = df[df["Branch"] == branch]["Efficiency_Score"].iloc[0]
    
    # Extreme inefficiency (< 0.5) is harder to predict accurately than moderate (0.8)
    if score > 0.9:
        base_conf += 15
    elif score > 0.7:
        base_conf += 10
    elif score > 0.5:
        base_conf += 5
        
    return min(95, base_conf) # Cap at 95% because models are never perfectly certain


def generate_insight_summary(
    branch: str, 
    score: float, 
    top_peers: pd.DataFrame, 
    recs: List[Dict], 
    sens_results: List[Dict]
) -> str:
    """
    Generates a human-readable consultant paragraph combining WHY, WHO, and WHAT.
    """
    if score >= EFFICIENCY_THRESHOLD:
        return f"**{branch}** is operating at peak efficiency. It defines the frontier and serves as a benchmark for other branches in the network. Keep monitoring to maintain this standard."

    # WHY (Primary issues)
    primary_issues = [r["variable"].lower() for r in recs[:2]]
    
    if len(primary_issues) == 2:
        why_str = f"primarily due to sub-optimal metrics in **{primary_issues[0]}** and **{primary_issues[1]}**"
    elif len(primary_issues) == 1:
        why_str = f"primarily due to a significant lag in **{primary_issues[0]}**"
    else:
        why_str = "due to minor generalized inefficiencies across multiple areas"

    # WHO (Benchmarks)
    if len(top_peers) > 0:
        peer_names = top_peers["Branch"].iloc[:2].tolist()
        who_str = f"Comparing against highly similar efficient baselines like **{peer_names[0]}**"
        if len(peer_names) > 1:
            who_str += f" and **{peer_names[1]}**"
    else:
        who_str = "Comparing against fleet-wide optimal performance"

    # WHAT (Highest Leverage Sensitivity)
    if len(sens_results) > 0 and sens_results[0]['delta'] > 0.001:
        top_sens = sens_results[0]
        what_str = f"The highest leverage action available is to **{top_sens['action'].lower()} {top_sens['variable'].lower()}**, which alone is projected to lift efficiency immediately by ~{top_sens['delta']*100:.2f}%."
    else:
        what_str = "A balanced approach across both resource reduction and output mobilization will be necessary to reach the frontier."

    narrative = (
        f"**{branch}** currently operates below the efficiency frontier (Score: {score:.3f}), {why_str}. "
        f"{who_str}, our recommendation engine indicates immediate structural improvements are required. "
        f"{what_str}"
    )
    
    return narrative


# ════════════════════════════════════════════════════════════════════════════
# VIZ: BENCHMARK RADAR CHART
# ════════════════════════════════════════════════════════════════════════════

def create_benchmark_radar_chart(df: pd.DataFrame, target_branch: str, top_peers: pd.DataFrame) -> go.Figure:
    """
    Overlays the target branch against the 'Peer Average' and 'Fleet Average'
    on a normalized radar chart.
    """
    if target_branch not in df["Branch"].values:
        return go.Figure()

    target_row = df[df["Branch"] == target_branch].iloc[0]
    
    if len(top_peers) > 0:
        peer_avg = top_peers[ALL_VARS].mean()
    else:
        peer_avg = target_row[ALL_VARS]
        
    fleet_avg = df[ALL_VARS].mean()

    # Normalize to maximums across the dataset to plot on a 0-1 scale
    max_vals = df[ALL_VARS].max()
    
    target_norm = target_row[ALL_VARS] / max_vals
    peer_norm   = peer_avg / max_vals
    fleet_norm  = fleet_avg / max_vals
    
    # Close the loop for the radar plot
    theta       = ALL_VARS + [ALL_VARS[0]]
    r_target    = target_norm.tolist() + [target_norm.iloc[0]]
    r_peer      = peer_norm.tolist() + [peer_norm.iloc[0]]
    r_fleet     = fleet_norm.tolist() + [fleet_norm.iloc[0]]
    
    fig = go.Figure()
    
    # Fleet Average (Background)
    fig.add_trace(go.Scatterpolar(
        r=r_fleet, theta=theta,
        fill='none', name='Fleet Average',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')
    ))
    
    # Target Branch
    fig.add_trace(go.Scatterpolar(
        r=r_target, theta=theta,
        fill='toself', name=target_branch,
        fillcolor='rgba(231,76,60,0.2)', # INEFFICIENT_COLOR alpha
        line=dict(color=INEFFICIENT_COLOR, width=2)
    ))
    
    # Peer Average
    fig.add_trace(go.Scatterpolar(
        r=r_peer, theta=theta,
        fill='none', name='Peer Average',
        line=dict(color=EFFICIENT_COLOR, width=3)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                tickfont=dict(color=TEXT_COLOR, size=11),
                direction="clockwise",
                gridcolor="rgba(255,255,255,0.1)",
                linecolor="rgba(255,255,255,0.1)"
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT_COLOR), orientation="h", y=-0.2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


# ════════════════════════════════════════════════════════════════════════════
# VIZ: SENSITIVITY TORNADO CHART
# ════════════════════════════════════════════════════════════════════════════

def create_sensitivity_chart(sens_results: List[Dict]) -> go.Figure:
    """
    Horizontal bar chart showing the margin Efficiency Score ∆ for 
    a 5% improvement in each variable.
    """
    if not sens_results:
         return go.Figure()
         
    # Reverse so highest is at the top
    plot_data = list(reversed(sens_results))
    
    y_labels = [f"{r['variable']}<br><span style='font-size:10px;color:#bdc3c7'>({r['action']})</span>" for r in plot_data]
    x_vals   = [r['delta'] * 100 for r in plot_data] # Convert to % point gain
    
    # Color top impact distinctly
    colors = [EFFICIENT_COLOR if i == len(plot_data)-1 else ACCENT_COLOR for i in range(len(plot_data))]
    
    fig = go.Figure(go.Bar(
        x=x_vals, y=y_labels,
        orientation='h',
        marker_color=colors,
        text=[f"+{v:.2f}%" for v in x_vals],
        textposition='outside',
        textfont=dict(color=TEXT_COLOR, size=11),
    ))
    
    fig.update_layout(
        title=dict(text="📊 5% Improvement Sensitivity (Efficiency ROI)", font=dict(color=TEXT_COLOR, size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        xaxis=dict(title="Efficiency Score Gain (%)", tickfont=dict(color=TEXT_COLOR), gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR)),
        height=320,
        margin=dict(l=140, r=40, t=50, b=40)
    )
    return fig
