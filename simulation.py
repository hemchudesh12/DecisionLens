"""
simulation.py — Phase 5: Decision Recommendation Engine
DecisionLens BI Platform

Modules
───────
1. Slack Analysis    — input excess & output shortfall per DMU
2. Recommendation    — convert slack into human-readable action bullets
3. What-If Simulator — re-score a single branch with adjusted values
4. Target Mode       — minimum input reduction to hit a target efficiency
5. Scenario Store    — save / compare simulation scenarios
6. Charts            — Before/After bar, Improvement Gap bar

Key design constraint
─────────────────────
Full DEA for 4 722 branches is expensive.  Phase 5 NEVER re-solves the
full fleet.  Slack is derived analytically from the CCR projection formula
and the existing dea_results.csv scores.  When simulating ONE branch we
solve a tiny LP against the pre-loaded data (fast: <1 sec).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# ── colour palette (mirrors visualization.py) ────────────────────────────────
EFFICIENT_COLOR   = "#2ecc71"
MODERATE_COLOR    = "#f1c40f"
INEFFICIENT_COLOR = "#e74c3c"
ACCENT_COLOR      = "#3498db"
DARK_BG           = "#1a1a2e"
CARD_BG           = "#16213e"
TEXT_COLOR        = "#edf2f4"
SIMULATED_COLOR   = "#a29bfe"   # purple for simulated values

# ── DEA model constants (must match dea_model.py) ────────────────────────────
INPUT_COLS  = ["Staff", "Operating_Cost"]
OUTPUT_COLS = ["Deposits_2016", "Deposit_Growth", "Avg_Deposits"]
EPSILON     = 1e-6
EFFICIENCY_THRESHOLD = 0.9999


# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — SLACK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_slack(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-DMU input excess and output shortfall using the CCR
    input-oriented projection formula.

    For an inefficient DMU with score θ:
        Projected input  = θ × actual_input
        Input excess     = actual_input − projected_input   = (1 − θ) × actual_input

    Output shortfall is computed by finding each DMU's dominant peers—
    branches on the efficient frontier with ≥ the same outputs per unit
    input—and using their per-unit-input performance as the target.

    Parameters
    ----------
    df  : merged DataFrame from visualization.load_data()
          must contain Staff, Operating_Cost, Deposits_2016,
          Deposit_Growth, Avg_Deposits, Efficiency_Score, Status

    Returns
    -------
    pd.DataFrame with the original df columns PLUS:
        Input_Excess_Staff       : float
        Input_Excess_OpCost      : float
        Output_Shortfall_Dep2016 : float
        Output_Shortfall_Growth  : float
        Output_Shortfall_AvgDep  : float
        Pct_Reduce_Staff         : float  (% reduction needed)
        Pct_Reduce_OpCost        : float
        Pct_Increase_Dep2016     : float  (% increase needed)
        Pct_Increase_Growth      : float
        Pct_Increase_AvgDep      : float
    """
    slack = df.copy()

    theta = slack["Efficiency_Score"].values

    # ── INPUT SLACKS ─────────────────────────────────────────────────────────
    # Projected input = θ × actual → excess = (1 − θ) × actual
    for col in INPUT_COLS:
        proj_col  = f"Projected_{col}"
        slack_col = f"Input_Excess_{_short(col)}"
        pct_col   = f"Pct_Reduce_{_short(col)}"

        slack[proj_col]  = theta * slack[col]
        slack[slack_col] = (slack[col] - slack[proj_col]).clip(lower=0)
        slack[pct_col]   = (slack[slack_col] / slack[col].replace(0, np.nan) * 100).fillna(0)

    # ── OUTPUT SLACKS ────────────────────────────────────────────────────────
    # For each DMU, its target output = actual_output / θ  (project UP to frontier)
    # Shortfall = target − actual = actual × (1/θ − 1)   (for θ < 1)
    for col in OUTPUT_COLS:
        proj_col  = f"Target_{col}"
        slack_col = f"Output_Shortfall_{_short(col)}"
        pct_col   = f"Pct_Increase_{_short(col)}"

        # safe divide: efficient branches have θ≈1 so shortfall ≈ 0
        slack[proj_col]  = slack[col] / theta.clip(min=EPSILON)
        slack[slack_col] = (slack[proj_col] - slack[col]).clip(lower=0)
        slack[pct_col]   = (slack[slack_col] / slack[col].replace(0, np.nan) * 100).fillna(0)

    return slack


def _short(col: str) -> str:
    """Abbreviate column names for compact slack column names."""
    mapping = {
        "Staff":          "Staff",
        "Operating_Cost": "OpCost",
        "Deposits_2016":  "Dep2016",
        "Deposit_Growth": "Growth",
        "Avg_Deposits":   "AvgDep",
    }
    return mapping.get(col, col)


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — DECISION RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_recommendations(row: pd.Series, min_pct: float = 0.01) -> list[dict]:
    """
    Convert a single DMU's slack row into a list of actionable recommendation
    dictionaries.

    Parameters
    ----------
    row      : one row of the DataFrame returned by compute_slack()
    min_pct  : ignore slack smaller than this % (noise filter)

    Returns
    -------
    list of dicts, each with keys:
        type     : "reduce" or "increase"
        variable : human-readable variable name
        amount   : absolute change (with unit)
        pct      : percentage change
        icon     : emoji
        priority : "High" / "Medium" / "Low"
    """
    score = float(row.get("Efficiency_Score", 1.0))
    if score >= EFFICIENCY_THRESHOLD:
        return [{"type": "info", "variable": "Status",
                 "amount": "No changes required",
                 "pct": 0.0, "icon": "✅", "priority": "None"}]

    recs = []

    # ── inputs: recommend reduction ──────────────────────────────────────────
    input_defs = [
        ("Staff",         "Input_Excess_Staff",  "Pct_Reduce_Staff",  "staff units",   "👥"),
        ("Operating_Cost","Input_Excess_OpCost", "Pct_Reduce_OpCost", "operating cost","⚙️"),
    ]
    for var, excess_col, pct_col, unit, icon in input_defs:
        pct    = float(row.get(pct_col, 0))
        amount = float(row.get(excess_col, 0))
        if pct < min_pct:
            continue
        recs.append({
            "type":     "reduce",
            "variable": var.replace("_", " "),
            "amount":   f"{amount:,.2f} {unit}",
            "pct":      round(pct, 2),
            "icon":     icon,
            "priority": _priority(pct),
        })

    # ── outputs: recommend increase ──────────────────────────────────────────
    output_defs = [
        ("Deposits_2016",  "Output_Shortfall_Dep2016", "Pct_Increase_Dep2016", "$",  "💰"),
        ("Deposit_Growth", "Output_Shortfall_Growth",  "Pct_Increase_Growth",  "$",  "📈"),
        ("Avg_Deposits",   "Output_Shortfall_AvgDep",  "Pct_Increase_AvgDep",  "$",  "🏦"),
    ]
    for var, short_col, pct_col, unit, icon in output_defs:
        pct    = float(row.get(pct_col, 0))
        amount = float(row.get(short_col, 0))
        if pct < min_pct:
            continue
        recs.append({
            "type":     "increase",
            "variable": var.replace("_", " "),
            "amount":   f"{unit}{amount:,.0f}",
            "pct":      round(pct, 2),
            "icon":     icon,
            "priority": _priority(pct),
        })

    # sort: High first, then by pct descending
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: (priority_order.get(r["priority"], 9), -r["pct"]))
    return recs


def _priority(pct: float) -> str:
    if pct >= 10:
        return "High"
    elif pct >= 2:
        return "Medium"
    return "Low"


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — WHAT-IF SINGLE-BRANCH SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

def simulate_branch(
    df: pd.DataFrame,
    branch: str,
    staff_adj:    float = 1.0,
    opcost_adj:   float = 1.0,
    dep2016_adj:  float = 1.0,
    growth_adj:   float = 1.0,
    avgdep_adj:   float = 1.0,
) -> float:
    """
    Re-score a single branch after applying multiplicative adjustments to its
    inputs and outputs.  All other branches serve as the reference set (their
    actual values unchanged).

    This solves ONE tiny CCR LP instead of n=4 722 — typically < 0.5 s.

    Parameters
    ----------
    df          : merged DataFrame from visualization.load_data()
    branch      : name of the branch to simulate
    *_adj       : multipliers (1.0 = no change, 0.9 = 10% reduction, etc.)

    Returns
    -------
    Simulated efficiency score in [0, 1]
    """
    # ── build normalised arrays (same as dea_model.py) ──────────────────────
    work = df[["Branch"] + INPUT_COLS + OUTPUT_COLS].dropna().copy()
    idx  = work[work["Branch"] == branch].index

    if len(idx) == 0:
        return 0.0

    row_idx = idx[0]

    # Apply adjustments to the target branch
    work.loc[row_idx, "Staff"]          *= staff_adj
    work.loc[row_idx, "Operating_Cost"] *= opcost_adj
    work.loc[row_idx, "Deposits_2016"]  *= dep2016_adj
    work.loc[row_idx, "Deposit_Growth"] *= growth_adj
    work.loc[row_idx, "Avg_Deposits"]   *= avgdep_adj

    X_raw = work[INPUT_COLS].values.astype(float)
    Y_raw = work[OUTPUT_COLS].values.astype(float)

    # column-wise normalisation (dea_model.py convention)
    x_scale = X_raw.mean(axis=0)
    y_scale = Y_raw.mean(axis=0)
    x_scale[x_scale == 0] = 1.0
    y_scale[y_scale == 0] = 1.0

    X = X_raw / x_scale
    Y = Y_raw / y_scale

    # find target row in normalised array
    k = work.index.get_loc(row_idx)
    n, m = X.shape
    s    = Y.shape[1]

    prob = pulp.LpProblem("Sim_CCR", pulp.LpMaximize)
    u = [pulp.LpVariable(f"u_{r}", lowBound=EPSILON) for r in range(s)]
    v = [pulp.LpVariable(f"v_{i}", lowBound=EPSILON) for i in range(m)]

    prob += pulp.lpSum(u[r] * Y[k, r] for r in range(s))
    prob += (pulp.lpSum(v[i] * X[k, i] for i in range(m)) == 1, "Norm")
    for j in range(n):
        prob += (
            pulp.lpSum(u[r] * Y[j, r] for r in range(s))
            - pulp.lpSum(v[i] * X[j, i] for i in range(m))
            <= 0,
            f"F_{j}",
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == "Optimal":
        return min(float(pulp.value(prob.objective)), 1.0)
    return 0.0


# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — TARGET EFFICIENCY MODE
# ════════════════════════════════════════════════════════════════════════════

def target_efficiency_mode(
    df: pd.DataFrame,
    branch: str,
    target_score: float = 1.0,
    max_iter: int = 15,
    tol: float = 0.0005,
) -> dict:
    """
    Find the **minimum proportional input reduction** that lifts the named
    branch to `target_score` (while keeping outputs unchanged).

    Uses binary search over the reduction factor α ∈ [0, 1]:
        Staff_new          = (1 − α) × Staff_actual
        Operating_Cost_new = (1 − α) × Operating_Cost_actual

    Parameters
    ----------
    df           : merged DataFrame
    branch       : branch name
    target_score : desired efficiency (default 1.0)
    max_iter     : binary-search iterations
    tol          : convergence tolerance on efficiency gap

    Returns
    -------
    dict with keys:
        alpha             : optimal reduction factor (0-1)
        pct_reduction     : alpha × 100  (percentage to cut all inputs)
        simulated_score   : achieved efficiency at alpha
        new_staff         : projected Staff value
        new_opcost        : projected Operating Cost value
        converged         : bool
    """
    row = df[df["Branch"] == branch]
    if len(row) == 0:
        return {"converged": False}

    actual_score = float(row["Efficiency_Score"].iloc[0])
    if actual_score >= target_score:
        return {
            "alpha": 0.0, "pct_reduction": 0.0,
            "simulated_score": actual_score,
            "new_staff": float(row["Staff"].iloc[0]),
            "new_opcost": float(row["Operating_Cost"].iloc[0]),
            "converged": True,
        }

    lo, hi = 0.0, 1.0   # reduction factor α
    best_alpha = hi
    best_score = 0.0

    for _ in range(max_iter):
        mid   = (lo + hi) / 2
        adj   = 1.0 - mid  # multiplier on inputs
        score = simulate_branch(df, branch, staff_adj=adj, opcost_adj=adj)

        if score >= target_score - tol:
            best_alpha = mid
            best_score = score
            hi = mid     # look for smaller reduction
        else:
            lo = mid     # need more reduction

    actual_staff  = float(row["Staff"].iloc[0])
    actual_opcost = float(row["Operating_Cost"].iloc[0])

    return {
        "alpha":           round(best_alpha, 4),
        "pct_reduction":   round(best_alpha * 100, 2),
        "simulated_score": round(best_score, 6),
        "new_staff":       round(actual_staff  * (1 - best_alpha), 3),
        "new_opcost":      round(actual_opcost * (1 - best_alpha), 0),
        "converged":       abs(best_score - target_score) <= tol * 2,
    }


# ════════════════════════════════════════════════════════════════════════════
# MODULE 5 — SCENARIO STORE
# ════════════════════════════════════════════════════════════════════════════

class ScenarioStore:
    """
    Lightweight in-memory scenario manager backed by a plain Python list.
    Designed for use with st.session_state in Streamlit.

    Usage
    -----
    store = ScenarioStore(st.session_state.get("scenarios", []))
    store.add("Branch A baseline", ...)
    st.session_state["scenarios"] = store.scenarios
    """

    def __init__(self, scenarios: Optional[list] = None):
        self.scenarios: list[dict] = scenarios or []

    def add(
        self,
        name: str,
        branch: str,
        original_score: float,
        simulated_score: float,
        adjustments: dict,
        notes: str = "",
    ) -> None:
        """Save a new scenario."""
        self.scenarios.append({
            "name":            name,
            "branch":          branch,
            "original_score":  round(original_score, 6),
            "simulated_score": round(simulated_score, 6),
            "delta":           round(simulated_score - original_score, 6),
            "delta_pct":       round((simulated_score - original_score) / max(original_score, 1e-9) * 100, 4),
            "adjustments":     adjustments,
            "notes":           notes,
        })

    def to_dataframe(self) -> pd.DataFrame:
        if not self.scenarios:
            return pd.DataFrame(columns=[
                "name", "branch", "original_score",
                "simulated_score", "delta", "delta_pct",
            ])
        return pd.DataFrame(self.scenarios)[[
            "name", "branch", "original_score",
            "simulated_score", "delta", "delta_pct",
        ]]

    def clear(self) -> None:
        self.scenarios = []

    def __len__(self) -> int:
        return len(self.scenarios)


# ════════════════════════════════════════════════════════════════════════════
# MODULE 6 — VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def create_before_after_chart(
    original_row: pd.Series,
    adjustments: dict,
) -> go.Figure:
    """
    Grouped bar chart comparing Actual vs Simulated values for
    all 5 input/output variables.

    Parameters
    ----------
    original_row : one row from the raw df (before adjustments)
    adjustments  : dict mapping column → multiplier
                   e.g. {"Staff": 0.9, "Operating_Cost": 0.85, ...}
    """
    all_vars = INPUT_COLS + OUTPUT_COLS
    labels   = ["Staff", "Op. Cost", "Deposits 2016", "Dep. Growth", "Avg Deposits"]
    actual   = [float(original_row.get(c, 0)) for c in all_vars]
    simulated = [actual[i] * float(adjustments.get(c, 1.0))
                 for i, c in enumerate(all_vars)]

    # normalise to % of actual so all bars share one axis
    pct_change = [
        ((s / a - 1) * 100) if a != 0 else 0
        for s, a in zip(simulated, actual)
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Absolute Values (normalised to %)", "% Change from Actual"],
        horizontal_spacing=0.12,
    )

    norm_actual = [100.0] * len(all_vars)
    norm_sim    = [(s / a * 100) if a != 0 else 100 for s, a in zip(simulated, actual)]

    # left panel — absolute (normalised)
    fig.add_trace(go.Bar(
        name="Actual", x=labels, y=norm_actual,
        marker_color=CARD_BG,
        marker_line=dict(color=TEXT_COLOR, width=1),
        text=["100%"] * len(labels),
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=9),
    ), row=1, col=1)

    bar_colors = [
        EFFICIENT_COLOR if v < 0 and c in INPUT_COLS
        else INEFFICIENT_COLOR if v > 0 and c in INPUT_COLS
        else EFFICIENT_COLOR if v > 0
        else INEFFICIENT_COLOR
        for v, c in zip(pct_change, all_vars)
    ]

    fig.add_trace(go.Bar(
        name="Simulated", x=labels, y=norm_sim,
        marker_color=SIMULATED_COLOR,
        text=[f"{v:+.1f}%" for v in pct_change],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=9),
    ), row=1, col=1)

    # right panel — % change
    fig.add_trace(go.Bar(
        name="% Change",
        x=labels,
        y=pct_change,
        marker_color=bar_colors,
        text=[f"{v:+.2f}%" for v in pct_change],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=9),
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", row=1, col=2)

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR),
        legend=dict(font=dict(color=TEXT_COLOR), bgcolor="rgba(0,0,0,0.4)"),
        height=380,
        margin=dict(l=50, r=30, t=60, b=60),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color=TEXT_COLOR, size=12)
    for axis in [fig.layout.xaxis, fig.layout.xaxis2,
                 fig.layout.yaxis, fig.layout.yaxis2]:
        axis.update(tickfont=dict(color=TEXT_COLOR),
                    gridcolor="rgba(255,255,255,0.06)")
    fig.layout.yaxis.update(title="% of actual", ticksuffix="%")
    fig.layout.yaxis2.update(title="% change", ticksuffix="%")

    return fig


def create_improvement_gap_chart(slack_row: pd.Series) -> go.Figure:
    """
    Horizontal bar chart showing the % gap (slack) between current
    performance and the efficiency frontier, for all 5 variables.

    Inputs  show reduction needed (negative = good direction).
    Outputs show increase needed  (positive = good direction).
    """
    labels = [
        "Staff (reduce)",
        "Op. Cost (reduce)",
        "Deposits 2016 (increase)",
        "Deposit Growth (increase)",
        "Avg Deposits (increase)",
    ]
    pct_cols = [
        "Pct_Reduce_Staff",
        "Pct_Reduce_OpCost",
        "Pct_Increase_Dep2016",
        "Pct_Increase_Growth",
        "Pct_Increase_AvgDep",
    ]
    values = [float(slack_row.get(c, 0)) for c in pct_cols]
    # Inputs: negate so they point left on the bar
    plot_vals = [-values[0], -values[1], values[2], values[3], values[4]]
    colors    = [
        INEFFICIENT_COLOR, INEFFICIENT_COLOR,
        EFFICIENT_COLOR, EFFICIENT_COLOR, EFFICIENT_COLOR,
    ]

    fig = go.Figure(go.Bar(
        y=labels,
        x=plot_vals,
        orientation="h",
        marker_color=colors,
        marker_line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        text=[f"{abs(v):.2f}%" for v in plot_vals],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=10),
        hovertemplate="<b>%{y}</b><br>Gap: %{customdata:.2f}%<extra></extra>",
        customdata=[abs(v) for v in plot_vals],
    ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)", line_width=1.5)

    fig.update_layout(
        title=dict(
            text="📊 Improvement Gap to Efficiency Frontier",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        xaxis=dict(
            title="← Reduce  |  Increase →",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            ticksuffix="%",
            gridcolor="rgba(255,255,255,0.07)",
        ),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=11)),
        height=340,
        margin=dict(l=180, r=80, t=60, b=50),
    )
    return fig


def create_scenario_comparison_chart(scenario_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: original vs simulated efficiency for each saved scenario.
    """
    if scenario_df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Original Score",
        x=scenario_df["name"],
        y=scenario_df["original_score"],
        marker_color=INEFFICIENT_COLOR,
        text=[f"{v:.6f}" for v in scenario_df["original_score"]],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=9),
        hovertemplate="<b>%{x}</b><br>Original: %{y:.6f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Simulated Score",
        x=scenario_df["name"],
        y=scenario_df["simulated_score"],
        marker_color=EFFICIENT_COLOR,
        text=[f"{v:.6f}" for v in scenario_df["simulated_score"]],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=9),
        hovertemplate="<b>%{x}</b><br>Simulated: %{y:.6f}<extra></extra>",
    ))

    fig.add_hline(
        y=EFFICIENCY_THRESHOLD,
        line_dash="dot",
        line_color=MODERATE_COLOR,
        annotation_text="Efficiency Threshold",
        annotation_font_color=MODERATE_COLOR,
        annotation_position="top right",
    )

    fig.update_layout(
        barmode="group",
        title=dict(text="📐 Scenario Comparison", font=dict(size=16, color=TEXT_COLOR)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        xaxis=dict(tickfont=dict(color=TEXT_COLOR, size=10),
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(
            title="Efficiency Score",
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(255,255,255,0.06)",
            range=[scenario_df["original_score"].min() - 0.0005,
                   min(scenario_df["simulated_score"].max() + 0.0005, 1.001)],
        ),
        legend=dict(font=dict(color=TEXT_COLOR), bgcolor="rgba(0,0,0,0.3)"),
        height=420,
        margin=dict(l=70, r=40, t=70, b=100),
    )
    return fig
