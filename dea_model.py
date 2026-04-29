"""
=============================================================
DecisionLens – Phase 2: DEA Optimization Engine
=============================================================
Model  : CCR (Charnes-Cooper-Rhodes) – Input-Oriented
Solver : PuLP  (COIN-BC via CBC)

For each DMU k the LP solved is:

    Maximise:    Σ_r  u_r · y_rk          (weighted outputs)

    Subject to:  Σ_i  v_i · x_ik = 1      (normalise own inputs)

                 Σ_r  u_r · y_rj
               − Σ_i  v_i · x_ij  ≤  0    ∀ j ∈ {1 … n}

                 u_r ≥ ε  ∀ r              (output weights strictly +)
                 v_i ≥ ε  ∀ i              (input  weights strictly +)

The optimal objective value IS the efficiency score θ_k ∈ (0, 1].

Inputs  : Staff, Operating_Cost
Outputs : Deposits_2016, Deposit_Growth, Avg_Deposits

Deliverables
  dea_results.csv  →  Branch | Efficiency_Score | Status
=============================================================
"""

import sys
import time
import numpy as np
import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV   = "bank_dea_dataset.csv"
OUTPUT_CSV  = "dea_results.csv"

INPUT_COLS  = ["Staff", "Operating_Cost"]
OUTPUT_COLS = ["Deposits_2016", "Deposit_Growth", "Avg_Deposits"]

# Non-Archimedean epsilon – prevents zero weights
EPSILON = 1e-6

# Score tolerance for classifying a branch as "Efficient"
EFFICIENCY_THRESHOLD = 0.9999


# ===========================================================================
# STEP 1 – Load Dataset
# ===========================================================================
def load_dataset(filepath: str) -> tuple:
    """
    Load bank_dea_dataset.csv and return normalised numpy arrays for
    inputs (X) and outputs (Y) together with branch labels.

    Normalisation (column-wise unit scaling) is applied before solving to
    keep LP coefficient magnitudes comparable and improve numerical stability.
    The scaling does NOT change relative efficiency scores.

    Returns
    -------
    X      : np.ndarray  shape (n, m)  – input  matrix, normalised
    Y      : np.ndarray  shape (n, s)  – output matrix, normalised
    labels : list of str               – branch names (length n)
    """
    print("\n" + "=" * 60)
    print("STEP 1 – Loading Dataset")
    print("=" * 60)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] '{filepath}' not found. Run data_preprocessing.py first.")
        sys.exit(1)

    # Validate expected columns
    required = ["Branch"] + INPUT_COLS + OUTPUT_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    print(f"  ✔  Loaded '{filepath}' → {len(df):,} DMUs")

    labels = df["Branch"].astype(str).tolist()

    X_raw = df[INPUT_COLS].values.astype(float)
    Y_raw = df[OUTPUT_COLS].values.astype(float)

    # Column-wise normalisation: divide each column by its mean
    # (avoids numerical issues from mixing values across wildly different scales)
    x_scale = X_raw.mean(axis=0)
    y_scale = Y_raw.mean(axis=0)

    x_scale[x_scale == 0] = 1.0   # guard against zero-mean columns
    y_scale[y_scale == 0] = 1.0

    X = X_raw / x_scale
    Y = Y_raw / y_scale

    print(f"  ✔  Inputs  ({len(INPUT_COLS)}): {INPUT_COLS}")
    print(f"  ✔  Outputs ({len(OUTPUT_COLS)}): {OUTPUT_COLS}")
    print(f"  ✔  Data normalised for numerical stability")

    return X, Y, labels


# ===========================================================================
# STEP 2 & 3 – Build & Solve CCR DEA LP  /  Compute Efficiency Scores
# ===========================================================================
def run_ccr_dea(X: np.ndarray, Y: np.ndarray, labels: list) -> list:
    """
    Solve the CCR input-oriented LP for every DMU.

    For n DMUs with m inputs and s outputs we solve n separate LP problems.
    Each LP has (m + s) variables and (1 + n) constraints.

    Parameters
    ----------
    X      : (n, m) normalised input  matrix
    Y      : (n, s) normalised output matrix
    labels : list of n branch name strings

    Returns
    -------
    scores : list of float, efficiency score for each DMU (same order)
    """
    print("\n" + "=" * 60)
    print("STEP 2 & 3 – Solving CCR DEA for every DMU")
    print("=" * 60)

    n, m = X.shape   # n = DMU count, m = inputs
    s    = Y.shape[1]  # s = outputs

    scores    = []
    failed    = 0
    start_t   = time.time()

    # Progress reporting interval
    report_every = max(1, n // 20)   # ~5 % increments

    print(f"  Solving {n:,} LP problems  "
          f"({m} inputs, {s} outputs, {n} constraints each) …\n")

    for k in range(n):

        # ── Define the LP ──────────────────────────────────────────────────
        prob = pulp.LpProblem(f"CCR_DMU_{k}", pulp.LpMaximize)

        # Decision variables: output weights u (s vars) + input weights v (m vars)
        u = [pulp.LpVariable(f"u_{r}", lowBound=EPSILON) for r in range(s)]
        v = [pulp.LpVariable(f"v_{i}", lowBound=EPSILON) for i in range(m)]

        # Objective: maximise weighted outputs of DMU k
        prob += pulp.lpSum(u[r] * Y[k, r] for r in range(s)), "Objective"

        # Constraint 1: normalise weighted inputs of DMU k to 1
        prob += (pulp.lpSum(v[i] * X[k, i] for i in range(m)) == 1,
                 "Normalise_Inputs")

        # Constraint 2: for every DMU j, weighted_output_j ≤ weighted_input_j
        for j in range(n):
            prob += (
                pulp.lpSum(u[r] * Y[j, r] for r in range(s))
                - pulp.lpSum(v[i] * X[j, i] for i in range(m))
                <= 0,
                f"Frontier_{j}"
            )

        # ── Solve ──────────────────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] == "Optimal":
            score = pulp.value(prob.objective)
            # Clip to [0, 1] to absorb tiny floating-point overshoots
            scores.append(min(float(score), 1.0))
        else:
            # Infeasible / unbounded – assign 0 and flag
            scores.append(0.0)
            failed += 1

        # ── Progress report ────────────────────────────────────────────────
        if (k + 1) % report_every == 0 or (k + 1) == n:
            elapsed   = time.time() - start_t
            pct       = (k + 1) / n * 100
            rate      = (k + 1) / elapsed if elapsed > 0 else 0
            remaining = (n - k - 1) / rate if rate > 0 else 0
            print(f"  [{pct:5.1f}%]  DMU {k+1:>5,}/{n:,}  "
                  f"| elapsed {elapsed:>6.1f}s  "
                  f"| ETA {remaining:>5.0f}s  "
                  f"| last score {scores[-1]:.4f}")

    total_time = time.time() - start_t
    print(f"\n  ✔  All {n:,} DMUs solved in {total_time:.1f}s")
    if failed:
        print(f"  ⚠  {failed} DMUs returned non-optimal status (assigned score=0)")

    return scores


# ===========================================================================
# STEP 4 – Assign Status (Efficient / Inefficient)
# ===========================================================================
def calculate_efficiency(scores: list, labels: list) -> pd.DataFrame:
    """
    Build the results DataFrame from raw LP scores.

    Columns
    -------
    Branch           : branch name
    Efficiency_Score : float in [0, 1]
    Status           : 'Efficient' if score ≥ EFFICIENCY_THRESHOLD, else 'Inefficient'
    """
    print("\n" + "=" * 60)
    print("STEP 4 – Assigning Efficiency Status")
    print("=" * 60)

    df = pd.DataFrame({
        "Branch"          : labels,
        "Efficiency_Score": scores,
    })

    df["Efficiency_Score"] = df["Efficiency_Score"].clip(0.0, 1.0).round(6)
    df["Status"] = df["Efficiency_Score"].apply(
        lambda s: "Efficient" if s >= EFFICIENCY_THRESHOLD else "Inefficient"
    )

    n_eff   = (df["Status"] == "Efficient").sum()
    n_ineff = (df["Status"] == "Inefficient").sum()

    print(f"  ✔  Efficient   branches : {n_eff:>6,}  ({n_eff/len(df)*100:.1f} %)")
    print(f"  ✔  Inefficient branches : {n_ineff:>6,}  ({n_ineff/len(df)*100:.1f} %)")
    return df


# ===========================================================================
# STEP 5 – Rank Branches
# ===========================================================================
def rank_branches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the results DataFrame by efficiency score (descending).
    Ties are broken alphabetically by Branch name.
    Adds a Rank column starting at 1.
    """
    print("\n" + "=" * 60)
    print("STEP 5 – Ranking Branches")
    print("=" * 60)

    df = (df.sort_values(["Efficiency_Score", "Branch"],
                          ascending=[False, True])
            .reset_index(drop=True))

    df.insert(0, "Rank", df.index + 1)

    print("  Top 10 branches:")
    print(df.head(10).to_string(index=False))
    print("\n  Bottom 5 branches:")
    print(df.tail(5).to_string(index=False))

    return df


# ===========================================================================
# STEP 6 – Save Results
# ===========================================================================
def save_results(df: pd.DataFrame, filepath: str) -> None:
    """
    Save the final ranked efficiency results to CSV.
    Only Branch, Efficiency_Score, and Status are written (Rank dropped).
    """
    print("\n" + "=" * 60)
    print("STEP 6 – Saving Results")
    print("=" * 60)

    out = df[["Branch", "Efficiency_Score", "Status"]].copy()
    out.to_csv(filepath, index=False)
    print(f"  ✔  Saved '{filepath}'  ({len(out):,} rows)")


# ===========================================================================
# STEP 8 – Validation Summary
# ===========================================================================
def print_validation_summary(df: pd.DataFrame) -> None:
    """
    Print a concise validation summary that satisfies DEA reporting requirements.
    """
    print("\n" + "=" * 60)
    print("STEP 8 – Validation Summary")
    print("=" * 60)

    n         = len(df)
    n_eff     = (df["Status"] == "Efficient").sum()
    avg_score = df["Efficiency_Score"].mean()
    min_score = df["Efficiency_Score"].min()
    max_score = df["Efficiency_Score"].max()

    print(f"  Total DMUs             : {n:,}")
    print(f"  Efficient  (score = 1) : {n_eff:,}  ({n_eff/n*100:.1f} %)")
    print(f"  Inefficient            : {n - n_eff:,}  ({(n - n_eff)/n*100:.1f} %)")
    print(f"  Average efficiency     : {avg_score:.4f}")
    print(f"  Min efficiency         : {min_score:.4f}")
    print(f"  Max efficiency         : {max_score:.4f}")

    # Score distribution buckets
    buckets = [(0.9, 1.01, "0.90 – 1.00 (near-efficient)"),
               (0.7, 0.9,  "0.70 – 0.90"),
               (0.5, 0.7,  "0.50 – 0.70"),
               (0.0, 0.5,  "0.00 – 0.50 (low efficiency)")]

    print("\n  Score distribution:")
    for lo, hi, label in buckets:
        count = ((df["Efficiency_Score"] >= lo) & (df["Efficiency_Score"] < hi)).sum()
        bar   = "█" * int(count / n * 40)
        print(f"    {label:<32} {count:>5,}  {bar}")


# ===========================================================================
# MAIN – Orchestrate all steps
# ===========================================================================
def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   DecisionLens  –  Phase 2: DEA Optimization Engine     ║")
    print("║   Model: CCR (Input-Oriented)   Solver: PuLP / CBC      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 1 – Load & normalise
    X, Y, labels = load_dataset(INPUT_CSV)

    # Steps 2 & 3 – Solve CCR LP for every DMU
    scores = run_ccr_dea(X, Y, labels)

    # Step 4 – Efficiency status
    results_df = calculate_efficiency(scores, labels)

    # Step 5 – Rank
    results_df = rank_branches(results_df)

    # Step 6 – Save
    save_results(results_df, OUTPUT_CSV)

    # Step 8 – Validation
    print_validation_summary(results_df)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 2 complete!  dea_results.csv is ready.          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
