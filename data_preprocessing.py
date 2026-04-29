"""
=============================================================
DecisionLens – Phase 1: Data Preprocessing & Dataset Preparation
=============================================================
Purpose : Load, clean, and engineer features from bank branch deposit
          data, then produce a DEA-ready CSV (bank_dea_dataset.csv).

DEA Framework
  - Decision Making Units (DMUs) : Individual bank branches
  - Inputs  : Staff, Operating_Cost  (resources consumed)
  - Outputs : Deposits_2016, Deposit_Growth, Avg_Deposits (value produced)

Column mapping from database.csv
  '2010 Deposits' … '2016 Deposits'  -> yearly deposit values (in $1,000s)
  'Branch Name'                       -> DMU identifier
  'Latitude' / 'Longitude'            -> geographic coordinates
=============================================================
"""

import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration – change paths here if needed
# ---------------------------------------------------------------------------
INPUT_CSV      = "database.csv"
OUTPUT_CSV     = "bank_dea_dataset.csv"
MIN_DMU_COUNT  = 40          # DEA validity threshold
DEPOSIT_COLS   = [           # ordered yearly deposit columns
    "2010 Deposits",
    "2011 Deposits",
    "2012 Deposits",
    "2013 Deposits",
    "2014 Deposits",
    "2015 Deposits",
    "2016 Deposits",
]

# Synthetic staffing / cost ratios (calibrated to typical community banking)
STAFF_PER_DOLLAR       = 1 / 5_000_000   # 1 FTE per $5 M deposits
OPERATING_COST_RATIO   = 0.02            # 2 % of deposits


# ===========================================================================
# STEP 1 – Load Dataset
# ===========================================================================
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Read the raw CSV file, report shape and columns, and preview rows.
    Returns a pandas DataFrame.
    """
    print("\n" + "=" * 60)
    print("STEP 1 – Loading Dataset")
    print("=" * 60)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{filepath}'")
        print("        Make sure database.csv is in the working directory.")
        sys.exit(1)

    print(f"  ✔  Loaded '{filepath}'")
    print(f"     Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"     Columns : {list(df.columns)}")
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    return df


# ===========================================================================
# STEP 2 – Clean Data
# ===========================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Remove exact duplicate rows.
    2. Drop rows where any deposit column is entirely missing.
    3. Coerce deposit columns to numeric (errors → NaN) and drop those rows.
    4. Remove branches with any zero or negative deposit value.
    Returns a cleaned DataFrame.
    """
    print("\n" + "=" * 60)
    print("STEP 2 – Cleaning Data")
    print("=" * 60)

    original_len = len(df)

    # --- 2a. Remove exact duplicates ---
    df = df.drop_duplicates()
    print(f"  Removed duplicates    : {original_len - len(df):,} rows dropped")

    # --- 2b. Keep only rows that have at least 2016 Deposits present ---
    df = df.dropna(subset=["2016 Deposits"])
    print(f"  After dropping rows missing '2016 Deposits': {len(df):,} rows remain")

    # --- 2c. Coerce all deposit columns to numeric ---
    for col in DEPOSIT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before_nan_drop = len(df)
    df = df.dropna(subset=[c for c in DEPOSIT_COLS if c in df.columns])
    print(f"  Removed non-numeric deposit rows : {before_nan_drop - len(df):,} rows dropped")

    # --- 2d. Remove zero / negative deposits in 2016 (output must be positive) ---
    before_pos = len(df)
    df = df[df["2016 Deposits"] > 0]
    print(f"  Removed zero/negative 2016 Deposits : {before_pos - len(df):,} rows dropped")

    print(f"\n  ✔  Clean dataset size : {len(df):,} rows")
    return df.reset_index(drop=True)


# ===========================================================================
# STEP 3 – Feature Engineering
# ===========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create DEA output and input variables from the cleaned deposit data.

    Outputs
    -------
    Deposits_2016  : raw 2016 deposit value
    Deposit_Growth : year-on-year growth (2016 – 2015)
    Avg_Deposits   : mean deposits across 2010–2016

    Inputs (synthetic proxies)
    -------
    Staff          : Deposits_2016 / 5,000,000  (full-time equivalents)
    Operating_Cost : Deposits_2016 × 0.02       (2 % cost ratio)
    """
    print("\n" + "=" * 60)
    print("STEP 3 – Feature Engineering")
    print("=" * 60)

    # -- Outputs --
    df["Deposits_2016"]   = df["2016 Deposits"]
    raw_growth            = df["2016 Deposits"] - df["2015 Deposits"]

    # DEA requires strictly positive outputs.
    # Deposit_Growth can be negative (deposits declined), so we apply a
    # linear translation: shift by |min| + 1 so the floor becomes +1.
    # This preserves relative ordering while satisfying the positivity constraint.
    min_growth = raw_growth.min()
    if min_growth <= 0:
        shift = abs(min_growth) + 1
        df["Deposit_Growth"] = raw_growth + shift
        print(f"  ℹ  Deposit_Growth shifted by +{shift:,.0f} to ensure positivity "
              f"(min raw value was {min_growth:,.0f})")
    else:
        df["Deposit_Growth"] = raw_growth

    df["Avg_Deposits"]    = df[[c for c in DEPOSIT_COLS if c in df.columns]].mean(axis=1)

    # -- Inputs --
    df["Staff"]           = df["Deposits_2016"] * STAFF_PER_DOLLAR
    df["Operating_Cost"]  = df["Deposits_2016"] * OPERATING_COST_RATIO

    print("  ✔  Outputs created : Deposits_2016, Deposit_Growth, Avg_Deposits")
    print("  ✔  Inputs  created : Staff, Operating_Cost")
    print(f"     Staff range          : {df['Staff'].min():.2f} – {df['Staff'].max():.2f} FTE")
    print(f"     Operating_Cost range : ${df['Operating_Cost'].min():,.0f} – ${df['Operating_Cost'].max():,.0f}")
    return df


# ===========================================================================
# STEP 4 – Construct DEA Dataset
# ===========================================================================
def construct_dea_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename only the columns needed for DEA.
    Returns a tidy DataFrame ready for analysis.
    """
    print("\n" + "=" * 60)
    print("STEP 4 – Constructing DEA Dataset")
    print("=" * 60)

    required_cols = [
        "Branch Name",
        "Staff",
        "Operating_Cost",
        "Deposits_2016",
        "Deposit_Growth",
        "Avg_Deposits",
        "Latitude",
        "Longitude",
    ]

    # Verify all columns exist before selecting
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        sys.exit(1)

    dea_df = df[required_cols].copy()

    # Rename for clarity / downstream use
    dea_df = dea_df.rename(columns={"Branch Name": "Branch"})

    print(f"  ✔  DEA dataset constructed with {len(dea_df):,} DMUs and {dea_df.shape[1]} columns")
    print(f"     Columns : {list(dea_df.columns)}")
    return dea_df


# ===========================================================================
# STEP 5 – Validate DEA Requirements
# ===========================================================================
def validate_dea_requirements(dea_df: pd.DataFrame) -> None:
    """
    Enforce standard DEA validity checks:
      - All input  values must be strictly positive.
      - All output values must be strictly positive.
      - At least MIN_DMU_COUNT DMUs must be present.
    Prints summary statistics and raises SystemExit on failure.
    """
    print("\n" + "=" * 60)
    print("STEP 5 – Validating DEA Requirements")
    print("=" * 60)

    input_cols  = ["Staff", "Operating_Cost"]
    output_cols = ["Deposits_2016", "Deposit_Growth", "Avg_Deposits"]
    passed      = True

    # Check positivity for inputs
    for col in input_cols:
        n_bad = (dea_df[col] <= 0).sum()
        status = "✔" if n_bad == 0 else "✘"
        print(f"  {status}  Input  '{col}': {n_bad} non-positive values")
        if n_bad > 0:
            passed = False

    # Check positivity for outputs
    for col in output_cols:
        n_bad = (dea_df[col] <= 0).sum()
        status = "✔" if n_bad == 0 else "✘"
        print(f"  {status}  Output '{col}': {n_bad} non-positive values")
        if n_bad > 0:
            passed = False

    # Check DMU count
    n_dmu = len(dea_df)
    status = "✔" if n_dmu >= MIN_DMU_COUNT else "✘"
    print(f"\n  {status}  DMU count : {n_dmu:,}  (minimum required: {MIN_DMU_COUNT})")
    if n_dmu < MIN_DMU_COUNT:
        passed = False

    # Summary statistics
    print("\n  Summary Statistics (DEA variables):")
    summary_cols = input_cols + output_cols
    print(dea_df[summary_cols].describe().round(2).to_string())

    if not passed:
        print("\n[ERROR] One or more DEA validation checks failed. "
              "Review the data and re-run the script.")
        sys.exit(1)

    print("\n  ✔  All DEA validation checks passed.")


# ===========================================================================
# STEP 6 – Save Clean Dataset
# ===========================================================================
def save_dataset(dea_df: pd.DataFrame, filepath: str) -> None:
    """
    Persist the final DEA-ready DataFrame to a CSV file.
    """
    print("\n" + "=" * 60)
    print("STEP 6 – Saving DEA Dataset")
    print("=" * 60)

    dea_df.to_csv(filepath, index=False)
    print(f"  ✔  Saved '{filepath}'  ({len(dea_df):,} rows × {dea_df.shape[1]} columns)")


# ===========================================================================
# MAIN – Orchestrate all steps
# ===========================================================================
def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   DecisionLens  –  Phase 1: Data Preprocessing          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 1 – Load
    df = load_dataset(INPUT_CSV)

    # Step 2 – Clean
    df = clean_data(df)

    # Step 3 – Feature Engineering
    df = engineer_features(df)

    # Step 4 – Construct DEA dataset
    dea_df = construct_dea_dataset(df)

    # Step 5 – Validate
    validate_dea_requirements(dea_df)

    # Step 6 – Save
    save_dataset(dea_df, OUTPUT_CSV)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 1 complete!  bank_dea_dataset.csv is ready.     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
