"""
causal_credit_risk/src/data_pipeline.py
========================================
Data scoping, cleaning, and feature engineering for the
Causal Feature Selection & HTE in Credit Risk project.

Author  : Senior Quant Researcher
Dataset : LendingClub Loan Data (2007–2018)
Purpose : Prepare data for PC Algorithm (causal-learn) and
          Double Machine Learning (EconML)

Design notes
------------
* Post-treatment mediators (total_pymnt, recoveries, etc.) are
  intentionally excluded to avoid collider bias.
* High-cardinality categoricals are ordinally encoded so that
  Fisher's Z conditional independence tests remain valid.
* All numeric features are winsorized (1–99 pct) before scaling
  to prevent CI test distortion from extreme outliers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Causal roles — documented explicitly for the research design record
TREATMENT: str = "int_rate"          # Continuous treatment (APR %)

OUTCOME: str = "default"             # Binary: 1 = Charged Off / Default

# Confounders: variables that cause BOTH treatment AND outcome
# (must be controlled for via Backdoor Criterion)
CONFOUNDERS: list[str] = [
    "annual_inc",     # Income → drives rate offer AND repayment capacity
    "dti",            # Debt-to-income → underwriting signal + default driver
    "emp_length_num", # Employment stability → rate pricing + default risk
    "fico_range_low", # Credit score → primary rate determinant
    "open_acc",       # # open credit lines → credit utilisation signal
    "revol_util",     # Revolving utilisation → liquidity stress indicator
]

# Effect modifiers: modulate HOW MUCH treatment affects outcome
# (used as X in CATE estimation — not controlled for, but conditioned on)
EFFECT_MODIFIERS: list[str] = [
    "annual_inc",
    "loan_amnt",
    "home_ownership_num",
    "purpose_num",
    "grade_num",
]

# Post-treatment mediators / descendants — MUST be excluded
# Conditioning on these opens collider paths → biases causal estimates
EXCLUDED_POST_TREATMENT: list[str] = [
    "funded_amnt", "funded_amnt_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "out_prncp", "out_prncp_inv",
]

# Raw string → ordinal integer mappings
GRADE_MAP: dict[str, int] = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

HOME_OWNERSHIP_MAP: dict[str, int] = {
    "OWN": 0, "MORTGAGE": 1, "RENT": 2, "OTHER": 3, "NONE": 4, "ANY": 4
}

PURPOSE_MAP: dict[str, int] = {
    "debt_consolidation": 0, "credit_card": 1, "home_improvement": 2,
    "other": 3, "major_purchase": 4, "medical": 5, "small_business": 6,
    "car": 7, "vacation": 8, "moving": 9, "wedding": 10,
    "house": 11, "renewable_energy": 12, "educational": 13,
}

EMP_LENGTH_MAP: dict[str, float] = {
    "< 1 year": 0.5, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
    "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
    "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0,
}

DEFAULT_STATUSES: set[str] = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}


# ── Core pipeline functions ───────────────────────────────────────────────────

def load_raw_data(filepath: str | Path, nrows: int | None = 150_000) -> pd.DataFrame:
    """
    Load the raw LendingClub CSV file.

    Parameters
    ----------
    filepath : path to the .csv file (e.g. loan.csv from Kaggle)
    nrows    : cap rows for fast iteration; set None for full dataset

    Returns
    -------
    Raw DataFrame with original column names.
    """
    path = Path(filepath)
    log.info("Loading data from %s (nrows=%s) ...", path.name, nrows)

    df = pd.read_csv(
        path,
        nrows=nrows,
        low_memory=False,
        # LendingClub CSVs sometimes have 2-row headers
        skiprows=lambda i: i == 1,
    )
    log.info("Raw shape: %s", df.shape)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode high-cardinality categoricals to ordinal integers.

    Rationale: Fisher's Z test (used inside PC Algorithm) operates on
    continuous / ordinal data. One-hot encoding bloats dimensionality
    and breaks the Gaussianity assumption of partial correlations.

    Parameters
    ----------
    df : raw DataFrame (must contain grade, home_ownership, purpose,
         emp_length, term columns)

    Returns
    -------
    DataFrame with new *_num columns added.
    """
    df = df.copy()

    # Grade: A (safest) → G (riskiest) as 1–7
    if "grade" in df.columns:
        df["grade_num"] = df["grade"].map(GRADE_MAP)

    # Home ownership
    if "home_ownership" in df.columns:
        df["home_ownership_num"] = df["home_ownership"].str.upper().map(HOME_OWNERSHIP_MAP)

    # Loan purpose
    if "purpose" in df.columns:
        df["purpose_num"] = df["purpose"].str.lower().str.strip().map(PURPOSE_MAP)

    # Employment length: "10+ years" → 10.0, etc.
    if "emp_length" in df.columns:
        df["emp_length_num"] = (
            df["emp_length"]
            .str.strip()
            .map(EMP_LENGTH_MAP)
        )

    # Term: "36 months" → 36
    if "term" in df.columns:
        df["term_num"] = df["term"].str.extract(r"(\d+)").astype(float)

    log.info("Categorical encoding complete.")
    return df


def build_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the binary default outcome from loan_status.

    Only rows with a *terminal* status are kept:
    - 'Fully Paid'  → default = 0
    - 'Charged Off' → default = 1
    Rows with 'Current' / 'In Grace Period' are dropped to avoid
    label leakage from censored observations.

    Parameters
    ----------
    df : DataFrame with loan_status column

    Returns
    -------
    DataFrame with binary `default` column; non-terminal rows removed.
    """
    df = df.copy()

    terminal_mask = df["loan_status"].isin(DEFAULT_STATUSES | {"Fully Paid"})
    n_dropped = (~terminal_mask).sum()
    log.info("Dropping %d non-terminal loan_status rows (censored).", n_dropped)
    df = df[terminal_mask].copy()

    df[OUTCOME] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    log.info(
        "Default rate: %.2f%% (%d / %d)",
        df[OUTCOME].mean() * 100,
        df[OUTCOME].sum(),
        len(df),
    )
    return df


def winsorize_and_log_transform(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Winsorize continuous features at [1%, 99%] and log-transform
    right-skewed variables to satisfy Gaussian CI test assumptions.

    The PC Algorithm's Fisher's Z statistic is:
        Z = 0.5 * ln[(1+r)/(1-r)] * sqrt(n - |S| - 3)
    where r is the partial correlation. This is asymptotically N(0,1)
    only under Gaussian or sub-Gaussian marginals — hence the need
    to normalize the distributions.

    Parameters
    ----------
    df       : DataFrame
    features : list of numeric column names to process

    Returns
    -------
    DataFrame with winsorized (and log-transformed where applicable) features.
    """
    df = df.copy()
    log_transform_cols = {"annual_inc", "loan_amnt", "revol_bal"}

    for col in features:
        if col not in df.columns:
            log.warning("Column '%s' not found — skipping.", col)
            continue

        # Winsorize: clip to [1st, 99th] percentile
        arr = df[col].values.astype(float)
        df[col] = mstats.winsorize(arr, limits=[0.01, 0.01])

        # Log-transform heavily right-skewed income / amount columns
        if col in log_transform_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
            log.debug("Log-transformed '%s'.", col)

    log.info("Winsorization complete for %d features.", len(features))
    return df


def prepare_causal_dataframe(
    df: pd.DataFrame,
    scale: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """
    Assemble the final causal DataFrame with all roles defined.

    Returns two DataFrames:
    1. `df_causal`  — for PC Algorithm (all features together, scaled)
    2. `df_econml`  — for EconML (T, Y, X, W columns clearly separated)

    Parameters
    ----------
    df    : cleaned DataFrame (output of prior pipeline steps)
    scale : if True, StandardScaler is applied to continuous features

    Returns
    -------
    df_causal : pd.DataFrame  — PC Algorithm input
    df_econml : pd.DataFrame  — EconML input
    scaler    : fitted StandardScaler or None
    """
    # All feature columns needed in downstream steps
    all_features = list(
        {TREATMENT, OUTCOME}
        | set(CONFOUNDERS)
        | set(EFFECT_MODIFIERS)
    )
    # Retain only columns that actually exist after encoding
    available = [c for c in all_features if c in df.columns]
    missing = set(all_features) - set(available)
    if missing:
        log.warning("Missing columns (will be excluded): %s", missing)

    df_work = df[available].copy()

    # Drop any row with NaN in our key columns
    before = len(df_work)
    df_work = df_work.dropna(subset=available)
    log.info("Dropped %d rows with NaN in causal feature set.", before - len(df_work))

    # Winsorize continuous features
    continuous_cols = [
        c for c in available
        if c not in {OUTCOME, "grade_num", "home_ownership_num",
                     "purpose_num", "emp_length_num"}
    ]
    df_work = winsorize_and_log_transform(df_work, continuous_cols)

    # ── PC Algorithm DataFrame ────────────────────────────────────────────────
    df_causal = df_work.copy()

    scaler: StandardScaler | None = None
    if scale:
        scale_cols = [c for c in available if c != OUTCOME]
        scaler = StandardScaler()
        df_causal[scale_cols] = scaler.fit_transform(df_causal[scale_cols])
        log.info("StandardScaler fitted on %d columns.", len(scale_cols))

    # ── EconML DataFrame ──────────────────────────────────────────────────────
    # EconML's DML interface expects: Y, T, X (effect modifiers), W (controls)
    # W = pure confounders not in effect modifier set
    w_cols = [c for c in CONFOUNDERS if c not in EFFECT_MODIFIERS and c in df_work.columns]
    x_cols = [c for c in EFFECT_MODIFIERS if c in df_work.columns]

    df_econml = df_work[
        [OUTCOME, TREATMENT] + x_cols + w_cols
    ].copy()
    df_econml.rename(columns={OUTCOME: "Y", TREATMENT: "T"}, inplace=True)

    log.info(
        "Final shapes — df_causal: %s | df_econml: %s",
        df_causal.shape,
        df_econml.shape,
    )
    log.info("EconML roles — Y: 1 col | T: 1 col | X: %d cols | W: %d cols",
             len(x_cols), len(w_cols))

    return df_causal, df_econml, scaler


def run_pipeline(
    filepath: str | Path,
    nrows: int | None = 150_000,
    scale: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """
    End-to-end data pipeline entry point.

    Usage
    -----
    >>> df_causal, df_econml, scaler = run_pipeline("data/loan.csv")

    Parameters
    ----------
    filepath : path to raw LendingClub CSV
    nrows    : row cap for development; None for production
    scale    : whether to StandardScale features

    Returns
    -------
    df_causal : pd.DataFrame — input to PC Algorithm
    df_econml : pd.DataFrame — input to EconML DML
    scaler    : fitted StandardScaler (for inverse-transform later)
    """
    df = load_raw_data(filepath, nrows=nrows)
    df = encode_categoricals(df)
    df = build_outcome(df)
    df_causal, df_econml, scaler = prepare_causal_dataframe(df, scale=scale)
    return df_causal, df_econml, scaler


# ── Diagnostic utilities ──────────────────────────────────────────────────────

def print_causal_summary(df_causal: pd.DataFrame, df_econml: pd.DataFrame) -> None:
    """Print a research-design summary of variable roles."""
    print("\n" + "="*65)
    print("  CAUSAL VARIABLE ROLE ASSIGNMENT SUMMARY")
    print("="*65)
    print(f"  Treatment  (T) : {TREATMENT}")
    print(f"  Outcome    (Y) : {OUTCOME}")
    print(f"  Confounders(W) : {CONFOUNDERS}")
    print(f"  Effect Mod.(X) : {EFFECT_MODIFIERS}")
    print("-"*65)
    print(f"  PC Algorithm input shape : {df_causal.shape}")
    print(f"  EconML DML input shape   : {df_econml.shape}")
    print(f"  Default rate             : {df_econml['Y'].mean():.3f}")
    print(f"  Mean int_rate (scaled T) : {df_econml['T'].mean():.3f}")
    print("="*65 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <path_to_loan.csv> [nrows]")
        sys.exit(1)

    fp = sys.argv[1]
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000

    df_c, df_e, sc = run_pipeline(fp, nrows=nr)
    print_causal_summary(df_c, df_e)

    # Quick sanity checks
    assert OUTCOME not in df_e.columns or "Y" in df_e.columns, \
        "Outcome column rename failed."
    assert df_c.isnull().sum().sum() == 0, \
        "NaN values remain in causal DataFrame — check pipeline."

    log.info("Pipeline validation passed. Ready for causal discovery.")
