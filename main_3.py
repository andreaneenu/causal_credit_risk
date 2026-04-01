"""
causal_credit_risk/main.py
===========================
MASTER ORCHESTRATION SCRIPT — Run the full causal inference pipeline.

  Step 1: Data loading and cleaning        (data_pipeline.py)
  Step 2: Causal discovery with PC algo    (causal_discovery.py)
  Step 3: HTE estimation with DML          (hte_estimation.py)
  Step 4: Validation & SHAP comparison     (validation.py)

HOW TO USE:
  Default (manual refutations, fast):
    python main.py

  With DoWhy refutations (requires: pip install dowhy):
    python main.py --use-dowhy

  Fast iteration on subset:
    python main.py --nrows 20000

  Real data:
    python main.py --data data/loan.csv --nrows 100000

Author  : Senior Quant Researcher
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def run_full_pipeline(
    data_path:  str   = "data/loan.csv",
    nrows:      int   = 100_000,
    output_dir: str   = "outputs/",
    use_dowhy:  bool  = False,
    pc_alpha:   float = 0.05,
    algorithm:  str   = "pc",
    ci_test:    str   = "fisherz",
    compare:    bool  = False,
) -> None:
    """
    Run all four steps of the causal credit risk pipeline.

    Parameters
    ----------
    data_path  : path to raw LendingClub CSV.
                 If missing, synthetic demo data is generated automatically.
    nrows      : number of rows to load (use 20_000 for fast iteration)
    output_dir : directory for all output plots, CSVs, and summaries
    use_dowhy  : if True, Step 4 uses DoWhy's CausalModel API for refutations
                 instead of the manual implementations.
                 Requires: pip install dowhy
                 The DAG is built from your PC Algorithm outputs (Step 2),
                 and the estimator uses the same config as your Step 3 model.
                 If DoWhy is not installed, falls back to manual automatically.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("\n" + "="*65)
    print("   CAUSAL CREDIT RISK PIPELINE — LendingClub")
    if use_dowhy:
        print("   Refutation mode: DoWhy CausalModel API")
    else:
        print("   Refutation mode: Manual implementations")
    print("="*65)

    # ── STEP 1: Data Pipeline ─────────────────────────────────────────────────
    print("\n[STEP 1] Loading and cleaning data...")
    from src.data_pipeline import (
        run_pipeline, print_causal_summary,
        CONFOUNDERS, EFFECT_MODIFIERS,
    )

    true_cate = None   # only available for synthetic data

    if not Path(data_path).exists():
        print(f"\n  Data file not found at '{data_path}'")
        print("   Running pipeline with SYNTHETIC demo data instead.")
        print("   See README.md for real data setup.\n")
        df_causal, df_econml, true_cate = _generate_synthetic_data(nrows=5000)
        pc_confounders = CONFOUNDERS
        pc_effect_mods = EFFECT_MODIFIERS
    else:
        df_causal, df_econml, scaler = run_pipeline(data_path, nrows=nrows)
        print_causal_summary(df_causal, df_econml)
        pc_confounders = CONFOUNDERS
        pc_effect_mods = EFFECT_MODIFIERS

    print(f"   Causal DataFrame : {df_causal.shape}")
    print(f"   EconML DataFrame : {df_econml.shape}")

    # ── STEP 2: Causal Discovery ──────────────────────────────────────────────
    print("\n[STEP 2] Running causal discovery (PC Algorithm)...")
    print("   This may take 2-5 minutes depending on dataset size.")
    from src.causal_discovery_3 import run_causal_discovery

    if compare:
        # Run all three methods and compare confounder sets.
        # Uses the agreed confounder set as W for downstream DML.
        from src.causal_discovery_3 import run_discovery_comparison
        comparison = run_discovery_comparison(
            df_causal,
            treatment        = "int_rate",
            outcome          = "default",
            confounders      = CONFOUNDERS,
            effect_modifiers = EFFECT_MODIFIERS,
            alpha            = pc_alpha,
            output_dir       = output_dir,
        )
        # Use GES DAG as the primary graph (recommended method)
        dag        = comparison["ges_dag"]
        undirected = comparison.get("ges_dag_undirected") or __import__("networkx").Graph()
        # Override pc_confounders with the cross-method agreed set
        pc_confounders = comparison["recommended_W"]
        print(f"\n   Cross-method agreed confounders → W: {pc_confounders}")
    else:
        dag, undirected = run_causal_discovery(
            df_causal,
            treatment        = "int_rate",
            outcome          = "default",
            confounders      = CONFOUNDERS,
            effect_modifiers = EFFECT_MODIFIERS,
            alpha            = pc_alpha,
            algorithm        = algorithm,
            ci_test          = ci_test,
            save_path        = f"{output_dir}/causal_dag.png",
        )
    print(f"   DAG edges: {dag.number_of_edges()} directed, "
          f"{undirected.number_of_edges()} undirected")
    if not compare:
        print(f"   Algorithm: {algorithm.upper()} | CI test: {ci_test} | alpha: {pc_alpha}")
        print(f"   Saved: {output_dir}/causal_dag.png")

    # Extract PC-discovered confounders from the DAG.
    # These are nodes with directed edges into the treatment node.
    # Passed to Step 4 so DoWhy's DAG reflects what PC actually found.
    #
    # Auto-detect the treatment node name — the PC Algorithm may have
    # run on df_causal (uses "int_rate") or df_econml (uses "T").
    # We try both candidate names and use whichever actually exists in
    # the DAG, so this never requires manual intervention.
    dag_nodes = set(dag.nodes())
    log.info("DAG nodes: %s", sorted(dag_nodes))

    # Candidate names for treatment and outcome, in priority order
    treatment_candidates = ["T", "int_rate"]
    outcome_candidates   = ["Y", "default"]

    treatment_node = next(
        (c for c in treatment_candidates if c in dag_nodes), None
    )
    outcome_node = next(
        (c for c in outcome_candidates if c in dag_nodes), None
    )

    if treatment_node is None:
        log.warning(
            "Could not find treatment node in DAG (tried %s). "
            "DAG nodes are: %s. Using pre-specified confounders.",
            treatment_candidates, sorted(dag_nodes)
        )
        pc_confounders = CONFOUNDERS
    else:
        log.info("Auto-detected treatment node in DAG: '%s'", treatment_node)
        exclude = {treatment_node, outcome_node} if outcome_node else {treatment_node}
        pc_confounders = [
            n for n in dag.predecessors(treatment_node)
            if n not in exclude
        ]
        if pc_confounders:
            log.info("PC-discovered confounders: %s", pc_confounders)
            print(f"   PC confounders (from DAG): {pc_confounders}")
        else:
            log.warning(
                "No predecessors found for '%s' in DAG. "
                "DAG may be undirected or treatment has no parents. "
                "Using pre-specified confounders.", treatment_node
            )
            pc_confounders = CONFOUNDERS

    pc_effect_mods = EFFECT_MODIFIERS

    # ── STEP 3: HTE Estimation ────────────────────────────────────────────────
    print("\n[STEP 3] Estimating Heterogeneous Treatment Effects via DML...")
    print("   Fitting CausalForestDML — this may take 3-5 minutes.")
    from src.hte_estimation import run_hte_estimation

    # Returns 6 values — all passed to Step 4 so validation uses
    # the exact same objects with no refitting.
    df_cate, fitted_model, X, Y, T, feature_names = run_hte_estimation(
        df_econml,
        income_col  = "annual_inc",
        output_dir  = output_dir,
        confounders = pc_confounders,   # ← Step 2 output: PC/GES-discovered W
    )

    print(f"   ATE (avg CATE):  {df_cate['cate'].mean():.4f}")
    print(f"   CATE std:        {df_cate['cate'].std():.4f}")
    print(f"   Features used:   {feature_names}")

    # ── STEP 4: Validation ────────────────────────────────────────────────────
    print("\n[STEP 4] Running validation (refutations + SHAP comparison)...")
    from src.validation_2 import run_full_validation

    # What gets passed and why:
    #
    #   cate_estimates / feature_cols / fitted_model / X / Y / T
    #     -> from Step 3: these are the exact objects produced by your
    #        pipeline. Validation uses them directly — no re-estimation.
    #
    #   confounders (pc_confounders)
    #     -> from Step 2 PC Algorithm: the variables that PC found to
    #        cause both T and Y. Used to build the DoWhy DAG when
    #        use_dowhy=True, so the DoWhy graph matches your discovery.
    #
    #   effect_modifiers (pc_effect_mods)
    #     -> from Step 3: the X columns that CATE conditions on.
    #        Passed to DoWhy so it enforces the correct X/W split
    #        inside CausalForestDML during refutation refits.
    #
    #   pipeline_n_estimators / pipeline_min_samples_leaf
    #     -> match your Step 3 config so DoWhy's internal refits
    #        use the same hyperparameters as your original model.
    #
    #   use_dowhy=False (default): manual numpy/scipy implementations.
    #   use_dowhy=True:            DoWhy CausalModel API.
    #   Both produce identical output DataFrames — nothing else changes.

    validation_results = run_full_validation(
        df             = df_econml,
        cate_estimates = df_cate["cate"].values,
        true_cate      = true_cate,
        feature_cols   = feature_names,
        fitted_model   = fitted_model,
        X_pipeline     = X,
        Y_pipeline     = Y,
        T_pipeline     = T,
        income_col     = "annual_inc",
        output_dir     = output_dir,
        n_refute_runs  = 8,
        use_dowhy                 = use_dowhy,
        confounders               = pc_confounders,
        effect_modifiers          = pc_effect_mods,
        pipeline_n_estimators     = 200,
        pipeline_min_samples_leaf = 20,
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    n_passed = validation_results["n_refutations_passed"]
    n_shap   = validation_results["n_shap_wins"]

    print("\n" + "="*65)
    print("   RESULTS SUMMARY")
    print("="*65)
    print(f"   ATE  (avg causal effect):     {df_cate['cate'].mean():.4f}")
    print(f"   CATE std  (heterogeneity):    {df_cate['cate'].std():.4f}")
    print(f"   CATE max  (most sensitive):   {df_cate['cate'].max():.4f}")
    print(f"   CATE min  (least sensitive):  {df_cate['cate'].min():.4f}")
    print(f"   Refutation tests passed:      {n_passed}/4")
    print(f"   SHAP comparison DML wins:     {n_shap}/4")
    print()
    print("   Interpretation:")
    print(f"   A 1-unit increase in int_rate changes default probability")
    print(f"   by {df_cate['cate'].mean():.4f} on average across all borrowers.")
    print(f"   Most sensitive borrowers: {df_cate['cate'].max():.4f} change.")
    print(f"   Least sensitive borrowers: {df_cate['cate'].min():.4f} change.")
    print()
    print("   Output files:")
    expected = [
        "causal_dag.png",
        "cate_by_income_decile.png",
        "cate_distribution.png",
        "hte_feature_importance.png",
        "validation_refutations.png",
        "shap_vs_dml_comparison.png",
        "validation_summary.csv",
        "shap_comparison_results.csv",
        "cate_results.csv",
    ]
    for fname in expected:
        fpath  = f"{output_dir}/{fname}"
        status = "OK" if Path(fpath).exists() else "MISSING"
        print(f"   [{status}]  {fpath}")
    print("="*65 + "\n")

    results_path = f"{output_dir}/cate_results.csv"
    df_cate.to_csv(results_path, index=False)
    print(f"   CATE results saved to {results_path}")


def _generate_synthetic_data(nrows: int = 5000):
    """
    Generate synthetic LendingClub-like data for demo purposes.
    Used automatically when the real dataset is not available.

    Returns
    -------
    df_causal  : DataFrame with original column names (int_rate, default, ...)
    df_econml  : Same data renamed to T/Y (required by EconML)
    true_cate  : Ground truth individual treatment effects (numpy array).
                 Available only for synthetic data — used by Step 4 to
                 measure bias directly against the true causal effect.

    True causal structure:
        tau(income) = 0.08 - 0.04 * income
        poor  borrowers (income=-2): tau = 0.16  (very rate-sensitive)
        median borrowers (income=0): tau = 0.08
        rich  borrowers (income=+2): tau = 0.00  (barely rate-sensitive)
    """
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = nrows

    def std(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    # All variables standardised to mean=0, std=1 before building
    # causal relationships — ensures coefficients are on same scale
    emp_length  = std(np.random.uniform(0.5, 10, n))
    fico        = std(np.random.normal(680, 60, n).clip(580, 850))
    revol_util  = std(np.random.beta(2, 3, n))
    open_acc    = std(np.random.poisson(10, n).clip(1, 40).astype(float))
    grade_num   = std(np.round(np.random.uniform(1, 7, n)))
    dti         = std((revol_util + np.random.normal(0, 1, n)).clip(-3, 3))
    annual_inc  = std(np.random.normal(0, 1, n))
    loan_amnt   = std(np.random.normal(0, 1, n))
    home_own    = std(np.random.randint(0, 4, n).astype(float))
    purpose     = std(np.random.randint(0, 8, n).astype(float))

    # Treatment: confounded by grade, fico, dti, income
    int_rate = std(
        0.5 * grade_num
        - 0.4 * fico
        + 0.3 * dti
        - 0.2 * annual_inc
        + np.random.normal(0, 1.0, n)
    )

    # True heterogeneous causal effect (income-driven)
    true_cate = 0.08 - 0.04 * annual_inc

    # Binary outcome via threshold (approx 20% default rate)
    default_latent = (
        true_cate * int_rate
        + 0.15 * grade_num
        - 0.20 * fico
        + 0.10 * dti
        - 0.15 * annual_inc
        + 0.05 * revol_util
        + np.random.normal(0, 0.8, n)
    )
    default = (default_latent > np.percentile(default_latent, 80)).astype(float)

    log.info(
        "Synthetic data — n=%d | default rate: %.1f%% | "
        "true CATE range: [%.3f, %.3f]",
        n, default.mean() * 100, true_cate.min(), true_cate.max(),
    )

    df_causal = pd.DataFrame({
        "int_rate":           int_rate,
        "default":            default,
        "annual_inc":         annual_inc,
        "dti":                dti,
        "emp_length_num":     emp_length,
        "fico_range_low":     fico,
        "open_acc":           open_acc,
        "revol_util":         revol_util,
        "grade_num":          grade_num,
        "loan_amnt":          loan_amnt,
        "home_ownership_num": home_own,
        "purpose_num":        purpose,
    })
    df_econml = df_causal.rename(columns={"int_rate": "T", "default": "Y"})

    return df_causal, df_econml, true_cate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Causal Credit Risk Pipeline")
    parser.add_argument(
        "--data",      default="data/loan.csv",
        help="Path to LendingClub CSV (default: data/loan.csv)"
    )
    parser.add_argument(
        "--nrows",     default=100_000, type=int,
        help="Number of rows to load (default: 100000; use 20000 for fast iteration)"
    )
    parser.add_argument(
        "--output",    default="outputs/",
        help="Output directory (default: outputs/)"
    )
    parser.add_argument(
        "--use-dowhy", action="store_true", default=False,
        help=(
            "Use DoWhy CausalModel API for Step 4 refutations. "
            "Requires: pip install dowhy. "
            "DAG built from PC Algorithm outputs (Step 2). "
            "Estimator config matches Step 3 CausalForestDML exactly. "
            "Falls back to manual implementations if DoWhy not installed."
        )
    )
    parser.add_argument(
        "--pc-alpha", default=0.05, type=float,
        help=(
            "Significance level for PC Algorithm independence tests "
            "(default: 0.05). "
            "Use 0.10 if the Random Common Cause refutation test fails. "
            "Ignored if --algorithm ges is used."
        )
    )
    parser.add_argument(
        "--algorithm", default="pc", choices=["pc", "ges"],
        help=(
            "Causal discovery algorithm (default: pc). "
            "pc:  PC Algorithm — constraint-based, uses conditional independence tests. "
            "     Struggles with binary outcomes (Fisher Z assumption violation). "
            "ges: Greedy Equivalence Search — score-based (BIC), no CI tests. "
            "     More robust to binary outcomes and multicollinearity. "
            "     Recommended if PC finds too few confounders."
        )
    )
    parser.add_argument(
        "--ci-test", default="fisherz", choices=["fisherz", "chisq", "kci"],
        help=(
            "Conditional independence test for PC Algorithm (default: fisherz). "
            "fisherz: fast, assumes Gaussian — standard choice for large n. "
            "chisq:   better for discrete/binary variables (recommended for LendingClub). "
            "kci:     non-parametric, no assumptions — subsamples to 3000 rows (slow). "
            "Ignored if --algorithm ges is used."
        )
    )
    parser.add_argument(
        "--compare", action="store_true", default=False,
        help=(
            "Run all three causal discovery methods (GES, PC+ChiSq, FCI) "
            "and compare their confounder sets. "
            "Produces a comparison table showing which confounders are found "
            "by each method. The cross-method agreed set is used as W in DML. "
            "GES DAG is used as the primary graph. "
            "Saves comparison report to outputs/discovery_comparison.csv. "
            "Recommended: use this before finalising your pipeline."
        )
    )
    args = parser.parse_args()

    run_full_pipeline(
        data_path  = args.data,
        nrows      = args.nrows,
        output_dir = args.output,
        use_dowhy  = args.use_dowhy,
        pc_alpha   = args.pc_alpha,
        algorithm  = args.algorithm,
        ci_test    = args.ci_test,
        compare    = args.compare,
    )
