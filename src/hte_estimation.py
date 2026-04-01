"""
causal_credit_risk/src/hte_estimation.py
=========================================
STEP 3 — Heterogeneous Treatment Effect Estimation via Double Machine Learning

What this file does (plain English):
--------------------------------------
We want to know: "If we raise someone's interest rate by 1%, how much more
likely are they to default?" — AND crucially, does this effect differ
across borrower types (rich vs poor, high vs low FICO)?

The answer is the CATE: Conditional Average Treatment Effect.
  τ(x) = E[Y(1) - Y(0) | X = x]
  = "For a borrower with characteristics x, how much does a 1-unit
    increase in interest rate change their default probability?"

Why Double Machine Learning (DML)?
------------------------------------
Naive regression (regress default on int_rate + controls) is biased because:
  - The ML model for default will "soak up" variation in int_rate
  - Leading to attenuation bias (underestimating the true causal effect)

DML fixes this in 2 stages (Robinson 1988 / Chernozhukov 2018):
  Stage 1 — Residualize:
    Ṽ = int_rate - E[int_rate | X, W]   ← "surprise" in the rate
    Ỹ = default   - E[default   | X, W] ← "surprise" in default
  Stage 2 — Regress residuals:
    Ỹ = τ(X) · Ṽ + ε
    τ(X) is now unbiased because we've removed confounder influence

Key result: τ(x) tells us WHICH borrowers are most sensitive to rate hikes.

Author  : Senior Quant Researcher
Dataset : LendingClub (output of data_pipeline.py)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Try importing EconML ──────────────────────────────────────────────────────
try:
    from econml.dml import CausalForestDML, LinearDML
    from econml.inference import BootstrapInference
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    log.warning(
        "EconML not installed. Run: pip install econml\n"
        "Will use a simplified manual DML fallback for demonstration."
    )


# ── DML Estimator ─────────────────────────────────────────────────────────────

def fit_causal_forest_dml(
    df_econml: pd.DataFrame,
    n_estimators: int   = 200,
    min_samples_leaf: int = 20,
    n_crossfit_splits: int = 5,
    random_state: int   = 42,
    confounders: list   = None,
):
    """
    Fit a CausalForestDML model to estimate heterogeneous treatment effects.

    Architecture:
    -------------
    CausalForestDML combines:
      1. DML residualization (removes confounding via cross-fitting)
      2. Causal Forest (estimates τ(x) non-parametrically, like a random forest
         but trained to maximize heterogeneity in treatment effects, not prediction)

    Cross-fitting (why it matters):
    --------------------------------
    Naively fitting Stage 1 on the same data as Stage 2 causes overfitting.
    Cross-fitting uses K-fold: Stage 1 trains on fold k, predicts on fold k+1.
    This is the "Double" in DML — double robustness against model misspecification.

    Parameters
    ----------
    df_econml       : DataFrame with columns Y, T, and feature columns
    n_estimators    : number of trees in the causal forest
    min_samples_leaf: minimum leaf size (regularization — increase to avoid overfitting)
    n_crossfit_splits: number of cross-fitting folds
    random_state    : reproducibility seed

    Returns
    -------
    model : fitted CausalForestDML
    X     : effect modifier array
    Y     : outcome array
    T     : treatment array
    """
    Y = df_econml["Y"].values
    T = df_econml["T"].values

    all_feature_cols = [c for c in df_econml.columns if c not in {"Y", "T"}]

    if confounders:
        # Use PC/GES-discovered confounders as W (backdoor adjustment set)
        # X = all features except pure confounders that aren't effect modifiers
        w_feature_cols = [c for c in confounders if c in df_econml.columns]
        # X = remaining features (effect modifiers + other predictors)
        # We keep confounders in X too if they also moderate the effect
        x_feature_cols = all_feature_cols  # X uses all features for CATE conditioning
        log.info("Using PC/GES-discovered confounders as W: %s", w_feature_cols)
    else:
        # Fallback: w_ prefix heuristic
        x_feature_cols = [c for c in all_feature_cols if not c.startswith("w_")]
        w_feature_cols = [c for c in all_feature_cols if c.startswith("w_")]
        if not x_feature_cols:
            x_feature_cols = all_feature_cols
        if not w_feature_cols:
            w_feature_cols = None
        log.info("No confounders passed — using w_ prefix heuristic. "
                 "Pass confounders=pc_confounders from Step 2 for proper W.")

    X = df_econml[x_feature_cols].values
    W = df_econml[w_feature_cols].values if w_feature_cols else None

    log.info(
        "DML setup — Y: %s | T: %s | X: %s cols | W: %s cols",
        Y.shape, T.shape, X.shape[1], W.shape[1] if W is not None else 0,
    )

    if not ECONML_AVAILABLE:
        log.warning("EconML not available — using manual DML fallback.")
        return _manual_dml_fallback(Y, T, X, x_feature_cols)

    # ── Stage 1 models: predict T and Y from confounders ─────────────────────
    # IMPORTANT: Use GradientBoostingREGRESSOR for Y even though default is
    # binary. DML operates on the continuous probability scale internally —
    # using a Classifier triggers sklearn's single-class fold error when
    # cross-fitting splits happen to land on imbalanced folds.
    model_t = GradientBoostingRegressor(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=random_state,
    )
    model_y = GradientBoostingRegressor(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=random_state,
    )

    # ── Build a KFold splitter ────────────────────────────────────────────────
    # Plain KFold with shuffle=True gives different random splits each fold,
    # which is enough to avoid single-class folds on 5000 rows.
    # (StratifiedKFold would fail here because T is continuous, not categorical)
    from sklearn.model_selection import KFold
    cv_splitter = KFold(
        n_splits=n_crossfit_splits,
        shuffle=True,
        random_state=random_state,
    )

    # ── CausalForestDML ───────────────────────────────────────────────────────
    # Note: EconML >= 0.15 renamed n_crossfit_splits → cv
    # We try the new name first, fall back to old name for older installs
    try:
        model = CausalForestDML(
            model_t=model_t,
            model_y=model_y,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=None,
            cv=cv_splitter,
            random_state=random_state,
            verbose=0,
        )
    except TypeError:
        model = CausalForestDML(
            model_t=model_t,
            model_y=model_y,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=None,
            n_crossfit_splits=n_crossfit_splits,
            random_state=random_state,
            verbose=0,
        )

    log.info("Fitting CausalForestDML (this may take 1–3 minutes)...")
    if W is not None:
        model.fit(Y, T, X=X, W=W)
    else:
        model.fit(Y, T, X=X)

    log.info("CausalForestDML fit complete.")
    return model, X, Y, T, x_feature_cols


def _manual_dml_fallback(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
):
    """
    Simplified manual DML for environments without EconML.
    Implements the core Robinson (1988) residualization by hand.

    Stage 1: Predict T and Y from X using cross-validation.
    Stage 2: Regress Ỹ on Ṽ (residuals) to get ATE.
    Returns a dict mimicking the EconML model interface.
    """
    log.info("Running manual DML fallback...")

    nuisance_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Stage 1: residualize via 5-fold cross-prediction
    T_hat = cross_val_predict(nuisance_model, X, T, cv=5)
    Y_hat = cross_val_predict(nuisance_model, X, Y, cv=5)

    T_tilde = T - T_hat   # "Surprise" in treatment
    Y_tilde = Y - Y_hat   # "Surprise" in outcome

    # Stage 2: regress Y_tilde on T_tilde (OLS)
    # τ_hat = Cov(Ỹ, Ṽ) / Var(Ṽ)  — the Robinson estimator
    tau_ate = np.cov(Y_tilde, T_tilde)[0, 1] / np.var(T_tilde)
    log.info("Manual DML ATE estimate: τ = %.4f", tau_ate)

    # For CATE: fit a gradient boosted model on T_tilde / Y_tilde
    # This approximates what CausalForestDML does internally
    cate_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    # Weighted pseudo-outcome: Ỹ / Ṽ  (where Ṽ ≠ 0)
    mask = np.abs(T_tilde) > 1e-8
    pseudo_outcome = np.where(mask, Y_tilde / T_tilde, tau_ate)
    cate_model.fit(X[mask], pseudo_outcome[mask])

    # Return a dict that mimics EconML model interface
    return {
        "type": "manual_dml",
        "ate": tau_ate,
        "cate_model": cate_model,
        "T_tilde": T_tilde,
        "Y_tilde": Y_tilde,
        "feature_names": feature_names,
    }, X, Y, T, feature_names


# ── CATE Extraction ───────────────────────────────────────────────────────────

def get_cate_estimates(
    model,
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract individual CATE estimates τ̂(xᵢ) for each borrower.

    Parameters
    ----------
    model         : fitted CausalForestDML or manual DML dict
    X             : effect modifier array
    feature_names : column names corresponding to X

    Returns
    -------
    df_cate : DataFrame with feature columns + cate + cate_lower + cate_upper
    """
    df_out = pd.DataFrame(X, columns=feature_names)

    if isinstance(model, dict) and model.get("type") == "manual_dml":
        # Manual DML fallback
        cate = model["cate_model"].predict(X)
        df_out["cate"]       = cate
        df_out["cate_lower"] = cate - 0.02   # Placeholder CI
        df_out["cate_upper"] = cate + 0.02
    else:
        # EconML CausalForestDML
        cate_results = model.effect_interval(X, alpha=0.1)  # 90% CI
        df_out["cate"]       = model.effect(X).flatten()
        df_out["cate_lower"] = cate_results[0].flatten()
        df_out["cate_upper"] = cate_results[1].flatten()

    log.info(
        "CATE summary — mean: %.4f | std: %.4f | min: %.4f | max: %.4f",
        df_out["cate"].mean(), df_out["cate"].std(),
        df_out["cate"].min(),  df_out["cate"].max(),
    )
    return df_out


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_cate_by_income_decile(
    df_cate: pd.DataFrame,
    income_col: str = "annual_inc",
    save_path: Optional[str] = "outputs/cate_by_income_decile.png",
) -> None:
    """
    Plot "Treatment Effect by Income Decile" — the key HTE result.

    What this shows:
    ----------------
    Each bar = average CATE for borrowers in that income decile.
    A higher bar means those borrowers are MORE sensitive to rate hikes
    (a 1% rate increase causes a larger increase in default probability).

    Business interpretation:
    ------------------------
    If low-income borrowers (D1–D3) have high CATE → rate hikes disproportionately
    hurt subprime borrowers → policy insight for pricing / risk management.

    Parameters
    ----------
    df_cate     : DataFrame with `cate` column and income column
    income_col  : name of the income column in df_cate
    save_path   : path to save the figure
    """
    if income_col not in df_cate.columns:
        log.warning("Income column '%s' not in df_cate. Skipping decile plot.", income_col)
        return

    df = df_cate.copy()
    df["income_decile"] = pd.qcut(
        df[income_col], q=10, labels=[f"D{i}" for i in range(1, 11)]
    )

    # Average CATE per decile + 90% CI approximation
    summary = df.groupby("income_decile", observed=True).agg(
        cate_mean  = ("cate",       "mean"),
        cate_lower = ("cate_lower", "mean"),
        cate_upper = ("cate_upper", "mean"),
        n          = ("cate",       "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    colors = ["#E74C3C" if v > summary["cate_mean"].median() else "#3498DB"
              for v in summary["cate_mean"]]

    bars = ax.bar(
        summary["income_decile"].astype(str),
        summary["cate_mean"],
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # Error bars (90% CI)
    ax.errorbar(
        x=range(len(summary)),
        y=summary["cate_mean"],
        yerr=[
            summary["cate_mean"] - summary["cate_lower"],
            summary["cate_upper"] - summary["cate_mean"],
        ],
        fmt="none",
        color="#2C3E50",
        capsize=5,
        linewidth=1.5,
        zorder=4,
    )

    # Add value labels on bars
    for bar, val in zip(bars, summary["cate_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#2C3E50",
        )

    # Reference line at zero
    ax.axhline(0, color="#2C3E50", linewidth=1, linestyle="--", alpha=0.5)

    # Add ATE as horizontal line
    ate = df["cate"].mean()
    ax.axhline(ate, color="#E67E22", linewidth=2, linestyle="-", alpha=0.8,
               label=f"Population ATE = {ate:.4f}")

    ax.set_xlabel("Income Decile (D1 = Lowest Income, D10 = Highest)", fontsize=11)
    ax.set_ylabel("CATE — Δ P(Default) per 1% Rate Increase", fontsize=11)
    ax.set_title(
        "Heterogeneous Treatment Effects by Income Decile\n"
        "How sensitive is each income group to interest rate hikes?",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Annotation for interpretation
    min_decile = summary.loc[summary["cate_mean"].idxmin(), "income_decile"]
    max_decile = summary.loc[summary["cate_mean"].idxmax(), "income_decile"]
    ax.annotate(
        f"Most sensitive:\n{max_decile}",
        xy=(summary["income_decile"].astype(str).tolist().index(str(max_decile)),
            summary["cate_mean"].max()),
        xytext=(1, summary["cate_mean"].max() + 0.01),
        fontsize=8, color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C"),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("CATE decile plot saved to %s", save_path)
    plt.show()


def plot_cate_distribution(
    df_cate: pd.DataFrame,
    save_path: Optional[str] = "outputs/cate_distribution.png",
) -> None:
    """
    Plot the full distribution of individual CATE estimates.

    What this shows:
    ----------------
    If all borrowers had the same treatment effect, this would be a spike.
    A wide distribution = significant heterogeneity = HTE is real and meaningful.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Distribution of Individual Treatment Effects τ̂(xᵢ)",
                 fontsize=13, fontweight="bold")

    # Left: histogram
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    ax.hist(df_cate["cate"], bins=50, color="#3498DB", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(df_cate["cate"].mean(), color="#E74C3C", linewidth=2,
               label=f"Mean ATE = {df_cate['cate'].mean():.4f}")
    ax.axvline(0, color="#2C3E50", linewidth=1.5, linestyle="--",
               label="Zero effect")
    ax.set_xlabel("CATE τ̂(x)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("CATE Histogram", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: sorted CATE (policy curve)
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    sorted_cate = np.sort(df_cate["cate"].values)
    ax2.plot(np.linspace(0, 100, len(sorted_cate)), sorted_cate,
             color="#3498DB", linewidth=2)
    ax2.fill_between(np.linspace(0, 100, len(sorted_cate)),
                     sorted_cate, 0,
                     where=sorted_cate > 0, alpha=0.3, color="#E74C3C",
                     label="Positive effect (rate hike → more default)")
    ax2.fill_between(np.linspace(0, 100, len(sorted_cate)),
                     sorted_cate, 0,
                     where=sorted_cate <= 0, alpha=0.3, color="#2ECC71",
                     label="Negative/null effect")
    ax2.axhline(0, color="#2C3E50", linewidth=1, linestyle="--")
    ax2.set_xlabel("Borrower Percentile (sorted by CATE)", fontsize=10)
    ax2.set_ylabel("CATE τ̂(x)", fontsize=10)
    ax2.set_title("Policy Curve — Sorted CATEs", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("CATE distribution plot saved to %s", save_path)
    plt.show()


def plot_feature_importance_for_hte(
    model,
    feature_names: list[str],
    save_path: Optional[str] = "outputs/hte_feature_importance.png",
) -> None:
    """
    Plot which features DRIVE heterogeneity in treatment effects.

    This is different from standard feature importance!
    Here importance = "which variables predict HOW MUCH the rate affects default?"
    High importance for `annual_inc` means income drives rate sensitivity.
    """
    if isinstance(model, dict):
        importances = model["cate_model"].feature_importances_
    elif ECONML_AVAILABLE:
        importances = model.feature_importances_
    else:
        return

    idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_features)))[::-1]
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Feature Importance for HTE (Causal Forest)", fontsize=10)
    ax.set_title(
        "What Drives Heterogeneity in Treatment Effects?\n"
        "(Higher = variable explains more variation in rate sensitivity)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("HTE feature importance saved to %s", save_path)
    plt.show()


# ── Main runner ───────────────────────────────────────────────────────────────

def run_hte_estimation(
    df_econml: pd.DataFrame,
    income_col:  str  = "annual_inc",
    output_dir:  str  = "outputs/",
    confounders: list = None,
) -> tuple:
    """
    Full Step 3 pipeline: fit DML, extract CATEs, generate plots.

    Parameters
    ----------
    df_econml  : EconML-ready DataFrame (from data_pipeline.py)
    income_col : column to use for income-decile plot
    output_dir : directory to save plots

    Returns
    -------
    df_cate       : DataFrame with individual CATE estimates per borrower
    model         : fitted CausalForestDML (or manual DML dict) — passed to Step 4
    X             : effect modifier array used during fitting  — passed to Step 4
    Y             : outcome array                              — passed to Step 4
    T             : treatment array                            — passed to Step 4
    feature_names : list of feature column names               — passed to Step 4

    NOTE: returning the fitted model is intentional.
    Step 4 (validation.py) needs the SAME model that produced df_cate — not
    a freshly re-fitted one — so that refutation tests and SHAP comparisons
    run against the exact object whose CATEs appear in your paper.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Fit model — pass confounders so DML uses the PC/GES-discovered W
    result = fit_causal_forest_dml(df_econml, confounders=confounders)
    model, X, Y, T, feature_names = result

    # Extract CATEs
    df_cate = get_cate_estimates(model, X, feature_names)

    # Attach income column for decile plot
    if income_col in df_econml.columns and income_col not in df_cate.columns:
        df_cate[income_col] = df_econml[income_col].values[:len(df_cate)]

    # Generate plots
    plot_cate_by_income_decile(
        df_cate, income_col=income_col,
        save_path=f"{output_dir}/cate_by_income_decile.png",
    )
    plot_cate_distribution(
        df_cate,
        save_path=f"{output_dir}/cate_distribution.png",
    )
    plot_feature_importance_for_hte(
        model, feature_names,
        save_path=f"{output_dir}/hte_feature_importance.png",
    )

    return df_cate, model, X, Y, T, feature_names


if __name__ == "__main__":
    # Synthetic demo — runs without real data
    np.random.seed(42)
    n = 2000

    # Simulate confounded observational data
    # True model: default ~ 0.02 * int_rate + noise, but high-income borrowers
    # are less sensitive (true CATE = 0.02 - 0.005 * income)
    income  = np.random.lognormal(10, 0.8, n)
    fico    = np.random.normal(680, 60, n)
    dti     = np.random.normal(18, 8, n).clip(0, 60)
    grade   = np.random.randint(1, 8, n).astype(float)

    # Interest rate = f(grade, fico, dti) + noise  [confounded assignment]
    int_rate = 5 + 2 * grade - 0.005 * fico + 0.1 * dti + np.random.normal(0, 1, n)
    int_rate = int_rate.clip(5, 30)

    # Default = f(int_rate, income, fico) [heterogeneous effect of int_rate]
    true_cate = 0.02 - 0.000003 * income   # Lower income → higher sensitivity
    default_prob = 0.05 + true_cate * int_rate - 0.0003 * fico + np.random.normal(0, 0.05, n)
    default = (default_prob > 0.15).astype(float)

    demo_df = pd.DataFrame({
        "Y":          default,
        "T":          int_rate,
        "annual_inc": income,
        "fico_range_low": fico,
        "dti":        dti,
        "grade_num":  grade,
        "loan_amnt":  np.random.lognormal(9, 0.7, n),
    })

    print(f"Demo data shape: {demo_df.shape}")
    print(f"Default rate: {demo_df['Y'].mean():.3f}")
    print(f"Mean int_rate: {demo_df['T'].mean():.2f}%\n")

    df_cate = run_hte_estimation(demo_df, income_col="annual_inc", output_dir="outputs")
    print("\nCate sample:")
    print(df_cate[["annual_inc", "cate", "cate_lower", "cate_upper"]].head(10).to_string())