"""
verify_confounders.py
─────────────────────
Quick empirical check: do the GES-discovered confounders actually
correlate with BOTH int_rate AND default in the real data?

A genuine confounder must:
  1. Correlate with int_rate (treatment)      — partial corr > threshold
  2. Correlate with default (outcome)         — point-biserial corr > threshold
  3. Be temporally prior to both              — domain logic check

Run: python verify_confounders.py --data data/loan.csv --nrows 100000
"""
import argparse
import pandas as pd
import numpy as np
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data/loan.csv")
    parser.add_argument("--nrows", default=100000, type=int)
    args = parser.parse_args()

    # Load the pipeline output (already cleaned)
    # We replicate just enough of data_pipeline to get scaled data
    import sys
    sys.path.insert(0, "src")
    from data_pipeline import run_pipeline
    df_causal, _, _ = run_pipeline(args.data, nrows=args.nrows)

    candidates = ["annual_inc", "dti", "fico_range_low", "grade_num",
                  "emp_length_num", "open_acc", "revol_util"]
    treatment  = "int_rate"
    outcome    = "default"

    print("\n" + "="*70)
    print("  CONFOUNDER VERIFICATION — Empirical correlations in real data")
    print("="*70)
    print(f"  {'Variable':<20} {'Corr→int_rate':>15} {'Corr→default':>15} "
          f"{'Confounder?':>12}")
    print("-"*70)

    confirmed = []
    for var in candidates:
        if var not in df_causal.columns:
            continue

        # Correlation with int_rate (continuous → continuous: Pearson)
        r_T, p_T = stats.pearsonr(df_causal[var], df_causal[treatment])

        # Correlation with default (continuous → binary: point-biserial)
        # scipy's pointbiserialr is equivalent to Pearson for binary Y
        r_Y, p_Y = stats.pointbiserialr(df_causal[outcome], df_causal[var])

        # Both correlations significant AND meaningful (|r| > 0.05)
        is_confounder = (abs(r_T) > 0.05 and p_T < 0.01 and
                         abs(r_Y) > 0.05 and p_Y < 0.01)
        if is_confounder:
            confirmed.append(var)

        flag = "✓ CONFIRMED" if is_confounder else (
               "→ T only"   if abs(r_T) > 0.05 and p_T < 0.01 else
               "→ Y only"   if abs(r_Y) > 0.05 and p_Y < 0.01 else
               "✗ weak")

        print(f"  {var:<20} {r_T:>+14.3f}  {r_Y:>+14.3f}  {flag:>12}")

    print("="*70)
    print(f"\n  GES discovered:  ['annual_inc', 'dti', 'fico_range_low', 'grade_num']")
    print(f"  Data confirms:   {confirmed}")

    agreed = set(confirmed) & {"annual_inc", "dti", "fico_range_low", "grade_num"}
    missed = {"annual_inc", "dti", "fico_range_low", "grade_num"} - set(confirmed)
    extra  = set(confirmed) - {"annual_inc", "dti", "fico_range_low", "grade_num"}

    print(f"\n  Agreement:       {sorted(agreed)}")
    if missed:
        print(f"  GES found but weak in data: {sorted(missed)}")
    if extra:
        print(f"  In data but GES missed:     {sorted(extra)}  ← consider adding to W")
    print()

if __name__ == "__main__":
    main()
