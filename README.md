# Causal Feature Selection & Heterogeneous Treatment Effects in Credit Risk
### *Moving beyond black-box ML to identify the true causal drivers of loan default*

---

## What is this project?

Most credit risk models (XGBoost, LightGBM) are excellent at **predicting** who will default but they can't answer: *"If we raise this borrower's interest rate by 1%, will they be more likely to default?"*

That question requires **causal inference**, not just prediction.

This project builds a full causal inference pipeline on the LendingClub dataset to answer:

> **"How does a change in interest rate causally affect default probability  and does this effect differ across borrower types?"**

The answer  called the **Heterogeneous Treatment Effect (HTE)**  has direct applications in:
- Loan pricing decisions
- Fair lending analysis (do rate hikes hurt low-income borrowers disproportionately?)
- Stress testing (what happens to default rates under a 200bps rate shock?)

---

## The Core Idea 

Imagine you have data showing that borrowers with higher interest rates default more often. Does that mean the rate *caused* the default?

**Not necessarily.** Maybe riskier borrowers *both* got higher rates *and* were more likely to default anyway  because they had poor credit scores. This is called **confounding**.

To find the *true causal effect* of the interest rate, we need to:
1. **Map out the causal relationships** (the DAG  like a flowchart of what causes what)
2. **Control for the confounders** (variables that influence both the rate AND default)
3. **Estimate the effect** in a way that's robust to our ML models being imperfect (this is DML)

---

## Project Architecture

```
causal_credit_risk/
│
├── data/                          ← Put your loan.csv here (see "Getting the Data")
│   └── loan.csv                   ← Raw LendingClub CSV (not included in repo, ~1.6 GB)
│
├── src/                           ← All source code (modular, importable)
│   ├── data_pipeline.py           ← STEP 1: Load, clean, encode, split into causal roles
│   ├── causal_discovery_3.py      ← STEP 2: GES/PC Algorithm → DAG visualization
│   ├── hte_estimation.py          ← STEP 3: Double ML → CATE → plots
│   └── validation_2.py            ← STEP 4: DoWhy refutations + SHAP vs DML comparison
│
├── outputs/                       ← Generated automatically when you run main_3.py
│   ├── causal_dag.png             ← The learned causal graph
│   ├── dag_ges.png                ← GES-specific DAG (when --algorithm ges)
│   ├── cate_by_income_decile.png  ← Key HTE result: rate sensitivity by income
│   ├── cate_distribution.png      ← Distribution of individual treatment effects
│   ├── hte_feature_importance.png ← What drives treatment effect heterogeneity?
│   ├── validation_refutations.png ← Four-panel DoWhy refutation test results
│   ├── shap_vs_dml_comparison.png ← SHAP vs DML four-panel comparison
│   ├── cate_results.csv           ← Individual CATE estimates (one row per borrower)
│   ├── shap_comparison_results.csv← SHAP vs DML numeric summary
│   └── validation_summary.csv     ← Refutation test numeric results
│
├── verify_confounders.py          ← Quick confounder validation script
├── main_3.py                      ← MASTER SCRIPT: runs the whole pipeline
├── requirements.txt               ← All Python dependencies
└── README.md                      ← You are here
```

**Each file is self-contained.** You can run just Step 2 without Step 3, etc.

---

## How the Four Steps Connect

```
  loan.csv
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: data_pipeline.py                                   │
│  • Encode categoricals (grade A→1, B→2... for math)         │
│  • Build binary outcome (default = 1 / 0)                   │
│  • Winsorize outliers (prevents bad stats in CI tests)       │
│  • Output 1: df_causal  → for causal discovery              │
│  • Output 2: df_econml  → for Double ML                     │
└───────────────────┬─────────────────────┬───────────────────┘
                    │                     │
                    ▼                     ▼
┌────────────────────────┐   ┌────────────────────────────────┐
│  STEP 2:               │   │  STEP 3:                       │
│  causal_discovery_3.py │   │  hte_estimation.py             │
│                        │   │                                │
│  GES/PC Algorithm      │   │  Double ML residualizes        │
│  learns which vars     │   │  confounders, then estimates   │
│  cause which others →  │   │  CATE τ(x) per borrower →     │
│  outputs a DAG         │   │  outputs treatment effect      │
│                        │   │  plots                         │
│  Tells us: "what must  │   │                                │
│  we control for?"      │   │  Tells us: "how large is       │
│                        │   │  the effect and for whom?"     │
└───────────┬────────────┘   └──────────────┬─────────────────┘
            │                               │
            └──────────────┬────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: validation_2.py                                    │
│  Part A: DoWhy refutation tests (4 tests  can we trust it?)│
│  Part B: SHAP vs DML comparison (why is SHAP wrong?)        │
│  Outputs: validation_refutations.png, shap_vs_dml.png       │
└─────────────────────────────────────────────────────────────┘
```

**The key link:** The DAG from Step 2 tells us *which variables are confounders*. We feed those into Step 3's DML estimator. Step 4 then validates the estimate is causally identified (not just a statistical fit) and benchmarks it against SHAP.

---

## The Causal Variables: What Role Does Each Play?

| Variable | Causal Role | Why |
|---|---|---|
| `int_rate` | **Treatment (T)** | The intervention we're studying  does it cause default? |
| `default` | **Outcome (Y)** | What we're trying to explain causally |
| `annual_inc`, `dti`, `fico_range_low`, `emp_length`, `open_acc`, `revol_util` | **Confounders (W)** | These influence *both* the interest rate assigned AND the likelihood of default  if we don't control for them, we get a biased estimate |
| `annual_inc`, `loan_amnt`, `grade_num`, `purpose`, `home_ownership` | **Effect Modifiers (X)** | These change *how much* the treatment affects the outcome  used to calculate CATE |
| `total_pymnt`, `recoveries`, `funded_amnt` | **Excluded** ⚠️ | These happen *after* the loan is issued. Including them would open a statistical "backdoor" and break our estimates |

> **Why are `total_pymnt` etc. excluded?** Because they're consequences of both the interest rate AND default. Controlling for a consequence of both cause and effect is a classic error called **collider bias**  it creates a spurious correlation that didn't exist in reality.

---

## Key Concepts

### The Backdoor Criterion
When we ask "what causes what," we need to block all *non-causal* paths between treatment and outcome. In a causal graph, if income affects both `int_rate` (via underwriting) and `default` (via repayment capacity), then income is a "backdoor path" from rate to default. We close it by including income as a control. The **Backdoor Criterion** is the formal mathematical rule for which variables to include as controls.

### Double Machine Learning (DML)  The "Why" Not Just "How"
Standard regression: `default = α + β·int_rate + γ·income + ε`
The problem: the ML model might learn that income is so predictive that it "explains away" the rate's effect, underestimating β.

DML fixes this by working in two stages:
- **Stage 1 (Residualize):** Build a ML model of `int_rate ~ income, dti, fico...` and extract the *residual* (the "surprise" in the rate that can't be explained by confounders). Do the same for `default`.
- **Stage 2 (Regress residuals):** Regress the default residual on the rate residual. Now β is the *pure causal effect*, free of confounding.

The mathematical insight: after residualization, the residual of `int_rate` is uncorrelated with all confounders  it's *as if* rates were randomly assigned.

### CATE: Why One Number Isn't Enough
The Average Treatment Effect (ATE) gives you one number: "a 1% rate increase raises default probability by X% on average." But CATE asks: *for whom?*

```
τ(x) = E[Y(1) - Y(0) | X = x]
```

Where `Y(1)` = default probability at raised rate, `Y(0)` = default probability at current rate, and `x` = this specific borrower's characteristics. The Causal Forest learns this function non-parametrically  a different τ for every borrower.

---

## Getting the Data

1. Go to [Kaggle: LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Click **Download** (you need a free Kaggle account)
3. Unzip and place `loan.csv` (or `accepted_2007_to_2018Q4.csv`) inside the `data/` folder
4. Rename it to `loan.csv` if needed

The dataset has ~2.2 million rows and 150 columns. The pipeline uses only ~12 columns and defaults to loading 100,000 rows for speed.

---

## Installation

**Step 1: Create a virtual environment** (keeps your packages isolated)

```bash
# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

This installs: `causal-learn` (PC Algorithm), `econml` (DML), `dowhy` (identification), plus standard data science libraries.

> ⚠️ **Note on install time:** EconML installs several sub-packages and may take 3–5 minutes. This is normal.

> ⚠️ **Python version:** Use Python 3.9, 3.10, or 3.11. Some packages have limited Python 3.12 support.

---

## Running the Project

> **The master script is `main_3.py`.** All four steps run sequentially from this single entry point.

### Option A: Run the full pipeline (recommended)

```bash
python main_3.py --data data/loan.csv --nrows 100000 --algorithm ges --use-dowhy
```

This is the **recommended production command**. It runs all four steps with:
- `--data data/loan.csv`  full real dataset
- `--nrows 100000`  100k rows (the full usable sample)
- `--algorithm ges`  GES causal discovery (more robust than PC for binary outcomes)
- `--use-dowhy`  DoWhy API for Step 4 refutation tests

Saves all plots and CSVs to `outputs/`.

### Option B: Fast iteration (development)

```bash
python main_3.py --data data/loan.csv --nrows 20000 --algorithm ges
```

Use `--nrows 20000` for fast local iteration. Drop `--use-dowhy` to skip the slower DoWhy API calls.

### Option C: Run without real data (synthetic demo)

```bash
python main_3.py
```

If `data/loan.csv` is not found, the pipeline automatically generates synthetic LendingClub-like data (5,000 rows). All four steps run identically  useful for testing your setup before downloading the dataset.

### Option D: Compare all causal discovery methods

```bash
python main_3.py --data data/loan.csv --nrows 100000 --algorithm ges --use-dowhy --compare
```

Runs GES, PC+ChiSq, and empirical correlation together, compares their confounder sets, and uses the cross-method agreed set as W in DML. Saves `outputs/discovery_comparison.csv`. Useful for validating that the confounder selection is robust across methods.

### Full CLI Reference

```
python main_3.py [OPTIONS]

Options:
  --data PATH        Path to LendingClub CSV           (default: data/loan.csv)
  --nrows INT        Rows to load                      (default: 100000)
  --output DIR       Output directory                  (default: outputs/)
  --algorithm STR    Causal discovery: pc or ges       (default: pc)
  --ci-test STR      CI test for PC: fisherz, chisq, kci  (default: fisherz)
  --pc-alpha FLOAT   Significance level for PC tests   (default: 0.05)
  --compare          Run all discovery methods and compare confounder sets
  --use-dowhy        Use DoWhy API for Step 4 refutations
```

**CI test guide:**
- `fisherz`  fast, assumes Gaussian. Best for large n with continuous features.
- `chisq`  better for discrete/binary variables. Recommended for LendingClub's binary `default`.
- `kci`  non-parametric, no assumptions. Very slow (subsamples to 3,000 rows).

> ⚠️ `--ci-test` and `--pc-alpha` are ignored when `--algorithm ges` is used.

### Running individual modules

```bash
# Quick confounder validation check
python verify_confounders.py

# Step 3 only (HTE estimation on synthetic data)
python src/hte_estimation.py
```

---

## Expected Outputs

After running `main_3.py`, check the `outputs/` folder:

| File | What It Shows |
|---|---|
| `causal_dag.png` | The learned causal graph. Red=treatment, Green=outcome, Blue=confounder, Purple=effect modifier. Solid arrows = directed causal edges. |
| `dag_ges.png` | GES-specific DAG with domain constraints applied (produced when `--algorithm ges` or `--compare`). |
| `cate_by_income_decile.png` | **The key result.** Each bar = average treatment effect for that income group. Higher bar = that group is more sensitive to rate hikes. |
| `cate_distribution.png` | Left: histogram of all individual τ̂(x) values. Right: sorted "policy curve" showing what fraction of borrowers have positive vs negative effects. |
| `hte_feature_importance.png` | Which features *drive* heterogeneity? High importance means that variable explains *why* some borrowers are more rate-sensitive than others. |
| `validation_refutations.png` | **Step 4A.** Four-panel plot of DoWhy refutation tests (placebo, random cause, bootstrap, tiered sensitivity). |
| `shap_vs_dml_comparison.png` | **Step 4B.** Four-panel comparison of SHAP attribution vs DML causal estimation. |
| `cate_results.csv` | One row per borrower. Columns: `cate` (point estimate), `cate_lower`, `cate_upper` (90% confidence interval). |
| `validation_summary.csv` | Numeric results for all four refutation tests. |
| `shap_comparison_results.csv` | Numeric SHAP vs DML comparison across the four demonstrations. |
| `discovery_comparison.csv` | Cross-method confounder comparison (only with `--compare`). |

---

## Interpreting the Results

### The CATE by Income Decile Plot

- **D1 (lowest income) has the highest bar** → Low-income borrowers are most sensitive to rate increases. A 1% rate hike raises their default probability more than it does for high-income borrowers. This is the classic financial vulnerability story.
- **D10 (highest income) has the lowest bar** → High-income borrowers can absorb rate increases; their default probability barely moves.

**Business insight:** Uniform rate increases disproportionately impact subprime borrowers. A risk-adjusted pricing model should weight the CATE when setting rates, not just the predicted default probability.

### The CATE Distribution Plot

- **Wide distribution** → Strong heterogeneity  the single ATE number is hiding a lot of variation across borrowers. HTE analysis is valuable here.
- **Narrow distribution** → Weak heterogeneity  all borrowers respond similarly to rate changes. A single ATE may be sufficient.

---

## Common Errors & Fixes

| Error | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: causal_learn` | Package not installed | `pip install causal-learn` (note the hyphen) |
| `ModuleNotFoundError: econml` | Package not installed | `pip install econml` |
| `FileNotFoundError: data/loan.csv` | Data not downloaded | See "Getting the Data" above; or just run without data for demo |
| `KeyError: 'loan_status'` | Wrong CSV file format | Make sure you downloaded the correct LendingClub file from Kaggle |
| PC Algorithm runs for 30+ min | Dataset too large | Use `--nrows 30000` to limit rows |
| All CATE values are identical | EconML not installed, using fallback | The manual DML fallback gives ATE only. Install EconML for true CATE. |

---

## Validation Results (Step 4)

Step 4 is fully implemented in `src/validation_2.py` and runs automatically as part of `main_3.py`.

**Part A : DoWhy Refutation Tests (4/4 passed):**

| Test | Result | Key Number |
|---|---|---|
| Placebo Treatment | ✓ PASSED | Fake T gives ATE = −0.0002 (p = 0.30) |
| Random Common Cause | ✓ PASSED | 0.1% ATE change when adding random confounder |
| Data Subset Bootstrap | ✓ PASSED | ATE = 0.0240 across 80% subsamples |
| Tiered Sensitivity | ✓ PASSED | Robust through moderate unmeasured confounding |

**Part B : SHAP vs DML Comparison (DML wins 3/4):**

| Demo | SHAP | DML | Winner |
|---|---|---|---|
| Placebo test | 0.0125 fake effect | 0.0034 ≈ 0 | DML |
| Confounding bias | 0.0924 bias | 0.0118 bias | DML |
| HTE recovery | N/A on real data | N/A on real data |  |
| Stability | 0.0242 mean error | 0.0084 mean error | DML |

---

## References

- Chernozhukov et al. (2018)  *Double/Debiased Machine Learning* (the DML paper)
- Wager & Athey (2018)  *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests* (Causal Forests)
- Pearl (2009)  *Causality* (foundational causal DAG theory)
- Spirtes, Glymour & Scheines (2000)  *Causation, Prediction and Search* (PC Algorithm)
