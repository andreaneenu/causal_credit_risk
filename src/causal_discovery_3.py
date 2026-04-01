"""
causal_credit_risk/src/causal_discovery.py
===========================================
STEP 2 — Causal Discovery: PC Algorithm + DAG Visualization

What this file does (plain English):
-------------------------------------
We feed our cleaned data into the PC Algorithm, which runs statistical
independence tests between every pair of variables to figure out:
  "Does knowing X tell me something about Y, even after I know Z?"
If the answer is NO for all sets Z, then X and Y are NOT causally linked.
The result is a Directed Acyclic Graph (DAG) — a picture of what causes what.

Key concept — CPDAG vs DAG:
  The PC Algorithm often can't determine the direction of some edges.
  It returns a CPDAG (Completed Partially Directed Acyclic Graph).
  We then use domain knowledge to orient ambiguous edges manually.

Author  : Senior Quant Researcher
Dataset : LendingClub (output of data_pipeline.py)
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Try importing causal-learn (optional at import time) ─────────────────────
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    log.warning(
        "causal-learn not installed. Run: pip install causal-learn\n"
        "Falling back to demo DAG for visualization."
    )

# Optional non-parametric CI tests
try:
    from causallearn.utils.cit import chisq
    CHISQ_AVAILABLE = True
except ImportError:
    CHISQ_AVAILABLE = False

try:
    from causallearn.utils.cit import kci
    KCI_AVAILABLE = True
except ImportError:
    KCI_AVAILABLE = False

# Optional GES (score-based, robust to non-Gaussian data)
try:
    from causallearn.search.ScoreBased.GES import ges as run_ges_causallearn
    GES_AVAILABLE = True
except ImportError:
    GES_AVAILABLE = False

# Optional FCI (handles hidden confounders, returns PAG)
try:
    from causallearn.search.ConstraintBased.FCI import fci as run_fci_causallearn
    FCI_AVAILABLE = True
except ImportError:
    FCI_AVAILABLE = False


# ── Constants: domain knowledge constraints ───────────────────────────────────

# Edges that are IMPOSSIBLE by domain logic.
# Format: (cause, effect) — these will be forced or forbidden.
# The PC algorithm may propose these; we override them.
FORBIDDEN_EDGES: list[tuple[str, str]] = [
    # Temporal constraint: default occurs AFTER all loan origination variables
    # are set. Nothing can be caused BY default except post-loan variables.
    ("default", "int_rate"),
    ("default", "grade_num"),
    ("default", "fico_range_low"),
    ("default", "annual_inc"),
    ("default", "dti"),
    ("default", "emp_length_num"),
    ("default", "loan_amnt"),
    ("default", "open_acc"),
    ("default", "revol_util"),
    ("default", "home_ownership_num"),
    ("default", "purpose_num"),
    # Employment length is a pre-existing borrower characteristic —
    # financial variables cannot cause it
    ("annual_inc",  "emp_length_num"),
    ("dti",         "emp_length_num"),
    ("int_rate",    "emp_length_num"),
    ("grade_num",   "emp_length_num"),
    # Interest rate is set by lender — borrower characteristics cause it,
    # it cannot cause the borrower's pre-existing financial profile
    ("int_rate", "annual_inc"),
    ("int_rate", "dti"),
    ("int_rate", "fico_range_low"),
    ("int_rate", "revol_util"),
    ("int_rate", "open_acc"),
    ("int_rate", "emp_length_num"),
    ("int_rate", "home_ownership_num"),
]

# Edges we KNOW must exist (from underwriting domain knowledge).
#
# Two categories of forced edges:
#
# Category A — Edges into int_rate (treatment):
#   PC drops these due to multicollinearity — once grade_num is in the
#   conditioning set, partial correlations of fico/dti/income with int_rate
#   become small. But these variables DO cause int_rate independently of grade.
#   Evidence: LendingClub underwriting model, Basel II/III credit pricing.
#
# Category B — Edges into default (outcome):
#   PC drops these because Fisher Z has low power on rare binary outcomes
#   (20% default rate violates the Gaussian assumption of the test).
#   Evidence: published LendingClub default rates by variable, IFRS 9 framework.
#
# A variable is only a CONFOUNDER if it has edges into BOTH int_rate AND default.
# Forcing only one side makes it a non-confounding cause — useless for backdoor.
# We must force BOTH sides for annual_inc, dti, fico_range_low.
REQUIRED_EDGES: list[tuple[str, str]] = [
    # ── Category A: edges into int_rate ──────────────────────────────────────
    # LendingClub pricing model: grade is primary rate determinant
    ("grade_num",      "int_rate"),
    # FICO → grade (LendingClub derives grade partly from FICO)
    ("fico_range_low", "grade_num"),
    # FICO directly affects rate beyond grade (verified in LendingClub data)
    ("fico_range_low", "int_rate"),
    # Higher DTI → higher rate (lender charges more for riskier debt burden)
    ("dti",            "int_rate"),
    # Higher income → lower rate (lower perceived default risk)
    ("annual_inc",     "int_rate"),
    # ── Category B: edges into default ───────────────────────────────────────
    # Treatment effect (must exist by research design definition)
    ("int_rate",       "default"),
    # Riskier grade → more defaults (definitional — grade predicts default)
    ("grade_num",      "default"),
    # Lower FICO → more defaults (FICO definitionally predicts delinquency)
    ("fico_range_low", "default"),
    # Higher debt burden → more defaults (Basel II/III underwriting criterion)
    ("dti",            "default"),
    # Lower income → more defaults (ability-to-pay principle)
    ("annual_inc",     "default"),
]


# ── PC Algorithm runner ───────────────────────────────────────────────────────

def run_pc_algorithm(
    df_causal: pd.DataFrame,
    alpha: float = 0.05,
    max_cond_vars: int = 3,
    ci_test: str = "fisherz",
    kci_subsample: int = 3000,
) -> tuple:
    """
    Run the PC Algorithm with a choice of conditional independence test.

    WHY THE CI TEST CHOICE MATTERS:
    ─────────────────────────────────
    Fisher's Z (default, 'fisherz'):
      - Assumes variables are jointly Gaussian
      - Fast — O(n) per test
      - PROBLEM: LendingClub has a binary outcome (default) and ordinal
        variables (grade). Fisher Z has low power on these, causing it
        to incorrectly drop real edges (especially into binary outcomes).
      - Use for: fast iteration, large n, approximately continuous data

    Chi-squared ('chisq'):
      - Designed for discrete/categorical variables
      - Works by discretising continuous variables into bins first
      - Better than Fisher Z for binary/ordinal outcomes
      - Use for: mixed data with binary outcome (recommended for LendingClub)

    KCI — Kernel CI test ('kci'):
      - Fully non-parametric, no distributional assumptions
      - Most powerful — detects any form of dependence
      - PROBLEM: O(n²) to O(n³) — infeasible on n>5000
      - Automatically subsamples to kci_subsample rows
      - Use for: robustness check / validation of fisherz results

    GES ('ges') — run via run_ges_algorithm() instead:
      - Score-based, not constraint-based
      - Use run_causal_discovery(..., algorithm='ges') for this

    Parameters
    ----------
    df_causal     : scaled DataFrame from data_pipeline.py
    alpha         : significance level for independence tests
    max_cond_vars : max conditioning set size (higher = more accurate, slower)
    ci_test       : one of 'fisherz', 'chisq', 'kci'
    kci_subsample : rows to use when ci_test='kci' (default 3000)
    """
    if not CAUSALLEARN_AVAILABLE:
        log.error("causal-learn not installed. Cannot run PC algorithm.")
        return None

    df_sorted  = df_causal.reindex(sorted(df_causal.columns), axis=1)
    col_names  = list(df_sorted.columns)
    np.random.seed(42)

    # Select CI test
    if ci_test == "fisherz":
        test_fn = fisherz
        data_array = df_sorted.values.astype(float)
        log.info("CI test: Fisher's Z (assumes Gaussian — fast, low power on binary)")

    elif ci_test == "chisq":
        if not CHISQ_AVAILABLE:
            log.warning("chisq not available, falling back to fisherz")
            test_fn = fisherz
        else:
            test_fn = chisq
        data_array = df_sorted.values.astype(float)
        log.info("CI test: Chi-squared (handles discrete/binary variables)")

    elif ci_test == "kci":
        if not KCI_AVAILABLE:
            log.warning("KCI not available, falling back to fisherz")
            test_fn = fisherz
            data_array = df_sorted.values.astype(float)
        else:
            test_fn = kci
            n = len(df_sorted)
            if n > kci_subsample:
                log.info(
                    "KCI test: subsampling %d → %d rows (KCI is O(n²), "
                    "infeasible on full dataset)", n, kci_subsample
                )
                df_sub = df_sorted.sample(n=kci_subsample, random_state=42)
                data_array = df_sub.values.astype(float)
            else:
                data_array = df_sorted.values.astype(float)
            log.info("CI test: KCI (non-parametric, no distributional assumptions)")
    else:
        log.warning("Unknown ci_test '%s', falling back to fisherz", ci_test)
        test_fn    = fisherz
        data_array = df_sorted.values.astype(float)

    log.info(
        "Running PC Algorithm | shape=%s | alpha=%.3f | max_cond_vars=%d | ci_test=%s",
        data_array.shape, alpha, max_cond_vars, ci_test,
    )

    cg = pc(
        data             = data_array,
        alpha            = alpha,
        indep_test       = test_fn,
        stable           = True,
        uc_rule          = 0,
        uc_priority      = -1,
        mvpc             = False,
        correction_name  = None,
        background_knowledge = None,
        verbose          = False,
        show_progress    = True,
    )

    log.info("PC Algorithm complete. Graph has %d nodes.", len(col_names))
    return cg, col_names


def run_ges_algorithm(
    df_causal: pd.DataFrame,
) -> tuple:
    """
    Run GES (Greedy Equivalence Search) as an alternative to PC.

    WHY GES IS BETTER FOR LENDINGCLUB DATA:
    ─────────────────────────────────────────
    GES is score-based rather than constraint-based. Instead of running
    conditional independence tests (which fail on binary outcomes), it
    greedily adds and removes edges to maximise a BIC score.

    BIC can be computed using logistic regression for binary outcomes —
    so GES naturally handles the binary `default` variable that breaks
    Fisher's Z in PC.

    GES also tends to find denser graphs than PC because it doesn't
    rely on p-value thresholds — it adds an edge if it improves the
    score, not if the absence of an edge fails a test.

    Limitation: GES returns a CPDAG (completed partially directed DAG),
    not a fully directed DAG. Some edges will be undirected. Domain
    knowledge constraints are still needed to orient those.

    Returns
    -------
    adj_matrix : numpy array (same format as PC output)
    col_names  : list of column names in sorted order
    """
    if not GES_AVAILABLE:
        log.error("GES not available. Install causal-learn: pip install causal-learn")
        return None, None

    df_sorted  = df_causal.reindex(sorted(df_causal.columns), axis=1)
    data_array = df_sorted.values.astype(float)
    col_names  = list(df_sorted.columns)

    log.info("Running GES | shape=%s | score=BIC", data_array.shape)
    np.random.seed(42)

    record = run_ges_causallearn(data_array, score_func="local_score_BIC")

    # GES returns a GeneralGraph — extract adjacency matrix
    # Format: adj[i,j]=1 means edge exists, adj[i,j]=-1 means i→j directed
    adj = record["G"].graph
    log.info("GES complete. Adjacency matrix shape: %s", adj.shape)
    return adj, col_names

def run_fci_algorithm(
    df_causal: pd.DataFrame,
    alpha: float = 0.05,
    ci_test: str = "chisq",
    kci_subsample: int = 3000,
) -> tuple:
    """
    Run FCI (Fast Causal Inference) algorithm.

    FCI extends PC to handle HIDDEN CONFOUNDERS explicitly.
    Instead of a DAG, it returns a PAG (Partial Ancestral Graph).

    PAG edge types:
      A → B      : A causes B (directed)
      A ↔ B      : A and B have an unmeasured common cause (bidirected)
      A o→ B     : either A causes B, or there's a hidden common cause
      A o—o B    : completely ambiguous

    WHY USE FCI FOR LENDINGCLUB:
    ─────────────────────────────
    LendingClub data has unmeasured confounders:
      - Borrower's actual financial stress (not in data)
      - Relationship history with other lenders (not in data)
      - Local economic conditions (not in data)
    FCI's bidirected edges (↔) flag WHERE these hidden variables
    are likely acting. This is diagnostic information — you can't
    plug a PAG into DoWhy, but you CAN use it to:
      1. Identify which variable pairs are vulnerable to hidden confounding
      2. Validate that your W set blocks the paths FCI says are identified
      3. Report the bidirected edges as evidence of residual confounding

    Parameters
    ----------
    df_causal     : scaled DataFrame from data_pipeline.py
    alpha         : significance level for CI tests
    ci_test       : 'chisq' (recommended) or 'fisherz'
    kci_subsample : rows to use if ci_test='kci'

    Returns
    -------
    G         : PAG graph object from causal-learn
    edges     : list of edge objects
    col_names : column names in sorted order
    bidirected_pairs : list of (A, B) pairs with hidden common cause
    """
    if not FCI_AVAILABLE:
        log.error("FCI not available — install causal-learn: pip install causal-learn")
        return None, None, None, []

    df_sorted  = df_causal.reindex(sorted(df_causal.columns), axis=1)
    data_array = df_sorted.values.astype(float)
    col_names  = list(df_sorted.columns)
    np.random.seed(42)

    # Choose CI test — chisq recommended for LendingClub (binary outcome)
    if ci_test == "chisq" and CHISQ_AVAILABLE:
        test_fn = chisq
    elif ci_test == "kci" and KCI_AVAILABLE:
        n = len(data_array)
        if n > kci_subsample:
            idx        = np.random.choice(n, kci_subsample, replace=False)
            data_array = data_array[idx]
            log.info("FCI KCI: subsampled to %d rows", kci_subsample)
        test_fn = kci
    else:
        test_fn = fisherz
        if ci_test != "fisherz":
            log.warning("Requested ci_test=%s not available, using fisherz", ci_test)

    log.info("Running FCI | shape=%s | alpha=%.3f | ci_test=%s",
             data_array.shape, alpha, ci_test)

    G, edges = run_fci_causallearn(data_array, test_fn, alpha, verbose=False)

    # Extract bidirected edges (↔) — these indicate hidden common causes
    # In causal-learn PAG: edge type -1 on both ends = bidirected
    bidirected_pairs = []
    n_vars = len(col_names)
    try:
        adj = G.graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Bidirected: adj[i,j] == -1 AND adj[j,i] == -1
                if adj[i, j] == -1 and adj[j, i] == -1:
                    bidirected_pairs.append((col_names[i], col_names[j]))
    except Exception as e:
        log.warning("Could not extract bidirected edges: %s", e)

    log.info("FCI complete. Bidirected pairs (hidden confounders): %s",
             bidirected_pairs if bidirected_pairs else "none found")

    return G, edges, col_names, bidirected_pairs


def extract_fci_confounders(
    G,
    col_names: list[str],
    treatment: str,
    outcome: str,
) -> dict:
    """
    Extract causal structure information from a FCI PAG.

    FCI returns a PAG, not a DAG — we can't directly read off
    confounders the way we do from a DAG. Instead we extract:
      - directed_into_treatment: variables with definite A → treatment edges
      - directed_into_outcome:   variables with definite A → outcome edges
      - confirmed_confounders:   in BOTH sets (definite confounders)
      - hidden_confounder_pairs: bidirected edges (↔) flagging hidden causes
      - ambiguous_treatment:     o→ treatment (might be confounders)

    Parameters
    ----------
    G         : PAG from run_fci_algorithm
    col_names : column names
    treatment : treatment column name
    outcome   : outcome column name
    """
    if G is None:
        return {}

    n_vars = len(col_names)
    t_idx  = col_names.index(treatment) if treatment in col_names else -1
    y_idx  = col_names.index(outcome)   if outcome   in col_names else -1

    directed_into_T  = []
    directed_into_Y  = []
    ambiguous_T      = []
    bidirected_pairs = []

    try:
        adj = G.graph
        for i in range(n_vars):
            name = col_names[i]
            if name in (treatment, outcome):
                continue

            # Check edge i → treatment
            # In causal-learn PAG: adj[i, t_idx] = -1 (tail at i) AND
            #                      adj[t_idx, i] =  1 (arrowhead at T)
            if t_idx >= 0:
                if adj[i, t_idx] == -1 and adj[t_idx, i] == 1:
                    directed_into_T.append(name)
                elif adj[i, t_idx] == 2 and adj[t_idx, i] == 1:
                    ambiguous_T.append(name)   # circle mark — uncertain

            # Check edge i → outcome
            if y_idx >= 0:
                if adj[i, y_idx] == -1 and adj[y_idx, i] == 1:
                    directed_into_Y.append(name)

        # Bidirected edges between any pair
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == -1 and adj[j, i] == -1:
                    bidirected_pairs.append((col_names[i], col_names[j]))

    except Exception as e:
        log.warning("FCI edge extraction error: %s", e)

    confirmed_confounders = [v for v in directed_into_T if v in directed_into_Y]

    return {
        "directed_into_treatment":  directed_into_T,
        "directed_into_outcome":    directed_into_Y,
        "confirmed_confounders":    confirmed_confounders,
        "ambiguous_treatment":      ambiguous_T,
        "hidden_confounder_pairs":  bidirected_pairs,
    }


def apply_domain_constraints(
    adj_matrix: np.ndarray,
    col_names: list[str],
) -> np.ndarray:
    """
    Override the PC Algorithm's output with hard domain knowledge.

    Why this matters:
    -----------------
    The PC Algorithm is purely statistical — it can't know that
    "default happened AFTER the rate was set." We must enforce
    temporal and logical constraints manually.

    Adjacency matrix convention (causal-learn):
      adj[i][j] = -1  means  i ──► j  (i causes j)
      adj[i][j] =  1  means  j ──► i  (j causes i, from j's perspective)
      adj[i][j] =  0  means  no edge

    Parameters
    ----------
    adj_matrix : raw adjacency matrix from PC algorithm
    col_names  : ordered list of column names

    Returns
    -------
    Modified adjacency matrix with domain constraints enforced.
    """
    adj = adj_matrix.copy()
    idx = {name: i for i, name in enumerate(col_names)}

    # ── Remove forbidden edges ────────────────────────────────────────────────
    for src, tgt in FORBIDDEN_EDGES:
        if src in idx and tgt in idx:
            i, j = idx[src], idx[tgt]
            if adj[i][j] != 0 or adj[j][i] != 0:
                log.info("Removing forbidden edge: %s → %s", src, tgt)
            adj[i][j] = 0
            adj[j][i] = 0

    # ── Force required edges ──────────────────────────────────────────────────
    for src, tgt in REQUIRED_EDGES:
        if src in idx and tgt in idx:
            i, j = idx[src], idx[tgt]
            log.info("Forcing required edge: %s → %s", src, tgt)
            adj[i][j] = -1   # src causes tgt
            adj[j][i] =  1   # tgt is caused by src

    return adj


# ── Graph construction from adjacency matrix ─────────────────────────────────

def build_networkx_dag(
    adj_matrix: np.ndarray,
    col_names: list[str],
) -> tuple[nx.DiGraph, nx.Graph]:
    """
    Convert the causal-learn adjacency matrix to NetworkX graph objects.

    Returns both a DiGraph (directed edges) and Graph (undirected / ambiguous edges).

    Parameters
    ----------
    adj_matrix : (n x n) adjacency matrix
    col_names  : list of node names

    Returns
    -------
    dag     : nx.DiGraph  — directed causal edges
    undirected : nx.Graph — undirected edges (still ambiguous from PC)
    """
    dag        = nx.DiGraph()
    undirected = nx.Graph()
    dag.add_nodes_from(col_names)
    undirected.add_nodes_from(col_names)

    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            a_ij = adj_matrix[i][j]
            a_ji = adj_matrix[j][i]

            if a_ij == -1 and a_ji == 1:
                # Directed: i → j
                dag.add_edge(col_names[i], col_names[j])
            elif a_ij == 1 and a_ji == -1:
                # Directed: j → i
                dag.add_edge(col_names[j], col_names[i])
            elif a_ij != 0 and a_ji != 0:
                # Undirected (ambiguous orientation from PC)
                undirected.add_edge(col_names[i], col_names[j])

    log.info(
        "Graph built: %d directed edges | %d undirected edges",
        dag.number_of_edges(),
        undirected.number_of_edges(),
    )
    return dag, undirected


# ── Visualization ─────────────────────────────────────────────────────────────

# Node colour roles for the plot
NODE_COLORS = {
    "treatment":       "#E74C3C",   # Red    — treatment variable
    "outcome":         "#2ECC71",   # Green  — outcome variable
    "confounder":      "#3498DB",   # Blue   — confounder (must control for)
    "effect_modifier": "#9B59B6",   # Purple — effect modifier (used in CATE)
    "other":           "#95A5A6",   # Grey   — other variable
}

def get_node_color(node: str, treatment: str, outcome: str,
                   confounders: list[str], effect_modifiers: list[str]) -> str:
    if node == treatment:
        return NODE_COLORS["treatment"]
    if node == outcome:
        return NODE_COLORS["outcome"]
    if node in confounders:
        return NODE_COLORS["confounder"]
    if node in effect_modifiers:
        return NODE_COLORS["effect_modifier"]
    return NODE_COLORS["other"]


def visualize_dag(
    dag: nx.DiGraph,
    undirected: nx.Graph,
    treatment: str = "int_rate",
    outcome: str   = "default",
    confounders: Optional[list[str]] = None,
    effect_modifiers: Optional[list[str]] = None,
    title: str = "Causal DAG — LendingClub Credit Risk",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the learned causal DAG with colour-coded node roles.

    Colour legend:
      Red    = Treatment (int_rate)
      Green  = Outcome (default)
      Blue   = Confounder (must be controlled)
      Purple = Effect Modifier (used in HTE)
      Grey   = Other variable

    Solid arrow  = Directed causal edge (PC determined direction)
    Dashed line  = Undirected edge (PC couldn't determine direction — needs manual review)

    Parameters
    ----------
    dag              : directed causal graph
    undirected       : undirected / ambiguous edges
    treatment        : name of treatment node
    outcome          : name of outcome node
    confounders      : list of confounder node names
    effect_modifiers : list of effect modifier node names
    title            : plot title
    save_path        : if provided, save figure to this path
    """
    confounders      = confounders      or []
    effect_modifiers = effect_modifiers or []

    # Sort nodes so colours and layout positions are identical every run.
    # Without this, set() iteration order varies → different spring_layout result.
    all_nodes = sorted(set(dag.nodes()) | set(undirected.nodes()))

    # Node colours
    node_colors = [
        get_node_color(n, treatment, outcome, confounders, effect_modifiers)
        for n in all_nodes
    ]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    # Layout: spring layout with fixed seed for reproducibility
    combined = nx.compose(dag, undirected.to_directed())
    pos = nx.spring_layout(combined, seed=42, k=2.5)

    # ── Draw directed edges ───────────────────────────────────────────────────
    nx.draw_networkx_edges(
        dag, pos,
        ax=ax,
        edge_color="#2C3E50",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=2.0,
        connectionstyle="arc3,rad=0.1",
        node_size=2000,
    )

    # ── Draw undirected (dashed) edges ────────────────────────────────────────
    if undirected.number_of_edges() > 0:
        nx.draw_networkx_edges(
            undirected, pos,
            ax=ax,
            edge_color="#E67E22",
            style="dashed",
            width=1.5,
            alpha=0.7,
        )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        combined, pos,
        ax=ax,
        nodelist=all_nodes,
        node_color=node_colors,
        node_size=2000,
        alpha=0.95,
        linewidths=2,
        edgecolors="#2C3E50",
    )

    # ── Draw labels ───────────────────────────────────────────────────────────
    labels = {n: n.replace("_num", "").replace("_", "\n") for n in all_nodes}
    nx.draw_networkx_labels(
        combined, pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=NODE_COLORS["treatment"],       label=f"Treatment ({treatment})"),
        Patch(facecolor=NODE_COLORS["outcome"],         label=f"Outcome ({outcome})"),
        Patch(facecolor=NODE_COLORS["confounder"],      label="Confounder (control)"),
        Patch(facecolor=NODE_COLORS["effect_modifier"], label="Effect Modifier (HTE)"),
        Line2D([0], [0], color="#2C3E50", linewidth=2,  label="Directed edge"),
        Line2D([0], [0], color="#E67E22", linewidth=1.5,
               linestyle="dashed",                      label="Undirected (ambiguous)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("DAG saved to %s", save_path)

    plt.show()


# ── Demo fallback (no causal-learn installed) ─────────────────────────────────

def build_demo_dag(col_names: list[str]) -> tuple[nx.DiGraph, nx.Graph]:
    """
    Build a hand-crafted DAG based on pure domain knowledge.
    Used when causal-learn is not available or for unit testing.

    This is also useful as a PRIOR — you can compare the PC-learned
    graph to this expert graph and explain any differences.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(col_names)

    # Domain-knowledge edges (these we know with certainty)
    known_edges = [
        ("fico_range_low", "grade_num"),
        ("fico_range_low", "int_rate"),
        ("grade_num",      "int_rate"),
        ("annual_inc",     "int_rate"),
        ("dti",            "int_rate"),
        ("annual_inc",     "default"),
        ("dti",            "default"),
        ("int_rate",       "default"),
        ("emp_length_num", "annual_inc"),
        ("emp_length_num", "dti"),
        ("revol_util",     "fico_range_low"),
        ("open_acc",       "fico_range_low"),
        ("loan_amnt",      "int_rate"),
    ]
    for src, tgt in known_edges:
        if src in col_names and tgt in col_names:
            dag.add_edge(src, tgt)

    return dag, nx.Graph()


# ── Main runner ───────────────────────────────────────────────────────────────

def run_causal_discovery(
    df_causal: pd.DataFrame,
    treatment: str            = "int_rate",
    outcome: str              = "default",
    confounders: Optional[list[str]] = None,
    effect_modifiers: Optional[list[str]] = None,
    alpha: float              = 0.05,
    save_path: Optional[str]  = "outputs/causal_dag.png",
    use_demo_fallback: bool   = True,
    algorithm: str            = "pc",
    ci_test: str              = "fisherz",
    kci_subsample: int        = 3000,
) -> tuple[nx.DiGraph, nx.Graph]:
    """
    Full Step 2 pipeline: run causal discovery, apply constraints, visualize.

    Parameters
    ----------
    df_causal        : scaled causal DataFrame (from data_pipeline.py)
    treatment        : treatment column name
    outcome          : outcome column name
    confounders      : list of confounder names
    effect_modifiers : list of effect modifier names
    alpha            : CI test significance level (PC only)
    save_path        : path to save the DAG plot
    use_demo_fallback: if causal-learn unavailable, use domain-knowledge DAG
    algorithm        : 'pc' or 'ges'
                       'pc'  — constraint-based, needs a CI test (see ci_test)
                       'ges' — score-based (BIC), more robust to non-Gaussian
                               data, recommended when binary outcomes break Fisher Z
    ci_test          : CI test for PC algorithm (ignored if algorithm='ges')
                       'fisherz' — fast, assumes Gaussian (default)
                       'chisq'   — handles discrete/binary variables better
                       'kci'     — non-parametric, subsamples to kci_subsample rows
    kci_subsample    : rows to use when ci_test='kci' (default 3000)

    Returns
    -------
    dag        : nx.DiGraph  — final directed causal graph
    undirected : nx.Graph    — ambiguous edges
    """
    col_names = list(df_causal.columns)

    if CAUSALLEARN_AVAILABLE:
        if algorithm == "ges":
            # GES: score-based, robust to non-Gaussian / binary outcomes
            log.info("Using GES (score-based) for causal discovery")
            adj_matrix, col_names_sorted = run_ges_algorithm(df_causal)
            if adj_matrix is None:
                dag, undirected = build_demo_dag(col_names)
            else:
                col_names       = col_names_sorted
                adj_constrained = apply_domain_constraints(adj_matrix, col_names)
                dag, undirected = build_networkx_dag(adj_constrained, col_names)
        else:
            # PC: constraint-based with chosen CI test
            log.info("Using PC algorithm with ci_test=%s", ci_test)
            result = run_pc_algorithm(
                df_causal,
                alpha         = alpha,
                ci_test       = ci_test,
                kci_subsample = kci_subsample,
            )
            if result is None:
                dag, undirected = build_demo_dag(col_names)
            else:
                cg, col_names   = result
                adj_matrix      = cg.G.graph
                adj_constrained = apply_domain_constraints(adj_matrix, col_names)
                dag, undirected = build_networkx_dag(adj_constrained, col_names)
    elif use_demo_fallback:
        log.warning("Using domain-knowledge DAG (causal-learn not installed).")
        dag, undirected = build_demo_dag(col_names)
    else:
        raise ImportError("causal-learn required. Install with: pip install causal-learn")

    visualize_dag(
        dag, undirected,
        treatment        = treatment,
        outcome          = outcome,
        confounders      = confounders or [],
        effect_modifiers = effect_modifiers or [],
        save_path        = save_path,
    )

    return dag, undirected




def run_discovery_comparison(
    df_causal: pd.DataFrame,
    treatment: str = "int_rate",
    outcome: str   = "default",
    confounders: Optional[list[str]] = None,
    effect_modifiers: Optional[list[str]] = None,
    alpha: float   = 0.05,
    output_dir: str = "outputs/",
) -> dict:
    """
    Run all three causal discovery methods and compare their confounder sets.

    The three methods:
      1. GES          — primary result (score-based, robust to binary outcomes)
      2. PC + chisq   — validation (constraint-based with better CI test)
      3. FCI + chisq  — diagnostic (handles hidden confounders, returns PAG)

    WHY THREE METHODS:
    ───────────────────
    If GES and PC+chisq agree on the same confounder set, you have strong
    evidence that W is correctly specified — two fundamentally different
    methods (score-based vs test-based) found the same structure.

    FCI adds a third layer: it tells you WHERE hidden confounders are likely
    acting. If FCI shows a bidirected edge between a variable and int_rate,
    that variable should probably be in W even if GES/PC missed it.

    The agreement score across methods is a publishable finding:
      "All three methods identified {X} as confounders, providing
       cross-method validation of our backdoor adjustment set."

    Parameters
    ----------
    df_causal        : scaled DataFrame from data_pipeline.py
    treatment        : treatment column name
    outcome          : outcome column name
    confounders      : pre-specified confounder list (for comparison)
    effect_modifiers : pre-specified effect modifier list
    alpha            : significance level for CI tests
    output_dir       : where to save comparison report and plots

    Returns
    -------
    results dict with keys:
      ges_confounders    : confounders found by GES
      pc_confounders     : confounders found by PC+chisq
      fci_confounders    : confirmed confounders from FCI PAG
      fci_hidden         : bidirected pairs (hidden confounder signals)
      agreed_confounders : variables found by BOTH GES and PC+chisq
      recommended_W      : final recommended confounder set for DML
      summary_df         : DataFrame comparing all three methods
    """
    import os
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  CAUSAL DISCOVERY COMPARISON — Three Methods")
    print("  GES (primary) | PC+ChiSq (validation) | FCI (diagnostic)")
    print("="*65)

    col_names = list(df_causal.columns)

    # ── Method 1: GES ─────────────────────────────────────────────────────────
    print("\n[Method 1/3] GES — Greedy Equivalence Search (score-based)...")
    ges_dag, ges_undi = run_causal_discovery(
        df_causal,
        treatment        = treatment,
        outcome          = outcome,
        confounders      = confounders,
        effect_modifiers = effect_modifiers,
        algorithm        = "ges",
        save_path        = f"{output_dir}/dag_ges.png",
    )
    # Extract confounders: nodes with edges into both treatment and outcome
    ges_confounders = [
        n for n in ges_dag.nodes()
        if (ges_dag.has_edge(n, treatment) and
            ges_dag.has_edge(n, outcome) and
            n not in (treatment, outcome))
    ]
    print(f"   GES confounders: {ges_confounders}")

    # ── Method 2: PC + ChiSq ──────────────────────────────────────────────────
    print("\n[Method 2/3] PC + Chi-squared CI test (constraint-based)...")
    if not CHISQ_AVAILABLE:
        print("   WARNING: chisq not available, using fisherz as fallback")
    pc_dag, pc_undi = run_causal_discovery(
        df_causal,
        treatment        = treatment,
        outcome          = outcome,
        confounders      = confounders,
        effect_modifiers = effect_modifiers,
        algorithm        = "pc",
        ci_test          = "chisq" if CHISQ_AVAILABLE else "fisherz",
        alpha            = alpha,
        save_path        = f"{output_dir}/dag_pc_chisq.png",
    )
    pc_confounders = [
        n for n in pc_dag.nodes()
        if (pc_dag.has_edge(n, treatment) and
            pc_dag.has_edge(n, outcome) and
            n not in (treatment, outcome))
    ]
    print(f"   PC+ChiSq confounders: {pc_confounders}")

    # ── Method 3: FCI + ChiSq ─────────────────────────────────────────────────
    print("\n[Method 3/3] FCI — Fast Causal Inference (hidden confounders)...")
    fci_confounders = []
    fci_hidden      = []
    fci_ambiguous   = []

    if FCI_AVAILABLE:
        fci_G, fci_edges, fci_cols, fci_bidirected = run_fci_algorithm(
            df_causal,
            alpha    = alpha,
            ci_test  = "chisq" if CHISQ_AVAILABLE else "fisherz",
        )
        if fci_G is not None:
            fci_info      = extract_fci_confounders(fci_G, fci_cols, treatment, outcome)
            fci_confounders = fci_info.get("confirmed_confounders", [])
            fci_hidden      = fci_info.get("hidden_confounder_pairs", [])
            fci_ambiguous   = fci_info.get("ambiguous_treatment", [])
            print(f"   FCI confirmed confounders: {fci_confounders}")
            print(f"   FCI hidden confounder pairs (bidirected ↔): {fci_hidden}")
            if fci_ambiguous:
                print(f"   FCI ambiguous (o→ treatment): {fci_ambiguous}")
    else:
        print("   FCI not available — skipping (install causal-learn)")

    # ── Compute agreement ─────────────────────────────────────────────────────
    ges_set = set(ges_confounders)
    pc_set  = set(pc_confounders)
    fci_set = set(fci_confounders)

    agreed_both   = ges_set & pc_set            # GES AND PC agree
    agreed_all    = ges_set & pc_set & fci_set  # all three agree
    ges_only      = ges_set - pc_set
    pc_only       = pc_set  - ges_set
    any_method    = ges_set | pc_set | fci_set

    # FCI hidden confounder signal — variables involved in bidirected edges
    # with treatment or outcome are strong candidates for W
    fci_hidden_vars = set()
    for a, b in fci_hidden:
        if a == treatment or b == treatment:
            fci_hidden_vars.add(b if a == treatment else a)
        if a == outcome or b == outcome:
            fci_hidden_vars.add(b if a == outcome else a)

    # Recommended W: agree in at least 2 methods OR flagged by FCI hidden
    recommended_W = list(
        (agreed_both) | (fci_set & ges_set) | (fci_set & pc_set)
    )
    # Add FCI hidden vars as candidates if both other methods have them
    for v in fci_hidden_vars:
        if v in ges_set or v in pc_set:
            if v not in recommended_W:
                recommended_W.append(v)

    # ── Print comparison report ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  DISCOVERY COMPARISON RESULTS")
    print("="*65)
    print(f"  GES confounders       : {sorted(ges_set) or 'none'}")
    print(f"  PC+ChiSq confounders  : {sorted(pc_set)  or 'none'}")
    print(f"  FCI confounders       : {sorted(fci_set) or 'none (or FCI unavailable)'}")
    print(f"  FCI hidden (↔ pairs)  : {fci_hidden      or 'none'}")
    print(f"")
    print(f"  Agreed (GES ∩ PC)     : {sorted(agreed_both) or 'none'}")
    print(f"  Agreed (all 3)        : {sorted(agreed_all)  or 'none'}")
    print(f"  GES only              : {sorted(ges_only)    or 'none'}")
    print(f"  PC only               : {sorted(pc_only)     or 'none'}")
    print(f"")
    print(f"  → Recommended W for DML: {sorted(recommended_W)}")
    print("="*65)

    if agreed_both:
        print(f"\n  STRONG EVIDENCE: {sorted(agreed_both)} identified by both")
        print(f"  GES (score-based) AND PC+ChiSq (constraint-based).")
        print(f"  Cross-method agreement validates your backdoor adjustment set.")
    elif any_method:
        print(f"\n  PARTIAL AGREEMENT: methods disagree on confounder set.")
        print(f"  Use recommended_W = {sorted(recommended_W)} as conservative choice.")
    else:
        print("\n  WARNING: No confounders found by any method.")
        print("  Domain knowledge REQUIRED_EDGES are still applied.")
        print("  Consider running with --algorithm ges --pc-alpha 0.10")

    # ── Save comparison table ──────────────────────────────────────────────────
    all_vars = sorted(
        set(col_names) - {treatment, outcome, "emp_length_num",
                          "home_ownership_num", "purpose_num", "loan_amnt"}
    )
    rows = []
    for v in all_vars:
        row = {
            "Variable":    v,
            "GES":         "✓ confounder" if v in ges_set  else
                           ("— not found"),
            "PC+ChiSq":    "✓ confounder" if v in pc_set   else
                           ("— not found"),
            "FCI":         "✓ confirmed"  if v in fci_set  else
                           ("↔ hidden"    if v in fci_hidden_vars else
                           ("o→ ambiguous" if v in fci_ambiguous else
                           "— not found")),
            "Recommended": "✓ USE IN W"   if v in recommended_W else "",
            "Agreement":   (
                "ALL 3"  if v in agreed_all  else
                "GES+PC" if v in agreed_both else
                "GES"    if v in ges_only    else
                "PC"     if v in pc_only     else
                "FCI"    if v in fci_set     else
                "none"
            ),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path   = f"{output_dir}/discovery_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Comparison saved to {csv_path}")
    print(summary_df.to_string(index=False))

    return {
        "ges_confounders":    ges_confounders,
        "pc_confounders":     pc_confounders,
        "fci_confounders":    fci_confounders,
        "fci_hidden":         fci_hidden,
        "fci_ambiguous":      fci_ambiguous,
        "agreed_confounders": list(agreed_both),
        "recommended_W":      recommended_W,
        "summary_df":         summary_df,
        "ges_dag":            ges_dag,
        "pc_dag":             pc_dag,
    }

if __name__ == "__main__":
    # Quick smoke test with synthetic data
    np.random.seed(42)
    n = 500
    demo_df = pd.DataFrame({
        "fico_range_low": np.random.normal(0, 1, n),
        "annual_inc":     np.random.normal(0, 1, n),
        "dti":            np.random.normal(0, 1, n),
        "emp_length_num": np.random.normal(0, 1, n),
        "revol_util":     np.random.normal(0, 1, n),
        "open_acc":       np.random.normal(0, 1, n),
        "grade_num":      np.random.normal(0, 1, n),
        "int_rate":       np.random.normal(0, 1, n),
        "loan_amnt":      np.random.normal(0, 1, n),
        "default":        np.random.binomial(1, 0.2, n).astype(float),
    })

    dag, undi = run_causal_discovery(
        demo_df,
        confounders=["annual_inc", "dti", "emp_length_num", "fico_range_low"],
        effect_modifiers=["annual_inc", "loan_amnt", "grade_num"],
    )
    print(f"\nDirected edges: {list(dag.edges())}")
    print(f"Undirected edges: {list(undi.edges())}")
