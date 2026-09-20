"""
train_model.py
--------------
Model training & evaluation pipeline for the Pima Indians Diabetes Dataset.

Steps:
  1.  Import and run the full preprocessing pipeline from preprocess.py.
  2.  Train three classifiers:
        • Logistic Regression
        • Decision Tree
        • Random Forest
  3.  Evaluate each model on the held-out test set:
        Accuracy, Precision, Recall, F1-score  (macro-averaged)
  4.  Print a side-by-side comparison table.
  5.  Identify the best model by F1-score and save it to 'diabetes_model.pkl'
      using joblib.

Usage:
  Ensure preprocess.py is in the same directory, then run:
      python train_model.py
"""

import sys
import os
import joblib

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     classification_report)

# ── 0.  Make sure preprocess.py is importable ────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess import main as run_preprocessing
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Cannot import preprocess.py – make sure it lives in the "
        "same directory as train_model.py."
    ) from exc


# ── 1.  Configuration ─────────────────────────────────────────────────────────

MODEL_OUTPUT_PATH = "diabetes_model.pkl"
RANDOM_STATE      = 42

# Classifiers to benchmark  {display_name: estimator_instance}
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter     = 1000,
        random_state = RANDOM_STATE,
        solver       = "lbfgs",
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth    = 5,
        random_state = RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators = 200,
        max_depth    = 8,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    ),
}

# Metric used to pick the winner
SELECTION_METRIC = "F1-Score"


# ── 2.  Training ──────────────────────────────────────────────────────────────

def train_classifiers(classifiers: dict, X_train, y_train) -> dict:
    """Fit every classifier; return {name: fitted_model}."""
    fitted = {}
    print("\n[INFO] Training classifiers …")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        fitted[name] = clf
        print(f"  ✔  {name}")
    return fitted


# ── 3.  Evaluation ────────────────────────────────────────────────────────────

def evaluate_classifiers(fitted: dict, X_test, y_test) -> list[dict]:
    """
    Return a list of result dicts, one per classifier:
        {name, model, Accuracy, Precision, Recall, F1-Score}
    All metrics are macro-averaged to treat both classes equally.
    """
    results = []
    for name, clf in fitted.items():
        y_pred = clf.predict(X_test)
        results.append({
            "Model"    : name,
            "estimator": clf,
            "Accuracy" : accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro",
                                         zero_division=0),
            "Recall"   : recall_score(y_test, y_pred,    average="macro",
                                      zero_division=0),
            "F1-Score" : f1_score(y_test, y_pred,        average="macro",
                                  zero_division=0),
        })
    return results


# ── 4.  Comparison table ──────────────────────────────────────────────────────

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score"]

def print_comparison_table(results: list[dict], best_name: str) -> None:
    col_w   = 22
    met_w   = 12
    divider = "─" * (col_w + met_w * len(METRICS) + 3)

    header  = f"  {'Model':<{col_w}}" + "".join(f"{m:>{met_w}}" for m in METRICS)

    print("\n" + "=" * len(divider))
    print("  Model Comparison (macro-averaged metrics on test set)")
    print("=" * len(divider))
    print(header)
    print("  " + divider)

    for row in results:
        marker = " ★" if row["Model"] == best_name else "  "
        line   = f"{marker}{row['Model']:<{col_w}}"
        for m in METRICS:
            line += f"{row[m]:>{met_w}.4f}"
        print(line)

    print("  " + divider)
    print(f"\n  ★  Best model by {SELECTION_METRIC}: {best_name}\n")


# ── 5.  Per-class classification report for the best model ───────────────────

def print_best_model_report(fitted: dict, best_name: str,
                             X_test, y_test) -> None:
    print("─" * 55)
    print(f"  Detailed report – {best_name}")
    print("─" * 55)
    y_pred = fitted[best_name].predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=["No Diabetes (0)", "Diabetes (1)"],
    ))


# ── 6.  Save best model ───────────────────────────────────────────────────────

def save_best_model(fitted: dict, best_name: str, path: str) -> None:
    joblib.dump(fitted[best_name], path)
    print(f"[INFO] Best model ('{best_name}') saved to '{path}'.")


# ── 7.  Select best model ─────────────────────────────────────────────────────

def select_best(results: list[dict], metric: str = SELECTION_METRIC) -> str:
    return max(results, key=lambda r: r[metric])["Model"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pima Indians Diabetes – Model Training Pipeline")
    print("=" * 60)

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    artefacts = run_preprocessing()
    X_train   = artefacts["X_train"]
    X_test    = artefacts["X_test"]
    y_train   = artefacts["y_train"]
    y_test    = artefacts["y_test"]

    # ── Step 2: Train ────────────────────────────────────────────────────────
    fitted_models = train_classifiers(CLASSIFIERS, X_train, y_train)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────────
    results = evaluate_classifiers(fitted_models, X_test, y_test)

    # ── Step 4: Select best ──────────────────────────────────────────────────
    best_name = select_best(results)

    # ── Step 5: Print table ──────────────────────────────────────────────────
    print_comparison_table(results, best_name)

    # ── Step 6: Detailed report ──────────────────────────────────────────────
    print_best_model_report(fitted_models, best_name, X_test, y_test)

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    save_best_model(fitted_models, best_name, MODEL_OUTPUT_PATH)

    print("\n[INFO] Training pipeline complete.\n")

    return {
        "fitted_models": fitted_models,
        "results"      : results,
        "best_name"    : best_name,
    }


if __name__ == "__main__":
    main()
