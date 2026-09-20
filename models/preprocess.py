"""
preprocess.py
-------------
Preprocessing pipeline for the Pima Indians Diabetes Dataset.

Steps:
  1. Load the dataset (CSV expected in the same directory, or auto-downloaded).
  2. Replace biologically invalid zeros with column medians for:
       Glucose, BloodPressure, SkinThickness, Insulin, BMI
  3. Split into feature matrix X and target vector y.
  4. Perform an 80/20 stratified train-test split.
  5. Standardise features with StandardScaler (fit on train, transform both).
  6. Print shapes of every resulting array.
  7. Save the fitted scaler to 'scaler.pkl' for later reuse.

Usage:
  Place 'diabetes.csv' in the working directory (or let the script fetch it),
  then run:
      python preprocess.py
"""

import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Configuration ──────────────────────────────────────────────────────────

DATASET_PATH = "diabetes.csv"
SCALER_PATH  = "scaler.pkl"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Columns whose zero values are physiologically impossible
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


# ── 2. Load dataset ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV from *path*.  If the file has no header row, inject column names."""
    if not os.path.exists(path):
        print(f"[INFO] '{path}' not found – downloading from GitHub …")
        urllib.request.urlretrieve(DATASET_URL, path)
        print(f"[INFO] Saved to '{path}'.")

    # Peek at the first line to detect a header
    with open(path) as fh:
        first_line = fh.readline()

    if first_line.startswith("Pregnancies"):          # has header
        df = pd.read_csv(path)
    else:                                              # raw numeric CSV
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    return df


# ── 3. Replace invalid zeros with column medians ──────────────────────────────

def replace_invalid_zeros(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    For each column in *cols*, treat 0 as a sentinel for a missing value,
    compute the median of the non-zero entries, and substitute.
    """
    df = df.copy()
    print("\n[INFO] Replacing invalid zeros with column medians:")
    print(f"  {'Column':<25} {'Zeros replaced':>15}  {'Median used':>12}")
    print("  " + "-" * 55)

    for col in cols:
        mask          = df[col] == 0
        n_zeros       = mask.sum()
        valid_median  = df.loc[~mask, col].median()
        df.loc[mask, col] = valid_median
        print(f"  {col:<25} {n_zeros:>15}  {valid_median:>12.4f}")

    return df


# ── 4. Split into X / y ───────────────────────────────────────────────────────

def split_features_target(df: pd.DataFrame, target: str = "Outcome"):
    X = df.drop(columns=[target]).values
    y = df[target].values
    return X, y


# ── 5. Train-test split ───────────────────────────────────────────────────────

def split_train_test(X, y):
    return train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,          # preserve class balance
    )


# ── 6. Standardise ────────────────────────────────────────────────────────────

def standardise(X_train, X_test):
    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ── 7. Save scaler ────────────────────────────────────────────────────────────

def save_scaler(scaler: StandardScaler, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"\n[INFO] Scaler saved to '{path}'.")


# ── 8. Print shapes ───────────────────────────────────────────────────────────

def print_shapes(X, y, X_train, X_test, y_train, y_test,
                 X_train_sc, X_test_sc) -> None:
    print("\n[INFO] Dataset shapes:")
    print(f"  {'Array':<30} {'Shape':>15}")
    print("  " + "-" * 47)
    rows = [
        ("Full feature matrix  (X)",        X.shape),
        ("Full target vector   (y)",         y.shape),
        ("Train features       (X_train)",   X_train.shape),
        ("Test  features       (X_test)",    X_test.shape),
        ("Train labels         (y_train)",   y_train.shape),
        ("Test  labels         (y_test)",    y_test.shape),
        ("Scaled train features",            X_train_sc.shape),
        ("Scaled test  features",            X_test_sc.shape),
    ]
    for name, shape in rows:
        print(f"  {name:<30} {str(shape):>15}")

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n[INFO] Class distribution in y_train:")
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / len(y_train)
        print(f"  Class {int(cls)}: {cnt} samples ({pct:.1f} %)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pima Indians Diabetes – Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_dataset(DATASET_PATH)
    print(f"\n[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # 2. Clean
    df_clean = replace_invalid_zeros(df, ZERO_INVALID_COLS)

    # 3. Split features / target
    X, y = split_features_target(df_clean)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 5. Standardise
    X_train_sc, X_test_sc, scaler = standardise(X_train, X_test)

    # 6. Print shapes
    print_shapes(X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)

    # 7. Save scaler
    save_scaler(scaler, SCALER_PATH)

    print("\n[INFO] Preprocessing complete.\n")

    # Return artefacts for downstream use
    return {
        "X_train": X_train_sc,
        "X_test":  X_test_sc,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
    }


if __name__ == "__main__":
    main()
