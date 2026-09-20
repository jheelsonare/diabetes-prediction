"""
Diabetes Prediction System — Exploratory Data Analysis
=======================================================
Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE   = {"positive": "#E05C5C", "negative": "#4A90D9", "neutral": "#6C7A89"}
CMAP_CORR = "coolwarm"
plt.rcParams.update({
    "figure.facecolor": "#F7F9FC",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#CCCCCC",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data(filepath: str = "diabetes.csv") -> pd.DataFrame:
    """Load dataset; replace biologically impossible zeros with NaN."""
    df = pd.read_csv(filepath)
    # Zero is physiologically impossible for these columns → treat as missing
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_invalid] = df[zero_invalid].replace(0, np.nan)
    print(f"✔ Loaded dataset  →  {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def summary_statistics(df: pd.DataFrame) -> None:
    sep = "─" * 70
    print(sep)
    print("  SUMMARY STATISTICS")
    print(sep)
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurt()
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.max_columns", 20):
        print(stats.to_string())
    print()

    print(sep)
    print("  CLASS BALANCE  (Outcome)")
    print(sep)
    vc = df[TARGET].value_counts()
    for label, count in vc.items():
        tag = "Diabetic    (1)" if label == 1 else "Non-Diabetic (0)"
        bar = "█" * int(count / df.shape[0] * 40)
        print(f"  {tag}: {count:4d}  {bar}  ({count/df.shape[0]*100:.1f}%)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def missing_value_analysis(df: pd.DataFrame) -> None:
    missing = (
        pd.DataFrame({
            "Missing Count":   df.isnull().sum(),
            "Missing %":       df.isnull().mean() * 100,
            "Data Type":       df.dtypes,
            "Unique Values":   df.nunique(),
        })
        .query("`Missing Count` > 0")
        .sort_values("Missing %", ascending=False)
    )

    print("─" * 70)
    print("  MISSING VALUE REPORT")
    print("─" * 70)
    if missing.empty:
        print("  ✔ No missing values detected.\n")
    else:
        print(missing.to_string())
        print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    miss_all = df.isnull().mean() * 100
    colors   = ["#E05C5C" if v > 0 else "#4A90D9" for v in miss_all]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Missing Value Analysis", fontsize=15, fontweight="bold", y=1.01)

    # Bar chart
    ax = axes[0]
    bars = ax.barh(miss_all.index, miss_all.values, color=colors, edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values per Feature")
    ax.axvline(0, color="#333", linewidth=0.8)
    for bar, val in zip(bars, miss_all.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(miss_all.values) * 1.25 + 1)

    # Heatmap of nulls
    ax2 = axes[1]
    null_matrix = df.isnull().astype(int)
    im = ax2.imshow(null_matrix.T, aspect="auto", cmap="RdYlGn_r",
                    interpolation="nearest", vmin=0, vmax=1)
    ax2.set_yticks(range(len(df.columns)))
    ax2.set_yticklabels(df.columns, fontsize=9)
    ax2.set_xlabel("Sample Index")
    ax2.set_title("Null Pattern Heatmap")
    plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04,
                 ticks=[0, 1]).ax.set_yticklabels(["Present", "Missing"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_missing_values.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: 01_missing_values.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASS DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
def class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    colors = [PALETTE["negative"], PALETTE["positive"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Class Distribution — Diabetes Outcome", fontsize=14, fontweight="bold")

    # Pie
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    axes[0].set_title("Proportion")

    # Bar
    axes[1].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sample Counts")
    axes[1].set_ylim(0, max(counts.values) * 1.15)
    axes[1].tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: 02_class_distribution.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
def feature_distributions(df: pd.DataFrame) -> None:
    n_cols, n_rows = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9))
    fig.suptitle("Feature Distributions  (Non-Diabetic vs Diabetic)",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, feat in zip(axes.flat, FEATURES):
        for outcome, color, label in [
            (0, PALETTE["negative"], "Non-Diabetic"),
            (1, PALETTE["positive"], "Diabetic"),
        ]:
            subset = df.loc[df[TARGET] == outcome, feat].dropna()
            ax.hist(subset, bins=25, alpha=0.55, color=color,
                    label=label, edgecolor="white", density=True)
            # KDE overlay
            if len(subset) > 5:
                from scipy.stats import gaussian_kde
                xs = np.linspace(subset.min(), subset.max(), 300)
                kde = gaussian_kde(subset)
                ax.plot(xs, kde(xs), color=color, linewidth=2)

        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: 03_feature_distributions.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. BOX PLOTS — OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Box Plots — Outlier Detection by Outcome",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, feat in zip(axes.flat, FEATURES):
        data = [
            df.loc[df[TARGET] == 0, feat].dropna(),
            df.loc[df[TARGET] == 1, feat].dropna(),
        ]
        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        medianprops={"color": "white", "linewidth": 2},
                        whiskerprops={"linewidth": 1.5},
                        capprops={"linewidth": 1.5})
        for patch, color in zip(bp["boxes"],
                                 [PALETTE["negative"], PALETTE["positive"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=9)
        ax.set_title(feat)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_boxplots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: 04_boxplots.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURES + [TARGET]].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Correlation Analysis", fontsize=15, fontweight="bold")

    # Full heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=CMAP_CORR,
        center=0, linewidths=0.5, linecolor="#EEEEEE",
        annot_kws={"size": 9}, ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Full Correlation Matrix (lower triangle)")
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=9)

    # Correlation with Outcome
    outcome_corr = corr[TARGET].drop(TARGET).sort_values()
    colors = [PALETTE["positive"] if v > 0 else PALETTE["negative"]
              for v in outcome_corr]
    axes[1].barh(outcome_corr.index, outcome_corr.values,
                 color=colors, edgecolor="white")
    axes[1].axvline(0, color="#333", linewidth=1)
    axes[1].set_xlabel("Pearson r  (with Outcome)")
    axes[1].set_title("Feature Correlation with Outcome")
    for i, (feat, val) in enumerate(outcome_corr.items()):
        axes[1].text(val + (0.005 if val >= 0 else -0.005), i,
                     f"{val:+.3f}", va="center",
                     ha="left" if val >= 0 else "right", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: 05_correlation_heatmap.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PAIRPLOT (top correlated features)
# ══════════════════════════════════════════════════════════════════════════════
def pairplot_top_features(df: pd.DataFrame, top_n: int = 5) -> None:
    corr = df[FEATURES + [TARGET]].corr()
    top_feats = (
        corr[TARGET].drop(TARGET).abs()
        .sort_values(ascending=False)
        .head(top_n).index.tolist()
    )
    print(f"  Top-{top_n} features correlated with Outcome: {top_feats}")

    subset = df[top_feats + [TARGET]].dropna()
    g = sns.pairplot(
        subset, hue=TARGET,
        palette={0: PALETTE["negative"], 1: PALETTE["positive"]},
        diag_kind="kde", plot_kws={"alpha": 0.45, "s": 20},
        diag_kws={"fill": True},
    )
    g.figure.suptitle(
        f"Pair Plot — Top {top_n} Features vs Outcome",
        y=1.01, fontsize=14, fontweight="bold",
    )
    g.figure.savefig(OUTPUT_DIR / "06_pairplot.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: 06_pairplot.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE STATISTICS TABLE (per class)
# ══════════════════════════════════════════════════════════════════════════════
def feature_stats_by_class(df: pd.DataFrame) -> None:
    print("─" * 70)
    print("  FEATURE STATISTICS BY CLASS")
    print("─" * 70)
    grouped = df.groupby(TARGET)[FEATURES].agg(["mean", "median", "std"])
    grouped.columns = [f"{feat} {stat}" for feat, stat in grouped.columns]
    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.max_columns", 30):
        print(grouped.T.to_string())
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 10. OUTLIER SUMMARY (IQR method)
# ══════════════════════════════════════════════════════════════════════════════
def outlier_summary(df: pd.DataFrame) -> None:
    print("─" * 70)
    print("  OUTLIER SUMMARY  (IQR method)")
    print("─" * 70)
    rows = []
    for feat in FEATURES:
        col = df[feat].dropna()
        Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((col < Q1 - 1.5 * IQR) | (col > Q3 + 1.5 * IQR)).sum()
        rows.append({
            "Feature": feat, "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "Lower Fence": Q1 - 1.5 * IQR,
            "Upper Fence": Q3 + 1.5 * IQR,
            "Outlier Count": n_out,
            "Outlier %": f"{n_out / len(col) * 100:.1f}%",
        })
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(pd.DataFrame(rows).set_index("Feature").to_string())
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_eda(filepath: str = "diabetes.csv") -> pd.DataFrame:
    """Run the complete EDA pipeline and return the processed DataFrame."""
    df = load_data(filepath)

    # ── Console reports ───────────────────────────────────────────────────────
    summary_statistics(df)
    feature_stats_by_class(df)
    outlier_summary(df)

    # ── Visual reports ────────────────────────────────────────────────────────
    missing_value_analysis(df)
    class_distribution(df)
    feature_distributions(df)
    boxplots(df)
    correlation_heatmap(df)
    pairplot_top_features(df)

    print("=" * 70)
    print(f"  EDA complete. All plots saved to → {OUTPUT_DIR.resolve()}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    # ── Change this path to your actual CSV file location ─────────────────────
    df = run_eda("diabetes.csv")
