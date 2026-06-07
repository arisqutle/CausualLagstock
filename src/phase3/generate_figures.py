"""Generate Nature/ICLR-style EDA figures for the CausalStock framework docs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "docs" / "figures"
DATA_PATH = PROJECT_ROOT / "data" / "stage3" / "phase3_dataset_ticker_precomputed.pt"

# ---- Style ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
sns.set_theme(style="ticks", context="paper")

PALETTE_3 = {"UP": "#0072B2", "DOWN": "#D55E00", "FLAT": "#009E73"}
PALETTE_2 = ["#0072B2", "#D55E00"]
LABEL_NAMES = {0: "UP", 1: "DOWN", 2: "FLAT"}

# ---- Load data ----
print(f"Loading {DATA_PATH}...")
data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
y = data["direction_labels"].cpu().numpy()
r = data["magnitude_labels"].cpu().numpy()
ts = pd.to_datetime(data["prediction_timestamps"].cpu().numpy(), unit="s", utc=True).tz_convert(None)
tickers = data["tickers"]

df = pd.DataFrame({
    "date": ts,
    "year": ts.year,
    "label_id": y,
    "label": [LABEL_NAMES[int(v)] for v in y],
    "ret": r,
    "abs_ret": np.abs(r),
    "ticker": tickers,
})
n = len(df)
print(f"N = {n}, {df['ticker'].nunique()} tickers, {df['year'].min()}–{df['year'].max()}")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helper ----
def save(name: str):
    for fmt in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{fmt}"
        plt.savefig(path)
        print(f"  saved {path}")


# ============================================================
# Figure 1: Class distribution
# ============================================================
print("\n[1/8] Class distribution")
counts = df["label"].value_counts()
fig, ax = plt.subplots(figsize=(3.2, 2.4))
bars = ax.bar(
    ["UP", "DOWN", "FLAT"],
    [counts.get("UP", 0), counts.get("DOWN", 0), counts.get("FLAT", 0)],
    color=[PALETTE_3["UP"], PALETTE_3["DOWN"], PALETTE_3["FLAT"]],
    width=0.55,
    edgecolor="white",
    linewidth=0.5,
)
for bar, count in zip(bars, [counts.get("UP", 0), counts.get("DOWN", 0), counts.get("FLAT", 0)]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
            f"{count}\n({count/n:.1%})", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Examples")
ax.set_title("Direction label distribution")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
plt.tight_layout()
save("fig1_class_distribution")

# ============================================================
# Figure 2: Signed return distribution with ±0.5% threshold
# ============================================================
print("[2/8] Return distribution")
THRESHOLD = 0.005
fig, ax = plt.subplots(figsize=(4.8, 2.6))
sns.histplot(
    data=df, x="ret", bins=100, stat="density",
    color="#333333", fill=True, alpha=0.15, edgecolor="#555555", linewidth=0.3,
    ax=ax,
)
sns.kdeplot(data=df, x="ret", color="#0072B2", linewidth=1.2, ax=ax)
ax.axvline(-THRESHOLD, color="#D55E00", linestyle="--", linewidth=0.9)
ax.axvline(THRESHOLD, color="#D55E00", linestyle="--", linewidth=0.9)
ax.text(-THRESHOLD - 0.002, ax.get_ylim()[1] * 0.92, r"$-\delta$", color="#D55E00", fontsize=8, ha="right")
ax.text(THRESHOLD + 0.002, ax.get_ylim()[1] * 0.92, r"$+\delta$", color="#D55E00", fontsize=8, ha="left")
ax.set_xlim(-0.08, 0.08)
ax.set_xlabel("Next-day close-to-close return $r$")
ax.set_ylabel("Density")
ax.set_title(f"Signed return distribution ($\\delta = {THRESHOLD}$)")
# Shade threshold regions
ylim = ax.get_ylim()
ax.fill_betweenx(ylim, -0.08, -THRESHOLD, color="#D55E00", alpha=0.06)
ax.fill_betweenx(ylim, THRESHOLD, 0.08, color="#0072B2", alpha=0.06)
ax.fill_betweenx(ylim, -THRESHOLD, THRESHOLD, color="#009E73", alpha=0.06)
plt.tight_layout()
save("fig2_return_distribution")

# ============================================================
# Figure 3: Temporal class mix by year
# ============================================================
print("[3/8] Temporal class mix")
year_counts = df.groupby(["year", "label"]).size().rename("n").reset_index()
year_totals = year_counts.groupby("year")["n"].transform("sum")
year_counts["prop"] = year_counts["n"] / year_totals

fig, ax = plt.subplots(figsize=(5.4, 2.4))
for label, color in PALETTE_3.items():
    sub = year_counts[year_counts["label"] == label]
    ax.plot(sub["year"], sub["prop"], color=color, marker="o", markersize=2.5,
            linewidth=1.0, label=label)
ax.set_xlabel("Year")
ax.set_ylabel("Class proportion")
ax.set_title("Direction-label mix by year")
ax.legend(frameon=False, loc="upper left", ncol=3)
ax.set_ylim(0, 0.65)
plt.tight_layout()
save("fig3_temporal_class_mix")

# ============================================================
# Figure 4: Volatility regime drift
# ============================================================
print("[4/8] Volatility drift")
year_vol = df.groupby("year").agg(
    n=("ret", "size"), ret_std=("ret", "std"), abs_ret_median=("abs_ret", "median"),
).reset_index()

fig, ax = plt.subplots(figsize=(5.4, 2.2))
ax.bar(year_vol["year"], year_vol["ret_std"], color="#555555", width=0.6, alpha=0.7)
ax.set_xlabel("Year")
ax.set_ylabel("std($r$)")
ax.set_title("Return volatility by year")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
save("fig4_volatility_drift")

# ============================================================
# Figure 5: Ticker concentration (top 12)
# ============================================================
print("[5/8] Ticker concentration")
ticker_counts = df["ticker"].value_counts().head(12)

fig, ax = plt.subplots(figsize=(5.4, 2.4))
sns.barplot(x=ticker_counts.index, y=ticker_counts.values, color="#0072B2", ax=ax)
ax.set_xlabel("Target ticker")
ax.set_ylabel("Examples")
ax.set_title("Top 12 tickers by example count")
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
plt.tight_layout()
save("fig5_ticker_concentration")

# ============================================================
# Figure 6: HGB feature importance (top 20)
# ============================================================
print("[6/8] Feature importance")
from sklearn.ensemble import HistGradientBoostingClassifier

# Build the same features as tabular_baseline
price = data["price_history"].float().numpy()
event_types = data["event_types"].long().numpy()
magnitudes = data["magnitudes"].float().numpy()
stock_ids = data["target_stock_ids"].long().numpy()
n_et = int(event_types.max()) + 1
n_st = int(stock_ids.max()) + 1

close = price[:, :, 3]
returns = np.diff(close, axis=1) / np.maximum(np.abs(close[:, :-1]), 1e-8)
volume = np.log1p(np.maximum(price[:, :, 4], 0.0))
base_close = np.maximum(np.abs(price[:, 0:1, 3:4]), 1e-8)
relative_ohlc = price[:, :, :4] / base_close - 1.0

price_feat = np.column_stack([
    returns[:, -10:], returns.mean(axis=1), returns.std(axis=1),
    returns[:, -5:].mean(axis=1), returns[:, -5:].std(axis=1),
    volume[:, -10:].mean(axis=1), volume[:, -10:].std(axis=1),
    relative_ohlc[:, -1, :],
    relative_ohlc.mean(axis=1).reshape(n, -1),
    relative_ohlc.std(axis=1).reshape(n, -1),
])
counts = np.zeros((n, n_et), dtype=np.float32)
mag_sum = np.zeros((n, n_et), dtype=np.float32)
mag_abs = np.zeros((n, n_et), dtype=np.float32)
last_type = np.zeros((n, n_et), dtype=np.float32)
for i in range(n):
    for et, mag in zip(event_types[i], magnitudes[i]):
        counts[i, int(et)] += 1.0
        mag_sum[i, int(et)] += float(mag)
        mag_abs[i, int(et)] += abs(float(mag))
    last_type[i, int(event_types[i, -1])] = 1.0

event_feat = np.concatenate([
    counts, mag_sum / event_types.shape[1], mag_abs / event_types.shape[1],
    last_type, magnitudes.mean(axis=1, keepdims=True),
    np.abs(magnitudes).mean(axis=1, keepdims=True),
], axis=1)
stock_feat = np.eye(n_st, dtype=np.float32)[stock_ids]
X = np.concatenate([price_feat, event_feat, stock_feat], axis=1).astype(np.float32)

# Feature names
fnames = (
    [f"ret_{i}" for i in range(1, 11)]
    + ["ret_mean", "ret_std", "ret_5d_mean", "ret_5d_std",
       "logvol_10d_mean", "logvol_10d_std"]
    + [f"rel_{ch}_last" for ch in ["O", "H", "L", "C"]]
    + [f"rel_{ch}_mean" for ch in ["O", "H", "L", "C"]]
    + [f"rel_{ch}_std" for ch in ["O", "H", "L", "C"]]
    + [f"count_{i}" for i in range(n_et)]
    + [f"magsum_{i}" for i in range(n_et)]
    + [f"absmag_{i}" for i in range(n_et)]
    + [f"last_type_{i}" for i in range(n_et)]
    + ["global_mag_mean", "global_abs_mag_mean"]
    + [f"stock_{i}" for i in range(n_st)]
)

from sklearn.inspection import permutation_importance

model = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.04, l2_regularization=0.05, random_state=42,
)
model.fit(X, y)
rng = np.random.RandomState(42)
pi = permutation_importance(model, X, y, n_repeats=3, random_state=rng, n_jobs=1)
importances = pi.importances_mean
top_idx = np.argsort(importances)[-20:][::-1]

fig, ax = plt.subplots(figsize=(4.2, 4.0))
colors = []
for name in [fnames[i] for i in top_idx]:
    if name.startswith("ret_") or name.startswith("rel_") or name.startswith("logvol_"):
        colors.append("#0072B2")
    elif name.startswith("stock_"):
        colors.append("#009E73")
    else:
        colors.append("#D55E00")
ax.barh(range(19, -1, -1), importances[top_idx], color=colors[::-1], height=0.65)
ax.set_yticks(range(19, -1, -1))
ax.set_yticklabels([fnames[i] for i in top_idx][::-1], fontsize=6.5, family="monospace")
ax.set_xlabel("Permutation importance")
ax.set_title("Top 20 features (HGB, stratified)")
ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Price"),
        plt.Rectangle((0, 0), 1, 1, color="#D55E00", label="News"),
        plt.Rectangle((0, 0), 1, 1, color="#009E73", label="Stock ID"),
    ],
    frameon=False, fontsize=6.5, loc="lower right",
)
plt.tight_layout()
save("fig6_feature_importance")

# ============================================================
# Figure 7: Confusion matrix
# ============================================================
print("[7/8] Confusion matrix")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y_test, y_pred):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(3.0, 2.8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap=sns.cubehelix_palette(as_cmap=True, rot=-0.3),
    xticklabels=["UP", "DOWN", "FLAT"], yticklabels=["UP", "DOWN", "FLAT"],
    linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax,
    annot_kws={"fontsize": 8},
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("HGB test confusion matrix")
plt.tight_layout()
save("fig7_confusion_matrix")

# ============================================================
# Figure 8: Stratified vs Chronological performance
# ============================================================
print("[8/8] Stratified vs chronological performance")
metrics_data = {
    "Model": ["HGB", "HGB", "Price-only HGB", "Price-only HGB"],
    "Split": ["Stratified", "Chronological", "Stratified", "Chronological"],
    "Macro-F1": [0.6929, 0.3366, 0.705, 0.338],
}

fig, ax = plt.subplots(figsize=(4.0, 2.8))
x = np.arange(2)
width = 0.30
bars1 = ax.bar(x - width/2, [0.6929, 0.705], width, color="#333333", label="Tabular (news+price)")
bars2 = ax.bar(x + width/2, [0.3366, 0.338], width, color="#999999", label="Tabular (news+price)")
ax.set_xticks(x)
ax.set_xticklabels(["Stratified random", "Chronological"])
ax.set_ylabel("Macro-F1")
ax.set_title("Performance by split type")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False, fontsize=7)
# Add value labels
for bar, val in zip(bars1, [0.6929, 0.705]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=7)
for bar, val in zip(bars2, [0.3366, 0.338]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=7)
# Random baseline
ax.axhline(0.333, color="#D55E00", linestyle=":", linewidth=0.8)
ax.text(1.3, 0.345, "random\nbaseline", color="#D55E00", fontsize=6.5, ha="center")
plt.tight_layout()
save("fig8_split_comparison")

print(f"\nDone. All figures saved to {FIG_DIR}")
