import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from scipy import stats

"""Null hypothesis (H₀): Rephrasing does not reduce the error. The mean absolute error after rephrasing is greater than or equal to the mean before.
Alternative hypothesis (H₁): Rephrasing reduces the error. The mean absolute error after rephrasing is lower than before."""

df = load_dataset("lopotaru/anthropomimetic-uncertainty", split="train").to_pandas()

init = df["initial_perceived_confidence"]
reph = df["rephrased_perceived_confidence"]
act = df["actual_confidence"]

err_init = (init - act).abs()
err_reph = (reph - act).abs()
diff = err_reph - err_init

print(f"MAE initial:   {err_init.mean():.4f}")
print(f"MAE rephrased: {err_reph.mean():.4f}")

t, p = stats.ttest_rel(err_reph, err_init, alternative="less")
print(f"Paired t-test: t = {t:.3f}, p = {p:.4g}")

# One-sided Wilcoxon signed-rank
w, p_w = stats.wilcoxon(err_reph, err_init, alternative="less", zero_method="pratt")
print(f"Wilcoxon (one-sided): p = {p_w:.4g}")
print(f"Median change: {diff.median():.3f}")
print(f"Exact zeros: {(diff == 0).sum()} of {len(diff)}")

same_conf = (
    df["initial_perceived_confidence"] == df["rephrased_perceived_confidence"]
).mean()
print(f"Share with identical perceived confidence: {same_conf:.1%}")
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(
    ["Initial", "Rephrased"],
    [err_init.mean(), err_reph.mean()],
    color=["#4C72B0", "#DD8452"],
)
axes[0].set_ylabel("Mean absolute error")
axes[0].set_title("MAE before vs after rephrasing")

axes[1].scatter(act, init, s=10, alpha=0.5, label="Initial")
axes[1].scatter(act, reph, s=10, alpha=0.5, label="Rephrased")
axes[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[1].set_xlabel("Actual confidence")
axes[1].set_ylabel("Perceived confidence")
axes[1].set_title("Perceived vs actual")
axes[1].legend()

plt.tight_layout()
plt.savefig("rephrasing_error_one_sided.png", dpi=150)
plt.show()

diff = err_reph - err_init

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Overlaid histograms of absolute error before and after rephrasing
bins = 30
axes[0].hist(
    err_init,
    bins=bins,
    alpha=0.6,
    label=f"Initial (mean {err_init.mean():.2f})",
    color="#4C72B0",
)
axes[0].hist(
    err_reph,
    bins=bins,
    alpha=0.6,
    label=f"Rephrased (mean {err_reph.mean():.2f})",
    color="#DD8452",
)
axes[0].set_xlabel("Absolute error |perceived − actual|")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of absolute error")
axes[0].legend()

# Distribution of the per-item change in error
axes[1].hist(diff, bins=bins, color="grey", edgecolor="white")
axes[1].axvline(0, color="k", ls="--", lw=1)
axes[1].axvline(diff.mean(), color="red", lw=2, label=f"Mean = {diff.mean():+.2f}")
axes[1].set_xlabel("Error after rephrasing − error before (negative = improved)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of change in error")
axes[1].legend()

plt.tight_layout()
plt.savefig("error_distributions.png", dpi=150)
plt.show()
"""
