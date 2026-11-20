import matplotlib.pyplot as plt
import numpy as np
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

diff_all = err_reph - err_init
keep = (err_init <= 30) & (err_reph <= 30) & (diff_all.abs() <= 30)

diff = diff_all[keep]
ei = err_init[keep]
er = err_reph[keep]
print(f"Kept {keep.sum()} of {len(keep)} pairs ({(~keep).mean():.2%} dropped)")

print(f"MAE initial:   {ei.mean():.4f}")
print(f"MAE rephrased: {er.mean():.4f}")

# One-sided paired t-test
t, p_t = stats.ttest_rel(er, ei, alternative="less")
print(f"Paired t-test (one-sided): t = {t:.3f}, p = {p_t:.4g}")


# One-sided Wilcoxon signed-rank
w, p_w = stats.wilcoxon(er, ei, alternative="less", zero_method="pratt")
print(f"Wilcoxon (one-sided): p = {p_w:.4g}")
print(f"Median change: {diff.median():.3f}")
print(f"Exact zeros: {(diff == 0).sum()} of {len(diff)}")

"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

bins = 30
axes[0].hist(
    ei, bins=bins, alpha=0.6, label=f"Initial (mean {ei.mean():.2f})", color="#4C72B0"
)
axes[0].hist(
    er, bins=bins, alpha=0.6, label=f"Rephrased (mean {er.mean():.2f})", color="#DD8452"
)
axes[0].set_xlabel("Absolute error |perceived − actual|")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Distribution of absolute error (clipped, n = {keep.sum()})")
axes[0].legend()

axes[1].hist(diff, bins=bins, color="grey", edgecolor="white")
axes[1].axvline(0, color="k", ls="--", lw=1)
axes[1].axvline(diff.mean(), color="red", lw=2, label=f"Mean = {diff.mean():+.2f}")
axes[1].set_xlabel("Error after rephrasing − error before (negative = improved)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of change in error (clipped to ±30)")
axes[1].legend()

plt.tight_layout()
plt.savefig("error_distribution_clipped.png", dpi=150)
plt.show()
"""
