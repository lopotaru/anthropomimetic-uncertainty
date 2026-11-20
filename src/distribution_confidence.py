import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats

df = load_dataset("lopotaru/anthropomimetic-uncertainty", split="train").to_pandas()

init = df["initial_perceived_confidence"]
reph = df["rephrased_perceived_confidence"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

bins = 30
axes[0].hist(
    init,
    bins=bins,
    alpha=0.6,
    label=f"Initial (mean {init.mean():.2f})",
    color="#4C72B0",
)

axes[0].set_xlabel("Perceived Confidence")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of initial perceived confidence")
axes[0].legend()

axes[1].hist(
    reph, bins=bins, label=f"Rephrased (mean {reph.mean():.2f})", color="green"
)
axes[1].set_xlabel("Perceived Confidence")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of rephrased perceived confidence")
axes[1].legend()

plt.tight_layout()
plt.savefig("perceived_confidence_distributions.png", dpi=150)
plt.show()
