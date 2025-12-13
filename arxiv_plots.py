import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

methods = ["MACH (P=5,L=5)", "Linear", "Performer-256", "FlashAttention2"]
train_s = [999, 591, 952, 1645]
test_s  = [41.8, 22.8, 35.0, 47.0]
acc_pct = [97.40, 96.35, 96.61, 97.00]

x = np.arange(len(methods))

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# train time
axes[0].bar(x, train_s)
axes[0].set_ylabel("Seconds / epoch")
axes[0].set_title("ArXiv (64K) Train Time")

# test time
axes[1].bar(x, test_s)
axes[1].set_ylabel("Seconds / epoch")
axes[1].set_title("ArXiv (64K) Test Time")

# accuracy
axes[2].bar(x, acc_pct)
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("ArXiv (64K) Accuracy")
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods, rotation=20, ha="right")

for ax in axes:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

fig.tight_layout()
fig.savefig(OUTDIR / "arxiv_train_test_acc_separate.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "arxiv_train_test_acc_separate.png", dpi=300, bbox_inches="tight")
plt.close(fig)
