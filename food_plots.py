import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

methods = ["MACH (P=6,L=6)", "Linear", "Linformer-128", "Performer-256", "FlashAttention2"]
train_s = [2165, 1166, 1250, 2546, 2600]
test_s  = [74,   44,   49,   105,  95]
acc_pct = [43.6, 41.4, 20.2, 42.4, 42.1]

x = np.arange(len(methods))

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# train time
axes[0].bar(x, train_s)
axes[0].set_ylabel("Seconds / epoch")
axes[0].set_title("Food-101 (16K) Train Time")

# test time
axes[1].bar(x, test_s)
axes[1].set_ylabel("Seconds / epoch")
axes[1].set_title("Food-101 (16K) Test Time")

# accuracy
axes[2].bar(x, acc_pct)
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Food-101 (16K) Accuracy")
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods, rotation=20, ha="right")

for ax in axes:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

fig.tight_layout()
fig.savefig(OUTDIR / "food101_train_test_acc_separate.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()