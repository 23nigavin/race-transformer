import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

ctx = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

gpt       = [0.451760, 0.178421, 0.233416, 0.469251, 0.904438, 2.873320, 12.738708, 55.568188]
gpt_race  = [0.183179, 0.162963, 0.200218, 0.585275, 0.625789, 1.267680,  2.060607,  5.028605]
llama     = [0.197150, 0.190494, 0.192757, 0.553539, 0.841397, 3.017533, 12.672846, 51.505827]
llama_race= [0.195418, 0.157178, 0.244296, 0.401589, 0.657038, 1.384594,  3.063767,  5.759868]

fig, ax = plt.subplots(figsize=(8.5, 5))

ax.plot(ctx, gpt,        marker="o", label="GPT")
ax.plot(ctx, gpt_race,   marker="o", label="GPT RACE")
ax.plot(ctx, llama,      marker="o", label="LLaMA")
ax.plot(ctx, llama_race, marker="o", label="LLaMA RACE")

ax.set_xscale("log", base=2)
ax.set_xticks(ctx)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel("Context length")
ax.set_ylabel("Prefill + TTFT (seconds)")
ax.set_title("TTFT vs Context Length (CPU)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUTDIR / "ttft_vs_ctx.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "ttft_vs_ctx.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
