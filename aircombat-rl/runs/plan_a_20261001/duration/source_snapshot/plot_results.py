"""Render experiment JSON with the existing system matplotlib installation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
colors = ("#1769aa", "#df721b", "#18846b")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
for seed, color in enumerate(colors):
    history = json.loads((root / f"s{seed}/validation.json").read_text(encoding="utf-8"))
    axes[0].plot([r["step"] / 1e6 for r in history],
                 [100 * r["summary"]["kills"] / 20 for r in history],
                 color=color, marker="o", markersize=3, label=f"Seed {seed}")
axes[0].set(title="Validation during training\n20 matches per checkpoint",
            xlabel="Environment steps (millions)", ylabel="Win rate (%)")
axes[0].legend(loc="upper right", frameon=False)
for axis, kind, title in zip(axes[1:], ("validation_selected", "budget_endpoint"),
                             ("Held-out: validation-selected model", "Held-out: model at budget endpoint")):
    rows = [r for r in summary["groups"] if r["kind"] == kind]
    for seed, color in enumerate(colors):
        axis.plot([i + (seed - 1) * 0.035 for i in range(3)],
                  [r["wins_by_seed"][seed] * 100 / 40 for r in rows],
                  color=color, marker="o", markersize=5, linewidth=1, label=f"Seed {seed}")
    axis.plot(range(3), [100 * r["mean_win_rate"] for r in rows],
              color="#222222", marker="D", markersize=5, linestyle="--", label="3-seed mean")
    axis.set(title=title + "\n40 matches per model, same test conditions",
             xlabel="Training budget (steps)", xticks=range(3),
             xticklabels=("204,800", "512,000", "1,024,000"))
    axis.legend(loc="upper right", frameon=False)
for axis in axes:
    axis.set_ylim(-1.5, 40)
    axis.set_yticks(range(0, 41, 10))
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("FairFight DQN: longer training with observations, rewards and exploration held fixed", fontsize=14)
fig.savefig(root / "learning_curves.png", dpi=160)
fig.savefig(root / "learning_curves.svg")
plt.close(fig)
