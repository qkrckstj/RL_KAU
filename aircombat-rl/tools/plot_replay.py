"""Plot recorded replay-capacity comparison without importing the learner."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for capacity, color in ((50000,"#2864a0"),(500000,"#dc7428")):
        history=json.loads((out/f"buffer_{capacity}/validation.json").read_text(encoding="utf-8"))
        x=[h["step"] for h in history]
        label=f"Replay {capacity:,}"
        metrics=([h["summary"]["kills"] / h["summary"]["n"] for h in history],
                 [h["replay_diagnostics"]["positive_rewards"] for h in history],
                 [h["replay_diagnostics"]["q_max"] for h in history])
        for ax,y in zip(axes,metrics):
            ax.plot(x,y,marker=".",color=color,label=label)
    for ax,title in zip(axes,("Validation win rate", "Positive-reward transitions in replay", "Maximum sampled Q")):
        ax.set(title=title,xlabel="Total training steps")
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylim(-.02,1.02)
    axes[2].axhline(1,color="gray",linestyle="--",linewidth=1)
    fig.suptitle("Double DQN: same starting model and 50,000 experiences | one selected seed")
    fig.tight_layout()
    fig.savefig(out/"learning_curves.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    plot(p.parse_args().out.resolve())
