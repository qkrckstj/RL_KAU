"""Capture a fixed validation engagement, or plot completed stability artifacts.

Capture uses the training environment's Python. Plot uses any Python with
matplotlib/Pillow, so plotting need not modify the frozen training environment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def read(path):
    return Path(path).read_text(encoding="utf-8")


def capture(out):
    import torch
    from tools import policies
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    from aircombat_gym.wvr import obs as O
    torch.set_num_threads(1)
    design = out / "selected_policy"
    act, _, mode = policies.load(design, design / "policy_net.zip")
    records = []
    # Fixed before viewing any trajectories; never cherry-pick a winning match.
    for seat in ("red", "blue"):
        env = FairFightEnv(action_mode=mode, seat=seat)
        try:
            obs, info = env.reset(seed=900000)
            frames = []
            steps = 0
            while True:
                if steps % 10 == 0:
                    frames.append(dict(t=steps / 20,
                        **{name: float(obs[O.index(name)]) for name in (
                            "own_x", "own_y", "opp_x", "opp_y", "own_health", "opp_health")}))
                obs, _, term, trunc, info = env.step(act(obs))
                steps += 1
                if term or trunc:
                    frames.append(dict(t=steps / 20,
                        **{name: float(obs[O.index(name)]) for name in (
                            "own_x", "own_y", "opp_x", "opp_y", "own_health", "opp_health")}))
                    break
            records.append(dict(seed=900000, seat=seat, outcome=info.get("outcome"), frames=frames))
        finally:
            env.close()
    (out / "demonstrations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np
    summary = json.loads(read(out / "summary.json"))
    plan = json.loads(read(out / "plan.json"))
    conditions = list(plan["learning_rates"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for condition, color in zip(conditions, ("#2864a0", "#db722b")):
        histories = [json.loads(read(out / condition / f"s{s}/validation.json")) for s in plan["seeds"]]
        x = [h["step"] for h in histories[0]]
        values = np.array([[h["summary"]["kills"] / plan["val_n"] for h in hist] for hist in histories])
        for y in values:
            axes[0].plot(x, y, color=color, alpha=.22)
        label = f"lr={plan['learning_rates'][condition]}"
        axes[0].plot(x, values.mean(axis=0), color=color, label=label, linewidth=2)
        for ax, kind in zip(axes[1:], ("validation_selected", "budget_endpoint")):
            group = [g for g in summary["groups"] if g["condition"] == condition and g["kind"] == kind]
            ax.plot([g["budget"] for g in group], [g["mean_win_rate"] for g in group],
                marker="o", color=color, label=label)
    for ax, title in zip(axes, ("Validation (thin: individual seeds)", "Test: selected checkpoints", "Test: budget endpoints")):
        ax.set(title=title, xlabel="Training steps", ylabel="Win rate", ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("FairFight: learning rate only | 3 training seeds per condition")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    records = json.loads(read(out / "demonstrations.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    artists = []
    for ax, record in zip(axes, records):
        frames = record["frames"]
        x = np.array([[f["own_x"], f["opp_x"]] for f in frames]) / 1000
        y = np.array([[f["own_y"], f["opp_y"]] for f in frames]) / 1000
        extent = max(float(np.ptp(x)), float(np.ptp(y))) / 2 + 1
        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        ax.set(xlim=(cx-extent, cx+extent), ylim=(cy-extent, cy+extent),
            xlabel="East (km)", ylabel="North (km)", aspect="equal")
        ax.grid(alpha=.2)
        own, = ax.plot([], [], color="#2864a0", label="Policy", linewidth=1.5)
        opp, = ax.plot([], [], color="#ca4545", label="Opponent", linewidth=1.5)
        dots = ax.scatter(x[0], y[0], c=["#2864a0", "#ca4545"], s=35)
        title = ax.set_title("", fontsize=10)
        ax.legend(loc="lower right")
        artists.append((own, opp, dots, title, x, y, record))
    def update(i):
        for own, opp, dots, title, x, y, rec in artists:
            n = min(i + 1, len(x))
            own.set_data(x[:n, 0], y[:n, 0])
            opp.set_data(x[:n, 1], y[:n, 1])
            dots.set_offsets(np.column_stack((x[n-1], y[n-1])))
            frame = rec["frames"][n-1]
            ending = f" | {rec['outcome']}" if n == len(x) else ""
            title.set_text(f"Seat {rec['seat']}, t={frame['t']:.1f}s{ending}\nHealth {frame['own_health']:.2f} / {frame['opp_health']:.2f}")
    fig.suptitle("Fixed validation seed 900000, both seats | 5x playback", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .86))
    anim = FuncAnimation(fig, update, frames=max(len(r["frames"]) for r in records), interval=100)
    anim.save(out / "flight_replay.gif", writer=PillowWriter(fps=10), dpi=80)
    update(max(len(r["frames"]) for r in records)-1)
    fig.savefig(out / "flight_paths.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "plot"))
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    (capture if args.mode == "capture" else plot)(args.out.resolve())
