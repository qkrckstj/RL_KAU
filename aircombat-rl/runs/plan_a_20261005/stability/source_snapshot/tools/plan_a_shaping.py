"""A3: paired fresh DQN runs, fixed reward comparison, then one held-out test.

run: six full runs (terminal/potential_v1 x seeds 0/1/2).
smoke: two 2048-step runs; evaluation only on the already-used validation band.
Every run preserves its protocol, source snapshot, checkpoints, replay and RNG.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from tools.plan_a import ROOT, timestamp, worker, write_json
from tools.plan_a_long import select_at_budget

REWARDS = ("terminal", "potential_v1")
SOURCE_PATHS = (
    "experiments/plan_a/core.py", "tools/plan_a.py", "tools/plan_a_long.py",
    "tools/plan_a_shaping.py", "tools/grade.py", "tools/policies.py",
    "tools/resume_dqn.py", "tests/test_plan_a_shaping.py",
    "experiments/plan_a/shaping.md", "requirements-portable.txt",
)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protocol(smoke=False):
    paths = list(SOURCE_PATHS) + [
        p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "aircombat_gym").rglob("*.py"))]
    return dict(
        experiment="A3 reward only", smoke=smoke, task="fair", variant="observed_dqn",
        rewards=list(REWARDS), seeds=[0] if smoke else [0, 1, 2],
        budgets=[1024, 2048] if smoke else [204800, 512000, 1024000],
        epsilon_decay_steps=1024 if smoke else 20480,
        val_every=1024 if smoke else 51200, val_n=2 if smoke else 20, val_band=900000,
        test_band=900000 if smoke else 1300000, test_n=2 if smoke else 40,
        cpu_threads_per_worker=1,
        potential="0.25*(1-range_feature)*(own_cos-opp_cos)/2"
                  "+0.125*(own_health-opp_health+own_track-opp_track); bound [-0.5,0.5]",
        shaping="r + gamma*Phi(next)-Phi(current); gamma=0.999; actual terminal Phi=0",
        controls="Fresh terminal controls on this PC; no reused learner or replay.",
        selection="Within budget: validation wins, mean win time, earlier step. "
                  "Condition: sum of selected validation wins, pooled win time, terminal tie. "
                  "Representative: validation wins, win time, earlier step, lower seed.",
        next_step="If selected test mean improves >=0.10, >=2/3 paired seeds improve, "
                  "and endpoint test mean does not decrease: independent-seed confirmation. "
                  "Otherwise diagnose validation and training failures; no test-driven coefficient tuning.",
        source_sha256={name: sha(ROOT / name) for name in paths},
    )

def check_protocol(out):
    plan = read(out / "plan.json")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source changed since preregistration: {name}; use a new output folder.")
    return plan

def prepare(out, smoke):
    plan = protocol(smoke)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Existing protocol differs; use a new output folder.")
        return plan
    write_json(path, plan)
    write_json(out / "environment.json", dict(
        created_utc=timestamp(), python=sys.version, platform=platform.platform(),
        processor=platform.processor(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        packages={d.metadata["Name"]: d.version for d in distributions()},
    ))
    for name in plan["source_sha256"]:
        dest = out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    (out / "source_snapshot/git_diff.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    return plan

def train_worker(args):
    plan = check_protocol(Path(args.out))
    args.out = str(Path(args.out) / args.reward / f"s{args.seed}")
    args.variant, args.task = plan["variant"], plan["task"]
    args.steps, args.val_every, args.val_n = plan["budgets"][-1], plan["val_every"], plan["val_n"]
    args.test_band, args.epsilon_decay_steps = plan["test_band"], plan["epsilon_decay_steps"]
    args.save_checkpoints, args.resume_milestones = True, plan["budgets"]
    worker(args)

def child(mode, out, reward, seed):
    dest = out / reward / f"s{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    log = dest / ("train.log" if mode == "worker" else "test.log")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "tools.plan_a_shaping", mode,
             "--out", str(out), "--reward", reward, "--seed", str(seed)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"{mode} {reward}/s{seed} failed; see {log}")

def rank(choice):
    v = choice["validation"]
    return (v["kills"], -(v["t_kill"] if v["t_kill"] is not None else 1e9),
            -choice["step"], -choice["seed"])

def freeze(out, plan):
    choices = []
    for reward in REWARDS:
        for seed in plan["seeds"]:
            dest = out / reward / f"s{seed}"
            result = read(dest / "result.json")
            if result["status"] != "complete" or result["steps"] != plan["budgets"][-1]:
                raise ValueError("All planned training must finish before test selection.")
            history = read(dest / "validation.json")
            for budget in plan["budgets"]:
                selected = select_at_budget(history, budget)
                endpoint = next(row for row in history if row["step"] == budget)
                for kind, row in (("validation_selected", selected), ("budget_endpoint", endpoint)):
                    weights = dest / f"checkpoints/step_{row['step']}/policy_net.zip"
                    choices.append(dict(reward=reward, seed=seed, budget=budget, kind=kind,
                                        step=row["step"], validation=row["summary"],
                                        weights=weights.relative_to(out).as_posix(),
                                        weights_sha256=sha(weights)))
    finalists = [c for c in choices if c["budget"] == plan["budgets"][-1]
                 and c["kind"] == "validation_selected"]
    def condition_rank(reward):
        group = [c for c in finalists if c["reward"] == reward]
        wins = sum(c["validation"]["kills"] for c in group)
        t = sum((c["validation"]["t_kill"] or 0) * c["validation"]["kills"] for c in group)
        return wins, -(t / wins if wins else 1e9), reward == "terminal"
    chosen_reward = max(REWARDS, key=condition_rank)
    selected = max((c for c in finalists if c["reward"] == chosen_reward), key=rank)
    content = dict(test_band=plan["test_band"], test_n=plan["test_n"],
                   choices=choices, selected_reward=chosen_reward, selected_policy=selected)
    path = out / "selection.json"
    if path.exists():
        existing = read(path)
        if any(existing[k] != v for k, v in content.items()):
            raise ValueError("Frozen choices or checkpoint hashes changed.")
        return existing
    selection = dict(frozen_utc=timestamp(), **content)
    dest = out / "selected_policy"
    dest.mkdir(exist_ok=True)
    source = out / selected["reward"] / f"s{selected['seed']}"
    for name in ("policy.py", "wrappers.py", "utils.py", "config.json"):
        shutil.copyfile(source / name, dest / name)
    shutil.copyfile(out / selected["weights"], dest / "policy_net.zip")
    write_json(dest / "selection.json", selection)
    write_json(path, selection)
    return selection

def test_worker(args):
    import torch
    from tools import project
    from tools.grade import score
    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    plan = check_protocol(out)
    selection = read(out / "selection.json")
    if (selection["test_band"], selection["test_n"]) != (plan["test_band"], plan["test_n"]):
        raise ValueError("Test protocol changed.")
    dest = out / args.reward / f"s{args.seed}"
    proj = project.load(str(ROOT / "templates/project_04_fair"))
    rows = []
    for choice in selection["choices"]:
        if choice["reward"] != args.reward or choice["seed"] != args.seed:
            continue
        weights = out / choice["weights"]
        if sha(weights) != choice["weights_sha256"]:
            raise ValueError("Checkpoint changed after model selection.")
        cache = weights.parent / f"test_band_{plan['test_band']}.json"
        if cache.exists():
            test = read(cache)
            if (test["band"], test["summary"]["n"], test.get("weights_sha256")) != (
                    plan["test_band"], plan["test_n"], choice["weights_sha256"]):
                raise ValueError("Cached test does not match protocol and checkpoint.")
        else:
            test = score(dest, weights, proj, plan["test_band"], plan["test_n"])
            test["weights_sha256"] = choice["weights_sha256"]
            write_json(cache, test)
        rows.append(dict(**choice, test=test))
        print(f"TEST {args.reward}/s{args.seed} {choice['kind']} budget={choice['budget']} "
              f"step={choice['step']}: {test['summary']['kills']}/{plan['test_n']}", flush=True)
    write_json(dest / "test_summary.json", rows)

def summarize(out, plan):
    rows = [r for reward in REWARDS for seed in plan["seeds"]
            for r in read(out / reward / f"s{seed}/test_summary.json")]
    groups = []
    for reward in REWARDS:
        for kind in ("validation_selected", "budget_endpoint"):
            for budget in plan["budgets"]:
                matches = [r for r in rows if (r["reward"], r["kind"], r["budget"]) == (reward, kind, budget)]
                rates = [r["test"]["summary"]["kills"] / plan["test_n"] for r in matches]
                groups.append(dict(reward=reward, kind=kind, budget=budget,
                                   seeds=[r["seed"] for r in matches],
                                   wins_by_seed=[r["test"]["summary"]["kills"] for r in matches],
                                   checkpoint_steps=[r["step"] for r in matches],
                                   mean_win_rate=statistics.mean(rates),
                                   sample_std_win_rate=statistics.stdev(rates) if len(rates) > 1 else None))
    final = {(g["reward"], g["kind"]): g for g in groups if g["budget"] == plan["budgets"][-1]}
    deltas = [(a - b) / plan["test_n"] for a, b in zip(
        final[("potential_v1", "validation_selected")]["wins_by_seed"],
        final[("terminal", "validation_selected")]["wins_by_seed"])]
    endpoint_delta = (final[("potential_v1", "budget_endpoint")]["mean_win_rate"]
                      - final[("terminal", "budget_endpoint")]["mean_win_rate"])
    promising = statistics.mean(deltas) >= 0.10 - 1e-12 and sum(d > 0 for d in deltas) >= 2 and endpoint_delta >= 0
    decision = dict(paired_test_win_rate_deltas=deltas, mean_delta=statistics.mean(deltas),
                    endpoint_mean_delta=endpoint_delta,
                    next_step="independent_seed_confirmation" if promising and not plan["smoke"]
                              else "validation_and_training_diagnostics",
                    automatic_adoption=False)
    selection = read(out / "selection.json")
    selected = selection["selected_policy"]
    chosen = next(r for r in rows if all(r[k] == selected[k] for k in ("reward", "seed", "budget", "kind")))
    write_json(out / "selected_policy/test.json", chosen["test"])
    training = [dict(reward=reward, **read(out / reward / f"s{seed}/result.json"))
                for reward in REWARDS for seed in plan["seeds"]]
    summary = dict(completed_utc=timestamp(), smoke=plan["smoke"], groups=groups,
                   selected_policy=chosen, training=training, decision=decision,
                   limitations=["Three training seeds are screening, not a definitive population estimate.",
                                "Both seats share initial seeds; 40 matches are not 40 independent training runs.",
                                "Budget selections share trajectories and have differing candidate counts.",
                                "Archived PC controls are historical; the primary controls are fresh matched runs.",
                                "Frozen representative is selected from validation only."])
    write_json(out / "test_summary.json", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(dict(groups=groups, decision=decision), indent=2), flush=True)
    plot(out, plan, groups)
    return summary

def plot(out, plan, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    colors = {"terminal": "#315b97", "potential_v1": "#dc6b2e"}
    for reward in REWARDS:
        histories = [read(out / reward / f"s{s}/validation.json") for s in plan["seeds"]]
        x = [r["step"] for r in histories[0]]
        means = [statistics.mean(h[i]["summary"]["kills"] / plan["val_n"] for h in histories)
                 for i in range(len(x))]
        for history in histories:
            axes[0].plot(x, [r["summary"]["kills"] / plan["val_n"] for r in history],
                         color=colors[reward], alpha=0.2, linewidth=1)
        axes[0].plot(x, means, label=reward, color=colors[reward], linewidth=2)
        for ax, kind in zip(axes[1:], ("validation_selected", "budget_endpoint")):
            group = [g for g in groups if g["reward"] == reward and g["kind"] == kind]
            ax.errorbar([g["budget"] for g in group], [g["mean_win_rate"] for g in group],
                        yerr=[g["sample_std_win_rate"] or 0 for g in group], marker="o",
                        capsize=3, color=colors[reward], label=reward)
    for ax, title in zip(axes, ("Validation: every checkpoint", "Test: validation-selected", "Test: budget endpoint")):
        ax.set(title=title, xlabel="Training interactions", ylabel="Win rate", ylim=(-0.03, 1.03))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"A3 reward comparison | test band {plan['test_band']} | error bars: training-seed sample SD")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=170)
    fig.savefig(out / "learning_curves.svg")
    plt.close(fig)

def run(args):
    started = time.perf_counter()
    out = Path(args.out).resolve()
    plan = prepare(out, args.mode == "smoke")
    jobs = [(r, s) for s in plan["seeds"] for r in REWARDS]
    def train(reward, seed):
        dest = out / reward / f"s{seed}"
        if (dest / "result.json").exists():
            return
        if (dest / "config.json").exists():
            raise ValueError(f"Interrupted training at {dest}; preserve it and use a new output folder.")
        child("worker", out, reward, seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(train, r, s) for r, s in jobs]):
            f.result()
            print("A3 TRAINING JOB COMPLETE", flush=True)
    freeze(out, plan)
    print(f"ALL SELECTIONS FROZEN; opening test band {plan['test_band']}.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(child, "test-worker", out, r, s) for r, s in jobs]):
            f.result()
    summarize(out, plan)
    write_json(out / "timing.json", dict(wall_seconds=time.perf_counter() - started, finished_utc=timestamp()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "smoke", "worker", "test-worker", "report"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--reward", choices=REWARDS, default="terminal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("Use 1..6 worker processes.")
    if args.mode in ("run", "smoke"):
        run(args)
    elif args.mode == "worker":
        train_worker(args)
    elif args.mode == "test-worker":
        test_worker(args)
    else:
        out = Path(args.out).resolve()
        summarize(out, check_protocol(out))

if __name__ == "__main__":
    main()
