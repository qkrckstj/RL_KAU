"""Learning-rate-only DQN comparison; freeze validation choices before testing."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import distributions
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from tools.plan_a import ROOT, timestamp, worker, write_json
from tools import plan_a_shaping as archive
from tools.plan_a_shaping import read, sha

RATES = {"lr_1e3": 1e-3, "lr_1e4": 1e-4}


def protocol(smoke=False):
    paths = list(archive.SOURCE_PATHS) + ["tools/plan_a_stability.py", "tests/test_plan_a_stability.py",
        "experiments/plan_a/stability.md"]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "aircombat_gym").rglob("*.py"))]
    return dict(experiment="DQN learning rate only", created_utc=timestamp(), smoke=smoke,
        learning_rates=RATES, reward="terminal", seeds=[0] if smoke else [0, 1, 2],
        budgets=[2048] if smoke else [204800, 512000, 1024000],
        val_every=1024 if smoke else 51200, val_n=2 if smoke else 20,
        val_band=900000, test_band=900000 if smoke else 1400000, test_n=2 if smoke else 40,
        epsilon_decay_steps=1024 if smoke else 20480,
        controls="Fresh paired CPU runs on the same PC; only learning rate differs.",
        selection="Validation wins, mean win time, earlier checkpoint. Condition uses summed validation wins; exact tie prefers baseline.",
        extension="No automatic extension. Require positive final validation in >=2 seeds and nondecreasing aggregate mean over last five vs preceding five checkpoints before considering a separately registered extension.",
        source_sha256={name: sha(ROOT / name) for name in paths})


def train(args):
    plan = archive.check_protocol(Path(args.out))
    args.out = str(Path(args.out) / args.condition / f"s{args.seed}")
    args.variant, args.task, args.reward = "observed_dqn", "fair", "terminal"
    args.learning_rate = plan["learning_rates"][args.condition]
    args.steps, args.val_every, args.val_n = plan["budgets"][-1], plan["val_every"], plan["val_n"]
    args.test_band, args.epsilon_decay_steps = plan["test_band"], plan["epsilon_decay_steps"]
    args.save_checkpoints, args.resume_milestones, args.q_diagnostics = True, plan["budgets"], True
    worker(args)


def child(mode, out, condition, seed):
    dest = out / condition / f"s{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / f"{mode}.log").open("w", encoding="utf-8") as stream:
        proc = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.plan_a_stability", mode,
            "--out", str(out), "--condition", condition, "--seed", str(seed)], cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8"),
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode:
        raise RuntimeError(f"{condition}/s{seed} {mode} failed; see {dest}")


def report(out, plan):
    from tools.diagnose_shaping import replay_diagnostics
    rows = [r for c in RATES for s in plan["seeds"] for r in read(out / c / f"s{s}/test_summary.json")]
    groups, diagnostics = [], []
    for c in RATES:
        for kind in ("validation_selected", "budget_endpoint"):
            for budget in plan["budgets"]:
                subset = [r for r in rows if (r["reward"], r["kind"], r["budget"]) == (c, kind, budget)]
                wins = [r["test"]["summary"]["kills"] for r in subset]
                groups.append(dict(condition=c, kind=kind, budget=budget, wins=wins,
                    mean_win_rate=statistics.mean(wins) / plan["test_n"]))
        for s in plan["seeds"]:
            dest = out / c / f"s{s}"
            history = [h for h in read(dest / "validation.json") if h["step"] > 0]
            diagnostics.append(dict(condition=c, seed=s,
                late_validation_wins=[h["summary"]["kills"] for h in history[-5:]],
                previous_validation_wins=[h["summary"]["kills"] for h in history[-10:-5]],
                final=replay_diagnostics(dest, plan["budgets"][-1])))
    summary = dict(groups=groups, diagnostics=diagnostics, test_band=plan["test_band"],
        selected_policy=read(out / "selection.json")["selected_policy"], finished_utc=timestamp())
    write_json(out / "summary.json", summary)
    lines = ["# DQN learning-rate comparison", "", f"Test band: {plan['test_band']}; matches/model: {plan['test_n']}", "",
        "| Learning rate | Budget | Selection | Wins by training seed | Mean |",
        "|---|---:|---|---|---:|"]
    for g in groups:
        lines.append(f"| {RATES[g['condition']]} | {g['budget']} | {g['kind']} | {g['wins']} | {g['mean_win_rate']:.2%} |")
    lines += ["", "## Final replay diagnostics", "", "| Condition | Seed | Q max | Out-of-bound fraction | Last five validation wins |",
        "|---|---:|---:|---:|---|"]
    for d in diagnostics:
        q = d["final"]
        lines.append(f"| {d['condition']} | {d['seed']} | {q['q_max']:.3f} | {q['all_action_return_bound_violation_fraction']:.2%} | {d['late_validation_wins']} |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Use a new output folder; never overwrite experiment history.")
    out.mkdir(parents=True)
    plan = protocol(args.mode == "smoke")
    write_json(out / "plan.json", plan)
    write_json(out / "environment.json", dict(python=sys.version, executable=sys.executable,
        packages={d.metadata["Name"]: d.version for d in distributions()},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    for name in plan["source_sha256"]:
        dest = out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    start = time.perf_counter()
    jobs = [(c, s) for s in plan["seeds"] for c in RATES]
    for mode in ("worker", "test-worker"):
        if mode == "test-worker":
            # Reuse the audited A3 selection/evaluation machinery. Its legacy
            # 'reward' field identifies the condition here; both use terminal reward.
            archive.REWARDS = tuple(RATES)
            archive.freeze(out, plan)
            print("SELECTION FROZEN BEFORE TEST", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(child, mode, out, c, s): (c, s) for c, s in jobs}
            for f in as_completed(tasks):
                f.result()
                print(f"COMPLETE {mode} {tasks[f]}", flush=True)
    report(out, plan)
    write_json(out / "timing.json", dict(wall_seconds=time.perf_counter() - start))
    print("EXPERIMENT COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "smoke", "worker", "test-worker", "report"))
    p.add_argument("--out", required=True)
    p.add_argument("--condition", choices=RATES, default="lr_1e3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, choices=range(1, 7), default=3)
    args = p.parse_args()
    if args.mode in ("run", "smoke"):
        run(args)
    elif args.mode == "worker":
        train(args)
    elif args.mode == "test-worker":
        args.reward = args.condition
        archive.test_worker(args)
    else:
        report(Path(args.out), archive.check_protocol(Path(args.out)))


if __name__ == "__main__":
    main()
