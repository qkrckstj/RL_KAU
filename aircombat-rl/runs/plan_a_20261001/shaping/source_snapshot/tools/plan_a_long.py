"""Duration-only DQN follow-up. Run trains, freezes selections, then tests them."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from tools.plan_a import ROOT, timestamp, worker, write_json

BUDGETS = (204800, 512000, 1024000)
SEEDS = (0, 1, 2)
TEST_BAND = 1200000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_at_budget(history, budget):
    eligible = [row for row in history if 0 < row["step"] <= budget]
    if not eligible:
        raise ValueError("No trained checkpoint available within this budget.")
    return max(eligible, key=lambda r: (
        r["summary"]["kills"],
        -(r["summary"]["t_kill"] if r["summary"]["t_kill"] is not None else 1e9),
        -r["step"]))


def train_worker(args):
    args.variant, args.task = "observed_dqn", "fair"
    args.steps, args.val_every, args.val_n = BUDGETS[-1], 51200, 20
    args.test_band, args.epsilon_decay_steps = TEST_BAND, 20480
    args.save_checkpoints, args.resume_milestones = True, BUDGETS
    worker(args)


def child(mode, out, seed, log):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            [sys.executable, "-m", "tools.plan_a_long", mode,
             "--out", str(out), "--seed", str(seed)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode:
        raise RuntimeError(f"{mode} seed {seed} failed; see {log}")


def audit_prefix(out, seed):
    """Check whether the first 204800 interactions reproduced the earlier run."""
    import torch
    from stable_baselines3 import DQN
    torch.set_num_threads(1)
    current = out / f"s{seed}"
    previous = ROOT / "runs/plan_a_20261001/comparison/observed_dqn" / f"s{seed}"
    old = DQN.load(previous / "final_net.zip", device="cpu")
    new = DQN.load(current / "checkpoints/step_204800/policy_net.zip", device="cpu")
    old_state, new_state = old.policy.state_dict(), new.policy.state_dict()
    exact = all(torch.equal(old_state[k], new_state[k]) for k in old_state)
    maximum = max(float((old_state[k] - new_state[k]).abs().max()) for k in old_state)
    old_val, new_val = read(previous / "validation.json"), read(current / "validation.json")
    matching = all(a["episodes"] == b["episodes"] for a, b in zip(old_val, new_val))
    result = dict(seed=seed, policy_parameters_bitwise_equal=exact,
                  max_parameter_absolute_difference=maximum,
                  first_five_validation_episode_lists_equal=matching,
                  previous_model=str(previous / "final_net.zip"))
    write_json(current / "prefix_audit.json", result)
    return result


def freeze(out):
    if (out / "selection.json").exists():
        return read(out / "selection.json")
    choices = []
    for seed in SEEDS:
        dest = out / f"s{seed}"
        if not (dest / "result.json").exists():
            raise RuntimeError("Finish all training before selecting or testing.")
        history = read(dest / "validation.json")
        for budget in BUDGETS:
            best = select_at_budget(history, budget)
            choices.append(dict(seed=seed, budget=budget, kind="validation_selected",
                                step=best["step"], validation=best["summary"]))
            endpoint = next(r for r in history if r["step"] == budget)
            choices.append(dict(seed=seed, budget=budget, kind="budget_endpoint",
                                step=budget, validation=endpoint["summary"]))
    candidates = [c for c in choices if c["budget"] == BUDGETS[-1]
                  and c["kind"] == "validation_selected"]
    selected = max(candidates, key=lambda c: (
        c["validation"]["kills"],
        -(c["validation"]["t_kill"] if c["validation"]["t_kill"] is not None else 1e9),
        -c["step"], -c["seed"]))
    selection = dict(frozen_utc=timestamp(), test_band=TEST_BAND, test_n=40,
                     choices=choices, selected_policy=selected,
                     rule="Validation wins, mean win time, earlier step, lower seed; no test-based selection.")
    write_json(out / "selection.json", selection)
    source = out / f"s{selected['seed']}"
    dest = out / "selected_policy"
    dest.mkdir(exist_ok=True)
    for name in ("policy.py", "wrappers.py", "utils.py", "config.json"):
        shutil.copyfile(source / name, dest / name)
    shutil.copyfile(source / f"checkpoints/step_{selected['step']}/policy_net.zip",
                    dest / "policy_net.zip")
    write_json(dest / "selection.json", selection)
    return selection


def test_worker(args):
    import torch
    from tools import project
    from tools.grade import score
    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    selection = read(out / "selection.json")
    dest = out / f"s{args.seed}"
    proj = project.load(str(ROOT / "templates/project_04_fair"))
    rows = []
    for choice in (c for c in selection["choices"] if c["seed"] == args.seed):
        step = choice["step"]
        checkpoint = dest / f"checkpoints/step_{step}"
        path = checkpoint / "test.json"
        if path.exists():
            result = read(path)
        else:
            result = score(dest, checkpoint / "policy_net.zip", proj, TEST_BAND, 40)
            write_json(path, result)
        rows.append(dict(**choice, test=result))
        print(f"TEST s{args.seed} budget={choice['budget']} {choice['kind']} "
              f"step={step}: {result['summary']['kills']}/40", flush=True)
    write_json(dest / "test_summary.json", rows)


def summarize(out):
    rows = [r for seed in SEEDS for r in read(out / f"s{seed}/test_summary.json")]
    groups = []
    for kind in ("validation_selected", "budget_endpoint"):
        for budget in BUDGETS:
            matches = [r for r in rows if r["budget"] == budget and r["kind"] == kind]
            rates = [r["test"]["summary"]["kills"] / 40 for r in matches]
            groups.append(dict(kind=kind, budget=budget,
                               wins_by_seed=[r["test"]["summary"]["kills"] for r in matches],
                               checkpoint_steps=[r["step"] for r in matches],
                               mean_win_rate=statistics.mean(rates),
                               sample_std_win_rate=statistics.stdev(rates)))
    selection = read(out / "selection.json")
    selected = selection["selected_policy"]
    chosen = next(r for r in rows if all(r[k] == selected[k] for k in ("seed", "budget", "kind")))
    write_json(out / "selected_policy/test.json", chosen["test"])
    summary = dict(completed_utc=timestamp(), groups=groups, selected_policy=chosen,
                   prefix_audits=[read(out / f"s{s}/prefix_audit.json") for s in SEEDS],
                   training=[read(out / f"s{s}/result.json") for s in SEEDS],
                   limitations=["Three training seeds are a screening experiment.",
                                "More validation candidates at longer budgets; endpoint results reported separately.",
                                "All budgets share one test band and paired training trajectories.",
                                "No exact simulator-state resume; learner/replay/RNG snapshots are saved."])
    write_json(out / "test_summary.json", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(groups, indent=2), flush=True)
    return summary


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = ["tools/plan_a.py", "tools/plan_a_long.py", "experiments/plan_a/core.py",
               "tests/test_plan_a_long.py", "requirements-windows-xpu.lock.txt"]
    plan = dict(task="fair", variant="observed_dqn", budgets=BUDGETS, seeds=SEEDS,
                epsilon_decay_steps=20480, val_every=51200, val_n=20, val_band=900000,
                test_band=TEST_BAND, test_n=40, workers=3, cpu_threads_per_worker=1,
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                selection="Wins then win time; ties earlier checkpoint, then lower seed.",
                comparison="Fresh trajectories; selected-within-budget and fixed endpoints; same held-out test.")
    # JSON normalization makes tuple/list comparison consistent on rerun.
    plan = json.loads(json.dumps(plan))
    if (out / "plan.json").exists() and read(out / "plan.json") != plan:
        raise ValueError("Protocol/source differs; use a new output folder.")
    write_json(out / "plan.json", plan)
    for name in sources:
        dest = out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    (out / "source_snapshot/git_diff.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    def train(seed):
        dest = out / f"s{seed}"
        dest.mkdir(exist_ok=True)
        if not (dest / "result.json").exists():
            child("worker", dest, seed, dest / "train.log")
        return read(dest / "result.json")
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(train, s) for s in SEEDS]):
            result = future.result()
            print(f"TRAIN COMPLETE s{result['seed']}: best={result['best_val_kills']}/20 "
                  f"at {result['best_step']}", flush=True)
    for seed in SEEDS:
        print(f"PREFIX AUDIT {audit_prefix(out, seed)}", flush=True)
    freeze(out)
    print("SELECTION FROZEN; opening held-out test band 1200000.", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(child, "test-worker", out, s,
                                                out / f"s{s}/test.log") for s in SEEDS]):
            future.result()
    summarize(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "test-worker"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"run": run, "worker": train_worker, "test-worker": test_worker}[args.mode](args)


if __name__ == "__main__":
    main()
