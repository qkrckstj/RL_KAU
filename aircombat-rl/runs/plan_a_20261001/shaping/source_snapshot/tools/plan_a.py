"""Reproducible Plan A experiment runner for the course simulation.

python -m tools.plan_a matrix --out runs/plan_a_20261001/comparison --workers 3
python -m tools.plan_a test --out runs/plan_a_20261001/comparison
Test is a separate, final phase after all training/configuration decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("control_dqn", "observed_dqn", "observed_ppo")
VAL_BAND, TEST_BAND = 900000, 1100000


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def export_design(out: Path, spec: dict, total_steps=204800):
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "experiments/plan_a/core.py", out / "utils.py")
    (out / "wrappers.py").write_text(
        "from utils import State as _State, make_env as _make_env\n"
        + f"SPEC = {spec!r}\nACTION_MODE = 'discrete'\n"
        + "class State(_State):\n    def __init__(self):\n        super().__init__(SPEC)\n"
        + "def make_env(seed=None, shaped=True):\n    return _make_env(SPEC, seed, shaped)\n",
        encoding="utf-8")
    (out / "policy.py").write_text(
        "from utils import PolicyBase, make_learner as _make_learner\nfrom wrappers import SPEC\n"
        + f"class Policy(PolicyBase):\n    SPEC = SPEC\n    TOTAL_STEPS = {total_steps}\n"
        + "    @staticmethod\n    def make_learner(make_env, seed, device):\n"
        + "        return _make_learner(SPEC, seed, device)\n", encoding="utf-8")


def evaluate(model, spec, band, n, core_module=None):
    if core_module is None:
        from experiments.plan_a import core as core_module
    State, make_env = core_module.State, core_module.make_env
    from tools.grade import play, summarise
    state = State(spec)
    def act(obs):
        a, _ = model.predict(state(obs), deterministic=True)
        return int(a)
    seats = ("red", "blue") if spec["task"] == "fair" else ("red",)
    if n % len(seats):
        raise ValueError("Evaluation episodes must split evenly between seats.")
    rows = []
    for seat in seats:
        env = make_env(spec, shaped=False, seat=seat)
        try:
            current = play(env, act, band, n // len(seats), seat)
            for row in current:
                row["seat"] = seat
            rows.extend(current)
        finally:
            env.close()
    return dict(band=band, summary=summarise(rows),
                per_seat={seat: summarise([r for r in rows if r["seat"] == seat]) for seat in seats},
                episodes=rows)


def worker(args):
    import torch
    import stable_baselines3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from experiments.plan_a.core import State, make_learner, specification

    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    spec = specification(args.variant, args.task)
    if getattr(args, "reward", "terminal") != "terminal":
        spec["reward"] = args.reward
    metadata = dict(spec=spec, seed=args.seed, steps=args.steps, val_every=args.val_every,
                    val_n=args.val_n, val_band=VAL_BAND,
                    test_band_reserved=getattr(args, "test_band", TEST_BAND),
                    python=platform.python_version(), torch=torch.__version__,
                    stable_baselines3=stable_baselines3.__version__, device="cpu", cpu_threads=1,
                    source_sha256=hashlib.sha256((ROOT / "experiments/plan_a/core.py").read_bytes()).hexdigest(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    started_utc=timestamp())
    if (out / "result.json").exists():
        raise FileExistsError(f"Completed run already exists: {out}")
    export_design(out, spec, args.steps)
    write_json(out / "config.json", metadata)
    model = make_learner(spec, args.seed)
    decay_steps = getattr(args, "epsilon_decay_steps", None)
    if decay_steps is not None:
        set_exploration_duration(model, args.steps, decay_steps)
        metadata["epsilon_decay_steps"] = decay_steps
        metadata["effective_exploration_fraction"] = model.exploration_fraction
    metadata["resume_milestones"] = list(getattr(args, "resume_milestones", ()))
    model.set_logger(configure(str(out / "metrics"), ["csv"]))
    metadata["features"] = list(State(spec).names)
    metadata["parameter_count_including_targets"] = sum(p.numel() for p in model.policy.parameters())
    write_json(out / "config.json", metadata)
    start = time.perf_counter()

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history = []
            self.next_eval = args.val_every
            self.next_log = 10000
            self.best = (-1, -1e9)
            self.best_step = None
            self.eval_seconds = 0.0
            self.outcomes = Counter()

        def status(self, stage):
            write_json(out / "progress.json", dict(stage=stage, steps=self.model.num_timesteps,
                       target_steps=args.steps, elapsed_seconds=round(time.perf_counter() - start, 2),
                       evaluation_seconds=round(self.eval_seconds, 2), train_outcomes=dict(self.outcomes),
                       best_validation_kills=self.best[0], best_step=self.best_step, updated_utc=timestamp()))

        def validate(self, select=True):
            self.status("validation")
            t0 = time.perf_counter()
            result = evaluate(self.model, spec, VAL_BAND, args.val_n)
            self.eval_seconds += time.perf_counter() - t0
            result["step"] = self.model.num_timesteps
            result["gradient_update_counter"] = self.model._n_updates
            result["train_outcomes"] = dict(self.outcomes)
            result["exploration_rate"] = getattr(self.model, "exploration_rate", None)
            self.history.append(result)
            summary = result["summary"]
            score = (summary["kills"], -(summary["t_kill"] if summary["t_kill"] is not None else 1e9))
            if select and score > self.best:
                self.best, self.best_step = score, self.model.num_timesteps
                self.model.save(out / "policy_net.zip")
            if select and getattr(args, "save_checkpoints", False):
                checkpoint = out / "checkpoints" / f"step_{self.model.num_timesteps}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                self.model.save(checkpoint / "policy_net.zip")
                if self.model.num_timesteps in getattr(args, "resume_milestones", ()):
                    save_training_state(self.model, checkpoint)
            write_json(out / "validation.json", self.history)
            self.status("training")
            print(f"VAL {args.variant} s{args.seed} step={self.model.num_timesteps} "
                  f"kills={summary['kills']}/{args.val_n} outcomes={summary['outcomes']}", flush=True)

        def _on_training_start(self):
            self.validate(select=False)

        def _on_rollout_start(self):
            # This follows the preceding optimizer update for both DQN and PPO.
            if self.model.num_timesteps >= self.next_eval:
                self.validate()
                self.next_eval += args.val_every

        def _on_step(self):
            for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
                if done:
                    self.outcomes[info.get("outcome", "unknown")] += 1
            if self.model.num_timesteps >= self.next_log:
                self.status("training")
                print(f"TRAIN {args.variant} s{args.seed} step={self.model.num_timesteps} "
                      f"outcomes={dict(self.outcomes)}", flush=True)
                self.next_log += 10000
            return True

    callback = Progress()
    try:
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
        if not callback.history or callback.history[-1]["step"] != model.num_timesteps:
            callback.validate()
        if model.num_timesteps != args.steps:
            raise RuntimeError("Unexpected interaction budget; choose steps divisible by 1024.")
        if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
            raise RuntimeError("Non-finite learned parameters.")
        model.save(out / "final_net.zip")
        result = dict(status="complete", variant=args.variant, seed=args.seed, steps=model.num_timesteps,
                      best_step=callback.best_step, best_val_kills=callback.best[0], val_n=args.val_n,
                      train_outcomes=dict(callback.outcomes), gradient_update_counter=model._n_updates,
                      wall_seconds=time.perf_counter() - start, evaluation_seconds=callback.eval_seconds,
                      finished_utc=timestamp())
        write_json(out / "result.json", result)
        callback.status("complete")
        return result
    finally:
        model.get_env().close()


def set_exploration_duration(model, total_steps, decay_steps):
    """Keep epsilon's absolute decay duration fixed when extending one learn call."""
    from stable_baselines3.common.utils import LinearSchedule
    if not 0 < decay_steps <= total_steps:
        raise ValueError("Epsilon decay must finish within the positive training budget.")
    model.exploration_fraction = decay_steps / total_steps
    model.exploration_schedule = LinearSchedule(
        model.exploration_initial_eps, model.exploration_final_eps,
        model.exploration_fraction)


def save_training_state(model, checkpoint):
    """Save learner state; JSBSim's in-flight state is deliberately not claimed."""
    import pickle
    import random
    import numpy as np
    import torch
    model.save_replay_buffer(checkpoint / "replay_buffer.pkl")
    with (checkpoint / "rng_state.pkl").open("wb") as stream:
        pickle.dump(dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch_cpu=torch.get_rng_state()), stream)
    write_json(checkpoint / "resume.json", dict(
        steps=model.num_timesteps, gradient_updates=model._n_updates,
        exploration_rate=model.exploration_rate, replay_size=model.replay_buffer.size(),
        saved_utc=timestamp(), optimizer_in="policy_net.zip",
        simulator_state_saved=False,
        resume_semantics="Learner and replay continuation with a fresh episode; not an exact trajectory resume."))


def matrix(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(task=args.task, variants=list(VARIANTS), seeds=args.seeds, steps=args.steps,
                val_every=args.val_every, val_n=args.val_n, val_band=VAL_BAND,
                test_band=TEST_BAND, test_n=40, workers=args.workers,
                selection="validation kills, then mean time-to-kill; ties keep earlier trained checkpoint",
                notes="Same sparse rewards, gamma, finite-match handling, action space, "
                      "environment budget, validation episodes; fixed algorithm-specific optimizer settings.")
    if (out / "plan.json").exists() and json.loads((out / "plan.json").read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs; use a new output directory.")
    write_json(out / "plan.json", plan)

    def job(variant, seed):
        dest = out / variant / f"s{seed}"
        if (dest / "result.json").exists():
            return json.loads((dest / "result.json").read_text(encoding="utf-8"))
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "tools.plan_a", "worker", "--variant", variant,
               "--seed", str(seed), "--task", args.task, "--steps", str(args.steps),
               "--val-every", str(args.val_every), "--val-n", str(args.val_n), "--out", str(dest)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
        with (dest / "train.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode:
            raise RuntimeError(f"{variant} s{seed} failed: {dest / 'train.log'}")
        return json.loads((dest / "result.json").read_text(encoding="utf-8"))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(job, v, s) for s in args.seeds for v in VARIANTS}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                print(f"DONE {result['variant']} s{result['seed']}: "
                      f"val={result['best_val_kills']}/{result['val_n']}", flush=True)
            active = []
            for progress in out.glob("*/s*/progress.json"):
                data = json.loads(progress.read_text(encoding="utf-8"))
                if data["stage"] != "complete":
                    active.append(f"{progress.parent.parent.name}/{progress.parent.name}: "
                                  f"{data['steps']}/{args.steps} {data['stage']} "
                                  f"best={data['best_validation_kills']}/{args.val_n}")
            if active:
                print(" | ".join(active), flush=True)
    write_json(out / "training_summary.json", results)
    print(f"TRAINING COMPLETE: {out}", flush=True)


def test(args):
    import torch
    from tools import project as P
    from tools.grade import score
    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    name = "project_04_fair" if plan["task"] == "fair" else "project_01_circular"
    proj = P.load(str(ROOT / "templates" / name))
    for variant in plan["variants"]:
        for seed in plan["seeds"]:
            if not (out / variant / f"s{seed}" / "result.json").exists():
                raise RuntimeError("All planned runs must finish before opening the final test set.")
    results = []
    for variant in plan["variants"]:
        for seed in plan["seeds"]:
            dest = out / variant / f"s{seed}"
            if (dest / "test.json").exists():
                result = json.loads((dest / "test.json").read_text(encoding="utf-8"))
            else:
                result = score(dest, dest / "policy_net.zip", proj, plan["test_band"], plan["test_n"])
                result.update(variant=variant, training_seed=seed)
                write_json(dest / "test.json", result)
            results.append(result)
            print(f"TEST {variant} s{seed}: {result['summary']['kills']}/{plan['test_n']} "
                  f"{result['summary']['outcomes']}", flush=True)
    write_json(out / "test_summary.json", results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("worker", "matrix", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=("fair", "circular"), default="fair")
    ap.add_argument("--variant", choices=VARIANTS, default="control_dqn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=204800)
    ap.add_argument("--val-every", type=int, default=51200)
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.steps <= 0 or args.steps % 1024 or args.val_every <= 0 or args.val_every % 1024:
        raise ValueError("Steps and validation interval must be positive multiples of 1024.")
    if args.val_n < 2 or args.val_n % 2 or args.workers < 1:
        raise ValueError("Use a positive worker count and a positive, even validation count.")
    {"worker": worker, "matrix": matrix, "test": test}[args.mode](args)


if __name__ == "__main__":
    main()
