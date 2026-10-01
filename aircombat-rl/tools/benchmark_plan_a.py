"""Measure actual Plan A FairFight DQN throughput on CPU and CUDA.

No validation/test bands, no persistent trained weights. Sequential fresh workers.
Includes JSBSim, inference, replay, optimizer; excludes imports, warmup and I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from tools.plan_a import ROOT, timestamp, write_json


def weight_hash(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                  for p in model.policy.parameters())).hexdigest()


def worker(args):
    import torch
    from experiments.plan_a.core import make_learner, specification
    from tools.plan_a import set_exploration_duration
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    spec = specification("observed_dqn")
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    start = time.perf_counter()
    warmup = make_learner(spec, args.seed, args.device)
    try:
        warmup.learn(1024)
        sync()
    finally:
        warmup.get_env().close()
    del warmup
    warmup_seconds = time.perf_counter() - start
    model = make_learner(spec, args.seed, args.device)
    try:
        set_exploration_duration(model, args.steps, 20480)
        initial = weight_hash(model)
        if any(p.device.type != args.device for p in model.policy.parameters()):
            raise RuntimeError("Policy is not on requested device.")
        sync()
        start = time.perf_counter()
        model.learn(args.steps)
        sync()
        seconds = time.perf_counter() - start
        if model._n_updates <= 0 or initial == weight_hash(model):
            raise RuntimeError("Weights did not update.")
        if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
            raise RuntimeError("Nonfinite learned parameters.")
        return dict(device=args.device, torch=torch.__version__, python=sys.version,
                    seed=args.seed, steps=model.num_timesteps, updates=model._n_updates,
                    cpu_threads=torch.get_num_threads(), seconds=seconds,
                    steps_per_second=model.num_timesteps/seconds,
                    warmup_seconds=warmup_seconds, initial_weights_sha256=initial,
                    gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                    cuda_build=torch.version.cuda,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
    finally:
        model.get_env().close()


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Use a new benchmark output directory.")
    out.mkdir(parents=True)
    configs = [("cpu_cpu_build", str(Path(args.cpu_python).resolve()), "cpu"),
               ("cpu_cuda_build", sys.executable, "cpu"),
               ("cuda", sys.executable, "cuda")]
    report = dict(started_utc=timestamp(), steps=args.steps, repeats=args.repeats,
                  task="FairFight observed_dqn, terminal reward, 31 features, 64x64",
                  epsilon_decay_steps=20480, concurrent_work=args.concurrent_work,
                  method="Sequential fresh processes, rotated order, CPU 1 thread, discarded 1024-step warmup. "
                         "Timing includes environment collection/inference/replay/updates; excludes evaluation and I/O.",
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                 for p in ("experiments/plan_a/core.py", "tools/benchmark_plan_a.py")},
                  runs=[])
    for seed in range(args.repeats):
        order = configs[seed % len(configs):] + configs[:seed % len(configs)]
        for label, python, device in order:
            print(f"START {label} seed={seed}", flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
            proc = subprocess.run(
                [python, "-X", "utf8", "-m", "tools.benchmark_plan_a", "--worker",
                 "--device", device, "--seed", str(seed), "--steps", str(args.steps)],
                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            (out/f"{label}_s{seed}.log").write_text(proc.stdout+"\n"+proc.stderr, encoding="utf-8")
            if proc.returncode:
                raise RuntimeError(f"Failed {label}; inspect benchmark log.")
            result = json.loads(next(line[7:] for line in proc.stdout.splitlines()
                                     if line.startswith("RESULT ")))
            result["config"] = label
            report["runs"].append(result)
            write_json(out/"results.json", report)
            print(f"DONE {label}: {result['seconds']:.2f}s, {result['steps_per_second']:.1f} steps/s", flush=True)
    for seed in range(args.repeats):
        group = [r for r in report["runs"] if r["seed"] == seed]
        if len({r["initial_weights_sha256"] for r in group}) != 1:
            raise RuntimeError("Initial weights were not identical across devices/builds.")
        if len({(r["steps"],r["updates"]) for r in group}) != 1:
            raise RuntimeError("Learning budgets differed.")
    report["summary"] = {}
    for label, _, _ in configs:
        seconds = [r["seconds"] for r in report["runs"] if r["config"] == label]
        median = statistics.median(seconds)
        report["summary"][label] = dict(median_seconds=median, min_seconds=min(seconds),
                                       max_seconds=max(seconds), steps_per_second=args.steps/median)
    report["cuda_time_over_current_cpu"] = (report["summary"]["cuda"]["median_seconds"]
                                            / report["summary"]["cpu_cpu_build"]["median_seconds"])
    report["finished_utc"] = timestamp()
    report["initial_weights_and_budgets_match"] = True
    write_json(out/"results.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/cuda_benchmark_20261001")
    p.add_argument("--cpu-python", default=".venv/Scripts/python.exe")
    p.add_argument("--steps", type=int, default=32768)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--concurrent-work", default="none")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.steps < 20480 or args.steps % 4 or args.repeats < 1:
        p.error("Use >=20480 steps divisible by 4 and at least one repetition.")
    if args.worker:
        print("RESULT "+json.dumps(worker(args)), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
