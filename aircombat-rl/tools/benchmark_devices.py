"""Compare baseline DQN learning throughput in separate, sequential processes.

Run from the repository root: python -m tools.benchmark_devices
Results measure runtime, not policy quality. No trained checkpoint is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker(args) -> dict:
    start = time.perf_counter()
    import torch

    sys.path.insert(0, str(ROOT / "templates" / args.project / "baseline"))
    from policy import Policy
    from wrappers import make_env

    import_seconds = time.perf_counter() - start
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable.")

    def sync():
        if args.device == "xpu":
            torch.xpu.synchronize()

    # Use a throwaway model to initialize the runtime, kernels and optimizer.
    # The timed model below is newly initialized with the requested seed.
    start = time.perf_counter()
    warmup = Policy.make_learner(make_env, args.seed, args.device)
    try:
        warmup.learn(total_timesteps=1024, progress_bar=False)
        sync()
    finally:
        warmup.get_env().close()
    del warmup
    warmup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model = Policy.make_learner(make_env, args.seed, args.device)
    try:
        sync()
        model_setup_seconds = time.perf_counter() - start
        initial_hash = hashlib.sha256(b"".join(
            p.detach().cpu().numpy().tobytes() for p in model.q_net.parameters()
        )).hexdigest()
        if any(p.device.type != args.device for p in model.policy.parameters()):
            raise RuntimeError("Unexpected model device.")
        sync()
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, progress_bar=False)
        sync()
        training_seconds = time.perf_counter() - start
        if model._n_updates < 1 or not all(
            bool(torch.isfinite(p).all()) for p in model.policy.parameters()
        ):
            raise RuntimeError("Training did not produce finite, updated weights.")
        return {
            "device": args.device,
            "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "seed": args.seed,
            "steps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "episodes": model._episode_num,
            "training_seconds": training_seconds,
            "steps_per_second": model.num_timesteps / training_seconds,
            "imports_seconds": import_seconds,
            "throwaway_warmup_seconds": warmup_seconds,
            "model_setup_seconds": model_setup_seconds,
            "initial_weights_sha256": initial_hash,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.xpu.get_device_name(0) if args.device == "xpu" else None,
        }
    finally:
        model.get_env().close()


def run(args) -> int:
    if args.steps <= 1000 or args.steps % 4:
        raise ValueError("--steps must exceed 1000 and be a multiple of 4.")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    configs = [("cpu_default", "cpu", 0), ("cpu_1_thread", "cpu", 1),
               ("xpu_default", "xpu", 0)]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "steps_per_run": args.steps,
        "repeats": args.repeats,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "method": "Sequential fresh processes; order rotated per seed; discard a "
                  "1024-step warmup model; initialize a fresh baseline for each timed "
                  "run. Timing includes JSBSim, policy inference, replay and optimizer "
                  "updates, with XPU synchronization at boundaries. Excludes imports, "
                  "warmup, model construction, validation and checkpoint I/O.",
        "runs": [],
    }
    for repeat in range(args.repeats):
        offset = repeat % len(configs)
        for label, device, threads in configs[offset:] + configs[:offset]:
            command = [sys.executable, "-m", "tools.benchmark_devices", "--worker",
                       "--project", args.project, "--device", device,
                       "--threads", str(threads), "--seed", str(repeat),
                       "--steps", str(args.steps)]
            print(f"START {label} seed={repeat} steps={args.steps}", flush=True)
            start = time.perf_counter()
            proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            wall_seconds = time.perf_counter() - start
            (out / f"{label}_seed{repeat}.log").write_text(
                proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            if proc.returncode:
                raise RuntimeError(f"Worker failed; see {out / f'{label}_seed{repeat}.log'}")
            payloads = [line[7:] for line in proc.stdout.splitlines()
                        if line.startswith("RESULT ")]
            result = json.loads(payloads[-1])
            result.update(config=label, process_wall_seconds=wall_seconds)
            report["runs"].append(result)
            (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"DONE {label} seed={repeat}: {result['training_seconds']:.3f}s, "
                  f"{result['steps_per_second']:.1f} steps/s, "
                  f"{result['gradient_updates']} updates", flush=True)

    for seed in range(args.repeats):
        runs = [r for r in report["runs"] if r["seed"] == seed]
        if len({r["initial_weights_sha256"] for r in runs}) != 1:
            raise RuntimeError("Initial weights differ between devices for the same seed.")
        if len({(r["steps"], r["gradient_updates"]) for r in runs}) != 1:
            raise RuntimeError("Step/update budgets differ between devices.")
    report["summary"] = {}
    for label, _, _ in configs:
        runs = [r for r in report["runs"] if r["config"] == label]
        times = [r["training_seconds"] for r in runs]
        median = statistics.median(times)
        report["summary"][label] = {
            "median_seconds": median,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "steps_per_second_at_median": args.steps / median,
            "cpu_threads": runs[0]["cpu_threads"],
        }
    report["same_initial_weights_and_update_budgets_verified"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Results: {out / 'results.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="project_01_circular",
                        choices=("project_01_circular", "project_02_evader",
                                 "project_03_advantaged", "project_04_fair"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="runs/device_benchmark")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu",
                        help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print("RESULT " + json.dumps(worker(args)), flush=True)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
