"""A4: independent-seed confirmation, only after A3's preregistered gate passes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import distributions
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from tools.plan_a import ROOT, timestamp, write_json
from tools.plan_a_shaping import (REWARDS, child, check_protocol, freeze,
                                   protocol, read, sha, summarize)


def run(args):
    parent = Path(args.parent).resolve()
    parent_plan = check_protocol(parent)
    parent_summary = read(parent / "summary.json")
    if parent_plan["smoke"] or parent_summary["decision"]["next_step"] != "independent_seed_confirmation":
        raise ValueError("A3's preregistered confirmation gate has not passed.")
    seeds = [3, 4, 5]
    if set(seeds) & set(parent_plan["seeds"]):
        raise ValueError("Confirmation seeds must be independent of A3.")
    out = Path(args.out).resolve()
    plan = protocol(False)
    plan.update(experiment="A4 independent-seed confirmation",
                seeds=seeds, budgets=[1024000], test_band=1400000,
                parent_summary_sha256=sha(parent / "summary.json"),
                parent_plan_sha256=sha(parent / "plan.json"),
                next_step="After this fixed confirmation, report the independent result and stop "
                          "additional training; no coefficient or exploration tuning from test scores.")
    for name in ("tools/plan_a_confirm.py", "experiments/plan_a/confirmation.md"):
        plan["source_sha256"][name] = sha(ROOT / name)
    if out.exists():
        if not (out / "plan.json").exists() or read(out / "plan.json") != plan:
            raise ValueError("Use a new output directory for a different protocol.")
    else:
        out.mkdir(parents=True)
        write_json(out / "environment.json", dict(
            created_utc=timestamp(), python=sys.version, platform=platform.platform(),
            packages={d.metadata["Name"]:d.version for d in distributions()},
            parent=str(parent), seeds=seeds))
        for name in plan["source_sha256"]:
            dest = out / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        write_json(out / "plan.json", plan)
    started = time.perf_counter()
    jobs = [(reward, seed) for seed in seeds for reward in REWARDS]
    def train(reward, seed):
        dest = out / reward / f"s{seed}"
        if (dest / "result.json").exists():
            return
        if (dest / "config.json").exists():
            raise ValueError("Interrupted confirmation must be preserved; choose a new output folder.")
        child("worker", out, reward, seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(train, reward, seed) for reward, seed in jobs]):
            future.result()
            print("A4 TRAINING JOB COMPLETE", flush=True)
    freeze(out, plan)
    print("A4 SELECTION FROZEN; opening previously unused test band 1400000.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(child, "test-worker", out, reward, seed)
                                    for reward, seed in jobs]):
            future.result()
    summary = summarize(out, plan)
    passed = summary["decision"]["next_step"] == "independent_seed_confirmation"
    summary["decision"]["confirmation_gate_passed"] = passed
    summary["decision"]["next_step"] = "retain_validated_candidate" if passed else "benefit_not_confirmed"
    summary["decision"]["new_training_seeds"] = seeds
    summary["decision"]["selection_remains_validation_only"] = True
    write_json(out / "summary.json", summary)
    write_json(out / "timing.json", dict(wall_seconds=time.perf_counter()-started, finished_utc=timestamp()))
    print(summary["decision"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("Use 1..6 workers.")
    run(args)


if __name__ == "__main__":
    main()
