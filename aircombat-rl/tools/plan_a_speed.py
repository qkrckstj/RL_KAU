"""Heading-only Double DQN with forced deceleration; paired with archived unrestricted DDQN."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tools.plan_a import ROOT, timestamp, worker, write_json
from tools.plan_a_long import select_at_budget
from tools import plan_a_shaping as archive
from tools.plan_a_shaping import read, sha


def should_extend(history):
    h = {r["step"]: r for r in history if r["step"] > 0}
    wins = [h[s]["summary"]["kills"] for s in sorted(h)]
    return (len(wins) >= 10 and wins[-1] > 0 and sum(w > 0 for w in wins[-5:]) >= 3
            and sum(wins[-5:]) >= sum(wins[-10:-5]))


def train(args):
    plan = archive.check_protocol(Path(args.out))
    args.out = str(Path(args.out) / "heading_only" / f"s{args.seed}")
    args.variant, args.task, args.reward = "observed_dqn", "fair", "terminal"
    args.double_dqn, args.learning_rate, args.q_diagnostics = True, 0.001, True
    args.force_decelerate = True
    args.stage_budgets = plan["stages"]
    args.steps, args.val_every, args.val_n = plan["stages"][-1], plan["val_every"], plan["val_n"]
    args.test_band, args.epsilon_decay_steps = plan["test_band"], plan["epsilon_decay_steps"]
    args.save_checkpoints, args.resume_milestones = True, plan["stages"]
    worker(args)


def child(mode, out, seed, condition="heading_only"):
    dest = out / condition / f"s{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / f"{mode}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.plan_a_speed", mode,
            "--out", str(out), "--seed", str(seed), "--condition", condition], cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8"),
            stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Failed {condition}/{seed}: {mode}, see {dest}")


def freeze(out, plan):
    choices = []
    for condition in ("heading_only", "ddqn_reference"):
        for seed in plan["seeds"]:
            dest = out / condition / f"s{seed}"
            if condition == "ddqn_reference":
                source = ROOT / plan["reference"] / "ddqn" / f"s{seed}"
                dest.mkdir(parents=True)
                for name in ("utils.py", "wrappers.py", "policy.py", "config.json", "validation.json", "result.json"):
                    shutil.copyfile(source / name, dest / name)
            else:
                source = dest
            result, history = read(dest / "result.json"), read(dest / "validation.json")
            if result["status"] != "complete":
                raise ValueError("Training incomplete")
            budgets = sorted(set([plan["stages"][0], result["steps"]])) if condition == "heading_only" else [plan["stages"][0]]
            for budget in budgets:
                best = select_at_budget(history, budget)
                endpoint = next(r for r in history if r["step"] == budget)
                for kind, row in (("validation_selected", best), ("budget_endpoint", endpoint)):
                    rel = f"checkpoints/step_{row['step']}/policy_net.zip"
                    weights = dest / rel
                    if condition == "ddqn_reference" and not weights.exists():
                        weights.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source / rel, weights)
                    choices.append(dict(reward=condition, seed=seed, budget=budget, kind=kind,
                        step=row["step"], validation=row["summary"], weights=weights.relative_to(out).as_posix(),
                        weights_sha256=sha(weights)))
    finalists = [c for c in choices if c["kind"] == "validation_selected"]
    selected = max(finalists, key=archive.rank)
    dest = out / "selected_policy"
    dest.mkdir()
    for name in ("policy.py", "utils.py", "wrappers.py", "config.json"):
        shutil.copyfile(out / selected["reward"] / f"s{selected['seed']}" / name, dest / name)
    shutil.copyfile(out / selected["weights"], dest / "policy_net.zip")
    write_json(out / "selection.json", dict(frozen_utc=timestamp(), choices=choices,
        selected_policy=selected, test_band=plan["test_band"], test_n=plan["test_n"]))


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory")
    out.mkdir(parents=True)
    smoke = args.mode == "smoke"
    paths = ["experiments/plan_a/core.py", "tools/plan_a.py", "tools/plan_a_speed.py",
        "tools/plan_a_shaping.py", "tools/plan_a_long.py", "tools/plan_a_double.py", "tools/grade.py", "tools/policies.py", "tools/resume_dqn.py",
        "tests/test_plan_a_speed.py", "experiments/plan_a/speed.md"]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / "aircombat_gym").rglob("*.py")]
    reference = "runs/plan_a_20261005/double_smoke" if smoke else "runs/plan_a_20261005/double"
    plan = dict(created_utc=timestamp(), stages=[2048] if smoke else [1024000],
        seeds=[0] if smoke else [0,1,2], val_every=1024 if smoke else 51200, val_n=2 if smoke else 20,
        epsilon_decay_steps=1024 if smoke else 20480, test_band=900000 if smoke else 1900000,
        test_n=2 if smoke else 40, reference=reference,
        gate="Fixed budget, no automatic extension", intervention="Internal 3 heading actions map to raw {0,3,6}; speed delta always -20kt",
        source_sha256={name:sha(ROOT / name) for name in paths})
    # Reference artifacts are fixed before new training, including their validation histories.
    plan["reference_sha256"] = {p.relative_to(ROOT).as_posix():sha(p)
        for s in plan["seeds"] for p in (ROOT / reference / "ddqn" / f"s{s}").rglob("*")
        if p.is_file() and p.suffix in (".zip", ".py", ".json")}
    write_json(out / "plan.json", plan)
    for name in paths:
        dest = out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(child,"worker",out,s) for s in plan["seeds"]]):
                f.result()
        for name, expected in plan["reference_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Reference artifacts changed")
        freeze(out, plan)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(child,"test-worker",out,s,c)
                    for c in ("heading_only","ddqn_reference") for s in plan["seeds"]]):
                f.result()
        rows = [r for c in ("heading_only","ddqn_reference") for s in plan["seeds"]
                for r in read(out / c / f"s{s}/test_summary.json")]
        from tools.grade import play, summarise
        from aircombat_gym.wvr.envs.fair import FairFightEnv
        episodes=[]
        for seat in ("red", "blue"):
            env=FairFightEnv(action_mode="discrete",seat=seat)
            try:
                current=play(env,lambda obs:0,plan["test_band"],plan["test_n"]//2,seat)
                for episode in current: episode["seat"]=seat
                episodes+=current
            finally: env.close()
        fixed=dict(summary=summarise(episodes),episodes=episodes)
        write_json(out / "fixed_baseline.json",fixed)
        write_json(out / "summary.json", dict(rows=rows,fixed_baseline=fixed,finished_utc=timestamp()))
        lines = ["# 감속 고정·선회 학습 결과", "", "같은 100만 스텝 예산에서 3행동 감속 고정과 기존 9행동 Double DQN을 비교했다. 시험 점수로 모델을 재선택하지 않았다.", "",
            "| 조건 | 시드 | 예산 | 선택 방식 | 시험 승리 |", "|---|---:|---:|---|---:|"]
        for r in rows:
            lines.append(f"| {r['reward']} | {r['seed']} | {r['budget']} | {r['kind']} | {r['test']['summary']['kills']}/{plan['test_n']} |")
        lines += ["", f"고정 좌선회·감속 기준: {fixed['summary']['kills']}/{plan['test_n']}승."]
        (out / "결과.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        write_json(out / "completion.json", dict(status="complete", finished_utc=timestamp()))
    except Exception as exc:
        write_json(out / "completion.json", dict(status="failed", error=str(exc), updated_utc=timestamp()))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run","smoke","worker","test-worker"))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="heading_only")
    args = p.parse_args()
    if args.mode in ("run","smoke"):
        run(args)
    elif args.mode == "worker":
        train(args)
    else:
        args.reward = args.condition
        archive.test_worker(args)
