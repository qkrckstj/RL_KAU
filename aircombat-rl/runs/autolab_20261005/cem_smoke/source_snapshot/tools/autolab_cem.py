"""Direct episodic controller search. Development screening; no held-out test.

Independent searches run in three processes with disjoint training seeds.
Each generation uses the same fresh initial conditions for all candidates.
Validation selects snapshots; it never updates the CEM parameter distribution.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.plan_a.tactical import Policy, PARAMETERS, LOW, HIGH, INITIAL, ANCHOR, normalise
from tools.grade import play, summarise

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(parameters, band, n, fixed=False):
    if n <= 0 or n % 2:
        raise ValueError("Evaluation must contain a positive even match count")
    policy = Policy(parameters=parameters)
    rows = []
    for seat in ("red", "blue"):
        env = FairFightEnv(action_mode="discrete", seat=seat)
        try:
            part = play(env, (lambda obs: 0) if fixed else policy.act, band, n // 2, seat)
            for row in part:
                row["seat"] = seat
            rows.extend(part)
        finally:
            env.close()
    summary = summarise(rows)
    return dict(band=band, summary=summary, episodes=rows,
                per_seat={s: summarise([r for r in rows if r["seat"] == s]) for s in ("red", "blue")})


def rank(result):
    s = result["summary"]
    # Kill count dominates: health only breaks ties, even for the largest
    # possible health difference. No shaped reward changes the official score.
    return s["kills"] + .05 * (s["own_health"] - s["opp_health"])


def export(directory, parameters):
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "experiments/plan_a/tactical.py", directory / "policy.py")
    (directory / "wrappers.py").write_text("class State:\n    def __call__(self, obs): return obs\n", encoding="utf-8")
    write(directory / "policy_net.json", dict(parameter_names=PARAMETERS, parameters=list(parameters),
                                            format="JSON controller parameters; use --policy policy_net.json"))


def worker(out, seed):
    plan = read(out / "plan.json")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    d = out / f"s{seed}"
    d.mkdir()
    rng = np.random.default_rng(seed)
    mean = normalise(INITIAL)
    std = np.full(7, .28)
    incumbent = normalise(ANCHOR)
    validation = []
    best_rank, best_generation = -float("inf"), None
    total_steps = total_episodes = total_wins = 0
    start = time.perf_counter()
    for generation in range(plan["generations"]):
        population = np.clip(rng.normal(mean, std, (plan["population"], 7)), 0, 1)
        population[0], population[1], population[2] = mean, incumbent, normalise(ANCHOR)
        band = plan["training_band"] + (seed - plan["seeds"][0]) * 10000 + generation * 100
        scores, candidates = [], []
        for i, unit in enumerate(population):
            parameters = LOW + unit * (HIGH - LOW)
            result = evaluate(parameters, band, plan["training_n"])
            total_steps += sum(r["steps"] for r in result["episodes"])
            total_episodes += plan["training_n"]
            total_wins += result["summary"]["kills"]
            scores.append(rank(result))
            candidates.append(dict(index=i, parameters=parameters.tolist(), score=scores[-1], result=result))
            write(d / "progress.json", dict(stage="training", generation=generation, candidate=i+1,
                population=plan["population"], generations=plan["generations"], training_steps=total_steps,
                train_episodes=total_episodes, train_wins=total_wins, elapsed_seconds=time.perf_counter()-start))
        order = np.argsort(scores)[::-1]
        elites = population[order[:plan["elites"]]]
        incumbent = population[order[0]].copy()
        mean = .5 * mean + .5 * elites.mean(axis=0)
        std = np.maximum(.06, .5 * std + .5 * elites.std(axis=0))
        parameters = LOW + incumbent * (HIGH - LOW)
        candidate = d / "checkpoints" / f"generation_{generation}"
        export(candidate, parameters)
        result = evaluate(parameters, plan["validation_band"], plan["validation_n"])
        validation.append(dict(generation=generation, training_steps=total_steps, parameters=parameters.tolist(), result=result))
        if rank(result) > best_rank:
            best_rank, best_generation = rank(result), generation
            export(d, parameters)
        write(d / f"generations/g{generation}.json", dict(candidates=candidates, mean=mean.tolist(), std=std.tolist(),
            rng_state=rng.bit_generator.state, validation=result, training_steps=total_steps))
        write(d / "validation.json", validation)
    export(d / "final", parameters)
    best = validation[best_generation]["result"]["summary"]
    write(d / "result.json", dict(status="complete", seed=seed, best_generation=best_generation,
        best_validation=best, final_validation=validation[-1]["result"]["summary"],
        training_steps=total_steps, train_episodes=total_episodes, train_wins=total_wins,
        elapsed_seconds=time.perf_counter()-start))
    write(d / "progress.json", dict(status="complete", training_steps=total_steps, generations=plan["generations"]))


def child(out, seed):
    with (out / f"s{seed}.log").open("w", encoding="utf-8") as log:
        process = subprocess.run([sys.executable,"-X","utf8","-m","tools.autolab_cem","--out",str(out),"--worker",str(seed)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if process.returncode:
        raise RuntimeError(f"CEM seed {seed} failed")


def run(out, smoke):
    if out.exists():
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True)
    paths = ["tools/autolab_cem.py", "experiments/plan_a/tactical.py", "tools/grade.py",
             "aircombat_gym/wvr/envs/fair.py"]
    plan = dict(algorithm="CEM direct episodic search", seeds=[800] if smoke else [800,801,802],
        population=4 if smoke else 12, elites=2 if smoke else 3, generations=1 if smoke else 6,
        training_band=3000000, training_n=2 if smoke else 6, validation_band=903000,
        validation_n=2 if smoke else 20, parameters=PARAMETERS, lower=LOW.tolist(), upper=HIGH.tolist(),
        hypothesis="Search a small reactive maneuver family directly by completed matches, avoiding critic instability",
        heldout_test="Not opened", source_sha256={n:sha(ROOT/n) for n in paths},
        selection="Training selects distribution; development validation selects saved deployment candidate")
    write(out / "plan.json", plan)
    for name in paths:
        destination = out / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda seed:child(out,seed),plan["seeds"]))
        baseline = evaluate(ANCHOR, plan["validation_band"], plan["validation_n"], fixed=True)
        write(out / "summary.json", dict(records=[read(out/f"s{s}/result.json") for s in plan["seeds"]],
            fixed_baseline=baseline, stage="Development screening only; restricted controller family, not neural PPO"))
        write(out / "completion.json", dict(status="complete"))
    except Exception as error:
        write(out / "completion.json", dict(status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.out.resolve(), args.worker)
    else:
        run(args.out.resolve(), args.smoke)
