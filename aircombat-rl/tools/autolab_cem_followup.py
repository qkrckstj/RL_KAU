"""Automatically replicate a promising CEM search, then freeze and test it.

Failed gates produce analysis_required.json, not a success claim. The active
Codex research goal then diagnoses and chooses a different experiment. No test
band is opened unless both development gates pass.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import time
import numpy as np
from tools.autolab_cem import ROOT, read, write, sha, child, evaluate, ANCHOR


def screening_gate(summary):
    rows = summary["records"]
    baseline = summary["fixed_baseline"]["summary"]["rate"]
    best = float(np.mean([r["best_validation"]["rate"] for r in rows]))
    final = float(np.mean([r["final_validation"]["rate"] for r in rows]))
    improved = sum(r["best_validation"]["rate"] > baseline for r in rows)
    passed = best >= .60 and best - baseline >= .15 and final >= .8*best and improved >= 2
    return dict(passed=bool(passed), best_mean=best, final_mean=final, baseline=baseline,
                advantage=best-baseline, retention=final/best if best else 0., improved_seeds=improved)


def holdout_job(job):
    result = evaluate(job["parameters"], job["band"], job["n"], fixed=job["name"]=="fixed")
    write(job["output"], result)
    return job["name"], result


def replication_plan(previous, pilot, warm=False):
    plan=dict(previous)
    plan.update(seeds=[903,904,905] if warm else [803,804,805],
        training_band=6000000 if warm else 4000000, validation_band=906000 if warm else 904000, validation_n=40,
        screening_source=str(pilot), purpose="Independent replication of frozen search method; all controller/search hyperparameters unchanged")
    # `parent` is the learned policy path for warm refinement, not the parent
    # experiment directory. Preserve it and its hash/parameters byte-for-byte.
    return plan


def analyse_holdout(results, seeds):
    baseline = results["fixed"]["summary"]["rate"]
    selected = [results[f"s{s}_selected"] for s in seeds]
    final = [results[f"s{s}_final"] for s in seeds]
    rates = [r["summary"]["rate"] for r in selected]
    best_mean = float(np.mean(rates))
    final_mean = float(np.mean([r["summary"]["rate"] for r in final]))
    # Paired cluster bootstrap: one IC seed is the unit, including both seats
    # and every training replicate. Do not pretend 600 matches are 600 ICs.
    baseline_by_seed = {}
    for row in results["fixed"]["episodes"]:
        baseline_by_seed.setdefault(row["seed"], []).append(int(bool(row["won"])))
    candidate_by_seed = {}
    for result in selected:
        for row in result["episodes"]:
            candidate_by_seed.setdefault(row["seed"], []).append(int(bool(row["won"])))
    differences = np.array([np.mean(candidate_by_seed[s])-np.mean(baseline_by_seed[s]) for s in sorted(baseline_by_seed)])
    rng = np.random.default_rng(9137)
    boot = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
    ci = np.quantile(boot,[.025,.975]).tolist()
    improved = sum(rate>baseline for rate in rates)
    passed = best_mean>=.60 and best_mean-baseline>=.15 and final_mean>=.8*best_mean and improved>=2 and ci[0]>0
    return dict(passed=bool(passed), selected_rates=rates, selected_mean=best_mean,
        final_rates=[r["summary"]["rate"] for r in final], final_mean=final_mean, baseline=baseline,
        advantage=best_mean-baseline, paired_ic_bootstrap_95_ci=ci, improved_seeds=improved,
        retention=final_mean/best_mean if best_mean else 0., independent_training_runs=len(seeds),
        shared_ic_seeds=len(differences), matches_per_policy=len(results["fixed"]["episodes"]),
        caveat="Bootstrap quantifies IC variation conditional on these three training runs, not all possible training runs")


def run(pilot, out, warm=False):
    if out.exists():
        raise FileExistsError("Use a fresh follow-up directory")
    out.mkdir(parents=True)
    write(out/"status.json", dict(stage="waiting_for_pilot", pilot=str(pilot)))
    while not (pilot/"completion.json").exists():
        time.sleep(5)
    if read(pilot/"completion.json")["status"]!="complete":
        raise RuntimeError("Pilot failed; inspect it before scheduling more")
    gate = screening_gate(read(pilot/"summary.json"))
    write(out/"pilot_gate.json",gate)
    if not gate["passed"]:
        write(out/"analysis_required.json",dict(reason="Pilot did not meet preregistered development gate", gate=gate))
        write(out/"status.json",dict(stage="analysis_required", heldout_opened=False))
        return
    # Exact search method, independent RNG and training initial conditions.
    # More development matches improve precision without tuning parameters.
    confirmation = out/"replication"
    confirmation.mkdir()
    plan = read(pilot/"plan.json")
    if warm:
        from tools.autolab_cem_warm import child as run_child
    else:
        run_child=child
    plan=replication_plan(plan,pilot,warm)
    for name,expected in plan["source_sha256"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f"Source changed before replication: {name}")
    write(confirmation/"plan.json",plan)
    shutil.copytree(pilot/"source_snapshot",confirmation/"source_snapshot")
    shutil.copyfile(__file__,out/"followup_source.py")
    write(out/"status.json",dict(stage="replication", heldout_opened=False))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda s:run_child(confirmation,s),plan["seeds"]))
    summary = dict(records=[read(confirmation/f"s{s}/result.json") for s in plan["seeds"]],
        fixed_baseline=evaluate(ANCHOR,plan["validation_band"],plan["validation_n"],fixed=True))
    write(confirmation/"summary.json",summary)
    write(confirmation/"completion.json",dict(status="complete"))
    gate = screening_gate(summary)
    write(out/"replication_gate.json",gate)
    if not gate["passed"]:
        write(out/"analysis_required.json",dict(reason="Independent development replication failed gate", gate=gate))
        write(out/"status.json",dict(stage="analysis_required", heldout_opened=False))
        return
    # Freeze all choices before any held-out episode. This stage consumes
    # band2000000 exactly once, regardless of its success or failure.
    sealed = out/"heldout"
    sealed.mkdir()
    jobs = []
    frozen = []
    for seed in plan["seeds"]:
        for name, directory in (("selected",confirmation/f"s{seed}"),("final",confirmation/f"s{seed}/final")):
            parameters = read(directory/"policy_net.json")["parameters"]
            label = f"s{seed}_{name}"
            frozen.append(dict(name=label,parameters=parameters,policy_sha256=sha(directory/"policy.py"),
                weights_sha256=sha(directory/"policy_net.json"),source=str(directory)))
            jobs.append(dict(name=label,parameters=parameters,band=2000000,n=200,output=str(sealed/f"{label}.json")))
    jobs.append(dict(name="fixed",parameters=ANCHOR.tolist(),band=2000000,n=200,output=str(sealed/"fixed.json")))
    write(sealed/"frozen_selection.json",dict(band=2000000,ic_seeds=100,seats=["red","blue"],policies=frozen,
        criteria="mean>=60%, advantage>=15pp, final>=80% of selected, >=2/3 repeats improve, paired IC bootstrap CI lower>0"))
    claim = ROOT/"runs/autolab_20261005/heldout_band_2000000.claim.json"
    with claim.open("x",encoding="utf-8") as stream:
        json.dump(dict(band=2000000,out=str(sealed),selection_sha256=sha(sealed/"frozen_selection.json")),stream,indent=2)
    write(out/"status.json",dict(stage="heldout",heldout_opened=True,band=2000000))
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(holdout_job,jobs))
    verdict = analyse_holdout(results,plan["seeds"])
    write(sealed/"verdict.json",verdict)
    write(out/"status.json",dict(stage="completed_success" if verdict["passed"] else "analysis_required",
        heldout_opened=True,band=2000000,verdict=verdict))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--warm",action="store_true")
    args=parser.parse_args()
    try:
        run(args.pilot.resolve(),args.out.resolve(),args.warm)
    except Exception as error:
        write(args.out.resolve()/"error.json",dict(error=repr(error)))
        raise
