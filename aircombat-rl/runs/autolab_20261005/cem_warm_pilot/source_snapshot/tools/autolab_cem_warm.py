"""CEM refinement around a development-verified parent, always retaining it as a candidate.

Independent refinement seeds share one learned parent. This is not independent
from-scratch discovery. Each generation doubles the original training IC count.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from experiments.plan_a.tactical import LOW,HIGH,normalise,ANCHOR
from tools.autolab_cem import ROOT,read,write,sha,evaluate,rank,export


def worker(out,seed):
    p=read(out/"plan.json")
    for name,expected in p["source_sha256"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f"Source changed: {name}")
    if sha(ROOT/p["parent"] / "policy_net.json")!=p["parent_sha256"]:
        raise ValueError("Parent changed")
    d=out/f"s{seed}"
    d.mkdir()
    parent=np.asarray(p["initial_parameters"])
    origin=normalise(parent)
    mean=origin.copy()
    std=np.full(7,p["initial_std"])
    incumbent=origin.copy()
    rng=np.random.default_rng(seed)
    start=time.perf_counter()
    initial=evaluate(parent,p["validation_band"],p["validation_n"])
    validation=[dict(generation=-1,training_steps=0,parameters=parent.tolist(),result=initial)]
    best_rank,best_generation=rank(initial),-1
    export(d,parent)
    export(d/"checkpoints/generation_initial",parent)
    total_steps=total_wins=total_episodes=0
    write(d/"validation.json",validation)
    for generation in range(p["generations"]):
        population=np.clip(rng.normal(mean,std,(p["population"],7)),0,1)
        population[0],population[1],population[2]=mean,incumbent,origin
        # Small continuing global exploration without discarding the parent.
        if p["immigrants"]:
            population[-p["immigrants"]:]=rng.random((p["immigrants"],7))
        band=p["training_band"]+(seed-p["seeds"][0])*10000+generation*100
        scores,candidates=[],[]
        for i,unit in enumerate(population):
            parameters=LOW+unit*(HIGH-LOW)
            result=evaluate(parameters,band,p["training_n"])
            total_steps+=sum(r["steps"] for r in result["episodes"])
            total_wins+=result["summary"]["kills"]
            total_episodes+=p["training_n"]
            scores.append(rank(result))
            candidates.append(dict(index=i,parameters=parameters.tolist(),score=scores[-1],result=result))
            write(d/"progress.json",dict(stage="training",generation=generation,candidate=i+1,
                generations=p["generations"],population=p["population"],training_steps=total_steps,
                train_wins=total_wins,train_episodes=total_episodes,elapsed_seconds=time.perf_counter()-start))
        order=np.argsort(scores)[::-1]
        elites=population[order[:p["elites"]]]
        incumbent=population[order[0]].copy()
        mean=.5*mean+.5*elites.mean(axis=0)
        std=np.maximum(p["minimum_std"],.5*std+.5*elites.std(axis=0))
        parameters=LOW+incumbent*(HIGH-LOW)
        export(d/f"checkpoints/generation_{generation}",parameters)
        result=evaluate(parameters,p["validation_band"],p["validation_n"])
        validation.append(dict(generation=generation,training_steps=total_steps,parameters=parameters.tolist(),result=result))
        if rank(result)>best_rank:
            best_rank,best_generation=rank(result),generation
            export(d,parameters)
        write(d/f"generations/g{generation}.json",dict(candidates=candidates,mean=mean.tolist(),std=std.tolist(),
            rng_state=rng.bit_generator.state,validation=result,training_steps=total_steps))
        write(d/"validation.json",validation)
    export(d/"final",parameters)
    best=next(r for r in validation if r["generation"]==best_generation)["result"]["summary"]
    write(d/"result.json",dict(status="complete",seed=seed,best_generation=best_generation,
        initial_validation=initial["summary"],best_validation=best,final_validation=validation[-1]["result"]["summary"],
        training_steps=total_steps,train_episodes=total_episodes,train_wins=total_wins,
        elapsed_seconds=time.perf_counter()-start,scope="Independent refinement with one shared learned parent"))
    write(d/"progress.json",dict(status="complete",training_steps=total_steps,generations=p["generations"]))


def child(out,seed):
    with (out/f"s{seed}.log").open("w",encoding="utf-8") as log:
        proc=subprocess.run([sys.executable,"-X","utf8","-m","tools.autolab_cem_warm","--out",str(out),"--worker",str(seed)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if proc.returncode:
        raise RuntimeError(f"Warm CEM worker {seed} failed")


def run(out,smoke):
    if out.exists():
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True)
    parent="runs/autolab_20261005/cem_pilot/s800"
    paths=["tools/autolab_cem_warm.py","tools/autolab_cem.py","experiments/plan_a/tactical.py","tools/grade.py","aircombat_gym/wvr/envs/fair.py"]
    p=dict(algorithm="Warm CEM parameter refinement",seeds=[900] if smoke else [900,901,902],
        population=4 if smoke else 12,elites=2 if smoke else 3,generations=1 if smoke else 4,
        training_band=5000000,training_n=2 if smoke else 12,validation_band=905000,validation_n=2 if smoke else 40,
        initial_std=.06,minimum_std=.025,immigrants=0 if smoke else 2,
        parent=parent,parent_sha256=sha(ROOT/parent/"policy_net.json"),
        initial_parameters=read(ROOT/parent/"policy_net.json")["parameters"],
        source_sha256={name:sha(ROOT/name) for name in paths},heldout_test="Not opened",
        hypothesis="Keep a verified parent in every training pool; use twice as many training ICs for candidate selection",
        scope="Independent refinements sharing pilot s800 parent; not independent original discovery")
    write(out/"plan.json",p)
    for name in paths:
        dest=out/"source_snapshot"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda seed:child(out,seed),p["seeds"]))
        summary=dict(records=[read(out/f"s{s}/result.json") for s in p["seeds"]],
            fixed_baseline=evaluate(ANCHOR,p["validation_band"],p["validation_n"],fixed=True))
        write(out/"summary.json",summary)
        write(out/"completion.json",dict(status="complete"))
    except Exception as error:
        write(out/"completion.json",dict(status="failed",error=repr(error)))
        raise


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.worker is not None:
        worker(args.out.resolve(),args.worker)
    else:
        run(args.out.resolve(),args.smoke)
