"""Predeclare untuned controller controls; score only after the sealed test opens.

These controls distinguish parameter learning from the hand-designed controller
family. They never select a model or affect the preceding search.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time
from experiments.plan_a.tactical import INITIAL, ANCHOR
from tools.autolab_cem import ROOT, read, write, sha
from tools.autolab_cem_followup import holdout_job


def run(followup, out):
    if out.exists():
        raise FileExistsError("Use a new control directory")
    out.mkdir(parents=True)
    controls = dict(untuned_initial=INITIAL.tolist(),untuned_anchor=ANCHOR.tolist())
    write(out/"plan.json",dict(controls=controls, band=2000000, matches=200,
        policy_sha256=sha(ROOT/"experiments/plan_a/tactical.py"),
        purpose="Frozen before heldout, untuned controls; never used for model selection"))
    shutil.copyfile(__file__,out/"source.py")
    while True:
        state = read(followup/"status.json")
        if state["stage"] in ("analysis_required","completed_success"):
            if not state.get("heldout_opened"):
                write(out/"completion.json",dict(status="skipped",reason="Development gate did not pass; heldout remains sealed"))
                return
            break
        if (followup/"error.json").exists():
            write(out/"completion.json",dict(status="skipped",reason="Follow-up errored; inspect before proceeding"))
            return
        time.sleep(5)
    if sha(ROOT/"experiments/plan_a/tactical.py")!=read(out/"plan.json")["policy_sha256"]:
        raise ValueError("Controller source changed after controls were frozen")
    jobs=[dict(name=name,parameters=parameters,band=2000000,n=200,output=str(out/f"{name}.json"))
          for name,parameters in controls.items()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=dict(pool.map(holdout_job,jobs))
    write(out/"summary.json",{name:result["summary"] for name,result in results.items()})
    write(out/"completion.json",dict(status="complete"))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--followup",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run(args.followup.resolve(),args.out.resolve())
