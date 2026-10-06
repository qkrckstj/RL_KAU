"""Development-only interventions on a frozen deployed controller; no selection."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tools.autolab_cem import read,write,sha
from tools.autolab_cem_followup import holdout_job


def run(design,out):
    if out.exists():
        raise FileExistsError("Use a fresh diagnostic directory")
    out.mkdir(parents=True)
    full=read(design/"policy_net.json")["parameters"]
    left=full.copy();left[0]=120.
    slow=full.copy();slow[1]=slow[2]=300.
    pursuit=full.copy();pursuit[0]=0.
    variants=dict(full=full,always_left_same_opening_speed=left,both_speeds_300=slow,no_opening=pursuit)
    write(out/"plan.json",dict(design=str(design),weights_sha256=sha(design/"policy_net.json"),
        variants=variants,band=906000,n=40,purpose="Development-only diagnostic; never updates or selects deployment"))
    jobs=[dict(name=name,parameters=parameters,band=906000,n=40,output=str(out/f"{name}.json"))
          for name,parameters in variants.items()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=dict(pool.map(holdout_job,jobs))
    write(out/"summary.json",{name:result["summary"] for name,result in results.items()})
    write(out/"completion.json",dict(status="complete"))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--design",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run(args.design.resolve(),args.out.resolve())
