"""Measure the effect of training-time epsilon on fixed existing validation matches."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from tools.plan_a import ROOT,write_json,timestamp
from tools.plan_a_shaping import read


def worker(out,epsilon,kind):
    from tools.policies import load
    from tools.grade import play,summarise
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    torch.set_num_threads(1)
    d=ROOT/"runs/plan_a_20261005/double/ddqn/s1"
    greedy,_,_=load(d,d/"checkpoints/step_1024000/policy_net.zip")
    rows=[]
    for seat in ("red","blue"):
        env=FairFightEnv(action_mode="discrete",seat=seat)
        try:
            for seed in range(900000,900010):
                rng=np.random.default_rng(seed+1729)
                counts=[0,0]
                def act(obs):
                    u,a=rng.random(),int(rng.integers(9))
                    counts[0]+=1
                    counts[1]+=int(u<epsilon)
                    return a if u<epsilon else (greedy(obs) if kind=="policy" else 0)
                row=play(env,act,seed,1,seat)[0]
                row.update(seat=seat,random_commands=counts[1],commands=counts[0])
                rows.append(row)
        finally:env.close()
    write_json(out/f"{kind}_{epsilon}.json",dict(kind=kind,epsilon=epsilon,summary=summarise(rows),episodes=rows))


def child(out,epsilon,kind):
    with (out/f"{kind}_{epsilon}.log").open("w",encoding="utf-8") as f:
        p=subprocess.run([sys.executable,"-X","utf8","-m","tools.probe_exploration","--out",str(out),
            "--epsilon",str(epsilon),"--kind",kind],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if p.returncode:raise RuntimeError(f"Probe failed: {kind}/{epsilon}")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--epsilon",type=float)
    p.add_argument("--kind",choices=("policy","constant"),default="policy")
    a=p.parse_args()
    if a.epsilon is not None:worker(a.out.resolve(),a.epsilon,a.kind)
    else:
        a.out.mkdir(parents=True)
        write_json(a.out/"plan.json",dict(epsilon=[0.,.005,.05],kinds=["policy","constant"],band=900000,n=20,
            scope="Fixed-policy diagnosis only; no training or fresh test queries",created_utc=timestamp()))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(child,a.out.resolve(),e,k) for k in ("policy","constant") for e in (0.,.005,.05)]
            for f in futures:f.result()
        write_json(a.out/"summary.json",[read(a.out/f"{k}_{e}.json") for k in ("policy","constant") for e in (0.,.005,.05)])
