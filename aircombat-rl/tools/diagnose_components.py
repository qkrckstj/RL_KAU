"""Fixed-action baselines and heading/speed swaps, using existing validation only."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tools.plan_a import ROOT,write_json,timestamp
from tools.plan_a_shaping import read

KINDS=("constant_left_decelerate","constant_right_decelerate","bad_heading_good_speed","good_heading_bad_speed")


def worker(out,kind):
    import torch
    from tools.policies import load
    from tools.grade import play,summarise
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    torch.set_num_threads(1)
    good_dir=ROOT/"runs/plan_a_20261005/double/ddqn/s1"
    bad_dir=ROOT/"runs/plan_a_20261005/double_continue/learner"
    good,_,_=load(good_dir,good_dir/"checkpoints/step_1024000/policy_net.zip")
    bad,_,_=load(bad_dir,bad_dir/"final_net.zip")
    def act(obs):
        if kind=="constant_left_decelerate": return 0
        if kind=="constant_right_decelerate": return 6
        ga,ba=good(obs),bad(obs)
        return (ba//3)*3+(ga%3) if kind=="bad_heading_good_speed" else (ga//3)*3+(ba%3)
    rows=[]
    for seat in ("red","blue"):
        env=FairFightEnv(action_mode="discrete",seat=seat)
        try:
            current=play(env,act,900000,10,seat)
            for r in current:r["seat"]=seat
            rows+=current
        finally:env.close()
    write_json(out/f"{kind}.json",dict(kind=kind,summary=summarise(rows),episodes=rows))


def child(out,kind):
    with (out/f"{kind}.log").open("w",encoding="utf-8") as f:
        p=subprocess.run([sys.executable,"-X","utf8","-m","tools.diagnose_components","--out",str(out),"--kind",kind],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if p.returncode:raise RuntimeError(f"Component diagnostic failed: {kind}")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--kind",choices=KINDS)
    a=p.parse_args()
    if a.kind:worker(a.out.resolve(),a.kind)
    else:
        a.out.mkdir(parents=True)
        write_json(a.out/"plan.json",dict(created_utc=timestamp(),kinds=KINDS,band=900000,n=20,
            purpose="Diagnose whether near-constant turning explains apparent learned performance; swap one action channel at a time"))
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda k:child(a.out.resolve(),k),KINDS))
        write_json(a.out/"summary.json",[read(a.out/f"{k}.json") for k in KINDS])
