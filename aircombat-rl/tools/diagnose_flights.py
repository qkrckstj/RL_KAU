"""Paired flight traces and a fixed 0.2s action-hold intervention on validation only."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import math
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
import torch
from tools.plan_a import ROOT, timestamp, write_json
from tools.plan_a_shaping import read, sha

MODELS = {
    "starting_good": ("runs/plan_a_20261005/double/ddqn/s1", "checkpoints/step_1024000/policy_net.zip"),
    "continued_bad": ("runs/plan_a_20261005/double_continue/learner", "final_net.zip"),
    "large_buffer_bad": ("runs/plan_a_20261005/replay_compare/buffer_500000", "final_net.zip"),
    "small_buffer_recovered": ("runs/plan_a_20261005/replay_compare/buffer_50000", "final_net.zip"),
}


def worker(out, name, hold):
    from tools import policies
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    from aircombat_gym.wvr import obs as O
    torch.set_num_threads(1)
    d,w=MODELS[name]
    act,_,mode=policies.load(ROOT/d,ROOT/d/w)
    policy=act.__self__
    dest=out/f"{name}_hold{hold}"
    dest.mkdir()
    episodes, traces, shared=[],[],[]
    for seat in ("red","blue"):
        env=FairFightEnv(action_mode=mode,seat=seat)
        try:
            for seed in range(900000,900010):
                obs,info=env.reset(seed=seed)
                steps,changes,reversals,speed_reversals=0,0,0,0
                last_action,last_h,last_v=None,0,0
                counts=Counter()
                qgaps,frames=[],[]
                max_track,entries,streak,short_windows=0.,0,0,0
                speed_sum=0.
                while True:
                    if steps%hold==0:
                        action=int(act(obs))
                    dh,dv,_=env.decode(action)
                    changes+=int(last_action is not None and action!=last_action)
                    reversals+=int(dh*last_h<0)
                    speed_reversals+=int(dv*last_v<0)
                    last_action,last_h,last_v=action,dh,dv
                    counts[action]+=1
                    if steps%10==0:
                        with torch.no_grad():
                            q=policy.model.q_net(torch.as_tensor(policy.state(obs)[None],dtype=torch.float32))
                        top=q.topk(2,dim=1).values[0]
                        qgaps.append(float(top[0]-top[1]))
                        if seed==900000:
                            shared.append(obs.copy())
                    obs,_,term,trunc,info=env.step(action)
                    steps+=1
                    max_track=max(max_track,info["track_time"])
                    speed_sum+=info["own_speed"]
                    if info["in_wez"]:
                        if streak==0:
                            entries+=1
                        streak+=1
                    elif streak:
                        short_windows+=int(streak<20)
                        streak=0
                    if steps%4==0 and seed<900002:
                        ac=env._combat.ac[seat]
                        frames.append(dict(t=info["t"],action=action,dh=dh,dv=dv,
                            **{n:float(obs[O.index(n)]) for n in ("own_x","own_y","opp_x","opp_y","own_health","opp_health")},
                            speed=info["own_speed"],range=info["range"],ata=info["ata"],lead_error=info["ata_lead"],
                            track=info["track_time"],under_track=info["under_track"],
                            heading_target=ac.psi_cmd,speed_target=ac.v_cmd_kt))
                    if term or trunc:
                        if streak:
                            short_windows+=int(streak<20)
                        break
                duration=steps/20
                episodes.append(dict(seed=seed,seat=seat,won=bool(info["won"]),outcome=info["outcome"],duration=duration,
                    changes_per_second=changes/duration,heading_reversals_per_second=reversals/duration,
                    speed_reversals_per_second=speed_reversals/duration,action_counts=dict(counts),
                    wez_time=env._combat.wez_time[seat],enemy_wez_time=env._combat.wez_time[env.foe_seat],max_track=max_track,
                    own_health=info["own_health"],opp_health=info["opp_health"],wez_entries=entries,
                    short_windows=short_windows,mean_speed=speed_sum/steps,mean_q_gap=statistics.mean(qgaps)))
                if frames:
                    traces.append(dict(seed=seed,seat=seat,outcome=info["outcome"],frames=frames))
                write_json(dest/"progress.json",dict(episodes=len(episodes),target=20,updated_utc=timestamp()))
        finally:
            env.close()
    np.save(dest/"shared_obs.npy",np.asarray(shared))
    write_json(dest/"episodes.json",episodes)
    write_json(dest/"traces.json",traces)
    means={k:statistics.mean(e[k] for e in episodes) for k in (
        "changes_per_second","heading_reversals_per_second","speed_reversals_per_second","wez_time",
        "enemy_wez_time","max_track","wez_entries","short_windows","mean_speed","mean_q_gap")}
    write_json(dest/"summary.json",dict(model=name,hold=hold,wins=sum(e["won"] for e in episodes),n=len(episodes),**means))


def child(out,name,hold):
    with (out/f"{name}_hold{hold}.log").open("w",encoding="utf-8") as log:
        p=subprocess.run([sys.executable,"-X","utf8","-m","tools.diagnose_flights","--out",str(out),
            "--model",name,"--hold",str(hold)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if p.returncode:
        raise RuntimeError(f"Flight diagnostics failed: {name} hold={hold}")


def run(out):
    if out.exists():
        raise FileExistsError("Use a new output directory")
    out.mkdir(parents=True)
    write_json(out/"plan.json",dict(created_utc=timestamp(),seeds=list(range(900000,900010)),seats=["red","blue"],
        holds=[1,4],models={n:dict(design=d,weights=w,sha256=sha(ROOT/d/w)) for n,(d,w) in MODELS.items()},
        purpose="Diagnostic intervention on existing validation only; no held-out testing or model reselection"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(child,out,n,h) for n in MODELS for h in (1,4)]
        for f in futures:
            f.result()
    rows=[read(out/f"{n}_hold{h}/summary.json") for n in MODELS for h in (1,4)]
    # Compare decisions on exactly the same public states, not only each policy's own rollouts.
    from tools import policies
    torch.set_num_threads(1)
    obs=np.load(out/"starting_good_hold1/shared_obs.npy")
    actions={}
    for n,(d,w) in MODELS.items():
        act,_,_=policies.load(ROOT/d,ROOT/d/w)
        actions[n]=np.asarray([act(o) for o in obs])
    agreement={n:float((a==actions["starting_good"]).mean()) for n,a in actions.items()}
    write_json(out/"summary.json",dict(rows=rows,shared_state_action_agreement=agreement,shared_states=len(obs)))
    lines=["# Flight diagnostics", "", "Same 20 validation engagements; hold4 is a diagnostic intervention, not retraining.","",
        "| Model | Hold ticks | Wins | Action changes/s | Heading reversals/s | Own WEZ s | Enemy WEZ s | Longest track mean s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['hold']} | {r['wins']}/20 | {r['changes_per_second']:.2f} | {r['heading_reversals_per_second']:.2f} | {r['wez_time']:.2f} | {r['enemy_wez_time']:.2f} | {r['max_track']:.2f} |")
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(out/"completion.json",dict(status="complete",finished_utc=timestamp()))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--hold",type=int,default=1)
    a=p.parse_args()
    (worker(a.out.resolve(),a.model,a.hold) if a.model else run(a.out.resolve()))
