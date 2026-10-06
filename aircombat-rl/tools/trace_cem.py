"""Record a prechosen development engagement for the frozen policy and control."""
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
from aircombat_gym.wvr.envs.fair import FairFightEnv
from aircombat_gym.wvr import obs as O
from tools.policies import load
from tools.autolab_cem import write, sha


def run(design, out, seed):
    if out.exists():
        raise FileExistsError("Use a new trace directory")
    out.mkdir(parents=True)
    act, _, mode = load(design, design/"policy_net.json")
    write(out/"plan.json",dict(seed=seed,purpose="Prechosen development seed, no model selection",design=str(design),
                              weights_sha256=sha(design/"policy_net.json")))
    traces=[]
    for name, policy in (("learned",act),("fixed_left",lambda obs:0)):
        for seat in ("red","blue"):
            env=FairFightEnv(action_mode=mode,seat=seat)
            try:
                obs, info=env.reset(seed=seed)
                frames=[]
                steps=0
                counts=Counter()
                while True:
                    action=int(policy(obs))
                    counts[action]+=1
                    dh,dv,_=env.decode(action)
                    obs,_,term,trunc,info=env.step(action)
                    steps+=1
                    if steps%4==0 or term or trunc:
                        frames.append(dict(t=info["t"],action=action,dh=dh,dv=dv,
                            **{k:float(obs[O.index(k)]) for k in ("own_x","own_y","opp_x","opp_y","own_health","opp_health")},
                            speed=info["own_speed"],range=info["range"],track=info["track_time"],under_track=info["under_track"]))
                    if term or trunc:
                        break
                traces.append(dict(model=name,seat=seat,seed=seed,outcome=info["outcome"],won=bool(info["won"]),
                    action_counts=dict(counts),steps=steps,wez_time=env._combat.wez_time[seat],frames=frames))
            finally:
                env.close()
    write(out/"traces.json",traces)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--design",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seed",default=904000,type=int)
    a=p.parse_args()
    run(a.design.resolve(),a.out.resolve(),a.seed)
