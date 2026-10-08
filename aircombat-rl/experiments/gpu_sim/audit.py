"""Record GPU throughput separately from original-JSBSim fidelity.

Never treats a failed fidelity gate as a drop-in simulator replacement.
Only previously consumed development initial conditions are used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from aircombat_gym.core.aircraft import Aircraft
from aircombat_gym.wvr.obs import _aircraft
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.league.controller import embed
from tools.league_matches import MatchEnv
from .env import BatchedFairFight,BACKEND_COMMIT
from .geometry import reactive_actions


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def initial(seed):
    ic,_=FairFightEnv.sample(None,np.random.default_rng(seed))
    return [[*ic.xy[s],math.radians(ic.psi_deg[s]),ic.v_kt[s]] for s in range(2)]


def flight_fidelity(out,dtype=torch.float64):
    """Same open-loop actions; three programs, both aircraft, 120 simulated seconds."""
    env=BatchedFairFight(3,dtype=dtype)
    ic=np.array([initial(33030000+i) for i in range(3)])
    refs=[]
    for x,y,psi,v in ic.reshape(-1,4):
        a=Aircraft();a.reset(x,y,psi,v);refs.append(a)
    env.reset(initial=torch.tensor(ic,device='cuda',dtype=dtype))
    env.capture()
    rows=[]
    for step in range(2400):
        acts=[4,8,(step//40)%9]
        for i,a in enumerate(refs):
            k=acts[i//2];a.step((k//3-1)*30,(k%3-1)*20)
        env.step(torch.tensor([[a,a] for a in acts],device='cuda'))
        if step+1 in (1,20,200,1200,2400):
            ref=np.array([_aircraft(a.state) for a in refs]).reshape(3,2,15)
            got=env.raw.cpu().numpy()
            delta=got-ref
            angles=np.arctan2(np.sin(delta[:,:,9:12]),np.cos(delta[:,:,9:12]))
            record=dict(seconds=(step+1)/20,
                position_m=np.linalg.norm(delta[:,:,:3],axis=-1).tolist(),
                speed_ms=np.abs(np.linalg.norm(got[:,:,3:6],axis=-1)-np.linalg.norm(ref[:,:,3:6],axis=-1)).tolist(),
                attitude_deg=np.abs(np.degrees(angles)).max(-1).tolist(),
                invalid=env.invalid.tolist(),done=env.done.tolist(),reference=ref.tolist(),gpu=got.tolist())
            rows.append(record)
            print('FLIGHT',record['seconds'],'max position',np.max(record['position_m']),flush=True)
    # Predeclared conservative drop-in criteria, applied across all measured horizons.
    gate=all(np.max(r['position_m'])<=10 and np.max(r['speed_ms'])<=.514444
             and np.max(r['attitude_deg'])<=1 and not any(r['invalid']) and not any(r['done']) for r in rows)
    result=dict(programs=['hold','right_accelerate','nine_actions_2s'],initial=ic.tolist(),
        dtype=str(dtype),thresholds=dict(position_m=10,speed_kt=1,attitude_deg=1),passed=bool(gate),runs=rows)
    write(out/'flight_fidelity.json',result)
    return result


def game_fidelity(out,dtype=torch.float64):
    root=Path(__file__).resolve().parents[2]
    final=json.loads((root/'experiments/league/bundle/models/final/policy_net.json').read_text())['parameters']
    old=json.loads((root/'experiments/plan_a/cem_policy/policy_net.json').read_text())
    old=old['parameters']
    if len(old)==7: old=embed(old).tolist()
    own=dict(id='frozen_final',kind='reactive',parameters=final)
    foe=dict(id='original_cem',kind='reactive',parameters=old)
    cases=[(seed,seat) for seed in (33030000,33030001) for seat in ('red','blue')]
    rows=[]
    for seed,seat in cases:
        ref=MatchEnv(own,foe,seat)
        try:
            obs,_=ref.reset(seed=seed)
            for steps in range(1,2402):
                obs,_,term,trunc,info=ref.step(ref.own_action(obs))
                if term or trunc: break
            rows.append(dict(seed=seed,seat=seat,cpu_outcome=info.get('outcome'),cpu_steps=steps,
                             cpu_health=[ref._combat.health[s] for s in ('red','blue')]))
        finally: ref.close()
    gpu=BatchedFairFight(len(cases),dtype=dtype)
    gpu.reset(initial=torch.tensor([initial(seed) for seed,_ in cases],device='cuda',dtype=dtype))
    gpu.capture()
    params=torch.tensor([[final,old] if seat=='red' else [old,final] for _,seat in cases],device='cuda',dtype=dtype)
    first_done=torch.zeros(len(cases),device='cuda',dtype=torch.int64)
    for step in range(1,2402):
        gpu.step(reactive_actions(gpu.obs,params))
        first_done=torch.where(gpu.done & (first_done==0),step,first_done)
    codes=gpu.outcome.tolist()
    for i,(_,seat) in enumerate(cases):
        code=codes[i]
        outcome={0:'unfinished',2:'mutual',3:'timeout',4:'invalid'}.get(code)
        if outcome is None: outcome='kill' if ((code==1)==(seat=='red')) else 'died'
        rows[i].update(gpu_outcome=outcome,gpu_steps=int(first_done[i]),gpu_health=gpu.health[i].tolist(),
                       same_verdict=outcome==rows[i]['cpu_outcome'])
    result=dict(dtype=str(dtype),scope='Two consumed development ICs, both seats; not unseen policy evaluation',
                agreement=sum(r['same_verdict'] for r in rows),games=len(rows),runs=rows)
    write(out/'game_fidelity.json',result)
    return result


class Hold:
    def reset(self): pass
    def act(self,*args): return (0.,0.,0.)


def throughput(out,batches,steps,repeats):
    """Identical hold commands, full 2-aircraft environment decisions, no policy/learning."""
    cpu=[];rows=[]
    for repeat in range(repeats):
        env=FairFightEnv()
        try:
            env.reset(seed=33030000);env._foe=Hold()
            for _ in range(20): env.step(4)
            env.reset(seed=33030000);env._foe=Hold()
            start=time.perf_counter()
            for _ in range(steps): env.step(4)
            cpu.append(time.perf_counter()-start)
        finally: env.close()
    for n in batches:
        start=time.perf_counter()
        env=BatchedFairFight(n)
        ic=torch.tensor(initial(33030000),device='cuda').expand(n,2,4)
        env.reset(initial=ic);env.capture()
        action=torch.full((n,2),4,device='cuda',dtype=torch.int64)
        for _ in range(20): env.step(action)
        torch.cuda.synchronize()
        setup=time.perf_counter()-start
        timings=[]
        for repeat in range(repeats):
            env.reset(initial=ic)
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(steps): env.step(action)
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        if bool(env.invalid.any()) or not bool(torch.isfinite(env.obs).all()):
            raise RuntimeError('Invalid GPU benchmark states')
        median=statistics.median(timings)
        record=dict(environments=n,aircraft=2*n,decision_steps=steps,env_steps=n*steps,
            physics_frames=n*steps*12,seconds=timings,median_seconds=median,
            setup_seconds=setup,env_steps_per_second=n*steps/median,
            vs_serial_cpu=n*statistics.median(cpu)/median)
        rows.append(record)
        print('BENCH',json.dumps(record),flush=True)
    result=dict(scope='Full environment stepping, fixed actions; excludes policy and optimizer. Different physics backend, not an equal-fidelity guarantee.',
        cpu_seconds=cpu,cpu_env_steps_per_second=steps/statistics.median(cpu),steps_per_run=steps,runs=rows)
    write(out/'throughput.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batches',nargs='+',type=int,default=[64,256,1024,4096])
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--fidelity-only',action='store_true')
    a=p.parse_args(argv)
    if not 1<=a.steps<=2000 or a.repeats<1 or min(a.batches)<1: p.error('Invalid benchmark sizes')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    sources={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    write(a.out/'manifest.json',dict(backend_commit=BACKEND_COMMIT,source_sha256=sources,
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),fidelity_thresholds_declared_before_run=True))
    flight=flight_fidelity(a.out,getattr(torch,a.dtype))
    games=game_fidelity(a.out,getattr(torch,a.dtype))
    timing=None if a.fidelity_only else throughput(a.out,a.batches,a.steps,a.repeats)
    write(a.out/'completion.json',dict(status='audit_complete',drop_in_fidelity_passed=flight['passed'] and games['agreement']==games['games'],
        official_backend_unchanged=True,fastest_simulator_throughput=None if timing is None else max(r['env_steps_per_second'] for r in timing['runs'])))


if __name__=='__main__': main()
