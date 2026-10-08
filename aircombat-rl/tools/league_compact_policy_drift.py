"""Replay32 consumed games exactly; retain public states for actor-drift diagnosis."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import os
import zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_matches import MatchEnv, Actor
from tools.grade import play
from experiments.league.residual_features import features, numpy_logits


class Probe(MatchEnv):
    def __init__(self, own, foe, seat, teacher):
        self.shadow=Actor(teacher);self.states=[];self.raw_states=[];self.labels=[]
        super().__init__(own,foe,seat)

    def reset(self, *, seed=None, options=None):
        self.shadow.seed(seed,0 if self.seat=='red' else 1);self.shadow.reset()
        self.tick=0;self.episode_seed=seed
        return super().reset(seed=seed,options=options)

    def own_action(self, obs):
        prior=int(self.encode_action(self.shadow.act(self._last[self.seat],obs)))
        if self.tick%20==0:
            self.states.append(features(obs,prior));self.raw_states.append(np.asarray(obs,dtype=np.float32).copy())
            self.labels.append([self.episode_seed,0 if self.seat=='red' else 1,self.tick,prior])
        self.tick+=1
        return super().own_action(obs)


def replay(job,teacher):
    rows=[];states=[];raw=[];labels=[]
    for seat in ('red','blue'):
        env=Probe(job['own'],job['foe'],seat,teacher)
        try:
            part=play(env,env.own_action,job['band'],2,seat)
            rows.extend(dict(r,seat=seat) for r in part)
            states.extend(env.states);raw.extend(env.raw_states);labels.extend(env.labels)
        finally:env.close()
    return rows,np.asarray(states),np.asarray(raw),np.asarray(labels,dtype=np.int64)


def run(source,out):
    assert not out.exists() and all(os.environ.get(k)=='1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    assert min(io.resources()[k] for k in ('commit_headroom_gib','available_memory_gib'))>=6
    p=io.read(source/'plan.json');io.verify(p)
    verified=io.read(source/'raw_budget_screen.json');assert verified['status']=='budget_screen_raw_verified'
    for name,digest in verified['input_sha256'].items():assert io.sha(io.ROOT/name)==digest
    trained=io.read(io.ROOT/p['source']/'plan.json');teacher=trained['teacher']
    ids=['untrained_parameter_01','reactive_brake_pursuit_right','switching_range_close','novel_offset_left']
    own=[p['roles'][k][0] for k in ('baseline','candidate')]
    jobs=[dict(own=s,foe=next(f for f in p['opponents'] if f['id']==key),band=p['band']) for s in own for key in ids]
    models={'baseline':own[0],'rotated':own[1]}
    old=io.read(io.ROOT/'runs/league_compact_inactive_assess_20261008/plan.json')
    models['compact']=old['roles']['candidate'][0]
    paths=[source/'plan.json',source/'raw_budget_screen.json',Path(__file__).resolve()]
    arrays={};teacher_bytes=None
    for key,spec in models.items():
        path=io.ROOT/spec['weights'];paths.append(path)
        with zipfile.ZipFile(path) as z:
            with np.load(BytesIO(z.read('residual_actor.npz')),allow_pickle=False) as data:
                arrays[key]={k:data[k].copy() for k in data.files}
            payload=z.read('parameters.json')
            if teacher_bytes is None:teacher_bytes=payload
            else:assert payload==teacher_bytes
    plan=dict(source=io.relative(source),jobs=jobs,models=models,teacher=teacher,workers=4,
        capture_every_steps=20,episodes=32,scope='Replay consumed113M first2ICs/bothseats on4known foes. Baseline and rotated trajectories; three actors scored on identical captured public states. State-distribution conditional diagnostic,not new performance evidence or proof of causality.',
        input_sha256={io.relative(f):io.sha(f) for f in paths})
    io.write(out/'plan.json',plan);io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
    expected={}
    for f in (source/'matches').glob('match_*.json'):
        d=io.read(f);expected[d['result']['own'],d['result']['foe']]=(f,d['result'])
    blocks=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(replay,j,teacher):(i,j) for i,j in enumerate(jobs)}
        for future in as_completed(futures):
            i,job=futures[future];rows,x,raw,labels=future.result()
            path,old=expected[job['own']['id'],job['foe']['id']]
            wanted=[r for r in old['episodes'] if r['seed']<job['band']+2]
            assert rows==wanted,'Instrumentation changed official episode records'
            f=out/f'states_{i:02d}.npz';np.savez_compressed(f,features=x,observations=raw,labels=labels)
            io.write(out/f'episodes_{i:02d}.json',dict(job=job,episodes=rows,reproduced_from=io.relative(path),source_sha256=io.sha(path)))
            paths.extend([f,out/f'episodes_{i:02d}.json',path]);blocks.append((i,x))
    stats={}
    for label,selected in [('all',blocks),('baseline_trajectories',[(i,x) for i,x in blocks if i<4]),('rotated_trajectories',[(i,x) for i,x in blocks if i>=4])]:
        x=np.concatenate([x for i,x in sorted(selected)]);prior=x[:,-9:].argmax(axis=1)
        probabilities={}
        for name,params in arrays.items():
            logits=numpy_logits(x,params).astype(np.float64);weights=np.exp(logits-logits.max(axis=1,keepdims=True))
            probabilities[name]=weights/weights.sum(axis=1,keepdims=True)
        ref=probabilities['baseline'];entries={}
        for name,probs in probabilities.items():
            kl=np.sum(ref*(np.log(ref)-np.log(probs)),axis=1)
            entries[name]=dict(mean_kl_ref_to_policy=float(kl.mean()),p95_kl=float(np.quantile(kl,.95)),
                mean_total_variation=float((np.abs(ref-probs).sum(axis=1)/2).mean()),
                greedy_disagreement=float((ref.argmax(axis=1)!=probs.argmax(axis=1)).mean()),
                mean_teacher_action_probability=float(probs[np.arange(len(x)),prior].mean()),
                mean_entropy=float(-(probs*np.log(probs)).sum(axis=1).mean()))
        stats[label]=dict(states=len(x),actors=entries)
    io.write(out/'completion.json',dict(status='policy_drift_replay_verified',exact_replayed_games=32,statistics=stats,
        scope=plan['scope'],input_sha256={io.relative(f):io.sha(f) for f in paths+[out/'plan.json']}))
    print('policy drift replay verified',stats,flush=True)


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--source',type=Path,default=Path('runs/league_compact_rotated_assess_20261008'))
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();run(a.source.resolve(),a.out.resolve())
