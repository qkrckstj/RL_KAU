"""Inspect existing public-observation gate decisions on consumed selection seeds.

Runs only the engagement prefix until the fixed gate decides. No wins/losses
are inferred from these truncated diagnostic prefixes; physics is unchanged.
"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import shutil
from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_matches import MatchEnv
from tools.league_train import verify


def run(source, out):
    if out.exists(): raise FileExistsError('Use a new diagnostic folder')
    if read(source/'completion.json')['status'] != 'no_profile_selection':
        raise ValueError('This diagnostic follows the completed failed selection')
    old=read(source/'plan.json');verify(old)
    names=('refine_2200','gate_s2302_g1_c9','novel_opening_left','ace','evader','circler')
    foes=[f for f in old['opponents'] if f['id'] in names]+old['candidates']
    models=[old['anchor'],old['candidates'][1]]
    sources=dict(old['source_sha256']);sources['tools/league_routing_probe.py']=sha(__file__)
    inputs=dict(old['input_sha256'])
    for path in (source/'plan.json',source/'completion.json'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    plan=dict(models=models,opponents=foes,band=old['selection_band'],n=old['selection_n'],
        source_sha256=sources,input_sha256=inputs,
        scope='Reused selection ICs, both seats, only prefixes until fixed public-motion gate selects. Not complete matches, performance validation, or new unseen evidence. Policy state is observed by the diagnostic only.')
    immutable_json(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    records=[]
    for own in models:
        for foe in foes:
            episodes=[]
            for seat in ('red','blue'):
                env=MatchEnv(own,foe,seat)
                try:
                    for seed in range(plan['band'],plan['band']+plan['n']//2):
                        obs,_=env.reset(seed=seed);policy=env.own_actor.predict.__self__
                        for step in range(30):
                            action=env.own_action(obs)
                            obs,_,term,trunc,info=env.step(action)
                            if policy.selected is not None:
                                episodes.append(dict(seed=seed,seat=seat,selected=int(policy.selected),
                                    features=policy.features.tolist(),gate=policy.gate.tolist(),
                                    margin=float(policy.gate[1:]@policy.features),observed_through_t=info['t']))
                                break
                            if term or trunc: raise ValueError('Unexpected terminal before gate decision')
                        else: raise ValueError('Gate did not select within probe prefix')
                finally: env.close()
            record=dict(own=own['id'],foe=foe['id'],routes=dict(Counter(e['selected'] for e in episodes)),episodes=episodes)
            write(out/f'pair_{len(records):03d}.json',record);records.append(record)
    verify(plan)
    result=dict(status='diagnostic_complete',prefixes=sum(len(r['episodes']) for r in records),full_matches=0,
        routes=[{k:r[k] for k in ('own','foe','routes')} for r in records],scope=plan['scope'])
    write(out/'completion.json',result)
    return result


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(run(a.source.resolve(),a.out.resolve()))
