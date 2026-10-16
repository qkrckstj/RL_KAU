"""Prepare an explicitly verified local continuation of a completed league batch."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

from tools.seven_model_league_v3 import ROOT, rank, reserve, verify_hashes, authenticated_result, model_identity, specs
from experiments.rear_async.storage import read,write,sha,verify,identity


def continuation_choice(initial,baseline,results):
    parent=initial;best=baseline;stalled=0
    for row in results:
        if rank(row['score'])>rank(best):
            parent=row['checkpoint'];best=row['score'];stalled=0
        else:stalled+=1
    return dict(parent=parent,score=best,stalled=stalled,learning_rate_factor=.5 if stalled>=2 else 1.)


def validate_panel_binding(result,checkpoint,pool,screen_band):
    p=result['protocol']
    required=dict(stage='screen',smoke=False,opponents=[x['id'] for x in pool],
                  seeds=list(range(screen_band,screen_band+8)),seats=['red','blue'],
                  action_seeds=[4900,4901],pool_id=identity(pool))
    if any(p.get(k)!=v for k,v in required.items()):
        raise ValueError('Screen protocol is not bound to expected pool/conditions')
    inputs={str(f):sha(f) for entry in [dict(sampled_specs=specs(checkpoint)),*pool]
            for spec in entry['sampled_specs'].values() for f in Path(spec['design']).iterdir() if f.is_file()}
    if result['input_hashes']!=inputs:
        raise ValueError('Evaluation inputs do not match claimed candidate and pool')


def prepare(source,out,training_band,confirmation_band,seed,rounds):
    import psutil
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Use a fresh local runs directory')
    if not 1<=rounds<=4:raise ValueError('Continuation requires a bounded1..4iteration review')
    with (source/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (source/'failure.json').exists() or not (source/'batch_complete.json').exists():
            raise ValueError('Source batch not successfully terminal')
        for p in psutil.process_iter(['cmdline','status']):
            args=p.info['cmdline'] or []
            if p.info['status']!=psutil.STATUS_ZOMBIE and any(a.startswith('tools.seven_model_league') for a in args) and any(a in args for a in ('run','train')):
                raise RuntimeError('Another league controller/trainer is live')
        previous=read(source/'config.json');verify_hashes(previous['frozen_sha256'])
        if previous['smoke']:raise ValueError('Smoke cannot select a production parent')
        pool=read(source/'pool.json')
        if pool!=previous['initial_pool'] or len(pool)!=7:raise ValueError('Expanded pool requires its own verified baseline plan')
        base=authenticated_result(source/'baseline');results=[]
        validate_panel_binding(base,Path(previous['initial_parent']),pool,previous['screen_band'])
        paths=sorted(source.glob('iteration_*/result.json'))
        if len(paths)!=previous['rounds']:raise ValueError('Incomplete bounded batch')
        for ordinal,f in enumerate(paths):
            d=read(f);screen=authenticated_result(f.parent/'screen');t=read(f.parent/'training_complete.json')
            if d['screen']!=screen or d['training']!=t or d['iteration']!=ordinal:raise ValueError('Changed result accounting')
            checkpoint=Path(t['checkpoint']);verify(checkpoint)
            validate_panel_binding(screen,checkpoint,pool,previous['screen_band'])
            if screen['protocol']['seeds']!=list(range(previous['screen_band'],previous['screen_band']+8)):
                raise ValueError('Source used another screen panel')
            if screen['protocol']['pool_id']!=identity(pool):raise ValueError('Source pool changed')
            results.append(dict(checkpoint=str(checkpoint),score=screen['score']))
        choice=continuation_choice(previous['initial_parent'],base['score'],results)
        if choice['parent']!=read(paths[-1])['selected_parent']:raise ValueError('Parent selection differs from audited evidence')
        parent=Path(choice['parent']);verify(parent)
        config=copy.deepcopy(previous)
        config.update(rounds=rounds,training_band=training_band,confirmation_band=confirmation_band,seed=seed,
            initial_parent=str(parent),initial_pool=pool,
            continuation=dict(source=str(source),source_config_sha256=sha(source/'config.json'),selection=choice,
                screen_scope='Previously consumed screen reused; fresh baseline replay is counted as evaluation, never unseen evidence.',
                hypothesis='After consecutive non-improvement, smaller PPO updates may retain the achieved opponent pass while reducing regression. Otherwise extend the improving parent without copying historical schedules.'))
        config['ppo']['learning_rate']=previous['ppo']['learning_rate']*choice['learning_rate_factor']
        if not 1e-7<=config['ppo']['learning_rate']<=3e-4:raise ValueError('Learning-rate floor reached; select a different hypothesis')
        # Screen reservation intentionally belongs to the source and is explicitly consumed.
        ranges=[('continuation_training',training_band,training_band+rounds*1000000),
                ('fresh_confirmation',confirmation_band,confirmation_band+rounds*32)]
        screen_range=(previous['screen_band'],previous['screen_band']+8)
        if any(a<screen_range[1] and b>screen_range[0] for _,a,b in ranges):raise ValueError('New range overlaps screen')
        reserve(out,ranges)
        files=list(map(Path,previous['frozen_sha256']))
        files += [Path(__file__).resolve(),ROOT/'tools/seven_model_league_v3.py',source/'config.json',source/'batch_complete.json']
        files += paths
        files += [p for p in parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        config['frozen_sha256']={str(p):sha(p) for p in files if '__pycache__' not in p.parts}
        out.mkdir();write(out/'config.json',config);write(out/'pool.json',pool)
        write(out/'prepared.json',dict(time=time.time(),parent_identity=model_identity(parent),choice=choice,
            source_completed_iterations=len(paths),screen_replayed_not_unseen=True,training_started=False))
        return config


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--training-band',type=int,required=True);p.add_argument('--confirmation-band',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--rounds',type=int,default=2);a=p.parse_args()
    prepare(a.source.resolve(),a.out.resolve(),a.training_band,a.confirmation_band,a.seed,a.rounds)
