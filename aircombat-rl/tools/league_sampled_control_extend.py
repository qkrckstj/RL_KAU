"""A matched extra-budget probe after the completed three-treatment pilots.

Only control branches continue, including the weaker latest branch. Prior best
policies remain the development references. This is an exploratory budget
decision, not evidence that a robust promotion gate has passed.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import gc, os, shutil, traceback
from tools import league_thread_benchmark as io
from tools.league_sampled_strategy import train_seed


def freeze(source, out, smoke, smoke_run):
    if out.exists(): raise FileExistsError('Use a fresh budget-probe directory')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS): raise ValueError('Set thread limits first')
    if io.process_live(io.read(source/'runtime.json')['pid']): raise RuntimeError('Pilot controller still live')
    old=io.read(source/'plan.json');io.verify(old)
    if io.read(source/'completion.json')['status']!='matched_strategy_pilots_complete': raise ValueError('Completed pilots required')
    analysis=io.read(source/'raw_batch_analysis.json')
    if analysis['status']!='raw_matched_pilots_verified':raise ValueError('Verified raw pilot analysis required')
    for path,digest in analysis['input_sha256'].items():
        if io.sha(io.ROOT/path)!=digest:raise ValueError('Changed raw analysis input')
    control=next(a for a in old['arms'] if a['name']=='control')
    branches=[]
    for seed in old['seeds']:
        result=io.read(source/'control'/f's{seed}'/'result.json')
        latest=result['history'][-1]['candidate']
        branches.append(dict(pilot_seed=seed,warm_start={k:latest[k] for k in ('learner','greedy','replicas')},
                             previous_best=result['selected']))
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for name in ('tools/league_sampled_control_extend.py','tools/league_sampled_strategy_analysis.py'):
        sources[name]=io.sha(io.ROOT/name)
    if not smoke:
        if io.read(smoke_run/'completion.json')['status']!='control_budget_integration_passed' or io.read(smoke_run/'plan.json')['source_sha256']!=sources:
            raise ValueError('Matching completed integration required')
    development=old['development_opponents']
    if smoke:development=[next(s for s in old['opponents'] if s['id']==k) for k in ('ace','ddqn_s0','temporal_extend_right','evader')]
    plan=dict(old,smoke=smoke,source=io.relative(source),branches=branches[:1] if smoke else branches,
        seeds=[5799] if smoke else [5700,5701],training_bands=[170130000] if smoke else [272000000,273000000],
        source_checkpoint_steps=old['chunk_steps'],workers=8 if smoke else 32,processes=2 if smoke else 8,
        ppo=dict(old['ppo'],n_steps=512 if smoke else 128),chunk_steps=32768 if smoke else 1048576,
        base_chunks=1,maximum_chunks=1,coverage_period=control['coverage_period'],probabilities=control['probabilities'],prior_bias=6,
        development_opponents=development,development_band=170140000 if smoke else old['development_band'],
        development_n=2 if smoke else old['development_n'],source_sha256=sources,input_sha256=inputs,
        budget_rule='Exactly one extra matched1M block per control lineage. No automatic unequal extension.',
        quality_scope='Exploratory extra-budget probe: control mean uniform win improved but group mean regressed and the second pilot did not improve. No robust gate passed; preserve all prior best policies.',
        resume_scope='Restore latest control actor/critic/optimizer exactly, then new learner/IC streams. Local steps reset; pilot1M plus this1M is2M per lineage. Not exact trajectory/RNG continuation.',
        experiment_scope='Same consumed82M development panel used for learning-curve comparisons; no fresh final/holdout opened.',
        environment=io.environment())
    extra=[source/'plan.json',source/'completion.json',source/'raw_batch_analysis.json']
    for b in branches:
        extra += [source/'control'/f"s{b['pilot_seed']}"/'result.json',io.ROOT/b['warm_start']['learner']]
        for choice in (b['warm_start'],b['previous_best']):
            for spec in [choice['greedy']]+[r['spec'] for r in choice['replicas']]:
                extra+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    if not smoke:extra += [smoke_run/'plan.json',smoke_run/'completion.json']
    for p in extra:inputs[io.relative(p)]=io.sha(p)
    if not smoke:
        reservations=[io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]
        if any(p.exists() for p in reservations):raise FileExistsError('Training band already reserved')
        for p,b in zip(reservations,plan['training_bands']):io.write(p,dict(run=io.relative(out),start=b,stop_exclusive=b+1000000))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,target)
    return plan


def run(source,out,smoke,smoke_run):
    import torch
    torch.set_num_threads(1)
    if torch.__version__!='2.14.1+cpu':raise ValueError('Qualified CPU environment required')
    plan=freeze(source,out,smoke,smoke_run)
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[]
    try:
        for ordinal,(seed,branch) in enumerate(zip(plan['seeds'],plan['branches'],strict=True)):
            io.verify(plan);gc.collect()
            local=dict(plan,warm_start=branch['warm_start'])
            root=out/f"control_from_s{branch['pilot_seed']}";root.mkdir()
            baseline=None if smoke else branch['previous_best']
            if baseline is not None:io.write(root/'retained_before_extension.json',baseline)
            print('control extra-budget learning start',seed,plan['chunk_steps'],flush=True)
            result,_=train_seed(root,local,seed,ordinal,baseline)
            results.append(dict(result,pilot_seed=branch['pilot_seed'],cumulative_lineage_additional_steps=plan['source_checkpoint_steps']+result['additional_steps']))
            io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan)
        io.write(out/'completion.json',dict(status='control_budget_integration_passed' if smoke else 'control_budget_probe_complete',
            searches=results,additional_training_steps=sum(r['additional_steps'] for r in results),
            shared_prefix_interactions=plan['shared_prefix_interactions'],prior_pilot_training_steps=6291456,
            quality_scope=plan['quality_scope'],final_opened=False,heldout_opened=False,promoted=False))
        print('control extra-budget complete',out,flush=True)
    except BaseException:
        io.write(out/'failure.json',dict(at=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc(),resources=io.resources()))
        raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-run',type=Path,default=Path('runs/league_sampled_control_extend_smoke_20261007'))
    a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.smoke,a.smoke_run.resolve())
