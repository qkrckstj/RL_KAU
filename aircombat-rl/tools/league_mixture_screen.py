"""Fresh-condition stratified screen of episode mixture and matched PPO controls."""
from argparse import ArgumentParser
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools import league_policy_assessment_metrics as metrics
from tools.league_tournament_metrics import code_groups,group_weights
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_numpy_matches import numpy_capable


def analyze(plan,rows):
    result=metrics.analyze(plan,rows)
    reference={}
    for foe in plan['opponents']:
        name=foe['id'];a=result['roles']['teacher']['opponents'][name];b=result['roles']['cem']['opponents'][name]
        reference[name]=dict(win=.6*a['wins']/a['games']+.4*b['wins']/b['games'],
            score=.6*(a['wins']+.5*a['draws'])/a['games']+.4*(b['wins']+.5*b['draws'])/b['games'])
    weights=group_weights(plan['groups'])
    result['component_weighted_reference']=dict(opponents=reference,
        mean_win=sum(v['win'] for v in reference.values())/len(reference),
        group_win=sum(weights[k]*v['win'] for k,v in reference.items()),
        worst_score=min(v['score'] for v in reference.values()),
        scope='60/40 arithmetic weighting of separately flown deterministic components. Not the observed win rate of either fixed action-seed mixture; report actual candidate separately. No performance fitting on this screen.')
    return result


def freeze(source,qualification,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    for p in [source,qualification]:
        if io.process_live(io.read(p/'runtime.json')['pid']):raise RuntimeError('Predecessor live')
    if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<6:raise MemoryError('Split16 headroom')
    old=io.read(source/'plan.json');io.verify(old);raw=io.read(source/'raw_discount_analysis.json');done=io.read(source/'completion.json')
    assert raw['status']=='discount_raw_verified'
    for p,h in raw['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    matched=io.read(source/'matched_discount_analysis.json');assert matched['status']=='matched_discount_comparison_verified'
    for p,h in matched['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    qp=io.read(qualification/'plan.json');io.verify(qp);qc=io.read(qualification/'completion.json');eq=io.read(qualification/'endpoint_reproduction.json')
    assert qc['status']=='sampled_numpy_execution_qualified' and qc['exact_episode_and_trace_equality'] and eq['status']=='exact_original_expert_reproduction'
    for p,h in eq['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    manifest_path=io.ROOT/qp['source']/'manifest.json';manifest=io.read(manifest_path);io.verify(manifest)
    controls=sorted([r for r in done['searches'] if r['arm']=='control'],key=lambda r:tuple(r['history'][0]['candidate']['rank']),reverse=True)
    assert len(controls)==2
    roles=dict(candidate=[v['spec'] for v in manifest['variants'][:2]],
        ppo_new=[a['spec'] for a in controls[0]['history'][0]['candidate']['replicas']],
        ppo_repeat=[a['spec'] for a in controls[1]['history'][0]['candidate']['replicas']],
        baseline=[a['spec'] for a in done['baseline']['replicas']],teacher=[manifest['original_experts'][0]],cem=[manifest['original_experts'][1]])
    peers=[a['spec'] for r in done['searches'] for a in r['history'][0]['candidate']['replicas']]
    archive=old['opponents']+peers;assert len({s['id'] for s in archive})==len(archive)==154
    prior=io.read(io.ROOT/old['source']/'raw_expanded_comparison.json')['analysis']['roles']['candidate']['opponents']
    weak=sorted(prior,key=lambda k:((prior[k]['wins']+.5*prior[k]['draws'])/prior[k]['games'],k))[:12]
    required=set(weak+[s['id'] for s in peers]+[s['id'] for s in archive if s['id'].startswith('switching_')])
    groups=code_groups(archive);selected=set(required)
    for group in sorted(set(groups.values())):
        if not any(groups[k]==group for k in selected):selected.add(next(s['id'] for s in archive if groups[s['id']]==group))
    target=max(48,len(selected))
    for s in archive:
        if len(selected)>=target:break
        selected.add(s['id'])
    opponents=[s for s in archive if s['id'] in selected];band,n=104000000,8
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Conditions reserved')
    own=[s for specs in roles.values() for s in specs];assert len({s['id'] for s in own})==len(own)==10
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for s in own for f in opponents]
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256'])
    for p in [Path(__file__).resolve(),Path(metrics.__file__).resolve()]:sources[io.relative(p)]=io.sha(p)
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for p in [source/'plan.json',source/'completion.json',source/'raw_discount_analysis.json',source/'matched_discount_analysis.json',qualification/'plan.json',qualification/'completion.json',qualification/'endpoint_reproduction.json',manifest_path]:inputs[io.relative(p)]=io.sha(p)
    for s in archive+own:
        if s['kind']=='submission':
            for p in list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]:inputs[io.relative(p)]=io.sha(p)
    pure=sum(all(numpy_capable(j[k],qc['approved_signatures']) for k in ['own','foe']) for j in jobs)
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),roles=roles,opponents=opponents,groups=code_groups(opponents),full_retained_archive=archive,
        partitions=dict(old_archive_selected=[s['id'] for s in opponents if s['id'] not in {x['id'] for x in peers}],new_pilot_peers=[s['id'] for s in peers],switching_consumed=[s['id'] for s in opponents if s['id'].startswith('switching_')]),
        band=band,n=n,jobs=jobs,workers=16,neural_workers=2,approved_signatures=qc['approved_signatures'],pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],
        heldout_opponent_ids=[],source_sha256=sources,input_sha256=inputs,environment=io.environment(),routing=dict(pure=pure,ordinary=len(jobs)-pure),
        frozen_ppo_nominee_seed=controls[0]['seed'],selection_rule='Prior consumed102M worst12 plus8new pilot peers plus8consumed switching rules,at least one per source-code group,then archive-order fill to48. All154 retained;selected subset is a diagnostic screen,not a full-archive win estimate.',
        scope='Fresh104M fourICs/both seats. Episode mixture60/40 fixed using prior102M;PPO control nominees fixed by103M development;both conditional training repeats retained. Deterministic experts once,stochastic policies two action replicas. All opponents known/consumed. No unseen/final/promotion/GitHub claim.',
        next_rule='Compare actual mixture versus both new PPOs,source PPO and components;report arithmetic component reference separately. If mean/tail tradeoff improves on fresh conditions,confirm broader archive and untouched opponents before promotion. Otherwise preserve policies and change strategy;do not tune mixture on this screen and call it unseen.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def verify_results(out):
    from tools.league_residual_train import validate_records
    p=io.read(out/'plan.json');io.verify(p);rows=[];paths=[out/'plan.json',out/'completion.json',out/'analysis.json',Path(__file__).resolve()]
    for i,j in enumerate(p['jobs']):
        f=out/f'matches/match_{i:04d}.json';d=io.read(f);paths.append(f)
        assert d['job']==j;validate_job_result(j,d['result']);rows.append(d['result'])
    for specs in p['roles'].values():
        for s in specs:validate_records([r for r in rows if r['own']==s['id']],p['opponents'],p['band'],p['n'])
    result=analyze(p,rows);assert result==io.read(out/'analysis.json')
    games=sum(len(r['episodes']) for r in rows);assert games==len(p['jobs'])*p['n']==io.read(out/'completion.json')['games']
    io.write(out/'raw_mixture_screen.json',dict(status='mixture_screen_raw_verified',games=games,analysis=result,input_sha256={io.relative(f):io.sha(f) for f in paths}))
    print('mixture screen raw verified',games,flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,default=Path('runs/league_episode_mixture_qualification_20261008'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();engine.freeze=freeze;engine.analyze=analyze
    try:engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve());verify_results(a.out.resolve())
    except BaseException:io.write(a.out/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
