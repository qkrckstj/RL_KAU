"""Freeze CEM nominees before a broad comparison with the preserved PPO."""
from argparse import ArgumentParser
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools import league_policy_assessment_metrics as metrics
from tools.league_tournament_metrics import code_groups
from tools.league_sampled_numpy_qualify import validate_job_result


def freeze(source,qualification,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Learning still live')
    old=io.read(source/'plan.json');io.verify(old);raw=io.read(source/'raw_parameter_analysis.json')
    assert raw['status']=='extend_parameter_raw_verified' and raw['games']==7120
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed raw learning evidence')
    transfer=io.read(io.ROOT/old['source']/'plan.json');io.verify(transfer)
    qp=io.read(qualification/'plan.json');q=io.read(qualification/'completion.json');io.verify(qp)
    assert q['status']=='sampled_numpy_execution_qualified' and q['exact_episode_and_trace_equality']
    checks=io.ROOT/'runs/policy_assessment_metrics_checks_20261007.json';check=io.read(checks)
    assert check['status']=='passed' and check['source_sha256']==io.sha(Path(metrics.__file__))
    selected=sorted(raw['searches'],key=lambda r:tuple(r['rank']),reverse=True)
    assert len(selected)==2
    roles=dict(candidate=[selected[0]['spec']],warm=[selected[1]['spec']],early=transfer['roles']['candidate'],teacher=[old['teacher']])
    peers=[s['spec'] for s in selected];opponents=old['opponents']+peers
    assert len({s['id'] for s in opponents})==len(opponents)
    partitions=dict(archive=[s['id'] for s in old['opponents']],new_cem_peers=[s['id'] for s in peers],reactive_consumed=transfer['partitions']['reactive_untrained'])
    band,n=100000000,16
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Conditions already reserved')
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for specs in roles.values() for s in specs for f in opponents]
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256'])
    for p in [Path(__file__).resolve(),Path(metrics.__file__).resolve()]:sources[io.relative(p)]=io.sha(p)
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for p in [source/'plan.json',source/'completion.json',source/'raw_parameter_analysis.json',qualification/'plan.json',qualification/'completion.json',checks]:inputs[io.relative(p)]=io.sha(p)
    for s in opponents+[s for r in roles.values() for s in r]:
        if s['kind']=='submission':
            for p in list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]:inputs[io.relative(p)]=io.sha(p)
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),roles=roles,opponents=opponents,groups=code_groups(opponents),partitions=partitions,
        nominee_seed=selected[0]['seed'],repeat_seed=selected[1]['seed'],band=band,n=n,jobs=jobs,workers=8,neural_workers=4,
        approved_signatures=q['approved_signatures'],pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),heldout_opponent_ids=[],
        role_meanings=dict(candidate='CEM6800 nominee fixed by development rank',warm='Other independent CEM6801 search nominee',early='Preserved PPO6700, two action RNGs',teacher='Original pure extend-right teacher'),
        scope='Fresh100M8IC/both seats,132consumed archived foes plus2new CEM peers. Deterministic CEM/script policies evaluated once, PPO two action RNGs. Reactive-consumed partition overlaps archive and is no longer heldout. Broader transfer comparison, not final/universal-entrant evidence. No promotion or GitHub upload.',
        next_rule='Compare nominee against preserved PPO and pure teacher on archive mean, group-balanced mean, tail, and paired uncertainty; inspect other independent search separately. If broad gains are absent, retain baselines and use weaknesses for the next learning design. If strong, freeze candidate for untouched opponent/condition confirmation before promotion.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def verify_results(out):
    from tools.league_residual_train import validate_records
    plan=io.read(out/'plan.json');io.verify(plan);rows=[];paths=[out/'plan.json',out/'completion.json',out/'analysis.json',Path(__file__).resolve()]
    for i,j in enumerate(plan['jobs']):
        p=out/f'matches/match_{i:04d}.json';d=io.read(p);paths.append(p)
        assert d['job']==j;validate_job_result(j,d['result']);rows.append(d['result'])
    for specs in plan['roles'].values():
        for s in specs:validate_records([r for r in rows if r['own']==s['id']],plan['opponents'],plan['band'],plan['n'])
    result=metrics.analyze(plan,rows);assert result==io.read(out/'analysis.json')
    games=sum(len(r['episodes']) for r in rows);assert games==io.read(out/'completion.json')['games']==10720
    io.write(out/'raw_comparison_analysis.json',dict(status='extend_parameter_assessment_raw_verified',games=games,analysis=result,input_sha256={io.relative(p):io.sha(p) for p in paths}))
    print('tactical comparison raw verified',games,flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,default=Path('runs/league_sampled_numpy_qualification_20261007'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    engine.freeze=freeze;engine.analyze=metrics.analyze
    try:
        engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve());verify_results(a.out.resolve())
    except BaseException:io.write(a.out/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
