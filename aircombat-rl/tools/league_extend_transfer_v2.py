"""Fresh-condition archive/peer comparison and first reactive-opponent audit."""
from argparse import ArgumentParser
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools.league_extend_entropy_analysis import verified
from tools.league_tournament_metrics import code_groups
from tools.league_sampled_numpy_qualify import validate_job_result

plain_analyze=engine.analyze


def analyze(plan,rows):
    result=plain_analyze(plan,rows)
    result['heldout_opened']=True
    result['partitions']={}
    for name,ids in plan['partitions'].items():
        opponents=[s for s in plan['opponents'] if s['id'] in ids]
        local=dict(plan,opponents=opponents,groups=code_groups(opponents))
        result['partitions'][name]=plain_analyze(local,[r for r in rows if r['foe'] in ids])
        result['partitions'][name]['heldout_opened']=name=='reactive_untrained'
        result['partitions'][name]['scope']='First use of newly frozen reactive scripts.' if name=='reactive_untrained' else 'Consumed archived/trained policies on fresh98M conditions.'
    return result


def freeze(source,qualification,out):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Learning still live')
    old,raw,done=verified(source)
    comparison=io.read(source/'matched_entropy_analysis.json')
    if comparison['status']!='matched_entropy_comparison_verified':raise ValueError('Matched analysis required')
    for name,digest in comparison['input_sha256'].items():
        if io.sha(io.ROOT/name)!=digest:raise ValueError('Changed comparison')
    panel_path=io.ROOT/'experiments/league/reactive_audit_20261007/panel.json'
    panel=io.read(panel_path);io.verify(panel)
    assert panel['training_source']==io.relative(source) and panel['outcomes_observed'] is False
    qp=io.read(qualification/'plan.json');q=io.read(qualification/'completion.json');io.verify(qp)
    assert q['status']=='sampled_numpy_execution_qualified' and q['exact_episode_and_trace_equality']
    checks=io.ROOT/'runs/sampled_assessment_analysis_checks_20261007.json'
    check=io.read(checks)
    assert check['status']=='passed' and check['source_sha256']==io.sha(Path(engine.__file__))
    partition_checks=io.ROOT/'runs/extend_transfer_v2_analysis_checks_20261007.json'
    checked=io.read(partition_checks)
    assert checked['status']=='passed' and checked['source_sha256']==io.sha(Path(__file__)) and checked['numpy_imports_clean'] and checked['partitions_verified_on_existing_4800_game_data']
    if out.exists():raise FileExistsError(out)
    searches=sorted(done['searches'],key=lambda r:tuple(r['history'][-1]['candidate']['rank']),reverse=True)
    assert len(searches)==2
    candidates=[r['history'][-1]['candidate'] for r in searches]
    initial=old['warm_starts']['prior6'][str(searches[0]['seed'])]
    roles=dict(candidate=[r['spec'] for r in candidates[0]['replicas']],warm=[r['spec'] for r in candidates[1]['replicas']],early=[r['spec'] for r in initial['replicas']],teacher=[old['teacher']])
    peers=roles['candidate']+roles['warm']
    opponents=old['opponents']+peers+panel['opponents']
    assert len({s['id'] for s in opponents})==len(opponents)
    partitions=dict(archive=[s['id'] for s in old['opponents']],learner_peers=[s['id'] for s in peers],reactive_untrained=[s['id'] for s in panel['opponents']])
    band,n=98000000,8
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Fresh conditions reserved')
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for specs in roles.values() for s in specs for f in opponents]
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256']);sources.update(panel['source_sha256'])
    sources[io.relative(Path(__file__).resolve())]=io.sha(Path(__file__))
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256']);inputs.update(panel['input_sha256'])
    for p in [source/'plan.json',source/'completion.json',source/'raw_extend_analysis.json',source/'matched_entropy_analysis.json',panel_path,qualification/'plan.json',qualification/'completion.json',checks,partition_checks]:inputs[io.relative(p)]=io.sha(p)
    for spec in opponents+[s for specs in roles.values() for s in specs]:
        if spec['kind']=='submission':
            for p in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:inputs[io.relative(p)]=io.sha(p)
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),roles=roles,partitions=partitions,opponents=opponents,groups=code_groups(opponents),band=band,n=n,jobs=jobs,
        nominee_learner_seed=searches[0]['seed'],other_learner_seed=searches[1]['seed'],
        role_meanings=dict(candidate='Development-selected entropy-zero final checkpoint, two action RNGs',warm='Other independent entropy-zero final checkpoint, including regression; two action RNGs',early='Untrained initial sampled prior6, two action RNGs',teacher='Pure deterministic extend-right teacher'),
        workers=8,neural_workers=4,approved_signatures=q['approved_signatures'],pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],
        environment=io.environment(),source_sha256=sources,input_sha256=inputs,
        scope='Both independently initialized entropy-zero final policies frozen before outcomes; nominee chosen by prior development rank.120known foes,4new learned peer variants and8previously untrained reactive scripts;4fresh98M ICs/both seats. Report partitions separately. Action RNGs are not independent learners. Reactive scripts are four related heuristics, not real entrants or exhaustive tactics. No policy promotion, no final band, no GitHub upload.',
        next_rule='Judge learned gains against initial sampled policy AND pure teacher on archive and unseen reactive partition. If gains are absent or inconsistent, retain teacher and change to tactical controller parameter learning instead of extending low-level PPO blindly.')
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
    result=analyze(plan,rows);assert result==io.read(out/'analysis.json')
    games=sum(len(r['episodes']) for r in rows);assert games==io.read(out/'completion.json')['games']==7392
    io.write(out/'raw_transfer_analysis.json',dict(status='extend_transfer_raw_verified',games=games,analysis=result,input_sha256={io.relative(p):io.sha(p) for p in paths}))
    print('extend transfer raw verified',games,flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,default=Path('runs/league_sampled_numpy_qualification_20261007'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    engine.freeze=freeze;engine.analyze=analyze
    try:
        engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve())
        verify_results(a.out.resolve())
    except BaseException:
        io.write(a.out/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
