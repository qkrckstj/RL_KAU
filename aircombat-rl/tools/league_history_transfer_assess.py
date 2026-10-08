"""Freeze observation-arm nominees before fresh-condition paired matches."""
from argparse import ArgumentParser
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools.league_tournament_metrics import code_groups


def freeze(source,qualification,out):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    for p in (source,qualification):
        if io.process_live(io.read(p/'runtime.json')['pid']):raise RuntimeError('Predecessor still live')
    old=io.read(source/'plan.json');io.verify(old)
    complete=io.read(source/'completion.json');raw=io.read(source/'raw_budget_analysis.json')
    if complete['status']!='history_budget_complete' or raw['status']!='history_budget_raw_verified':raise ValueError('Verified completed comparison required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed raw evidence')
    qp=io.read(qualification/'plan.json');q=io.read(qualification/'completion.json');io.verify(qp)
    if q['status']!='sampled_numpy_execution_qualified' or not q['exact_episode_and_trace_equality']:raise ValueError('Qualified loader families required')
    checks_path=io.ROOT/'runs/sampled_assessment_analysis_checks_20261007.json';checks=io.read(checks_path)
    if checks['status']!='passed' or checks['source_sha256']!=io.sha(Path(engine.__file__)):raise ValueError('Aggregation checks mismatch')
    if (out/'plan.json').exists():
        saved=io.read(out/'plan.json');io.verify(saved)
        if saved['source']!=io.relative(source) or saved['qualification']!=io.relative(qualification) or saved['environment']!=io.environment():raise ValueError('Changed resume')
        return saved
    if out.exists():raise FileExistsError('Fresh output required')
    nominees={arm:max((r for r in complete['searches'] if r['arm']==arm),key=lambda r:tuple(r['selected']['rank']))['selected'] for arm in old['arms']}
    initial=old['warm_starts']['control']
    for nominee in [*nominees.values(),initial]:
        if [r['action_seed'] for r in nominee['replicas']]!=[4900,4901]:raise ValueError('Changed action replicas')
    roles=dict(candidate=[r['spec'] for r in nominees['control']['replicas']],warm=[r['spec'] for r in nominees['history']['replicas']],early=[r['spec'] for r in initial['replicas']],teacher=[old['teacher']])
    opponents=list(old['opponents']);band,n=86000000,8
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for specs in roles.values() for s in specs for f in opponents]
    jobs=list({(j['own']['id'],j['foe']['id']):j for j in jobs}.values())
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256'])
    for p in [Path(__file__).resolve(),Path(engine.__file__).resolve(),io.ROOT/'tools/league_history_budget_analysis.py']:sources[io.relative(p)]=io.sha(p)
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for p in [source/'plan.json',source/'completion.json',source/'raw_budget_analysis.json',qualification/'plan.json',qualification/'completion.json',checks_path]:inputs[io.relative(p)]=io.sha(p)
    for spec in [s for specs in roles.values() for s in specs]+opponents:
        if spec['kind']=='submission':
            for p in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:inputs[io.relative(p)]=io.sha(p)
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Conditions reserved')
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),chosen=nominees['control'],nominees=nominees,roles=roles,
        role_meanings=dict(candidate='selected current-state policy',warm='selected causal-history policy',early='shared pre-comparison current-state policy',teacher='preserved CEM teacher'),
        opponents=opponents,groups=code_groups(opponents),band=band,n=n,jobs=jobs,workers=8,neural_workers=4,
        approved_signatures=q['approved_signatures'],pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],
        environment=io.environment(),source_sha256=sources,input_sha256=inputs,
        rule='Each observation-arm nominee fixed by retained development rank before fresh games. Both action RNGs,4ICs/both seats,100 known foes.',
        next_rule='Use fresh transfer/group/weakness evidence to decide further control training versus a credit-horizon repair. No automatic promotion based on development.',
        qualification_scope='Reuse qualified NumPy families; history/other unknown source signatures use ordinary workers. Four ordinary workers avoid routing history to just two; total concurrency remains eight.',
        scope='Fresh initial conditions over known opponents, conditional on shared prefix and two action RNGs. Role warm denotes history, early denotes shared initial policy. No unseen-opponent/final/heldout evidence, promotion or GitHub upload.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),band=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();engine.freeze=freeze
    try:engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve())
    except BaseException:io.write(a.out/f'failure_{time.time_ns()}.json',dict(error=traceback.format_exc()));raise
