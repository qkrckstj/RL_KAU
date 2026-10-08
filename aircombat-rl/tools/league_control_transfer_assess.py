"""Freeze a post-budget nominee, then reuse the qualified paired match runner.

The original runner/aggregator is unchanged; this adapter supplies a new plan
for the completed control-budget experiment and a fresh condition reservation.
"""
from argparse import ArgumentParser
from pathlib import Path
import shutil,time,traceback,os
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools.league_tournament_metrics import code_groups


def freeze(source,qualification,out):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits first')
    for p in (source,qualification):
        if io.process_live(io.read(p/'runtime.json')['pid']):raise RuntimeError('Predecessor still live')
    old=io.read(source/'plan.json');io.verify(old)
    complete=io.read(source/'completion.json');raw=io.read(source/'raw_batch_analysis.json')
    if complete['status']!='control_budget_probe_complete' or raw['status']!='control_budget_raw_verified':raise ValueError('Completed verified budget probe required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed raw budget evidence')
    qp=io.read(qualification/'plan.json');q=io.read(qualification/'completion.json');io.verify(qp)
    if q['status']!='sampled_numpy_execution_qualified' or not q['exact_episode_and_trace_equality']:raise ValueError('Source-family loader qualification required')
    checks_path=io.ROOT/'runs/sampled_assessment_analysis_checks_20261007.json';checks=io.read(checks_path)
    if checks['status']!='passed' or checks['source_sha256']!=io.sha(Path(engine.__file__)):raise ValueError('Matching aggregation checks required')
    if (out/'plan.json').exists():
        saved=io.read(out/'plan.json');io.verify(saved)
        if saved['source']!=io.relative(source) or saved['qualification']!=io.relative(qualification) or saved['environment']!=io.environment():raise ValueError('Changed resume')
        return saved
    if out.exists():raise FileExistsError('Use fresh transfer-assessment folder')
    chosen=max(complete['searches'],key=lambda r:tuple(r['selected']['rank']))['selected']
    initial=old['warm_starts']['6']
    if any([r['action_seed'] for r in c['replicas']]!=[4900,4901] for c in (chosen,initial)):raise ValueError('Changed action replicas')
    roles=dict(candidate=[r['spec'] for r in chosen['replicas']],warm=[r['spec'] for r in initial['replicas']],early=[old['early_greedy_comparator']],teacher=[old['teacher']])
    opponents=list(old['opponents']);band,n=83000000,8
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for specs in roles.values() for s in specs for f in opponents]
    if len({(j['own']['id'],j['foe']['id']) for j in jobs})!=len(jobs):raise ValueError('Duplicate policy jobs')
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256'])
    for name in ('tools/league_control_transfer_assess.py','tools/league_sampled_control_analysis.py','tools/league_sampled_assess.py'):
        sources[name]=io.sha(io.ROOT/name)
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for p in (source/'plan.json',source/'completion.json',source/'raw_batch_analysis.json',qualification/'plan.json',qualification/'completion.json',checks_path):inputs[io.relative(p)]=io.sha(p)
    for spec in [s for specs in roles.values() for s in specs]+opponents:
        if spec['kind']=='submission':
            for p in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:inputs[io.relative(p)]=io.sha(p)
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Fresh conditions already reserved')
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),chosen=chosen,roles=roles,
        opponents=opponents,groups=code_groups(opponents),band=band,n=n,jobs=jobs,workers=8,neural_workers=2,
        approved_signatures=q['approved_signatures'],pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],
        environment=io.environment(),source_sha256=sources,input_sha256=inputs,
        rule='Nominee fixed by retained two-action-replica development rank before any83M games. All100 archived foes; four fresh ICs/both seats; no action-seed selection.',
        next_rule='Both1Mto2M budget probes failed to improve retained development models. Use fresh aggregate/per-foe transfer evidence to freeze the next strategy; no blind same-setting extension.',
        qualification_scope='Reuse inspected/trace-checked NumPy source families, with checkpoint export agreement for new weights. Unknown source signatures stay on ordinary workers. Not a new full episode qualification for every checkpoint.',
        scope='Fresh IC diagnostic over known archived opponents, comparing retained nominee, pre-pilot sampled policy, preserved early greedy policy and CEM teacher. No unseen-opponent/final/heldout test, promotion or GitHub publication.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),band=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    engine.freeze=freeze
    try:engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve())
    except BaseException:
        io.write(a.out/f'failure_{time.time_ns()}.json',dict(error=traceback.format_exc()));raise
