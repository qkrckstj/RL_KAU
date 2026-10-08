"""Qualify three reviewed new families after the frozen comparison exits."""
from argparse import ArgumentParser
from pathlib import Path
import os,shutil,traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_numpy_qualify as engine
from tools.league_numpy_matches import code_signature,numpy_capable


def freeze(source,out):
    if out.exists():raise FileExistsError(out)
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Assessment still live; no concurrent simulations')
    old=io.read(source/'plan.json');io.verify(old)
    raw=io.read(source/'raw_comparison_analysis.json')
    assert raw['status']=='extend_parameter_assessment_raw_verified'
    for name,digest in raw['input_sha256'].items():
        if io.sha(io.ROOT/name)!=digest:raise ValueError('Changed assessment evidence')
    review_path=io.ROOT/'runs/extend_numpy_source_review_20261007.json'
    review=io.read(review_path);representatives=[];pilots=[]
    for row in review['reviewed']:
        s=row['representative'];assert code_signature(s)==row['signature']
        for name,digest in row['source_sha256'].items():assert io.sha(io.ROOT/s['design']/name)==digest
        representatives.append(s)
        for s in row['specs']:
            assert code_signature(s)==row['signature'];pilots.append(s)
    assert len(review['reviewed'])==3 and len(pilots)==14
    approved=sorted(set(old['approved_signatures'])|{r['signature'] for r in review['reviewed']})
    foes={s['id']:s for s in old['opponents']}
    jobs=[]
    for pilot in pilots:
        for foe in [foes['ace'],foes['temporal_extend_left'],old['roles']['early'][1]]:
            jobs.append(dict(own=pilot,foe=foe,band=170420000,n=2,traced=pilot in representatives and foe['id']=='temporal_extend_left'))
    jobs += [dict(own=s,foe=foes['ddqn_s0'],band=170420000,n=2,traced=False) for s in representatives]
    for j in jobs:j['expected_pure_route']=all(numpy_capable(s,approved) for s in [j['own'],j['foe']])
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for p in [Path(__file__).resolve(),Path(engine.__file__).resolve()]:sources[io.relative(p)]=io.sha(p)
    for p in [source/'plan.json',source/'completion.json',source/'raw_comparison_analysis.json',review_path]:inputs[io.relative(p)]=io.sha(p)
    for s in pilots:
        for p in list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]:inputs[io.relative(p)]=io.sha(p)
    plan=dict(source=io.relative(source),jobs=jobs,approved_candidate_signatures=approved,new_candidate_signatures=sorted({r['signature'] for r in review['reviewed']}),
        pure_warmup=old['pure_warmup']+representatives,neural_warmup=old['neural_warmup']+representatives,
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),games_per_trial=sum(j['n'] for j in jobs),scenarios=['ordinary2','split4'],
        rule='Exact official episode/summary equality and selected full damage traces; pure workers must never import Torch. DQN and unreviewed history/hold/portfolio remain ordinary.',
        scope='Execution equivalence only,180game executions. No performance selection, training, final test, speed ranking, global profile change or GitHub upload. Frozen predecessor unchanged.')
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();engine.freeze=freeze
    try:engine.run(a.source.resolve(),a.out.resolve())
    except BaseException:io.write(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
