"""Assess both existing extend controllers as possible stronger learning priors."""
from argparse import ArgumentParser
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools.league_tournament_metrics import code_groups


def freeze(source,qualification,out):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    for p in [source,qualification]:
        if io.process_live(io.read(p/'runtime.json')['pid']):raise RuntimeError('Predecessor still live')
    old=io.read(source/'plan.json');io.verify(old);done=io.read(source/'completion.json');raw=io.read(source/'raw_verification.json')
    if done['status']!='recent_round_robin_complete' or raw['status']!='recent_round_robin_raw_verified':raise ValueError('Verified cross-play required')
    for name,digest in raw['input_sha256'].items():
        if io.sha(io.ROOT/name)!=digest:raise ValueError('Changed cross-play evidence')
    qp=io.read(qualification/'plan.json');q=io.read(qualification/'completion.json');io.verify(qp)
    if q['status']!='sampled_numpy_execution_qualified' or not q['exact_episode_and_trace_equality']:raise ValueError('Execution family qualification required')
    checks=io.ROOT/'runs/sampled_assessment_analysis_checks_20261007.json';check=io.read(checks)
    if check['status']!='passed' or check['source_sha256']!=io.sha(Path(engine.__file__)):raise ValueError('Aggregation checks mismatch')
    if (out/'plan.json').exists():
        plan=io.read(out/'plan.json');io.verify(plan)
        if plan['source']!=io.relative(source) or plan['qualification']!=io.relative(qualification) or plan['environment']!=io.environment():raise ValueError('Changed resume')
        return plan
    if out.exists():raise FileExistsError(out)
    opponents=old['expanded_training_archive']
    roles=dict(candidate=old['roles']['baseline'],warm=[next(s for s in opponents if s['id']=='temporal_extend_right')],early=[next(s for s in opponents if s['id']=='temporal_extend_left')],teacher=old['roles']['teacher'])
    band=95000000;n=8;jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for specs in roles.values() for s in specs for f in opponents]
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256']);inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for p in [Path(__file__).resolve(),Path(engine.__file__).resolve()]:sources[io.relative(p)]=io.sha(p)
    for p in [source/'plan.json',source/'completion.json',source/'raw_verification.json',qualification/'plan.json',qualification/'completion.json',checks]:inputs[io.relative(p)]=io.sha(p)
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Teacher assessment conditions reserved')
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),roles=roles,role_meanings=dict(candidate='preserved current PPO, two action RNGs',warm='existing deterministic extend-right controller',early='existing deterministic extend-left controller',teacher='preserved original CEM teacher'),opponents=opponents,groups=code_groups(opponents),band=band,n=n,jobs=jobs,workers=8,neural_workers=4,approved_signatures=q['approved_signatures'],pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],environment=io.environment(),source_sha256=sources,input_sha256=inputs,scope='Fresh95M conditions across all120 archived policies,4ICs/both seats. Both extend directions frozen before results. Role candidate is the CURRENT PPO baseline; comparisons are PPO minus each comparator, so negative gain favors extend/CEM. This assesses existing scripted policies, not new neural training. They were already consumed opponents and are no longer heldout. No automatic promotion, universal tournament claim, final test or GitHub upload.',next_rule='If an extend controller is substantially stronger across this broader archive, qualify it as a new frozen teacher for residual PPO or controller learning while preserving the original CEM. Otherwise retain current source and investigate retention-regularized learning.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,default=Path('runs/league_sampled_numpy_qualification_20261007'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();engine.freeze=freeze
    try:engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve())
    except BaseException:io.write(a.out/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
