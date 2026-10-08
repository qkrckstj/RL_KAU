"""Compare development-selected continuations on fresh conditions and switching foes."""
from argparse import ArgumentParser
from pathlib import Path
import os
import shutil
import time
import traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools import league_policy_assessment_metrics as metrics
from tools.league_tournament_metrics import code_groups
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_numpy_matches import numpy_capable


def freeze(source, qualification, out):
    if out.exists():
        raise FileExistsError(out)
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS):
        raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):
        raise RuntimeError('Predecessor still live')
    resources = io.resources()
    if min(resources[k] for k in ('commit_headroom_gib', 'available_memory_gib')) < 6:
        raise MemoryError('Insufficient split16 startup headroom')
    old = io.read(source/'plan.json'); io.verify(old)
    raw = io.read(source/'raw_expanded_analysis.json')
    done = io.read(source/'completion.json')
    assert raw['status'] == 'expanded_ppo_raw_verified'
    assert raw['additional_steps'] == done['additional_steps'] == 7864320
    for p, h in raw['input_sha256'].items():
        assert io.sha(io.ROOT/p) == h
    selected = sorted(raw['searches'], key=lambda r: tuple(r['selected']['rank']), reverse=True)
    assert len(selected) == 2 and all(r['selected']['step'] > 0 for r in selected)
    profile_path = io.ROOT/'experiments/league/execution_profile_extend_20261008.json'
    execution = io.read(profile_path)
    assert execution['qualification'] == io.relative(qualification)
    qp = io.read(qualification/'plan.json'); io.verify(qp)
    q = io.read(qualification/'completion.json')
    assert q['status'] == 'sampled_numpy_execution_qualified' and q['exact_episode_and_trace_equality']
    assert execution['qualification_plan_sha256'] == io.sha(qualification/'plan.json')
    assert execution['qualification_completion_sha256'] == io.sha(qualification/'completion.json')
    assert execution['approved_signatures'] == q['approved_signatures']
    panel_path = io.ROOT/'experiments/league/switching_audit_20261008/panel.json'
    panel = io.read(panel_path); io.verify(panel)
    assert not panel['outcomes_observed'] and len(panel['opponents']) == 8
    previous = io.read(io.ROOT/old['source']/'plan.json'); io.verify(previous)
    roles = dict(candidate=[a['spec'] for a in selected[0]['selected']['replicas']],
                 repeat=[a['spec'] for a in selected[1]['selected']['replicas']],
                 baseline=[a['spec'] for a in done['baseline']['replicas']],
                 teacher=[old['teacher']], cem=previous['roles']['candidate'])
    for r in [selected[0]['selected'], selected[1]['selected'], done['baseline']]:
        assert [a['action_seed'] for a in r['replicas']] == [4900, 4901]
    peers = roles['candidate'] + roles['repeat']
    opponents = old['opponents'] + peers + panel['opponents']
    assert len(opponents) == len({s['id'] for s in opponents}) == 146
    own = [s for specs in roles.values() for s in specs]
    assert len(own) == len({s['id'] for s in own}) == 8
    band, n = 102000000, 8
    claim = io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():
        raise FileExistsError('Conditions already reserved')
    jobs = [dict(own=s, foe=f, band=band, n=n, traced=False) for s in own for f in opponents]
    checks = io.ROOT/'runs/policy_assessment_metrics_checks_20261007.json'
    check = io.read(checks)
    assert check['status'] == 'passed' and check['source_sha256'] == io.sha(Path(metrics.__file__))
    sources = dict(old['source_sha256']); sources.update(qp['source_sha256']); sources.update(panel['source_sha256'])
    for p in [Path(__file__).resolve(), Path(metrics.__file__).resolve()]:
        sources[io.relative(p)] = io.sha(p)
    inputs = dict(old['input_sha256']); inputs.update(qp['input_sha256']); inputs.update(panel['input_sha256'])
    for p in [source/'plan.json', source/'completion.json', source/'raw_expanded_analysis.json',
              qualification/'plan.json', qualification/'completion.json', profile_path, panel_path, checks]:
        inputs[io.relative(p)] = io.sha(p)
    for s in opponents + own:
        if s['kind'] == 'submission':
            for p in list((io.ROOT/s['design']).glob('*.py')) + [io.ROOT/s['weights']]:
                inputs[io.relative(p)] = io.sha(p)
    pure_jobs = sum(all(numpy_capable(j[k], execution['approved_signatures']) for k in ('own','foe')) for j in jobs)
    plan = dict(source=io.relative(source), qualification=io.relative(qualification),
        roles=roles, opponents=opponents, groups=code_groups(opponents), band=band, n=n, jobs=jobs,
        workers=execution['workers'], neural_workers=execution['neural_workers'],
        approved_signatures=execution['approved_signatures'], pure_warmup=execution['pure_warmup'], neural_warmup=execution['neural_warmup'],
        routing=dict(pure_jobs=pure_jobs, ordinary_jobs=len(jobs)-pure_jobs), startup_resources=resources,
        partitions=dict(archive=[s['id'] for s in old['opponents']], new_ppo_peers=[s['id'] for s in peers], switching_untrained=[s['id'] for s in panel['opponents']]),
        heldout_opponent_ids=[s['id'] for s in panel['opponents']],
        nominee_seed=selected[0]['seed'], repeat_seed=selected[1]['seed'], selected_checkpoints=[r['selected'] for r in selected],
        source_sha256=sources, input_sha256=inputs, environment=io.environment(),
        role_meanings=dict(candidate='Higher development-rank continuation, selected before this audit', repeat='Other continuation from same learned ancestor', baseline='Shared starting PPO6700', teacher='Pure extend-right script', cem='Preserved CEM6800 specialist'),
        scope='Fresh102M four ICs, both seats, fixed paired action RNGs.134 known foes,4 new continuation variants,8 previously untrained switching rules over known expert components. Switching rules become consumed after this audit; no claim of wholly unseen architectures or arbitrary entrants. Shared-ancestor continuations are not independent from-scratch repeats. No final-test promotion or GitHub upload.',
        next_rule='Retain all checkpoints. Compare candidate and repeat to starting PPO on archive/group means and lower-tail, then inspect switching and peer partitions separately. Both final extended chunks regressed, so no blind further extension. If transfer is convincing, test independent fresh initializations; if narrow or absent, use newly exposed weakness for a changed strategy. Four ICs are exploratory; use new-condition confirmation for promising gains, never select by final test.')
    io.verify(plan)
    io.write(claim, dict(run=io.relative(out), start=band, stop_exclusive=band+n//2))
    io.write(out/'plan.json', plan)
    for name in sources:
        dest=out/'source_snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(io.ROOT/name,dest)
    return plan


def verify_results(out):
    # Keep Torch imports out of spawned NumPy-only workers.
    from tools.league_residual_train import validate_records
    plan=io.read(out/'plan.json'); io.verify(plan)
    rows=[]; paths=[out/'plan.json',out/'completion.json',out/'analysis.json',Path(__file__).resolve()]
    for i,j in enumerate(plan['jobs']):
        p=out/f'matches/match_{i:04d}.json'; d=io.read(p); paths.append(p)
        assert d['job']==j; validate_job_result(j,d['result']); rows.append(d['result'])
    for specs in plan['roles'].values():
        for s in specs:
            validate_records([r for r in rows if r['own']==s['id']],plan['opponents'],plan['band'],plan['n'])
    result=metrics.analyze(plan,rows); assert result==io.read(out/'analysis.json')
    games=sum(len(r['episodes']) for r in rows)
    assert games==io.read(out/'completion.json')['games']==9344
    io.write(out/'raw_expanded_comparison.json',dict(status='expanded_comparison_raw_verified',games=games,analysis=result,input_sha256={io.relative(p):io.sha(p) for p in paths}))
    print('expanded comparison raw verified',games,flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qualification',type=Path,default=Path('runs/league_extend_numpy_qualification_20261007'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    engine.freeze=freeze;engine.analyze=metrics.analyze
    try:
        engine.run(a.source.resolve(),a.qualification.resolve(),a.out.resolve());verify_results(a.out.resolve())
    except BaseException:
        io.write(a.out/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
