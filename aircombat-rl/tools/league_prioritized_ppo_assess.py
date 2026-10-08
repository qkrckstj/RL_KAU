"""Fresh-condition full-archive comparison of development-retained PPO controls."""
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
        raise RuntimeError('Predecessor live')
    resources = io.resources()
    if min(resources[k] for k in ('commit_headroom_gib', 'available_memory_gib')) < 6:
        raise MemoryError('Insufficient startup headroom')
    old = io.read(source/'plan.json'); io.verify(old)
    done = io.read(source/'completion.json')
    raw = io.read(source/'raw_prioritized_analysis.json')
    assert raw['status'] == 'prioritized_ppo_raw_verified'
    assert raw['additional_steps'] == done['additional_steps'] == 10485760
    for p, h in raw['input_sha256'].items():
        assert io.sha(io.ROOT/p) == h
    controls = sorted([r for r in done['searches'] if r['arm'] == 'control'],
                      key=lambda r: tuple(r['selected']['rank']), reverse=True)
    assert len(controls) == 2 and all(r['selected']['step'] == 1310720 for r in controls)
    assert all(r['selected'] == done['baseline'] for r in done['searches'] if r['arm'] == 'prioritized')
    qp = io.read(qualification/'plan.json'); io.verify(qp)
    qc = io.read(qualification/'completion.json')
    eq = io.read(qualification/'endpoint_reproduction.json')
    assert qc['status'] == 'sampled_numpy_execution_qualified' and qc['exact_episode_and_trace_equality']
    assert eq['status'] == 'exact_original_expert_reproduction'
    for p, h in eq['input_sha256'].items():
        assert io.sha(io.ROOT/p) == h
    screen_path = io.ROOT/old['source_screen']/'plan.json'
    screen = io.read(screen_path); io.verify(screen)
    roles = dict(candidate=[a['spec'] for a in controls[0]['selected']['replicas']],
                 repeat=[a['spec'] for a in controls[1]['selected']['replicas']],
                 baseline=[a['spec'] for a in done['baseline']['replicas']],
                 teacher=[old['teacher']], cem=screen['roles']['cem'])
    for r in [controls[0]['selected'], controls[1]['selected'], done['baseline']]:
        assert [a['action_seed'] for a in r['replicas']] == [4900, 4901]
    peers = [a['spec'] for r in done['searches'] for h in r['history'] for a in h['candidate']['replicas']]
    opponents = old['opponents'] + peers
    assert len(opponents) == len({s['id'] for s in opponents}) == 173
    own = [s for specs in roles.values() for s in specs]
    assert len(own) == len({s['id'] for s in own}) == 8
    band, n = 107000000, 8
    claim = io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():
        raise FileExistsError('Conditions reserved')
    jobs = [dict(own=s, foe=f, band=band, n=n, traced=False) for s in own for f in opponents]
    checks = io.ROOT/'runs/policy_assessment_metrics_checks_20261007.json'
    check = io.read(checks)
    assert check['status'] == 'passed' and check['source_sha256'] == io.sha(Path(metrics.__file__))
    sources = dict(old['source_sha256']); sources.update(qp['source_sha256'])
    for p in [Path(__file__).resolve(), Path(metrics.__file__).resolve()]:
        sources[io.relative(p)] = io.sha(p)
    inputs = dict(old['input_sha256']); inputs.update(qp['input_sha256'])
    for p in [source/'plan.json', source/'completion.json', source/'raw_prioritized_analysis.json',
              qualification/'plan.json', qualification/'completion.json', qualification/'endpoint_reproduction.json', screen_path, checks]:
        inputs[io.relative(p)] = io.sha(p)
    for s in opponents + own:
        if s['kind'] == 'submission':
            for p in list((io.ROOT/s['design']).glob('*.py')) + [io.ROOT/s['weights']]:
                inputs[io.relative(p)] = io.sha(p)
    pure = sum(all(numpy_capable(j[k], qc['approved_signatures']) for k in ('own', 'foe')) for j in jobs)
    plan = dict(source=io.relative(source), qualification=io.relative(qualification),
        roles=roles, opponents=opponents, groups=code_groups(opponents), band=band, n=n, jobs=jobs,
        workers=16, neural_workers=2, approved_signatures=qc['approved_signatures'],
        pure_warmup=qp['pure_warmup'], neural_warmup=qp['neural_warmup'],
        routing=dict(pure_jobs=pure, ordinary_jobs=len(jobs)-pure), startup_resources=resources,
        partitions=dict(archive=[s['id'] for s in old['opponents']],
                        new_ppo_peers=[s['id'] for s in peers],
                        switching_consumed=[s['id'] for s in old['opponents'] if s['id'].startswith('switching_')]),
        heldout_opponent_ids=[], nominee_seed=controls[0]['seed'], repeat_seed=controls[1]['seed'],
        selected_checkpoints=[r['selected'] for r in controls], source_sha256=sources, input_sha256=inputs,
        environment=io.environment(),
        role_meanings=dict(candidate='Higher development-rank control continuation', repeat='Other control continuation',
            baseline='Shared PPO7000 source; also retained winner of both prioritized branches',
            teacher='Pure extend-right controller', cem='Preserved CEM6800'),
        scope='Fresh107M four initial conditions, both seats, paired action RNGs. Full157 training archive plus16 new first/final checkpoint variants. All original foes consumed; new peers are not independent unseen architectures. Shared-ancestor continuations are not from-scratch replicates. Prioritized branches retained source, so do not duplicate it as extra independent policies. No final-test promotion or GitHub upload.',
        next_rule='Compare both development-selected controls with source across archive, code-group and tails; report peer and consumed-switching partitions separately. Training continuation beyond2chunks failed both arms. If improvement transfers convincingly, confirm with independent initializations and unused opponents before promotion; otherwise preserve all models and change strategy. Do not optimize on these outcomes and label them unseen.')
    io.verify(plan)
    io.write(claim, dict(run=io.relative(out), start=band, stop_exclusive=band+n//2))
    io.write(out/'plan.json', plan)
    for name in sources:
        dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT/name, dest)
    return plan


def verify_results(out):
    from tools.league_residual_train import validate_records
    plan = io.read(out/'plan.json'); io.verify(plan)
    rows = []; paths = [out/'plan.json', out/'completion.json', out/'analysis.json', Path(__file__).resolve()]
    for i, j in enumerate(plan['jobs']):
        p = out/f'matches/match_{i:04d}.json'; d = io.read(p); paths.append(p)
        assert d['job'] == j; validate_job_result(j, d['result']); rows.append(d['result'])
    for specs in plan['roles'].values():
        for s in specs:
            validate_records([r for r in rows if r['own'] == s['id']], plan['opponents'], plan['band'], plan['n'])
    result = metrics.analyze(plan, rows); assert result == io.read(out/'analysis.json')
    games = sum(len(r['episodes']) for r in rows)
    assert games == io.read(out/'completion.json')['games'] == 11072
    io.write(out/'raw_prioritized_comparison.json', dict(status='prioritized_comparison_raw_verified',
        games=games, analysis=result, input_sha256={io.relative(p): io.sha(p) for p in paths}))
    print('prioritized comparison raw verified', games, flush=True)


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--qualification', type=Path, default=Path('runs/league_episode_mixture_qualification_20261008'))
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    engine.freeze = freeze; engine.analyze = metrics.analyze
    try:
        engine.run(a.source.resolve(), a.qualification.resolve(), a.out.resolve()); verify_results(a.out.resolve())
    except BaseException:
        io.write(a.out/f'failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc())); raise
