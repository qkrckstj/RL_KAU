"""Eight-IC diagnostic screen following unchanged-settings PPO extra budget."""
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


def unique_specs(roles):
    specs = {}
    for values in roles.values():
        for s in values:
            if s['id'] in specs:
                assert specs[s['id']] == s, 'Role ID collision'
            specs[s['id']] = s
    return list(specs.values())


def freeze(source, qualification, out):
    if out.exists():
        raise FileExistsError(out)
    assert all(os.environ.get(k) == '1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    assert min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) >= 6
    old = io.read(source/'plan.json'); io.verify(old)
    done = io.read(source/'completion.json'); raw = io.read(source/'raw_rate_analysis.json')
    assert not old['smoke'] and raw['status'] == 'rate_ppo_raw_verified'
    assert done['additional_steps'] == raw['additional_steps'] == 5242880
    for p, h in raw['input_sha256'].items():
        assert io.sha(io.ROOT/p) == h
    protocol_path = io.ROOT/'experiments/league/rate_budget_screen_protocol_20261008.json'
    protocol = io.read(protocol_path); io.verify(protocol)
    assert protocol['training_run'] == io.relative(source)
    assert protocol['source_assessment'] == old['source']
    prior = io.read(io.ROOT/old['source']/'plan.json'); io.verify(prior)
    qp = io.read(qualification/'plan.json'); io.verify(qp)
    qc = io.read(qualification/'completion.json'); eq = io.read(qualification/'endpoint_reproduction.json')
    assert qc['status'] == 'sampled_numpy_execution_qualified' and qc['exact_episode_and_trace_equality']
    assert eq['status'] == 'exact_original_expert_reproduction'
    for p, h in eq['input_sha256'].items():
        assert io.sha(io.ROOT/p) == h
    searches = sorted(done['searches'], key=lambda r: tuple(r['selected']['rank']), reverse=True)
    assert len(searches) == 2 and {r['seed'] for r in searches} == {7600, 7601}
    roles = dict(candidate=[a['spec'] for a in searches[0]['selected']['replicas']],
                 repeat=[a['spec'] for a in searches[1]['selected']['replicas']],
                 baseline=[a['spec'] for a in done['baseline']['replicas']],
                 teacher=[old['teacher']], cem=prior['roles']['cem'])
    for c in [r['selected'] for r in searches]+[done['baseline']]:
        assert [a['action_seed'] for a in c['replicas']] == protocol['action_replicas']
    peers = [a['spec'] for r in done['searches'] for h in r['history'] for a in h['candidate']['replicas']]
    assert len(peers) == 8
    opponents = protocol['existing_opponents']+peers
    assert len(opponents) == len({s['id'] for s in opponents}) == 56
    assert len(set(code_groups(protocol['existing_opponents']).values())) == protocol['code_groups_covered'] == 27
    own = unique_specs(roles)
    band, n = protocol['proposed_band'], protocol['initial_conditions']*2
    assert n == 16
    jobs = [dict(own=s, foe=f, band=band, n=n, traced=False) for s in own for f in opponents]
    assert len(jobs)*n <= protocol['maximum_games'] == 7168
    claim = io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    assert not claim.exists(), 'Conditions already reserved'
    sources = dict(old['source_sha256']); sources.update(qp['source_sha256'])
    inputs = dict(old['input_sha256']); inputs.update(qp['input_sha256'])
    for p in [Path(__file__).resolve(), Path(metrics.__file__).resolve()]:
        sources[io.relative(p)] = io.sha(p)
    extra = [source/'plan.json', source/'completion.json', source/'raw_rate_analysis.json',
             protocol_path, qualification/'plan.json', qualification/'completion.json',
             qualification/'endpoint_reproduction.json']
    for s in opponents+own:
        if s['kind'] == 'submission':
            extra += list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
    for p in extra:
        inputs[io.relative(p)] = io.sha(p)
    pure = sum(all(numpy_capable(j[k], qc['approved_signatures']) for k in ('own', 'foe')) for j in jobs)
    plan = dict(source=io.relative(source), qualification=io.relative(qualification),
        roles=roles, selected_checkpoints=[r['selected'] for r in searches],
        nominee_seed=searches[0]['seed'], repeat_seed=searches[1]['seed'],
        opponents=opponents, groups=code_groups(opponents), full_retained_archive=prior['opponents']+peers,
        partitions=dict(existing_screen=[s['id'] for s in protocol['existing_opponents']],
            new_checkpoint_peers=[s['id'] for s in peers],
            parameter_combinations_consumed=prior['partitions']['untrained_parameters'],
            switching_consumed=prior['partitions']['switching_consumed']),
        heldout_opponent_ids=[], band=band, n=n, jobs=jobs, workers=16, neural_workers=2,
        approved_signatures=qc['approved_signatures'], pure_warmup=qp['pure_warmup'], neural_warmup=qp['neural_warmup'],
        routing=dict(pure_jobs=pure, ordinary_jobs=len(jobs)-pure),
        source_sha256=sources, input_sha256=inputs, environment=io.environment(),
        selection_protocol=io.relative(protocol_path),
        scope='Eight fresh110M ICs,both seats,paired action RNGs.48previously observed foes covering27code groups plus8new checkpoint peers. All old policies preserved. Not a full-archive win estimate or unseen-opponent architecture test. Nominees selected only on108Mdevelopment; shared learned ancestor. Identical policy IDs execute once even if several retained-role aliases refer to them. No final-test promotion or GitHub upload.',
        next_rule='Interpret both continuation trajectories and fresh screen,including tails and weakness regressions. Use new independent/unseen evidence before promotion; this diagnostic becomes consumed after use.')
    io.verify(plan)
    io.write(claim, dict(run=io.relative(out), start=band, stop_exclusive=band+n//2))
    io.write(out/'plan.json', plan)
    for name in sources:
        dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT/name, dest)
    return plan


def verify_results(out):
    from tools.league_residual_train import validate_records
    p = io.read(out/'plan.json'); io.verify(p)
    paths = [out/'plan.json', out/'completion.json', out/'analysis.json', Path(__file__).resolve()]
    rows = []
    for i, job in enumerate(p['jobs']):
        f = out/f'matches/match_{i:04d}.json'; d = io.read(f); paths.append(f)
        assert d['job'] == job; validate_job_result(job, d['result']); rows.append(d['result'])
    for s in unique_specs(p['roles']):
        validate_records([r for r in rows if r['own'] == s['id']], p['opponents'], p['band'], p['n'])
    result = metrics.analyze(p, rows); assert result == io.read(out/'analysis.json')
    games = sum(len(r['episodes']) for r in rows)
    assert games == len(p['jobs'])*p['n'] == io.read(out/'completion.json')['games']
    io.write(out/'raw_budget_screen.json', dict(status='budget_screen_raw_verified', games=games,
        analysis=result, input_sha256={io.relative(f): io.sha(f) for f in paths}))
    print('budget screen raw verified', games, flush=True)


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--qualification', type=Path, default=Path('runs/league_episode_mixture_qualification_20261008'))
    a = p.parse_args(); engine.freeze = freeze; engine.analyze = metrics.analyze
    try:
        engine.run(a.source.resolve(), a.qualification.resolve(), a.out.resolve())
        verify_results(a.out.resolve())
    except BaseException:
        io.write(a.out/f'failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc())); raise
