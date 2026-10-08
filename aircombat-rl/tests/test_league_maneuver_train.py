import numpy as np
import pytest
from tools import league_maneuver_train as module


def sample_record(objective=.5, win=.5, hp=.4, outcome='kill'):
    return dict(metrics=dict(objective=objective, mean_win_rate=win, health=0.),
                results=[dict(episodes=[dict(own_health=hp, outcome=outcome)])])


def test_health_bonus_does_not_reward_unhurt_draws_or_losses():
    won = sample_record(hp=.4)
    for outcome in ('timeout', 'died'):
        other = sample_record(hp=1., outcome=outcome)
        assert module.healthy_wins(other) == 0
        assert module.rank(won) > module.rank(other)


def test_extension_uses_recent_improvement_and_rejects_win_regression():
    old = sample_record()
    assert not module.extension_warranted(old, old)
    assert module.extension_warranted(old, sample_record(objective=.503))
    assert module.extension_warranted(old, sample_record(hp=.42))
    assert not module.extension_warranted(old, sample_record(objective=.51, win=.49, hp=.8))


def test_explicit_anchor_is_exact_and_other_parameters_remain_decoded():
    initial = module.LOW + np.random.default_rng(123).random(17) * (module.HIGH-module.LOW)
    unit = (initial-module.LOW)/(module.HIGH-module.LOW)
    assert np.array_equal(module.decode(unit, initial), initial)
    changed = unit.copy(); changed[11] = .6
    assert module.decode(changed, initial)[11] == pytest.approx(2100.)


def test_shared_requests_reuse_records_but_changed_conditions_execute(tmp_path, monkeypatch):
    calls = []
    def fake(pool, specs, opponents, band, n, out, weights):
        calls.append((band, specs))
        return [dict(request=dict(spec=s, opponents=opponents, band=band, n=n, weights=weights),
                     results=[dict(own=s['id'], foe=f['id'], summary=dict(n=n),
                         episodes=[dict(seed=band+i, seat=seat) for i in range(n//2)
                                   for seat in ('red', 'blue')]) for f in opponents], metrics={}) for s in specs]
    monkeypatch.setattr(module, 'evaluate_candidates', fake)
    evaluate = module.Evaluation()
    spec = dict(id='anchor'); foes = [dict(id='ace')]
    a = evaluate(None, [spec, spec], foes, 100, 2, tmp_path/'one')
    b = evaluate(None, [spec], foes, 100, 2, tmp_path/'two')
    assert a[0] == b[0] and len(calls) == 1 and len(calls[0][1]) == 1
    a[0]['metrics']['changed'] = True
    assert 'changed' not in a[1]['metrics']
    evaluate(None, [spec], foes, 200, 2, tmp_path/'three')
    assert len(calls) == 2
    with pytest.raises(ValueError, match='frozen record'):
        evaluate(None, [spec], foes, 300, 2, tmp_path/'one')


def test_trace_pool_dispatches_only_official_evaluation_jobs():
    class Pool:
        def submit(self, function, *args):
            return function, args
    proxy = module.TracePool(Pool())
    function, args = proxy.submit(module.duel, 'a', 'b', 100, 2)
    assert function is module.timed_observe and args == ('a', 'b', 100, 2)
    with pytest.raises(ValueError, match='Unexpected'):
        proxy.submit(lambda: None)


@pytest.mark.parametrize('offset', [-90., 90.])
def test_modified_parent_export_keeps_opening_defense_and_interceptor(tmp_path, offset):
    from pathlib import Path
    from experiments.league.interception_gate import Policy
    from tools.policies import load
    from tests.test_league_interception_gate import configuration
    from tests.test_league_adaptive import observation
    config = configuration()
    for index, value in {0:2., 1:550., 7:1., 11:1200., 12:30., 13:offset, 14:450.}.items():
        config['parent'][index] = value
    preserved_interceptor = list(config['interceptor'])
    direct = Policy(configuration=config)
    spec = module.export(tmp_path/'exported', config, 'modified_parent')
    action, _, mode = load(Path(spec['design']), Path(spec['weights']))
    assert mode == 'discrete'
    for turn in (.1, 0.):
        for remaining in np.linspace(120, 117, 61):
            x = observation(remaining, turn)
            x[26] += turn * (120-remaining)
            for distance, own_firing in [(10000., 0.), (1000., 0.), (1000., 1.)]:
                x[16], x[34] = distance, own_firing
                assert action(x) == direct.act(x)
        assert direct.selected == (0 if turn else 1)
    assert config['interceptor'] == preserved_interceptor


def test_run_freezes_choice_and_compares_both_baselines_with_traced_final(tmp_path, monkeypatch):
    from tools.league_train import metrics
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'verify', lambda plan: None)
    plan = dict(anchor=dict(id='anchor'), parent=dict(id='parent'),
        opponents=[dict(id='evader'), dict(id='circler')], seeds=[1, 2],
        development_band=100, development_n=4, selection_band=200, selection_n=4,
        final_band=300, final_n=4)
    def freeze(out, previous, damage, workers=3):
        plan['workers'] = workers
        module.write(out/'plan.json', plan)
        return plan
    monkeypatch.setattr(module, 'freeze', freeze)
    class Pool:
        def __init__(self, **kwargs): assert kwargs['max_workers'] == 6
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(module, 'search', lambda out, root, p, seed, *args:
                        dict(selected=dict(spec=dict(id='c0' if seed == 1 else 'c1'))))
    calls = []
    class Evaluation:
        def __call__(self, pool, specs, opponents, band, n, out, weights=None, traced=False):
            calls.append(dict(ids=[s['id'] for s in specs], traced=traced, band=band))
            records = []
            for spec in specs:
                results = []
                for foe in opponents:
                    rows, traces = [], []
                    for i in range(n//2):
                        win = spec['id'] == 'c0' or (spec['id'] == 'c1' and i == 0)
                        hp = .8 if win else .1
                        for seat in ('red', 'blue'):
                            rows.append(dict(seed=band+i, seat=seat, outcome='kill' if win else 'timeout',
                                won=win, own_health=hp, steps=800))
                            traces.append(dict(seed=band+i, seat=seat, first_hit=1., damage_seconds=.2,
                                timeline_1s=[dict(t=0., hp=1.)], terminal=dict(t=40., hp=hp),
                                snapshots={'30': dict(t=30., hp=hp)}))
                    wins = sum(r['won'] for r in rows)
                    result = dict(own=spec['id'], foe=foe['id'], episodes=rows,
                        summary=dict(n=n, wins=wins, losses=0, draws=n-wins, rate=wins/n,
                            score=(wins+.5*(n-wins))/n, own_health=np.mean([r['own_health'] for r in rows]), opp_health=0.))
                    if traced: result['damage_traces'] = traces
                    results.append(result)
                records.append(dict(request=dict(spec=spec), results=results, metrics=metrics(results)))
            return records
    monkeypatch.setattr(module, 'Evaluation', Evaluation)
    out = tmp_path/'run'; module.run(out, tmp_path/'prior', tmp_path/'damage', workers=6)
    chosen = module.read(out/'frozen_selection.json')['chosen']
    completion = module.read(out/'completion.json')
    assert chosen['id'] == completion['selected']['id'] == 'c0'
    assert completion['status'] == 'evaluation_passed' and completion['selected_before_test']
    assert calls[-1] == dict(ids=['anchor', 'parent', 'c0'], traced=True, band=300)
    assert (out/'damage_comparison.json').exists()
    assert module.read(out/'runtime.json')['workers'] == 6


def test_freeze_rejects_live_predecessor_before_creating_a_plan(tmp_path, monkeypatch):
    prior = tmp_path/'prior'; module.write(prior/'runtime.json', dict(pid=123))
    monkeypatch.setattr(module, 'process_live', lambda pid: True)
    with pytest.raises(ValueError, match='still live'):
        module.freeze(tmp_path/'new', prior, tmp_path/'damage')
    assert not (tmp_path/'new/plan.json').exists()


def test_completed_predecessors_do_not_treat_archived_pids_as_live_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'verify', lambda plan: None)
    monkeypatch.setattr(module, 'environment', lambda: {'test': 1})
    monkeypatch.setattr(module, 'process_live', lambda pid: pytest.fail('Archived PID must not be queried'))
    prior, damage, out = tmp_path/'prior', tmp_path/'damage', tmp_path/'new'
    for folder, status in [(prior, 'evaluation_passed'), (prior/'novel_audit', 'audit_complete'),
                           (damage, 'diagnostic_complete'), (damage/'analysis', 'analysis_complete')]:
        module.write(folder/'runtime.json', dict(pid=123))
        module.write(folder/'completion.json', dict(status=status))
    module.write(damage/'compatibility.json', dict(status='passed'))
    plan = dict(previous='prior', damage='damage', environment={'test': 1})
    module.write(out/'plan.json', plan)
    assert module.freeze(out, prior, damage) == plan
    with pytest.raises(ValueError, match='Changed predecessor'):
        module.freeze(out, prior, damage, workers=6)


def test_second_controller_cannot_bypass_a_live_successor_queue(tmp_path, monkeypatch):
    out = tmp_path/'new'; module.write(out/'queue_runtime.json', dict(pid=-123))
    monkeypatch.setattr(module, 'process_live', lambda pid: True)
    with pytest.raises(ValueError, match='owns the queued'):
        module.run(out, tmp_path/'prior', tmp_path/'damage')
