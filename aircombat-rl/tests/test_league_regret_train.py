from copy import deepcopy
import numpy as np
import pytest
import tools.league_regret_train as module
from tools.league_train import metrics


FOES = ['pursuit', 'evader', 'circler'] + [f'archive_{i}' for i in range(5)]
BASE = [1., .3, 1.] + [.4] * 5
TRAP = [.05, .3, 1.] + [.75] * 5


def record(rates, identity='candidate', n=20, band=100, weights=None):
    results = []
    for foe, rate in zip(FOES, rates):
        wins = round(n * rate)
        episodes = [dict(seed=band+i//2, seat=('red', 'blue')[i % 2],
            outcome='kill' if i < wins else 'died', own_health=.8 if i < wins else 0., steps=20)
            for i in range(n)]
        results.append(dict(foe=foe, episodes=episodes,
            summary=dict(n=n, wins=wins, losses=n-wins, draws=0, rate=wins/n,
                         score=wins/n, own_health=.8*wins/n, opp_health=1-wins/n)))
    return dict(request=dict(spec=dict(id=identity), opponents=[dict(id=f) for f in FOES],
                band=band, n=n, weights=weights), results=results, metrics=metrics(results, weights))


def test_higher_average_with_catastrophic_regression_is_penalized():
    base, trap = record(BASE, 'anchor'), record(TRAP)
    assert trap['metrics']['objective'] > base['metrics']['objective']
    assert module.regression_cost(base, base) == 0
    assert module.rank(trap, base) < module.rank(base, base)
    assert not module.checkpoint_admissible(trap, [base, base])


def test_broad_improvement_has_no_regression_cost_and_can_be_retained():
    base = record(BASE, 'anchor')
    good = record([1., .4, 1.] + [.5] * 5)
    assert module.regression_cost(good, base) == 0
    assert module.rank(good, base) > module.rank(base, base)
    assert module.checkpoint_admissible(good, [base, base])


@pytest.mark.parametrize('change', ['band', 'pair', 'order'])
def test_regression_rejects_unmatched_evidence(change):
    base = record(BASE, 'anchor'); new = record(TRAP)
    if change == 'band': new['request']['band'] += 1
    if change == 'pair': new['results'][0]['episodes'][0]['seed'] += 999
    if change == 'order': new['results'].reverse()
    with pytest.raises(ValueError): module.regression_cost(new, base)


def test_successful_predecessor_does_not_trigger_failure_branch(tmp_path, monkeypatch):
    prior, out = tmp_path/'prior', tmp_path/'out'
    module.write(prior/'completion.json', dict(status='evaluation_passed'))
    monkeypatch.setattr(module, 'freeze', lambda *a, **kw: pytest.fail('Must not start'))
    module.run(out, prior, tmp_path/'damage')
    assert module.read(out/'completion.json')['status'] == 'not_eligible_prior_passed'
    assert not (out/'plan.json').exists()


def test_incomplete_live_predecessor_blocks_freeze(tmp_path, monkeypatch):
    prior = tmp_path/'prior'; module.write(prior/'runtime.json', dict(pid=123))
    monkeypatch.setattr(module, 'process_live', lambda pid: True)
    with pytest.raises(ValueError, match='still live'):
        module.freeze(tmp_path/'out', prior, tmp_path/'damage')


def test_search_keeps_paired_anchor_in_confirmation_and_rejects_average_trap(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'verify', lambda plan: None)
    initial = (module.LOW + module.HIGH) / 2
    def candidate(root, plan, parameters):
        return dict(id='anchor' if np.array_equal(parameters, initial)
                    else module.digest(np.asarray(parameters).tolist()))
    monkeypatch.setattr(module, 'candidate', candidate)
    plan = dict(initial_configuration=dict(parent=initial.tolist()), population=12, elites=3,
        initial_std=.08, minimum_std=.035, maximum_generations=2, base_generations=6,
        training_band=1000, development_band=100, screen_n=4, confirmation_n=8,
        development_n=20, opponents=[dict(id=f) for f in FOES])
    calls = []
    def evaluate(pool, specs, opponents, band, n, out, weights=None):
        calls.append((out.name, [s['id'] for s in specs]))
        return [record(BASE if s['id']=='anchor' else TRAP, s['id'], n, band, weights)
                for s in specs]
    base = record(BASE, 'anchor')
    result = module.search(tmp_path/'s1', tmp_path, plan, 1, 0, [base, deepcopy(base)], evaluate, None)
    assert result['selected']['spec']['id'] == 'anchor'
    assert len(result['history']) == 2
    assert all('anchor' in ids for phase, ids in calls if phase in ('screen', 'confirm'))
    assert all(any(v > 0 for v in h['confirmation_regression_costs']) for h in result['history'])


def test_final_contract_still_freezes_choice_and_checks_both_baselines(tmp_path, monkeypatch):
    # Reuse the established external behavior assertions on this new driver.
    import tests.test_league_maneuver_train as prior_tests
    monkeypatch.setattr(prior_tests, 'module', module)
    prior_tests.test_run_freezes_choice_and_compares_both_baselines_with_traced_final(tmp_path, monkeypatch)
