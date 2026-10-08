from copy import deepcopy
import pytest
from tools import league_tournament_metrics as metric
from tools import league_tournament_assess as driver
from tests.test_league_regret_train import record, FOES


def results(rates):
    rows = record(rates)['results']
    for row in rows:
        for e in row['episodes']:
            e['won'] = e['outcome'] == 'kill'
    return rows


def test_absolute_improvement_can_pass_while_legacy_relative_cap_fails():
    groups = {f: f for f in FOES}
    base = results([1., .3, 1.] + [.3] * 5)
    new = results([.75, .4, 1.] + [.65] * 5)
    c = metric.compare(new, base, groups, final=True)
    assert c['profile_passed'] and not c['legacy_checkpoint_passed']
    assert c['per_opponent_win_gain']['pursuit'] == -.25
    bad = metric.compare(results([.1, .4, 1.] + [.9] * 5), base, groups)
    assert not bad['profile_passed']  # Higher average cannot mask a worse absolute floor.


def test_group_weighting_resists_extra_same_code_opponents():
    w = metric.group_weights({'a': 'one', 'b': 'one', 'c': 'two'})
    assert w == {'a': .25, 'b': .25, 'c': .5}
    a, b = results([.2] * 8), results([.4] * 8)
    gain = metric.paired_gain(b, a, {f: 1/8 for f in FOES})
    assert gain['mean'] == pytest.approx(.2)
    assert gain['shared_initial_conditions'] == 10


@pytest.mark.parametrize('corrupt', ['duplicate', 'seat', 'seed', 'foe'])
def test_paired_evidence_rejects_corruption(corrupt):
    a = results([.2] * 8); b = deepcopy(a)
    if corrupt == 'duplicate': b[0]['episodes'].append(b[0]['episodes'][0])
    if corrupt == 'seat': b[0]['episodes'].pop()
    if corrupt == 'seed': b[0]['episodes'][0]['seed'] += 999
    if corrupt == 'foe': b[0]['foe'] = 'unexpected'
    with pytest.raises(ValueError):
        metric.paired_gain(b, a, {f: 1/8 for f in FOES})


def test_selection_requires_both_references():
    wrap = lambda r: dict(results=results(r))
    plan = dict(groups={f: f for f in FOES}, candidates=[dict(id='challenger')])
    candidate = wrap([.7] * 8)
    assert driver.select([wrap([.4] * 8), wrap([.8] * 8), candidate], plan)['chosen'] is None
    assert driver.select([wrap([.4] * 8), wrap([.5] * 8), candidate], plan)['chosen']['id'] == 'challenger'


def test_live_predecessor_blocks_new_matches(tmp_path, monkeypatch):
    driver.write(tmp_path / 'runtime.json', dict(pid=123))
    monkeypatch.setattr(driver, 'process_live', lambda pid: True)
    with pytest.raises(ValueError, match='still live'):
        driver.terminal(tmp_path, {'benchmark_complete'})
