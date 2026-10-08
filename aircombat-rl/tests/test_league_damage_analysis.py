from copy import deepcopy
import pytest
from tools.league_damage_analysis import episode_rows, summarize, paired_change


def record():
    return dict(episodes=[dict(seed=100, seat=s, outcome=o) for s, o in
                          [('red', 'kill'), ('blue', 'died')]], damage_traces=[
        dict(seed=100, seat='red', first_hit=3., timeline_1s=[dict(t=0., hp=1.)],
             terminal=dict(t=12., hp=.8), snapshots={'30': None}, damage_seconds=.2),
        dict(seed=100, seat='blue', first_hit=2., timeline_1s=[dict(t=0., hp=1.)],
             terminal=dict(t=35., hp=0.), snapshots={'30': dict(t=30., hp=.1)}, damage_seconds=1.)])


def test_early_win_damage_is_counted_without_imputing_later_health():
    rows = episode_rows(record(), 100, 2)
    data = summarize(rows)
    assert data['damage_through_30_or_end'] == pytest.approx(.55)
    assert data['reached_30'] == 1 and data['hp_at_30_mean_among_reached'] == .1
    assert data['ended_before_30'] == 1 and data['early_end_wins'] == 1
    assert data['wins'] == data['losses'] == 1
    assert not any(paired_change(rows, rows).values())


def test_bad_pairing_or_missing_reached_snapshot_is_rejected():
    bad = record(); bad['damage_traces'][0]['seat'] = 'blue'
    with pytest.raises(ValueError, match='pairs'): episode_rows(bad, 100, 2)
    bad = record(); bad['damage_traces'][1]['snapshots']['30'] = None
    with pytest.raises(ValueError, match='Missing'): episode_rows(bad, 100, 2)
    rows = episode_rows(record(), 100, 2); other = deepcopy(rows); other[0]['seed'] = 101
    with pytest.raises(ValueError, match='identical'): paired_change(other, rows)
