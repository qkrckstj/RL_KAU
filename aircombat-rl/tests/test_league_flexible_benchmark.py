import pytest

from tools.league_flexible_benchmark import recommendation


def records(static=(10., 10.), flexible=(9., 9.), headroom=4.):
    return [dict(scenario=name, repeat=i, match_seconds=t,
                 resources=dict(minimum_commit_headroom_gib=headroom, minimum_available_memory_gib=8.))
            for name, times in (('static16', static), ('flexible16', flexible))
            for i, t in enumerate(times)]


def test_requires_repeated_practical_gain_and_memory_headroom():
    assert recommendation(records())['recommended'] == 'flexible16'
    assert recommendation(records(flexible=(9.9, 9.9)))['recommended'] == 'static16'
    assert recommendation(records(flexible=(7., 11.)))['recommended'] == 'static16'
    assert recommendation(records(headroom=2.9))['recommended'] == 'static16'
    with pytest.raises(ValueError):
        recommendation(records()[:-1])
