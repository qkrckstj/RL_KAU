import pytest
from tools.league_damage_trace import DamageTrace


def test_first_damage_and_time_snapshots_preserve_early_end():
    trace=DamageTrace(0.,1.,1.)
    trace.observe(9.95,1.,1.)
    trace.observe(10.,.8,1.)
    trace.observe(10.05,.7,.9)
    trace.observe(12.,.7,0.)
    result=trace.finish()
    assert result['first_hit']==10
    assert result['first_enemy_hit']==10.05
    assert result['damage_seconds']==pytest.approx(.1)
    assert result['snapshots']['10']['hp']==.8
    assert result['snapshots']['20'] is None
    assert result['minimum_hp']==.7
    assert result['terminal']['t']==12


def test_no_damage_and_duplicate_clock_do_not_create_damage_time():
    trace=DamageTrace(0.,1.,1.)
    for t in (0.,1.,10.,20.,30.,60.):trace.observe(t,1.,1.)
    result=trace.finish()
    assert result['first_hit'] is None and result['first_enemy_hit'] is None
    assert result['damage_seconds']==0
    assert all(x['hp']==1 for x in result['snapshots'].values())
    with pytest.raises(ValueError):trace.observe(59.,1.,1.)
