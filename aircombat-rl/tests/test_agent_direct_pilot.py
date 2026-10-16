"""Public-observation direct pilot;diagnostic controller,not neural training."""
import importlib.util
import numpy as np
import pytest


def cls():
    assert importlib.util.find_spec('experiments.direct_pilot') is not None,'Missing direct pilot package'
    from experiments.direct_pilot.agent_policy import Policy
    return Policy


def frame(t=0.,hp=1.):
    raw=np.zeros(39,np.float32);raw[15]=10000.;raw[4]=250.;raw[19]=-250.;raw[26]=np.pi
    raw[30]=hp;raw[31]=1.;raw[38]=120.-t
    return raw


def test_actions_use_only_public_state_and_remain_in_official_nine_actions():
    p=cls()()
    for t in np.arange(0,20,.05):
        a=p.act(frame(float(t)));assert type(a) is int and 0<=a<9


def test_break_starts_before_damage_and_rejoins_after_public_elapsed_duration():
    p=cls()(parameters=dict(break_range=5000.,break_duration=3.,break_turn_deg=-60.))
    raw=frame();p.act(raw);assert p.phase=='approach'
    raw[15]=4000.;raw[38]=110.;p.act(raw);assert p.phase=='break' and raw[30]==1.
    raw[38]=106.;p.act(raw);assert p.phase=='slot'


def test_clock_reset_replays_actions_and_state():
    p=cls()();tape=[frame(0.),frame(10.,.95),frame(20.,.9)]
    first=[p.act(x) for x in tape];assert [p.act(x) for x in tape]==first


@pytest.mark.parametrize('bad',[np.zeros(38),np.full(39,np.nan)])
def test_invalid_observation_rejected(bad):
    with pytest.raises(ValueError):cls()().act(bad)


def test_invalid_configuration_rejected():
    for parameters in [dict(unknown=2),dict(break_duration=float('nan')),dict(break_range=-1.)]:
        with pytest.raises(ValueError):cls()(parameters=parameters)
