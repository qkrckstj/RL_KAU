"""Remembered heading commands obey the official zero-delta hold semantics."""
import math
import numpy as np
import pytest

def test_hold_reuses_existing_target_even_while_heading_is_far_away():
    from experiments.direct_pilot.setpoint_policy import choose_heading
    action,target=choose_heading(math.radians(20),math.radians(50),math.radians(50))
    assert action==1
    assert target==pytest.approx(math.radians(50))

def test_turn_replaces_target_relative_to_current_heading():
    from experiments.direct_pilot.setpoint_policy import choose_heading
    action,target=choose_heading(math.radians(45),math.radians(5),math.radians(75))
    assert action==0
    assert target==pytest.approx(math.radians(15))

def test_heading_selection_wraps_and_ties_prefer_hold():
    from experiments.direct_pilot.setpoint_policy import choose_heading
    assert choose_heading(math.radians(179),math.radians(-179),math.radians(-179))[0]==1
    assert choose_heading(0.,math.radians(15),0.)[0]==1

def test_policy_target_memory_matches_executed_action_and_clock_reset():
    from experiments.direct_pilot.setpoint_policy import Policy,wrap
    p=Policy(parameters=dict(break_range=0,lead=6));x=np.zeros(39,np.float32);x[15]=5000;x[4]=x[19]=200;x[30:32]=1
    retained=None;first=None
    for i in range(100):
        x[38]=120.-i*.05;x[11]=.01*i;x[26]=.02*i
        action=p.act(x)
        if first is None:first=action
        assert type(action)is int and 0<=action<9
        if retained is None:retained=float(x[11])
        if action//3!=1:retained=wrap(float(x[11])+math.radians(30*(action//3-1)))
        assert p.command_heading==pytest.approx(retained)
    x[38]=120;x[11]=x[26]=0
    assert p.act(x)==first

def test_repeated_query_is_idempotent_and_external_executed_hold_is_synchronized():
    from experiments.direct_pilot.setpoint_policy import Policy
    p=Policy(parameters=dict(break_range=0,lead=6));x=np.zeros(39,np.float32);x[15]=5000;x[4]=x[19]=200;x[30:32]=1;x[38]=120
    action=p.act(x);target=p.command_heading
    assert action//3!=1
    assert p.act(x.copy())==action and p.command_heading==target
    p.synchronize_executed_action(x,4)
    assert p.command_heading==pytest.approx(0.)
    p.synchronize_executed_action(x,8)
    assert p.command_heading==pytest.approx(math.pi/6)
    with pytest.raises(ValueError):p.synchronize_executed_action(x,9)
    y=x.copy();y[38]-=.05
    with pytest.raises(ValueError):p.synchronize_executed_action(y,4)
