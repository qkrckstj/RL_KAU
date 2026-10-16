import math
import numpy as np
import pytest

def raw_state(t=0.):
    x=np.zeros(39,np.float32);x[15]=3000;x[4]=x[19]=230;x[30:32]=1;x[38]=120.-t
    return x

def test_opening_uses_public_clock_and_turn_then_transitions():
    from experiments.direct_pilot.opening_policy import Policy
    p=Policy(parameters=dict(opening_seconds=12.,opening_turn=-1.,opening_speed=350.))
    x=raw_state(11.95);assert p.act(x)//3==0 and p.act(x)%3==0
    x=raw_state(12.);assert p.act(x)//3==2
    assert p.act(raw_state(0.))==p.act(raw_state(0.))

def test_zero_opening_exactly_uses_tracking_even_after_later_calls():
    from experiments.direct_pilot.opening_policy import Policy
    a=Policy(parameters=dict(opening_seconds=0.,opening_turn=-1.));b=Policy(parameters=dict(opening_seconds=0.,opening_turn=1.))
    for t in (0.,12.,80.,0.):assert a.act(raw_state(t))==b.act(raw_state(t))

def test_euler_heading_rate_and_near_vertical_input_are_finite():
    from experiments.direct_pilot.opening_policy import heading_rate
    x=raw_state();x[9]=math.pi/3;x[13]=.2;x[14]=.1
    assert heading_rate(x)==pytest.approx(.2*math.sin(float(x[9]))+.1*math.cos(float(x[9])),abs=1e-8)
    x[10]=math.pi/2
    assert math.isfinite(heading_rate(x)) and abs(heading_rate(x))<=2.

def test_range_rate_negative_means_closing_and_controls_braking():
    from experiments.direct_pilot.opening_policy import Policy,range_rate
    p=Policy();x=raw_state();x[15]=0;x[16]=800;x[4]=250;x[19]=100
    assert range_rate(x)==pytest.approx(-150.) and p.act(x)%3==0
    x[19]=300;assert range_rate(x)>0 and p.act(x)%3==1
    x[16]=400;assert p.act(x)%3==0
    x[16]=1000;x[4]=500*.514444;assert p.act(x)%3==1

def test_ordinary_actions_and_parameter_validation():
    from experiments.direct_pilot.opening_policy import Policy
    with pytest.raises(ValueError):Policy(parameters=dict(opening_turn=.5))
    p=Policy();rng=np.random.default_rng(7)
    for _ in range(200):
        x=raw_state();x[11]=rng.uniform(-math.pi,math.pi);x[15:17]=rng.uniform(-6000,6000,2)
        action=p.act(x);assert type(action)is int and 0<=action<9
