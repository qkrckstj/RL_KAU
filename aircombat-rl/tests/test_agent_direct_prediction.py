import math
import numpy as np
import pytest

def test_constant_turn_prediction_reduces_to_linear_at_zero_rate():
    from experiments.direct_pilot.predictive_policy import future_displacement
    assert future_displacement(100.,200.,0.,3.)==pytest.approx((300.,600.))

def test_constant_turn_prediction_follows_public_clockwise_heading():
    from experiments.direct_pilot.predictive_policy import future_displacement
    x,y=future_displacement(0.,100.,math.pi/2,1.)
    assert (x,y)==pytest.approx((200/math.pi,200/math.pi))

def test_predictive_pilot_resets_and_uses_only_nine_actions():
    from experiments.direct_pilot.predictive_policy import Policy
    p=Policy(parameters=dict(lead=6.));obs=np.zeros(39,np.float32);obs[15]=5000;obs[19]=200;obs[4]=200;obs[30:32]=1
    actions=[]
    for i in range(150):
        obs[38]=120.-i*.05;obs[26]=i*.01;actions.append(p.act(obs))
    p.reset();obs[38]=120;obs[26]=0
    assert p.act(obs)==actions[0] and all(type(a)is int and 0<=a<9 for a in actions)
