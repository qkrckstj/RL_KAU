import numpy as np
import pytest
from experiments.league.risk_break_policy import Policy,BasePolicy,BASE_EXPERT,RISK_NAMES,RISK_BASELINE


def state(t=0.):
    x=np.zeros(39,np.float32);x[4]=250.;x[16]=1000.;x[19]=-250.;x[26]=np.pi
    x[30:32]=[.6,.8];x[32:34]=[.8,.9];x[34:36]=1.;x[38]=120.-t
    return x


def config(**kwargs):
    result=dict(zip(RISK_NAMES,RISK_BASELINE));result.update(kwargs);return result


def test_disabled_reproduces_base_and_clock_reset():
    a=Policy(configuration=config());b=BasePolicy(configuration=BASE_EXPERT);rng=np.random.default_rng(721)
    for ep in range(10):
        for tick in range(600):
            x=rng.normal(size=39).astype(np.float32);x[[0,1,15,16]]*=1800.;x[3:6]*=180.;x[18:21]*=180.
            x[30:32]=rng.uniform(0,1,2);x[32:34]=rng.uniform(0,3,2);x[34:36]=rng.integers(0,2,2);x[38]=120.-tick*.05
            assert a.act(x)==b.act(x)
        a.reset();b.reset();assert a.act(state())==b.act(state())


def test_can_break_while_both_have_firing_solution_and_honors_cooldown():
    p=Policy(configuration=config(break_seconds=1.,break_angle=90.,cooldown_seconds=2.))
    assert p.act(state()) in range(9) and p.break_until==1.
    p.act(state(.5));assert p.break_until==1.
    p.act(state(1.5));assert p.break_until==1.  # still threatened but cooldown holds
    p.act(state(3.5));assert p.break_until==4.5
    first=Policy(configuration=config(break_seconds=1.,break_angle=90.,cooldown_seconds=2.)).act(state())
    assert p.act(state())==first and p.break_until==1.  # clock rollback resets


@pytest.mark.parametrize('channel,value',[(35,0.),(33,.1),(30,1.)])
def test_public_threat_guards_do_not_activate_when_absent(channel,value):
    p=Policy(configuration=config(break_seconds=1.,break_angle=90.));x=state();x[channel]=value
    expected=BasePolicy(configuration=BASE_EXPERT).act(x)
    assert p.act(x)==expected and p.break_until==-1.


def test_reject_nonfinite_or_invalid_parameters():
    with pytest.raises(ValueError):Policy(configuration=config(break_seconds=float('nan')))
    p=Policy(configuration=config(break_seconds=1.));x=state();x[35]=np.nan
    with pytest.raises(ValueError):p.act(x)
