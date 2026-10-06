from pathlib import Path
import numpy as np
from experiments.league.adaptive import Policy
from experiments.league.controller import embed,Policy as Expert
from tools.league_matches import initial_roster


def config():
    p=embed(initial_roster()[0]['parameters'])
    other=p.copy()
    other[7]=1
    return dict(experts=[dict(parameters=p.tolist()),dict(parameters=other.tolist())],
                gate=[1.,0.,-1.,0.,0.,0.])


def observation(remaining,turn):
    x=np.zeros(39)
    x[16]=10000
    x[4]=x[19]=220
    x[26]=np.pi
    x[29]=turn
    x[38]=remaining
    return x


def test_public_motion_selects_different_strategy_and_clock_resets_it():
    policy=Policy(configuration=config())
    for t in np.linspace(120,119,21):
        policy.act(observation(t,.1))
    assert policy.selected==0
    # No external reset needed between grader episodes.
    for t in np.linspace(120,119,21):
        policy.act(observation(t,-.1))
    assert policy.selected==1
    assert policy.act(observation(118,-.1))//3==2


def test_duplicate_observation_does_not_advance_probe_clock():
    policy=Policy(configuration=config())
    obs=observation(120,-.1)
    for _ in range(100):
        policy.act(obs)
    assert policy.selected is None
    assert policy.probe_elapsed==0


def test_constant_zero_gate_reproduces_old_expert():
    c=config()
    c['gate']=[1,-8,0,0,0,0]
    policy=Policy(configuration=c)
    expert=Expert(parameters=c['experts'][0]['parameters'])
    for t in np.linspace(120,0,241):
        x=observation(t,.1)
        assert policy.act(x)==expert.act(x)


def test_standalone_export_keeps_state_and_matches_local_policy(tmp_path):
    from tools.league_adaptive_export import export
    from tools.policies import load
    spec=export(tmp_path/'model',config(),'test')
    act,_,mode=load(Path(spec['design']),Path(spec['weights']))
    direct=Policy(configuration=config())
    assert mode=='discrete'
    for sign in (.1,-.1):
        for t in np.linspace(120,117,61):
            obs=observation(t,sign)
            assert act(obs)==direct.act(obs)
