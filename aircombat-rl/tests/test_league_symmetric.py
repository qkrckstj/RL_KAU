from pathlib import Path
import numpy as np
from experiments.league.adaptive_symmetric import Policy
from experiments.league.controller import Policy as Expert
from tests.test_league_adaptive import config, observation


def test_symmetric_gate_treats_left_and_right_turns_equally_and_resets():
    c = config()
    c['gate'] = [1, .5, -1, 0, -1, 0]
    p = Policy(configuration=c)
    for turn, expected in ((.1, 0), (-.1, 0), (0., 1)):
        for t in np.linspace(120,119,21):
            obs = observation(t,turn)
            obs[26] += turn*(120-t)
            p.act(obs)
        assert p.selected == expected
        assert p.features[1] >= 0 and p.features[3] >= 0


def test_constant_parent_and_duplicate_clock_are_preserved():
    c = config()
    c['gate'] = [1,-8,0,0,0,0]
    p = Policy(configuration=c)
    expert = Expert(parameters=c['experts'][0]['parameters'])
    for _ in range(50):
        p.act(observation(120,-.1))
    assert p.probe_elapsed == 0 and p.selected is None
    for t in np.linspace(120,0,241):
        obs = observation(t,-.1)
        assert p.act(obs) == expert.act(obs)


def test_standalone_symmetric_export_matches_local_policy(tmp_path):
    from tools.league_symmetric_export import export
    from tools.policies import load
    c = config()
    c['gate'] = [1,.5,-1,0,-1,0]
    spec = export(tmp_path/'model',c,'symmetric')
    act, _, mode = load(Path(spec['design']),Path(spec['weights']))
    direct = Policy(configuration=c)
    assert mode == 'discrete'
    for turn in (.1,-.1,0.):
        for t in np.linspace(120,117,61):
            obs = observation(t,turn)
            obs[26] += turn*(120-t)
            assert act(obs) == direct.act(obs)
