from pathlib import Path
import numpy as np
from experiments.league.interception_gate import Policy
from experiments.league.controller import Policy as Expert
from experiments.league.interception import Policy as Interceptor
from tools.league_interception_gate_export import export
from tools.policies import load
from tests.test_league_adaptive import config, observation


def configuration():
    return dict(parent=config()['experts'][0]['parameters'],
                interceptor=[15,650,650,6,4000,.3,3])


def test_gate_and_export_match_both_branches_and_episode_reset(tmp_path):
    c = configuration()
    p = Policy(configuration=c)
    spec = export(tmp_path/'policy', c, 'test')
    act, _, mode = load(Path(spec['design']), Path(spec['weights']))
    assert mode == 'discrete'
    parent, pursuit = Expert(parameters=c['parent']), Interceptor(parameters=c['interceptor'])
    for turn, expected in ((.1, 0), (-.1, 0), (0., 1)):
        for t in np.linspace(120, 117, 61):
            obs = observation(t, turn)
            obs[26] += turn*(120-t)
            action = p.act(obs)
            assert act(obs) == action
            assert action == (parent if p.selected in (None, 0) else pursuit).act(obs)
        assert p.selected == expected
