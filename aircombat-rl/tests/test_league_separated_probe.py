from pathlib import Path
from copy import deepcopy
import numpy as np
from experiments.league.interception_gate import Policy as Original
from experiments.league.separated_probe import Policy
from tests.test_league_interception_gate import configuration
from tests.test_league_adaptive import observation
from tools.league_separated_export import export
from tools.policies import load


def sequences():
    for turn in (0., .1, -.1):
        yield [observation(t,turn) for t in np.linspace(120,117,61)]


def test_identity_parent_preserves_original_and_clock_reset():
    old=configuration();new=dict(**old,probe=old['parent'])
    a,b=Original(configuration=old),Policy(configuration=new)
    for seq in sequences():
        for obs in seq: assert a.act(obs)==b.act(obs)
        assert a.selected==b.selected
    a.reset();b.reset()
    for obs in next(sequences()): assert a.act(obs)==b.act(obs)


def test_changed_parent_cannot_change_predecision_actions_and_export_matches(tmp_path):
    old=configuration();cfg=dict(parent=list(old['parent']),probe=list(old['parent']),interceptor=list(old['interceptor']))
    cfg['parent'][4]=60.
    a,b=Original(configuration=old),Policy(configuration=cfg)
    assert np.array_equal(b.experts[0].probe.parameters,old['parent'])
    assert b.experts[0].parent.parameters[4]==60.
    spec=export(tmp_path/'model',cfg,'separated')
    loaded,_,mode=load(Path(spec['design']),Path(spec['weights']))
    assert mode=='discrete'
    for seq in sequences():
        for obs in seq:
            old_action=a.act(obs);new_action=b.act(obs)
            if 120-float(obs[38]) < .5-1e-5: assert old_action==new_action
            assert new_action==loaded(obs)
        assert a.selected==b.selected
