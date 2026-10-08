from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from experiments.league.separated_probe import Policy as Base
from experiments.league.contextual_speed import Policy,ContextExpert
from tests.test_league_interception_gate import configuration as original
from tests.test_league_adaptive import observation
from tools.league_contextual_speed_export import export
from tools.policies import load


def config(gain=1.):
    c=original();c['probe']=list(c['parent']);c['parent']=list(c['parent']);c['parent'][0]=0.
    c['context']=[10.,1500.,500.,gain]
    return c


def test_disabled_context_exactly_matches_base_in_both_routes_and_clock_reset():
    c=config(0);a=Policy(configuration=c);b=Base(configuration=c)
    for turn in (.1,0.,-.1):
        for remaining in np.linspace(120,0,241):
            x=observation(remaining,turn);x[26]+=turn*(120-remaining)
            assert a.act(x)==b.act(x)
        assert a.selected==b.selected


def test_speed_change_uses_public_conditions_and_preserves_heading_firing_defense():
    c=config();p=c['parent'];p[2]=325.;p[9]=0.;p[11]=0.
    expert=ContextExpert(p,c['context'])
    x=observation(100,.1);x[16]=4000.;x[4]=450*.514444;x[19]=600*.514444
    base=expert.base.act(x);fast=expert.act(x)
    assert base%3==0 and fast%3==2 and fast//3==base//3
    for index,value in ((38,115.),(16,1000.),(19,400*.514444),(34,1.)):
        y=x.copy();y[index]=value
        assert expert.act(y)==expert.base.act(y)
    defensive=deepcopy(p);defensive[11]=3000.;defensive[12]=90.;x[16]=2500.
    expert=ContextExpert(defensive,c['context'])
    assert expert.act(x)==expert.base.act(x)


def test_export_preserves_probe_and_matches_actions_in_relocated_folder(tmp_path):
    c=config();c['context']=[.5,0.,300.,1.5]
    direct=Policy(configuration=c);base=Base(configuration=c)
    spec=export(tmp_path/'relocated',c,'context_test')
    predict,_,mode=load(Path(spec['design']),Path(spec['weights']))
    assert mode=='discrete'
    assert 'from experiments.' not in (tmp_path/'relocated/policy.py').read_text('utf-8')
    for turn in (.1,0.,-.1):
        for remaining in np.linspace(120,110,101):
            x=observation(remaining,turn);x[26]+=turn*(120-remaining)
            a,b=direct.act(x),base.act(x)
            assert predict(x)==a
            if 120-remaining<.5-1e-6 or direct.selected==1:assert a==b
        assert direct.selected==base.selected
    with pytest.raises(ValueError):ContextExpert(c['parent'],[.5,0.,300.,float('nan')])
