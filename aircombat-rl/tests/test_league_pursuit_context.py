from pathlib import Path
import numpy as np
import pytest
from experiments.league.pursuit_context import Policy,PursuitExpert
from experiments.league.contextual_speed import Policy as SpeedPolicy
from experiments.league.separated_probe import Policy as Base
from tools.league_pursuit_context_export import export
from tools.policies import load
from tests.test_league_contextual_speed import config
from tests.test_league_adaptive import observation


def configuration(gate=0.):
    c=config();c['context']=[5.,1000.,500.,1.,gate]
    return c


def test_unrestricted_and_disabled_controls_and_reset():
    c=configuration(-1);legacy=dict(c,context=c['context'][:4])
    direct=Policy(configuration=c);old=SpeedPolicy(configuration=legacy)
    disabled=configuration(.5);disabled['context'][3]=0.
    neutral=Policy(configuration=disabled);base=Base(configuration=c)
    for turn in (.1,0.,-.1):
        for remaining in np.linspace(120,0,241):
            x=observation(remaining,turn);x[26]+=turn*(120-remaining)
            assert direct.act(x)==old.act(x)
            assert neutral.act(x)==base.act(x)
        assert direct.selected==old.selected==neutral.selected==base.selected


def test_away_direction_controls_only_boost_and_is_rotation_invariant():
    c=configuration(.1);p=c['parent'];p[2]=325.;p[9]=0.;p[11]=0.
    expert=PursuitExpert(p,c['context'])
    x=np.zeros(39,dtype=np.float32);x[16]=4000.;x[4]=450*.514444;x[19]=600*.514444;x[38]=100.
    assert expert.speed.base.act(x)%3==0
    assert expert.act(x)%3==2
    assert expert.act(x)//3==expert.speed.base.act(x)//3
    for vx,vy in ((0.,-600.),(600.,0.)):
        y=x.copy();y[18]=vx*.514444;y[19]=vy*.514444
        assert expert.act(y)==expert.speed.base.act(y)
    # Rotate east/north by90 degrees along with headings; gate and actions agree.
    y=x.copy();y[15]=4000.;y[16]=0.;y[3]=x[4];y[4]=0.;y[18]=x[19];y[19]=0.;y[11]=np.pi/2;y[26]=np.pi/2
    assert expert.act(y)==expert.act(x)
    x[34]=1.;assert expert.act(x)==expert.speed.base.act(x)
    with pytest.raises(ValueError):PursuitExpert(p,[5,1000,500,1,float('nan')])


def test_relocated_export_preserves_probe_routes_and_actions(tmp_path):
    c=configuration(.25);direct=Policy(configuration=c);base=Base(configuration=c)
    spec=export(tmp_path/'relocated',c,'pursuit_test')
    act,_,mode=load(Path(spec['design']),Path(spec['weights']))
    assert mode=='discrete'
    assert 'from experiments.' not in (tmp_path/'relocated/policy.py').read_text('utf-8')
    for turn in (.1,0.,-.1):
        for remaining in np.linspace(120,100,201):
            x=observation(remaining,turn);x[26]+=turn*(120-remaining)
            a,b=direct.act(x),base.act(x)
            assert act(x)==a
            if 120-remaining<.5-1e-6 or direct.selected==1:assert a==b
