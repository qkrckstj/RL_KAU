from copy import deepcopy
import numpy as np
import pytest

from tools.league_temporal_repair_train import ACTIVE, encode, configuration, proposals, extension
from experiments.league.controller import LOW,HIGH
from experiments.league.separated_probe import Policy
from tests.test_league_interception_gate import configuration as original_configuration
from tests.test_league_adaptive import observation


def test_active_search_preserves_probe_and_inactive_parameters_and_early_actions():
    cfg=original_configuration()
    cfg['probe']=list(cfg['parent']);cfg['parent']=list(cfg['parent']);cfg['parent'][0]=0.
    saved=deepcopy(cfg)
    assert configuration(encode(cfg['parent']),cfg)==cfg
    for vector in (np.zeros(len(ACTIVE)),np.ones(len(ACTIVE)),np.full(len(ACTIVE),.5)):
        changed=configuration(vector,cfg)
        assert changed['probe']==saved['probe'] and changed['interceptor']==saved['interceptor']
        assert all(changed['parent'][i]==saved['parent'][i] for i in set(range(17))-set(ACTIVE))
        a,b=Policy(configuration=saved),Policy(configuration=changed)
        for t in (120.,119.9,119.8,119.7,119.6):
            assert a.act(observation(t,.1))==b.act(observation(t,.1))
    assert cfg==saved
    with pytest.raises(ValueError):configuration(np.full(len(ACTIVE),np.nan),cfg)


def test_population_is_distinct_reproducible_and_independent_across_seeds():
    mean=np.zeros(len(ACTIVE));std=np.full(len(ACTIVE),.12)
    def generate(seed):return proposals(np.random.default_rng(seed),mean,std,[mean,mean,np.ones(len(ACTIVE))],12)
    a,indices,origins=generate(3000);b,_,_=generate(3000);c,_,_=generate(3001)
    assert a.shape==(12,11) and len(set(map(tuple,a)))==12
    assert np.array_equal(a,b) and not np.array_equal(a,c)
    assert indices[0]==indices[1] and np.array_equal(a[indices[2]],np.ones(11))
    assert np.all((a>=0)&(a<=1)) and origins.count('uniform')>=2


def test_budget_extension_needs_development_gain_without_win_regression():
    assert extension((.6,.5,.4),(.603,.5,.4))
    assert not extension((.6,.5,.4),(.603,.49,.4))
    assert not extension((.6,.5,.4),(.601,.7,.4))
    assert not extension((.6,.5,.4),(.6,.5,.4))
