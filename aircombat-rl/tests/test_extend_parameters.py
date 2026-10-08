import math
import numpy as np
import pytest
from experiments.league.extend_parameter_policy import Policy,NAMES,BASELINE
from experiments.league.temporal_opponents import Policy as Original


def test_baseline_reproduces_original_stateful_controller():
    old=Original(configuration=dict(mode='extend',side=1,period=8.,speed=600.))
    new=Policy(configuration=dict(zip(NAMES,BASELINE)))
    rng=np.random.default_rng(726)
    for i in range(12000):
        obs=rng.normal(size=39)
        obs[[0,1,15,16]]*=1800.
        obs[3:6]*=180.;obs[18:21]*=180.
        obs[34]=float(i%7==0);obs[38]=120.-(i%2400)*.05
        assert old.act(obs)==new.act(obs)
        assert old.escape_until==new.escape_until
        assert old.escape_heading==new.escape_heading
        assert old.cooldown_until==new.cooldown_until


def test_reject_nonfinite_configuration():
    config=dict(zip(NAMES,BASELINE));config['chase_bonus']=math.nan
    with pytest.raises(ValueError):Policy(configuration=config)
