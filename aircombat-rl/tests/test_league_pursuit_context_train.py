import numpy as np
import pytest
from tools.league_pursuit_context_train import panels,weights_for,encode,configuration,proposals
from tests.test_league_pursuit_context import configuration as policy_configuration


def test_context_parameter_roundtrip_and_preservation():
    c=policy_configuration(.35)
    rebuilt=configuration(encode(c['context']),c)
    assert np.allclose(rebuilt['context'],c['context'])
    for key in ('probe','parent','interceptor'):assert rebuilt[key]==c[key]
    with pytest.raises(ValueError):configuration(np.array([0,0,0,0,1.1]),c)
    with pytest.raises(ValueError):configuration(np.array([0,0,0,0,np.nan]),c)


def test_rotating_panels_preserve_weakness_groups_and_cover_archive():
    opponents=[{'id':f'f{i}'} for i in range(69)]
    groups={f'f{i}':'temporal' if i<8 else f'g{(i-8)%13}' for i in range(69)}
    temporal=[f'f{i}' for i in range(8)]
    schedule=panels(opponents,groups,temporal,'f8',3100,10,32)
    assert schedule==panels(opponents,groups,temporal,'f8',3100,10,32)
    assert schedule!=panels(opponents,groups,temporal,'f8',3101,10,32)
    assert set().union(*(set(p) for p in schedule[:6]))==set(groups)
    for row in schedule:
        assert len(row)==len(set(row))==32
        assert set(temporal+['f8']).issubset(row)
        assert {groups[n] for n in row}==set(groups.values())
        w=weights_for([{'id':n} for n in row],groups,{n:.5 for n in temporal})
        assert abs(sum(w)-1)<1e-12 and min(w)>0


def test_proposals_retain_controls_and_escape_duplicate_clipped_draws():
    anchors=[np.zeros(5),np.zeros(5),np.ones(5)]
    population,indices,origins=proposals(np.random.default_rng(4),np.zeros(5),np.zeros(5),anchors,12)
    assert population.shape==(12,5) and len(set(map(tuple,population)))==12
    assert indices==[0,0,1]
    assert np.isfinite(population).all() and np.all((population>=0)&(population<=1))
    assert origins.count('uniform')>=2
