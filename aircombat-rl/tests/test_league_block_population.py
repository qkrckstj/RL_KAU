import numpy as np
from tools.league_block_population import sample_population,BLOCKS


def test_block_changes_preserve_other_coordinates_and_distinctness():
    mean=np.full(17,.5);std=np.full(17,.015)
    pop,anchors,origins=sample_population(np.random.default_rng(7),mean,std,[mean,mean],12)
    assert anchors==[0,0] and len({tuple(v) for v in pop})==12
    assert len(BLOCKS)==4 and sorted(i for b in BLOCKS for i in b)==list(range(17))
    assert sum(o['kind']=='uniform' for o in origins)==1
    for v,o in zip(pop,origins):
        if o['kind']=='block':
            outside=[i for i in range(17) if i not in o['indices']]
            assert np.array_equal(v[outside],mean[outside])
    again=sample_population(np.random.default_rng(7),mean,std,[mean,mean],12)
    assert np.array_equal(pop,again[0])


def test_zero_variance_does_not_hang():
    mean=np.zeros(17)
    pop,_,_=sample_population(np.random.default_rng(8),mean,np.zeros(17),[mean],6)
    assert len({tuple(v) for v in pop})==6
