import random
import numpy as np
import pytest
import torch

from experiments.league.residual_worker import isolated_rng
from tools.league_residual_train import development_panel, extend_budget, validate_records


def test_evaluation_rng_restore_even_on_error():
    random.seed(331);np.random.seed(331);torch.manual_seed(331)
    expected=(random.random(),np.random.random(),torch.rand(3))
    random.seed(331);np.random.seed(331);torch.manual_seed(331)
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.random();np.random.random(900);torch.rand(100)
            raise RuntimeError('intentional evaluation failure')
    assert random.random()==expected[0]
    assert np.random.random()==expected[1]
    assert torch.equal(torch.rand(3),expected[2])


def test_sparse_panel_covers_families_and_extension_requires_recent_gain():
    foes=[dict(id=f'f{i}') for i in range(30)]
    groups={f['id']:str(i//3) for i,f in enumerate(foes)}
    chosen=development_panel(foes,groups,['f2','f5'],'f29',size=15)
    ids={s['id'] for s in chosen}
    assert len(ids)==15 and {'f2','f5','f29'}<=ids
    assert {groups[x] for x in ids}==set(groups.values())
    assert not extend_budget([.7,.4],[.7,.4],[.6,.3])
    assert not extend_budget([.7,.4],[.706,.39],[.6,.3])
    assert not extend_budget([.5,.4],[.506,.4],[.51,.3])
    assert extend_budget([.7,.4],[.706,.4],[.6,.3])


def test_evaluation_rejects_missing_seat_or_unpaired_initial_conditions():
    foes=[{'id':'example'}]
    with pytest.raises(ValueError,match='Incomplete evaluation'):
        validate_records([],foes,100,2)
    with pytest.raises(ValueError,match='IC/seat'):
        validate_records([dict(foe='example',episodes=[{'seed':100,'seat':'red'}, {'seed':101,'seat':'blue'}])],foes,100,2)
