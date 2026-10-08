from copy import deepcopy
from concurrent.futures import Future
import numpy as np
import pytest
from tools import league_repair_train as module
from tools import league_reliable_evaluation as reliable
from tools.league_train import evaluate_candidates as original_evaluate
from tests.test_league_regret_train import record,FOES,BASE,TRAP


@pytest.mark.parametrize('training_favors_warm_start', [False, True])
def test_rejected_warm_start_does_not_replace_reference_and_anchor_is_paired(tmp_path,monkeypatch,training_favors_warm_start):
    monkeypatch.setattr(module,'verify',lambda plan:None)
    anchor=(module.LOW+module.HIGH)/2
    warm=anchor.copy();warm[0]+=1
    conservative=anchor.copy();conservative[11]+=100
    cfg=lambda p:dict(parent=p.tolist(),interceptor=[0.]*7)
    plan=dict(initial_configuration=cfg(warm),anchor_configuration=cfg(anchor),
        conservative_configuration=cfg(conservative),anchor=dict(id='anchor'),
        warm_start=dict(id='warm'),conservative=dict(id='conservative'),
        population=12,elites=3,initial_std=.015,minimum_std=.003,
        uniform_immigrants=1,opening_proposal_scale=.05,repair_priorities={'pursuit':1.},
        maximum_generations=2,base_generations=6,training_band=1000,development_band=100,
        screen_n=4,confirmation_n=8,development_n=20,opponents=[dict(id=f) for f in FOES])
    monkeypatch.setattr(module,'candidate',lambda root,p,v:dict(id=module.digest(np.asarray(v).tolist())))
    calls=[]
    def evaluate(pool,specs,foes,band,n,out,weights=None):
        calls.append((out.name,[s['id'] for s in specs]))
        challenger_rates = ([1., .5, 1.] + [.6] * 5
                            if training_favors_warm_start and weights is not None else TRAP)
        return [record(BASE if s['id']=='anchor' else challenger_rates,s['id'],n,band,weights) for s in specs]
    base=record(BASE,'anchor')
    result=module.search(tmp_path/'search',tmp_path,plan,3,0,[base,deepcopy(base)],evaluate,None)
    assert result['selected']['spec']['id']=='anchor'
    assert all('anchor' in ids for stage,ids in calls if stage in ('screen','confirm'))
    assert all('warm' in ids for stage,ids in calls if stage=='screen')
    assert not module.checkpoint_admissible(record(TRAP,'warm'), [base, deepcopy(base)])
    assert result['history'][-1]['development_admissible'] is (not training_favors_warm_start)
    assert (result['history'][-1]['challenger']['id']=='anchor') is (not training_favors_warm_start)


def test_final_choice_and_both_reference_guards_remain(tmp_path,monkeypatch):
    import tests.test_league_maneuver_train as prior_tests
    monkeypatch.setattr(prior_tests,'module',module)
    prior_tests.test_run_freezes_choice_and_compares_both_baselines_with_traced_final(tmp_path,monkeypatch)


def test_reliable_evaluator_matches_original_and_resumes_without_rerunning(tmp_path):
    class Pool:
        def __init__(self,allowed=True):self.calls=0;self.allowed=allowed
        def submit(self,fn,spec,foe,band,n):
            assert self.allowed
            self.calls+=1
            row=record(BASE,spec['id'],n,band)['results'][FOES.index(foe['id'])]
            row.update(own=spec['id'],band=band,elapsed_seconds=0.)
            future=Future();future.set_result(row);return future
    candidates=[dict(id='one'),dict(id='two')];foes=[dict(id=f) for f in FOES[:3]]
    a,b=Pool(),Pool()
    old=original_evaluate(a,candidates,foes,100,8,tmp_path/'old')
    new=reliable.evaluate_candidates(b,candidates,foes,100,8,tmp_path/'new')
    assert old==new and a.calls==b.calls==6
    (tmp_path/'new/candidate_001.json').unlink()
    assert reliable.evaluate_candidates(Pool(False),candidates,foes,100,8,tmp_path/'new')==new
    with pytest.raises(ValueError):
        reliable.evaluate_candidates(Pool(False),candidates,foes,101,8,tmp_path/'new')
