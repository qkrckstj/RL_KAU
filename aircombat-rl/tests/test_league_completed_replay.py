from copy import deepcopy
import pytest
from tools import league_completed_replay as module


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'verify',lambda plan:None)
    module.write(tmp_path/'completion.json',dict(status='development_pilot_complete',final_opened=False))
    module.write(tmp_path/'plan.json',{})
    request=dict(spec={'id':'a'},opponents=[{'id':'f'}],band=100,n=2,weights=None)
    record=dict(request=request,results=[dict(own='a',foe='f',summary={'n':2},
        episodes=[{'seed':100,'seat':'red'},{'seed':100,'seat':'blue'}],elapsed_seconds=1.)])
    module.write(tmp_path/'confirmation/schedule.json',dict(requests=[request],traced=False))
    module.write(tmp_path/'confirmation/candidate_000.json',record)
    return record


def test_terminal_pilot_with_complete_pairs_can_be_replayed(tmp_path,monkeypatch):
    expected=fixture(tmp_path,monkeypatch)
    _,_,records=module.stage_inputs(tmp_path,'confirmation')
    assert records==[expected]


def test_active_experiment_unopened_holdout_and_missing_pairs_rejected(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='complete'):module.stage_inputs(tmp_path,'screen')
    record=fixture(tmp_path,monkeypatch)
    for stage in ('final','parameter_audit'):
        with pytest.raises(ValueError,match='never opened'):module.stage_inputs(tmp_path,stage)
    with pytest.raises(ValueError,match='Unknown'):module.stage_inputs(tmp_path,'../confirmation')
    record['results'][0]['episodes'].pop()
    module.write(tmp_path/'confirmation/candidate_000.json',record)
    with pytest.raises(ValueError,match='paired'):module.stage_inputs(tmp_path,'confirmation')


def test_only_runtime_timing_excluded_not_episode_changes():
    a=[dict(results=[dict(episodes=[{'steps':8,'own_health':.4}],summary={'n':1},elapsed_seconds=1.)])]
    b=deepcopy(a);b[0]['results'][0]['elapsed_seconds']=5.
    assert module.exact_results(a)==module.exact_results(b)
    b[0]['results'][0]['episodes'][0]['steps']=9
    assert module.exact_results(a)!=module.exact_results(b)
