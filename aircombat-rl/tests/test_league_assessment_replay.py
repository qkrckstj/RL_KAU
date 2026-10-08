from copy import deepcopy
import pytest
from tools import league_assessment_replay as module


def test_unopened_stages_are_not_run_as_reproduction(tmp_path):
    module.write(tmp_path/'completion.json',dict(status='no_profile_selection',final_opened=False,temporal_opened=False))
    for stage in ('final','temporal'):
        with pytest.raises(ValueError,match='never opened'): module.stage_inputs(tmp_path,stage)


def test_replay_requires_complete_records_and_matching_schedule(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'verify',lambda plan: None)
    module.write(tmp_path/'completion.json',dict(status='no_profile_selection',final_opened=False,temporal_opened=False))
    module.write(tmp_path/'plan.json',{})
    request=dict(spec={'id':'a'},opponents=[{'id':'f'}],band=100,n=2,weights=None)
    module.write(tmp_path/'selection/schedule.json',dict(requests=[request],traced=False))
    with pytest.raises(FileNotFoundError): module.stage_inputs(tmp_path,'selection')
    bad=deepcopy(request);bad['band']+=1
    module.write(tmp_path/'selection/candidate_000.json',dict(request=bad,results=[]))
    with pytest.raises(ValueError,match='schedule'): module.stage_inputs(tmp_path,'selection')


def test_only_timing_is_excluded_from_equality():
    old=[dict(results=[dict(summary={'wins':1},episodes=[{'won':True}],elapsed_seconds=1.)])]
    new=deepcopy(old);new[0]['results'][0]['elapsed_seconds']=100.
    assert module.exact_results(old)==module.exact_results(new)
    new[0]['results'][0]['episodes'][0]['won']=False
    assert module.exact_results(old)!=module.exact_results(new)
