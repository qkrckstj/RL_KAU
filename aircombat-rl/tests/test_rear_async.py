from pathlib import Path
import pytest
from experiments.rear_async.storage import publish,verify,identity
from experiments.rear_async.judgement import candidate_status,budget_decision,continuation_limit

def test_partial_checkpoint_hidden_and_corrupt_checkpoint_rejected(tmp_path):
    stage=tmp_path/'.writing';stage.mkdir();(stage/'learner.zip').write_bytes(b'weights')
    with pytest.raises(FileNotFoundError):verify(stage)
    final=tmp_path/'step_1';publish(stage,final,dict(step=1))
    assert verify(final)['step']==1
    (final/'learner.zip').write_bytes(b'changed')
    with pytest.raises(ValueError):verify(final)

def test_job_identity_changes_with_config_but_not_dictionary_order():
    assert identity(dict(step=1,config='a'))==identity(dict(config='a',step=1))
    assert identity(dict(step=1,config='a'))!=identity(dict(step=1,config='b'))

def test_one_point_gain_is_kept_without_claiming_significance():
    assert candidate_status(.01)['candidate']
    assert not candidate_status(.01)['improvement_confirmed']
    assert candidate_status(-.05)['model_retained']

def point(rear,received=.4,step=1):
    return dict(additional_steps=step,summary=dict(rear_damage=rear,clean_rear_wins=0,
        max_rear_wez_streak=0.,rear_damage_games=1,damage_received=received))

def test_minimum_budget_is_not_shortened_by_low_win_rate():
    assert budget_decision([point(0)],point(1)['summary'],100,5242880,10485760)['decision']=='continue_minimum'

def test_rear_progress_can_extend_and_plateau_cannot_extend_forever():
    baseline=point(.1)['summary'];points=[point(.1,step=2),point(.2,step=3),point(.3,step=4)]
    assert budget_decision(points,baseline,5242880,5242880,10485760)['decision']=='extend'
    assert budget_decision([point(.1)]*3,baseline,5242880,5242880,10485760)['decision']=='review_plateau'
    assert budget_decision(points,baseline,10485760,5242880,10485760)['decision']=='maximum_reached'

def test_resume_keeps_previously_approved_extension_limit():
    assert continuation_limit(5242880,10485760,dict(extend=True))==10485760
    assert continuation_limit(5242880,10485760,None)==5242880

def test_failed_evaluation_reuses_completed_cells_and_duplicate_job_does_no_work(tmp_path,monkeypatch):
    from experiments.rear_async.storage import write,read
    from experiments.rear_async import evaluate as metrics
    from tools.rear_async_cpu import evaluate_job
    stage=tmp_path/'.staging';stage.mkdir();(stage/'learner.zip').write_bytes(b'weight')
    checkpoint=tmp_path/'checkpoint'
    publish(stage,checkpoint,dict(config_id='config',checkpoint_id='id',seed=7,prefix_steps=0,
        additional_steps=10,baseline=False,sampled_specs={str(a):{} for a in (4900,4901)}))
    write(tmp_path/'started.json',dict(time=0))
    foe=dict(id='foe')
    write(tmp_path/'plan.json',dict(frozen_sha256={},development_opponents=[foe],development_ics=1,
        development_band=100,monitor_band=200,monitor_ics=1,monitor_opponents=[foe],processes=8))
    calls=[];fail=[True]
    def cell(own,foe,seed,seat):
        calls.append((own['weights'],seat))
        if len(calls)==2 and fail[0]:
            fail[0]=False;raise RuntimeError('Injected evaluation failure')
        return dict(seed=seed,seat=seat,opponent=foe['id'],result='win',own_health=1.,opp_health=0.,
            rear_damage=.1,rear_wez_seconds=1.,max_rear_wez_streak=1.,damage_dealt=1.,damage_received=0.,
            head_on_damage=0.,clean_rear_win=True,rear_before_first_received=True)
    monkeypatch.setattr(metrics,'cell',cell)
    job=dict(checkpoint=str(checkpoint),checkpoint_id='id',kind='development',config_id='config',job_id='job')
    with pytest.raises(RuntimeError):evaluate_job(tmp_path,job)
    assert len(list((tmp_path/'evaluations/job/cells').glob('*.json')))==1
    evaluate_job(tmp_path,job)
    assert len(calls)==5  # Onefailedcall plusfourdistinctsuccessfulcells.
    result=read(tmp_path/'evaluations/job/completion.json')
    assert result['actual_new_games']==4 and result['reused_cells_in_final_attempt']==1
    evaluate_job(tmp_path,job);assert len(calls)==5
