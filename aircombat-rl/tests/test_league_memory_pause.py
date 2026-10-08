import pytest
from experiments.league.memory_pause import guarded_step,wait_for_headroom,GUARD_MESSAGE


def fake_wait(samples,maximum=300):
    rows=iter(samples)
    clock=[0.]
    events=[]
    def sleep(seconds):clock[0]+=seconds
    result=wait_for_headroom(lambda:next(rows),events.append,clock=lambda:clock[0],sleep=sleep,max_seconds=maximum)
    return result,events


def sample(commit,physical=8.):
    return dict(commit_headroom_gib=commit,available_memory_gib=physical)


def test_transition_callback_not_replayed_and_two_healthy_samples_required():
    calls=[]
    observations=[]
    def step():
        calls.append('transition_counted')
        raise MemoryError(GUARD_MESSAGE)
    def pause():
        result,events=fake_wait([sample(1.05),sample(3.),sample(1.7),sample(3.),sample(3.1)])
        observations.append(result)
        assert events[-1]['stage']=='resource_pause_recovered'
    assert guarded_step(step,pause) is True
    assert calls==['transition_counted']
    assert observations[0]['elapsed_seconds']==20


def test_critical_memory_and_persistent_pressure_abort():
    with pytest.raises(MemoryError,match='Critical'):
        fake_wait([sample(.4)])
    with pytest.raises(MemoryError,match='Critical'):
        fake_wait([sample(3.,.7)])
    with pytest.raises(MemoryError,match='timeout'):
        fake_wait([sample(1.2)]*3,maximum=10)


def test_unrelated_memory_errors_are_not_swallowed():
    def unrelated():raise MemoryError('different allocation failure')
    with pytest.raises(MemoryError,match='different allocation'):
        guarded_step(unrelated,lambda:pytest.fail('Unexpected wait'))
    assert guarded_step(lambda:False,lambda:pytest.fail('Unexpected wait')) is False
