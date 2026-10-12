import numpy as np
import pytest
from tools import league_thread_benchmark as io
from experiments.league.tail_actual_reward import make_worker as baseline
from experiments.league.rear_geometry_opponents import make_worker,blend_weight
from experiments.league.periodic_coverage import PeriodicCoverageSampler

@pytest.mark.parametrize('index',[0,1])
def test_disabled_schedule_preserves_full_episodes_and_seed_reset(index):
    p=io.read(io.ROOT/'runs/league_tail_actual_reward_20261012/plan.json')
    p=dict(p,seed=21,ic_band=915000000,workers=2,rear_damage_coefficient=.5,opponents=[dict(id='ace',kind='bot',name='ace')],probabilities=[1.],curriculum=dict(stage='quarter',fraction=.5),opponent_schedule=dict(enabled=False,focus_ids=['ace'],quarter_focus_ids=['ace'],fade_steps_per_env=10240))
    a=baseline(p,index);b=make_worker(p,index)
    try:
        for seed in (None,123):
            x,_=a.reset(seed=seed);y,j=b.reset(seed=seed);np.testing.assert_array_equal(x,y)
            assert j['league_episode']['schedule_start_step']==0
            for step in range(2410):
                act=(step//17)%9
                x,rx,tx,ux,ix=a.step(act);y,ry,ty,uy,iy=b.step(act)
                np.testing.assert_array_equal(x,y);assert (rx,tx,ux)==(ry,ty,uy)
                assert (ix['own_health'],ix['opp_health'])==(iy['own_health'],iy['opp_health'])
                if tx or ux:
                    assert ix['result']==iy['result'] and ix['actual_rear_damage']==iy['actual_rear_damage'];break
            else:raise AssertionError('No terminal')
    finally:a.close();b.close()

def test_schedule_boundaries_and_archive_coverage():
    assert [blend_weight(n,10240,True) for n in (0,5120,10240,20480)]==[1.,.5,0.,0.]
    assert blend_weight(0,10240,False)==0
    for period in (2,4):
        a=PeriodicCoverageSampler([1,2,3,4],99,period)
        b=PeriodicCoverageSampler([1,2,3,4],99,period)
        b.probabilities=np.array([.5,.5,0,0])
        ca=[];cb=[]
        for i in range(160):
            x,y=a.next(),b.next()
            if i%period==0:ca.append(x);cb.append(y)
            else:assert y in (0,1)
        assert ca==cb
        for i in range(0,len(cb),4):assert sorted(cb[i:i+4])==[0,1,2,3]


def test_conditional_draws_preserve_geometry_and_coverage():
    from experiments.league.rear_geometry_opponents import GeometryOpponentArchive
    from experiments.league.tail_curriculum import CurriculumArchive
    teacher=dict(id='prior',kind='fixed',action=6)
    foes=[dict(id=n,kind='fixed',action=a) for n,a in [('a',0),('b',3),('c',6),('d',8)]]
    args=(teacher,foes,[.25]*4,12,441000000,0,1,'red')
    kw=dict(coverage_period=2,curriculum=dict(stage='quarter',fraction=.5))
    a=GeometryOpponentArchive(*args,**kw,opponent_schedule=dict(enabled=True,focus_ids=['a','b'],quarter_focus_ids=['c','d'],fade_steps_per_env=10240))
    b=CurriculumArchive(*args,**kw)
    seen=set()
    try:
        for k in range(32):
            x,i=a.reset();y,j=b.reset();np.testing.assert_array_equal(x,y)
            meta=i['league_episode'];seen.add(meta['curriculum_geometry'])
            if k%2==0:assert meta['opponent_id']==j['league_episode']['opponent_id']
            else:assert meta['opponent_id'] in meta['opponent_focus_ids']
        assert seen=={'neutral','training_tail_quarter'}
        a.physical_steps=10240
        _,i=a.reset();assert i['league_episode']['opponent_focus_weight']==0
        np.testing.assert_array_equal(a.sampler.probabilities,a.base_probabilities)
    finally:a.close();b.close()
