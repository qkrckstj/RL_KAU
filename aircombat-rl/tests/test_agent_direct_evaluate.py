"""Extra unilateral metrics preserve the existing native evaluator's decisions."""
import importlib.util
from experiments.aim_async import range_evaluate
from test_range_evaluate import RangeMatch


def test_observation_only_unilateral_metrics_preserve_original_episode(monkeypatch):
    assert importlib.util.find_spec('experiments.direct_pilot.evaluate') is not None,'Missing direct pilot evaluator'
    from experiments.direct_pilot import evaluate
    created=[]
    def factory(*args):
        env=RangeMatch(*args);created.append(env);return env
    for m in (range_evaluate,evaluate):
        monkeypatch.setattr(m,'MatchEnv',factory);monkeypatch.setattr(m,'geometry',lambda obs:dict(tail_attack=False,head_on=True))
    old=range_evaluate.cell({},dict(id='foe'),42,'red');new=evaluate.cell({},dict(id='foe'),42,'red')
    assert all(new[k]==v for k,v in old.items() if k!='wall_seconds')
    assert created[0].actions==created[1].actions and all(e.closed for e in created)
    assert new['first_posthit_unilateral_time'] is None and new['max_posthit_rear_unilateral_streak']==0.


def test_posthit_unilateral_rear_sequence_and_received_damage_are_measured(monkeypatch):
    import pytest
    from experiments.direct_pilot import evaluate
    class SuccessMatch(RangeMatch):
        def state(self):
            raw=super().state();raw[30]=[1.,.95,.95,.95,.95][self.index]
            raw[31]=[1.,1.,.9,.8,.7][self.index]
            raw[34]=[0.,0.,1.,1.,1.][self.index];raw[35]=[0.,1.,0.,0.,0.][self.index]
            return raw
    monkeypatch.setattr(evaluate,'MatchEnv',SuccessMatch)
    monkeypatch.setattr(evaluate,'geometry',lambda obs:dict(tail_attack=bool(obs[34]>.5),head_on=False))
    row=evaluate.cell({},dict(id='foe'),42,'blue')
    assert row['first_received_time']==pytest.approx(.05)
    assert row['first_posthit_unilateral_time']==pytest.approx(.1)
    assert row['damage_received_before_first_posthit_unilateral']==pytest.approx(.05)
    assert row['max_posthit_rear_unilateral_streak']==pytest.approx(.15)
    assert row['max_unilateral_wez_streak']==pytest.approx(.15)


def test_light_damage_and_two_second_hold_must_belong_to_same_segment(monkeypatch):
    from experiments.direct_pilot import evaluate
    import numpy as np
    class InterruptedMatch(RangeMatch):
        def state(self):
            raw=np.zeros(39,np.float32);raw[15]=900.;raw[30]=1. if self.index==0 else .95 if self.index<4 else .6
            raw[31]=1.-.002*self.index;raw[34]=float(self.index>=2);raw[38]=120.-.05*self.index
            return raw
        def step(self,action):
            self.actions.append(action);self.index+=1;self._combat.t=.05*self.index
            return self.state(),0.,self.index==50,False,self.info()
    monkeypatch.setattr(evaluate,'MatchEnv',InterruptedMatch)
    monkeypatch.setattr(evaluate,'geometry',lambda obs:dict(tail_attack=bool(obs[34]>.5),head_on=False))
    row=evaluate.cell({},dict(id='foe'),42,'red')
    assert row['damage_received_before_first_posthit_unilateral']<.25
    assert row['max_posthit_rear_unilateral_streak']>=2.
    assert row['max_light_posthit_rear_unilateral_streak']<2.
