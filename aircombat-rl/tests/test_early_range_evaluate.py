"""New evaluator observes both gates without changing existing episode records."""
from types import SimpleNamespace
import numpy as np
import pytest
from aircombat_gym.wvr import obs as O
from experiments.aim_async import wez_or_damaged_evaluate
import importlib.util



class ScriptedMatch:
    t_max = 120.
    track_lock = .9
    wez_r_min = 150.
    wez_r_max = 1500.

    def __init__(self, own, foe, seat):
        self._combat = SimpleNamespace(t=0.)
        self.actions = [];self.closed = False

    def state(self):
        raw = np.zeros(39, np.float32)
        raw[O.index('own_health')] = [1.,1.,1.,.75,0.][self.index]
        raw[O.index('opp_x')]=[5000.,4000.000244140625,5000.,5000.,0.][self.index]
        raw[O.index('opp_health')] = [1.,1.,1.,.8,0.][self.index]
        raw[O.index('opp_in_wez')] = [0.,0.,1.,0.,0.][self.index]
        return raw

    def info(self):
        state = self.state()
        return dict(own_health=float(state[O.index('own_health')]),
            opp_health=float(state[O.index('opp_health')]), t=self.index*.05,
            in_wez=False, track_time=0., range=1000., ata_lead=0.)

    def reset(self, seed):
        self.index = 0;self.seed = seed
        return self.state(), self.info()

    def own_action(self, obs):
        return self.index%9

    def step(self, action):
        self.actions.append(action);self.index += 1;self._combat.t = self.index*.05
        return self.state(), 0., self.index == 4, False, self.info()

    def close(self):
        self.closed = True


@pytest.mark.parametrize('seat', ['red', 'blue'])
def test_observation_only_metrics_preserve_old_actions_steps_and_records(monkeypatch, seat):
    assert importlib.util.find_spec('experiments.aim_async.early_range_evaluate') is not None,'Missing early-range evaluator'
    from experiments.aim_async import early_range_evaluate
    from experiments.aim_async.early_range_metrics import METRIC_NAMES
    created = []

    def factory(*args):
        env = ScriptedMatch(*args);created.append(env)
        return env

    for module in (wez_or_damaged_evaluate, early_range_evaluate):
        monkeypatch.setattr(module, 'MatchEnv', factory)
        monkeypatch.setattr(module, 'geometry', lambda obs:dict(tail_attack=False, head_on=True))
    old = wez_or_damaged_evaluate.cell({}, {'id':'foe'}, 42, seat)
    new = early_range_evaluate.cell({}, {'id':'foe'}, 42, seat)
    assert created[0].actions == created[1].actions == [0,1,2,3]
    assert all(env.closed for env in created)
    assert {k:v for k,v in old.items() if k != 'wall_seconds'} == {
        k:v for k,v in new.items() if k not in (*METRIC_NAMES, 'wall_seconds')}
    assert new['steps'] == new['threat_gate_decisions'] == new['wez_or_damaged_gate_decisions'] == new['early_range_gate_decisions'] == 4
    assert new['threat_gate_active_decisions'] == 1
    assert new['wez_or_damaged_gate_active_decisions'] == 2
    assert new['wez_or_damaged_gate_transitions'] == 1

    assert new['early_range_gate_active_decisions']==3 and new['early_range_gate_transitions']==1
