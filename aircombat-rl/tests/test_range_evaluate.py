"""8km accounting adds no native step or policy decision."""
import importlib.util
import numpy as np
import pytest
from aircombat_gym.wvr import obs as O
from experiments.aim_async import early_range_evaluate
from test_early_range_evaluate import ScriptedMatch


class RangeMatch(ScriptedMatch):
    def state(self):
        raw=super().state();raw[O.index('opp_x')]=[6000.,4000.,9000.,9000.,0.][self.index]
        return raw


@pytest.mark.parametrize('seat',['red','blue'])
def test_four_gates_leave_old_actions_and_episode_records_equal(monkeypatch,seat):
    assert importlib.util.find_spec('experiments.aim_async.range_evaluate') is not None,'Missing range evaluator'
    from experiments.aim_async import range_evaluate
    from experiments.aim_async.range_metrics import METRIC_NAMES
    created=[]
    def factory(*args):
        env=RangeMatch(*args);created.append(env);return env
    for module in (early_range_evaluate,range_evaluate):
        monkeypatch.setattr(module,'MatchEnv',factory)
        monkeypatch.setattr(module,'geometry',lambda obs:dict(tail_attack=False,head_on=True))
    old=early_range_evaluate.cell({},dict(id='foe'),42,seat);new=range_evaluate.cell({},dict(id='foe'),42,seat)
    assert created[0].actions==created[1].actions==[0,1,2,3] and all(env.closed for env in created)
    assert {k:v for k,v in old.items() if k!='wall_seconds'}=={k:v for k,v in new.items() if k not in (*METRIC_NAMES,'wall_seconds')}
    assert [new[p+'_active_decisions'] for p in ('threat_gate','wez_or_damaged_gate','early_range_gate','range_gate')]==[1,2,3,4]
    assert new['range_gate_decisions']==4 and new['range_gate_transitions']==0
