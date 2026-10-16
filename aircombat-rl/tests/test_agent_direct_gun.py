"""Public lead reconstruction and disabled-override control equivalence."""
from pathlib import Path
from types import SimpleNamespace
import json,math
import numpy as np
import pytest

def test_public_gun_heading_matches_official_lead_with_relative_velocity():
    from experiments.direct_pilot.gun_policy import gun_heading
    from aircombat_gym.wvr.engagement import look
    raw=np.zeros(39,np.float32);raw[:6]=[10,20,3000,200,10,0];raw[15:21]=[310,920,3010,-150,100,1]
    def aircraft(offset):
        return SimpleNamespace(x=float(raw[offset]),y=float(raw[offset+1]),h=float(raw[offset+2]),vx=float(raw[offset+3]),vy=float(raw[offset+4]),vz=float(raw[offset+5]),psi=0.,theta=0.)
    e=look(aircraft(0),aircraft(15))
    assert gun_heading(raw)==pytest.approx(math.atan2(e.aim_dx,e.aim_dy),abs=1e-12)

@pytest.mark.parametrize('mode',[0,1])
def test_disabled_gun_override_preserves_full_original_or_setpoint_sequence(mode):
    from experiments.direct_pilot.gun_policy import Policy
    from experiments.direct_pilot.predictive_policy import Policy as Original
    from experiments.direct_pilot.setpoint_policy import Policy as Setpoint
    reference=(Original if mode==0 else Setpoint)(parameters=dict(lead=6.,break_range=0.))
    candidate=Policy(parameters=dict(lead=6.,break_range=0.,gun_range=0.,command_mode=mode))
    root=Path(__file__).resolve().parents[1]/'runs/direct_pilot_20261016/traces_000'
    tapes=sorted(root.glob('agent*/trace.jsonl'));assert len(tapes)==3
    for tape in tapes:
        for line in tape.read_text().splitlines():
            raw=json.loads(line)['raw'];assert candidate.act(raw)==reference.act(raw)

def test_gun_command_mode_rejects_intermediate_value():
    from experiments.direct_pilot.gun_policy import Policy
    with pytest.raises(ValueError):Policy(parameters=dict(command_mode=.5))
