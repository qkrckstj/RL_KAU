"""A fixed manual demonstration must replay its exact chosen actions from public time."""
from pathlib import Path
import json
import numpy as np
import pytest
BASE=Path(__file__).resolve().parents[1]/'runs/direct_pilot_20261016/manual_001'

def parameters():return dict(commands=[dict(step=r['step'],action=r['action']) for r in map(json.loads,(BASE/'commands.jsonl').read_text().splitlines())])

def test_every_manual_action_replays_exactly_from_public_clock():
    from experiments.direct_pilot.manual_replay_policy import Policy
    p=Policy(parameters=parameters());count=0
    for line in (BASE/'trace.jsonl').read_text().splitlines():
        r=json.loads(line);assert p.act(r['raw_before'])==r['action'];count+=1
    assert count==2401

def test_clock_reset_and_repeat_queries_are_stateless():
    from experiments.direct_pilot.manual_replay_policy import Policy
    p=Policy(parameters=parameters());raw=np.zeros(39,np.float32)
    raw[38]=120;first=p.act(raw)
    raw[38]=1;p.act(raw);raw[38]=120
    assert p.act(raw)==first and p.act(raw)==first

@pytest.mark.parametrize('commands',[
    [],[dict(step=1,action=4)],[dict(step=0,action=9)],
    [dict(step=0,action=4),dict(step=0,action=5)],
    [dict(step=0,action=True)],[dict(step=0,action=4),dict(step=-1,action=5)]])
def test_invalid_manual_schedules_are_rejected(commands):
    from experiments.direct_pilot.manual_replay_policy import Policy
    with pytest.raises(ValueError):Policy(parameters=dict(commands=commands))
