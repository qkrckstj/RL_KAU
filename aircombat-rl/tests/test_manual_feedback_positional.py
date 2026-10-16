"""Catch unnecessary circling when a real recorded egress is already opening."""
import json
import math
from pathlib import Path
import numpy as np
import pytest

TAPE = Path(__file__).resolve().parents[1]/'runs/direct_pilot_20261016/feedback_001/trace_failure_000/trace.jsonl'


def before_missed_hold(cls):
    rows = [json.loads(s) for s in TAPE.read_text().splitlines()]
    p = cls()
    for row in rows[:1282]: p.act(row['raw_before'])
    return p, np.asarray(rows[1282]['raw_before'], dtype=np.float32)


def test_actual_opening_geometry_stops_turning_before_the_later_loss():
    from experiments.direct_pilot.manual_feedback_signed_policy import Policy as Previous
    from experiments.direct_pilot.manual_feedback_positional_policy import Policy
    old, x = before_missed_hold(Previous)
    new, _ = before_missed_hold(Policy)
    assert old.act(x) == 8 and old.phase == 'egress_turn'
    target = new.command_heading
    assert new.act(x) == 5 and new.phase == 'egress_hold'
    assert new.command_heading == target  # native hold retains the commanded target


@pytest.mark.parametrize('unsafe', ['enemy_can_shoot', 'closing', 'no_lead', 'target_ahead', 'enemy_points_at_own'])
def test_opening_egress_refuses_each_unsafe_geometry(unsafe):
    from experiments.direct_pilot.manual_feedback_positional_policy import Policy
    p, x = before_missed_hold(Policy)
    if unsafe == 'enemy_can_shoot': x[35] = 1
    if unsafe == 'closing':
        dx, dy = float(x[15]-x[0]), float(x[16]-x[1]); r = math.hypot(dx, dy)
        x[18], x[19] = x[3]-100*dx/r, x[4]-100*dy/r
    if unsafe == 'no_lead': x[30] = x[31]
    if unsafe == 'target_ahead': x[11] = math.atan2(float(x[15]-x[0]), float(x[16]-x[1]))
    if unsafe == 'enemy_points_at_own': x[26] = math.atan2(float(x[0]-x[15]), float(x[1]-x[16]))
    assert p.act(x) == 8 and p.phase == 'egress_turn'


def test_positional_change_preserves_earlier_actions_configuration_and_reset():
    from experiments.direct_pilot.manual_feedback_signed_policy import Policy as Previous
    from experiments.direct_pilot.manual_feedback_positional_policy import Policy
    old, new = Previous(), Policy()
    assert {k:v for k,v in old.parameters.items() if k != 'egress_heading_deg'} == {
        k:v for k,v in new.parameters.items() if k != 'egress_foe_bearing_deg'}
    assert 'egress_heading_deg' not in new.parameters
    for line in TAPE.read_text().splitlines():
        x = np.asarray(json.loads(line)['raw_before'], np.float32)
        a, b = old.act(x), new.act(x)
        if a != b:
            assert (a,b) == (8,5) and old.phase == 'egress_turn' and new.phase == 'egress_hold'
            break
    else: raise AssertionError('Recorded missed hold was not exercised')
    first = np.asarray(json.loads(TAPE.read_text().splitlines()[0])['raw_before'], np.float32)
    assert new.act(first) == 1 and new.phase == 'wide'


def test_positional_gate_rotation_and_repeated_query_remain_equivalent():
    from experiments.direct_pilot.manual_feedback_positional_policy import Policy
    p, q = Policy(), Policy()
    angle = -.73; s, c = math.sin(angle), math.cos(angle)
    for line in TAPE.read_text().splitlines():
        x = np.asarray(json.loads(line)['raw_before'], np.float32); y = x.copy()
        for k in (0,15):
            for i,j in ((k,k+1),(k+3,k+4)):
                y[i], y[j] = c*x[i]+s*x[j], -s*x[i]+c*x[j]
            y[k+11] = math.atan2(math.sin(float(x[k+11])+angle), math.cos(float(x[k+11])+angle))
        a = p.act(x)
        assert a == p.act(x) == q.act(y)
