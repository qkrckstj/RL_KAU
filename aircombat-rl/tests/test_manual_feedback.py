"""Public-state, event-driven teacher; no new simulator episodes in these tests."""
import json
import math
from pathlib import Path
import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1] / 'runs/direct_pilot_20261016'


def sample(name, tick):
    row = json.loads((BASE/name/'trace.jsonl').read_text().splitlines()[tick-1])
    return np.asarray(row['raw_after'], dtype=np.float32)


def route(name='manual_003'):
    # Actual demonstration observations at semantic maneuver boundaries.
    ticks = [1, 320, 500, 762, 822, 902, 1042, 1062]
    if name == 'manual_005': ticks = [1, 320, 505, 764, 822, 912, 1057, 1077]
    return [sample(name, n) for n in ticks]


def test_actual_public_events_reach_unilateral_tracking():
    from experiments.direct_pilot.manual_feedback_policy import Policy
    p = Policy()
    actions = [p.act(x) for x in route()]
    assert actions == [1, 2, 5, 6, 7, 8, 2, 2]
    assert p.phase == 'attack'


def test_clock_changes_do_not_replace_geometry_with_a_time_schedule():
    from experiments.direct_pilot.manual_feedback_policy import Policy
    a, b = Policy(), Policy()
    for i, x in enumerate(route()):
        moved = x.copy(); moved[38] = 110 - i * 2
        assert a.act(x) == b.act(moved)
    far = sample('manual_003', 1); far[38] = 5
    assert Policy().act(far) == 1


def rotated(x, angle):
    y = x.copy(); s, c = math.sin(angle), math.cos(angle)
    for k in (0, 15):
        for a, b in ((k, k+1), (k+3, k+4)):
            y[a], y[b] = c*x[a]+s*x[b], -s*x[a]+c*x[b]
        y[k] += 12345; y[k+1] -= 6789
        y[k+11] = math.atan2(math.sin(float(x[k+11])+angle), math.cos(float(x[k+11])+angle))
    return y


@pytest.mark.parametrize('angle', [math.pi, -.73])
def test_rotated_translated_world_has_identical_actions(angle):
    from experiments.direct_pilot.manual_feedback_policy import Policy
    a, b = Policy(), Policy()
    for x in route():
        assert a.act(x) == b.act(rotated(x, angle))


def test_repeat_query_does_not_advance_turn_progress_and_clock_reset_clears_state():
    from experiments.direct_pilot.manual_feedback_policy import Policy
    p = Policy()
    for x in route()[:5]: p.act(x)
    assert p.phase == 'right_hold'
    progress = p.turn_progress
    assert p.act(x) == 7 and p.act(x) == 7 and p.turn_progress == progress
    first = sample('manual_003', 1)
    assert p.act(first) == 1 and p.phase == 'wide'
    for x in route(): p.act(x)
    p.reset()
    assert p.act(first) == 1 and p.phase == 'wide'


def test_zero_heading_action_retains_previous_target():
    from experiments.direct_pilot.manual_feedback_policy import Policy
    p = Policy()
    xs = route()
    p.act(xs[0]); p.act(xs[1]); target = p.command_heading
    assert p.act(xs[2]) == 5
    assert p.command_heading == target
    assert abs(math.atan2(math.sin(target-xs[2][11]), math.cos(target-xs[2][11]))) > .1


def test_larger_foe_health_uses_demonstrated_braking_and_stays_frozen():
    from experiments.direct_pilot.manual_feedback_policy import Policy
    p = Policy()
    for x in route('manual_005'): action = p.act(x)
    assert p.phase == 'attack' and action == 0
    x = x.copy(); x[31] = .01; x[38] -= .05
    assert p.act(x) == 0


def test_earned_health_lead_egress_requires_geometry_and_no_lead_does_not_hold():
    from experiments.direct_pilot.manual_feedback_policy import Policy
    p = Policy()
    for x in route('manual_005'): p.act(x)
    ended = sample('manual_005', 1119)
    assert p.act(ended) == 8 and p.phase == 'egress_turn'
    safe = sample('manual_005', 1179)
    assert p.act(safe) == 5 and p.phase == 'egress_hold'
    q = Policy()
    for x in route('manual_005'): q.act(x)
    q.act(ended)
    no_lead = safe.copy(); no_lead[30] = no_lead[31]
    assert q.act(no_lead) == 8 and q.phase == 'egress_turn'


def test_gun_lead_uses_relative_velocity_and_three_dimensional_range():
    from experiments.direct_pilot.manual_feedback_policy import gun_heading
    x = np.zeros(39, dtype=np.float32); x[15:18] = [0, 1050, 1050]
    x[3] = 20; x[18] = 120
    assert gun_heading(x) == pytest.approx(math.atan2(100*math.sqrt(2), 1050))
    x[18] = x[3]
    assert gun_heading(x) == 0


@pytest.mark.parametrize('obs', [np.zeros(38), np.full(39, np.nan), np.full(39, np.inf)])
def test_invalid_observations_fail_closed(obs):
    from experiments.direct_pilot.manual_feedback_policy import Policy
    with pytest.raises(ValueError): Policy().act(obs)


@pytest.mark.parametrize('parameters', [{'seed': 4900}, {'merge_range': True}, {'attack_lead_deg': float('nan')}, {'merge_range': -1}])
def test_invalid_or_identity_based_configuration_rejected(parameters):
    from experiments.direct_pilot.manual_feedback_policy import Policy
    with pytest.raises(ValueError): Policy(parameters=parameters)
