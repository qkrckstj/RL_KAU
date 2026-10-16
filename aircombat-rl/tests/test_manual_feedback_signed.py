"""Regression for passing the gun-lead alignment before entering firing range."""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT/'runs/direct_pilot_20261016/feedback_000/trace_failure_000/trace.jsonl'


def test_late_unilateral_entry_reverses_left_instead_of_continuing_right():
    from experiments.direct_pilot.manual_feedback_policy import Policy as Previous, gun_heading, wrap
    from experiments.direct_pilot.manual_feedback_signed_policy import Policy
    old, new = Previous(), Policy()
    for line in TAPE.read_text().splitlines():
        r = json.loads(line); x = np.asarray(r['raw_before'], np.float32)
        before = old.phase; old_action = old.act(x); new_action = new.act(x)
        if before == 'right_intercept' and x[34] > .5 and x[35] <= .5:
            error = math.degrees(wrap(gun_heading(x)-x[11]))
            assert error < -12 and old_action == 8 and old.phase == 'right_intercept'
            assert new_action == 2 and new.phase == 'attack'
            return
    raise AssertionError('The actual missed-transition observation was not exercised')


def test_signed_revision_preserves_all_earlier_actions_and_original_configuration():
    from experiments.direct_pilot.manual_feedback_policy import Policy as Previous
    from experiments.direct_pilot.manual_feedback_signed_policy import Policy
    old, new = Previous(), Policy()
    assert old.parameters == new.parameters
    for line in TAPE.read_text().splitlines():
        x = np.asarray(json.loads(line)['raw_before'], np.float32)
        a, b = old.act(x), new.act(x)
        if a != b:
            assert old.phase == 'right_intercept' and new.phase == 'attack'
            assert (a, b) == (8, 2)
            break
    else:
        raise AssertionError('Expected actual late-alignment correction')
    first = np.asarray(json.loads(TAPE.read_text().splitlines()[0])['raw_before'], np.float32)
    assert new.act(first) == 1 and new.phase == 'wide'


def test_signed_revision_keeps_rotated_world_and_repeated_query_behavior():
    from experiments.direct_pilot.manual_feedback_signed_policy import Policy
    a, b = Policy(), Policy()
    for line in TAPE.read_text().splitlines():
        x = np.asarray(json.loads(line)['raw_before'], np.float32); y = x.copy()
        for k in (0, 15):
            for i in (k, k+1, k+3, k+4): y[i] = -x[i]
            y[k+11] = math.atan2(math.sin(float(x[k+11])+math.pi), math.cos(float(x[k+11])+math.pi))
        chosen = a.act(x)
        assert chosen == a.act(x) == b.act(y)
