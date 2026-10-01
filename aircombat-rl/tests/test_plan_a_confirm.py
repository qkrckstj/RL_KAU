"""Confirmation cannot launch from a smoke test, failed gate, or reused seeds."""
from argparse import Namespace

import pytest

from tools.plan_a import write_json
from tools.plan_a_confirm import run


@pytest.mark.parametrize("smoke,next_step,seeds,message", [
    (True, "independent_seed_confirmation", [0, 1, 2], "gate"),
    (False, "validation_and_training_diagnostics", [0, 1, 2], "gate"),
    (False, "independent_seed_confirmation", [0, 1, 3], "independent"),
])
def test_confirmation_gate_rejects_before_creating_run(tmp_path, smoke, next_step, seeds, message):
    parent, out = tmp_path / "parent", tmp_path / "confirmation"
    write_json(parent / "plan.json", dict(source_sha256={}, smoke=smoke, seeds=seeds))
    write_json(parent / "summary.json", dict(decision=dict(next_step=next_step)))
    with pytest.raises(ValueError, match=message):
        run(Namespace(parent=str(parent), out=str(out), workers=3))
    assert not out.exists()
