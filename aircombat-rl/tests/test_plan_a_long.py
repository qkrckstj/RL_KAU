"""Guard the duration-only comparison, especially epsilon and budget leakage."""
from types import SimpleNamespace

import pytest
from stable_baselines3.common.utils import LinearSchedule

from tools.plan_a import set_exploration_duration
from tools.plan_a_long import select_at_budget


def test_longer_budgets_preserve_original_absolute_exploration_schedule():
    original = LinearSchedule(1.0, 0.05, 0.1)
    for budget in (204800, 512000, 1024000):
        model = SimpleNamespace(exploration_initial_eps=1.0, exploration_final_eps=0.05)
        set_exploration_duration(model, budget, 20480)
        for step in (0, 1, 1000, 10240, 20479, 20480, 20481, 100000, 204800):
            assert model.exploration_schedule(1 - step / budget) == pytest.approx(
                original(1 - step / 204800), abs=1e-12)
        assert model.exploration_schedule(0) == 0.05


def test_selection_cannot_see_future_or_select_untrained_policy():
    def row(step, kills, time):
        return dict(step=step, summary=dict(kills=kills, t_kill=time))
    history = [row(0, 20, 1), row(51200, 2, 90), row(102400, 3, 100),
               row(153600, 3, 90), row(204800, 3, 90), row(512000, 10, 80)]
    assert select_at_budget(history, 204800)["step"] == 153600
    assert select_at_budget(history, 512000)["step"] == 512000
    assert select_at_budget([row(51200, 0, None), row(102400, 0, None)], 102400)["step"] == 51200
    with pytest.raises(ValueError):
        select_at_budget(history, 1)
