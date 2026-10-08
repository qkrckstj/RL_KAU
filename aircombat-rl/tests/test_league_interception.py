import math
import numpy as np
import pytest
from experiments.league.interception import Policy, intercept_time


@pytest.mark.parametrize('geometry,expected', [
    ((3000, 4000, 0, 0, 250, 90), 20),
    ((1000, 0, 100, 0, 200, 90), 10),
    ((1000, 0, -100, 0, 200, 90), 1000/300),
    ((1000, 0, -200, 0, 200, 90), 2.5),
    ((1000, 0, 200, 0, 200, 90), 90),
    ((1000, 0, 300, 0, 200, 90), 90),
    ((1000, 0, 0, 300, 200, 90), 90),
    ((0, 0, 100, 100, 200, 90), 0),
    ((10000, 0, 100, 0, 200, 20), 20),
])
def test_analytic_intercepts_and_unreachable_fallback(geometry, expected):
    assert intercept_time(*geometry) == pytest.approx(expected)


def test_crossing_target_meets_distance_equation():
    t = intercept_time(3000, 2000, -60, 100, 250, 90)
    assert math.hypot(3000-60*t, 2000+100*t) == pytest.approx(250*t)


def test_reflection_reverses_turn_without_changing_throttle():
    obs = np.zeros(39)
    obs[4] = 250
    obs[15:17] = [3000, 8000]
    obs[18:20] = [100, 100]
    mirrored = obs.copy()
    mirrored[[0, 3, 15, 18, 11, 14]] *= -1
    a, b = Policy().act(obs), Policy().act(mirrored)
    assert a//3 == 2 and b//3 == 0
    assert a%3 == b%3


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        Policy(parameters=[float('nan')]*7)
    with pytest.raises(ValueError):
        intercept_time(1, 0, 0, 0, -1, 30)


def test_standalone_export_uses_official_loader(tmp_path):
    from tools.league_interception_export import export
    from tools.policies import load
    from experiments.league.interception import INITIAL
    folder = tmp_path/'interceptor'
    export(folder, INITIAL, 'test')
    act, _, mode = load(folder, folder/'policy_net.zip')
    assert mode == 'discrete'
    direct = Policy()
    rng = np.random.default_rng(90)
    for _ in range(50):
        obs = rng.normal(size=39)
        obs[[0, 1, 15, 16]] *= 5000
        obs[[3, 4, 18, 19]] *= 200
        assert act(obs) == direct.act(obs)
    with pytest.raises(FileExistsError):
        export(folder, INITIAL, 'test')
