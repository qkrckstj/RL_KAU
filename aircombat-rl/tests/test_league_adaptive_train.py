import pytest
from tools.league_adaptive_train import development_pass, paired_ci, candidate
from tests.test_league_adaptive import config


def test_runner_improvement_cannot_hide_large_archive_regression():
    c = dict(per_opponent_win_gain={'evader': .1}, win_gain=.006,
             objective_gain=.005, lower_quarter_gain=0,
             worst_opponent_win_gain=0, evader_loss_increase=0)
    assert development_pass(c)
    assert not development_pass(dict(c, worst_opponent_win_gain=-.2))
    assert not development_pass(dict(c, lower_quarter_gain=-.03))
    assert not development_pass(dict(c, evader_loss_increase=1))
    assert not development_pass(dict(c, per_opponent_win_gain={'evader':0}))


def result(wins, foe='evader'):
    return dict(foe=foe, episodes=[dict(seed=s, seat=seat, won=win)
        for s, win in enumerate(wins) for seat in ('red', 'blue')])


def test_paired_interval_clusters_shared_conditions_across_opponents_and_seats():
    old = [result([False]*4), result([False]*4, 'ace')]
    new = [result([True]*4), result([False]*4, 'ace')]
    ci = paired_ci(new, old)
    assert ci['shared_initial_conditions'] == 4
    assert ci['mean'] == .5
    assert ci['ci95'] == [.5, .5]
    new[0]['episodes'].pop()
    with pytest.raises(ValueError, match='seed/seat'):
        paired_ci(new, old)


def test_candidate_resume_verifies_configuration_and_artifacts(tmp_path):
    c = config()
    folder = tmp_path/'model'
    one = candidate(folder, c['experts'], c['gate'], 'gate')
    assert candidate(folder, c['experts'], c['gate'], 'gate') == one
    with pytest.raises(ValueError, match='exported candidate'):
        candidate(folder, c['experts'], [1, -8, 0, 0, 0, 0], 'gate')
    (folder/'wrappers.py').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        candidate(folder, c['experts'], c['gate'], 'gate')
