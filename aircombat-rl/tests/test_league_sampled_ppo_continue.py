from tools.league_sampled_ppo_continue import pair_rank, should_extend


def test_replicas_are_averaged_and_weakness_uses_worse_replica():
    common = dict(mean_win_rate=.9, group_balanced_win_rate=.9, lower_quarter_score=.7, worst_score=.4, losing_matchups=1)
    weak = dict(mean_win_rate=.1, group_balanced_win_rate=.3, lower_quarter_score=.1, worst_score=0., losing_matchups=8)
    assert pair_rank([common, weak]) == [.55, .1, 0., -8]
    assert pair_rank([common, weak]) == pair_rank([weak, common])


def test_flat_progress_or_worsened_floor_does_not_extend_learning():
    baseline = [.5, .2, 0., -5]
    first = [.55, .3, .1, -4]
    assert not should_extend(first, first, baseline)
    assert not should_extend(first, [.6, .2, .1, -3], baseline)
    assert not should_extend(first, [.6, .3, 0., -3], baseline)
    assert should_extend(first, [.56, .3, .1, -3], baseline)
