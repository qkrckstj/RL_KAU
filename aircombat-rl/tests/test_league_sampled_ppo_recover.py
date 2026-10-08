from pathlib import Path
import ast


def test_recovery_keeps_running_guard_and_declares_disjoint_environment_seeds():
    source = Path('tools/league_sampled_ppo_recover.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_seed')
    text = ast.get_source_segment(source, train)
    assert "vec.seed(seed * 1000)" in text
    assert "plan['startup_headroom_gib']" in text
    assert "< 2.8" in text
    assert 'super()._on_step()' in text


def test_recovery_initial_comparator_does_not_discard_the_better_prior_policy():
    from tools.league_sampled_ppo_recover import choose_baseline
    earlier = dict(rank=[.64, .34, 0., -4], learner='earlier')
    resumed = dict(rank=[.58, .25, 0., -5], learner='resumed')
    assert choose_baseline(resumed, earlier) is earlier
    assert choose_baseline(earlier, resumed) is earlier
    assert choose_baseline(resumed, None) is resumed
