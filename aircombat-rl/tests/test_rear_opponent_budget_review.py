import copy

from pytest import raises


def metric(rear, streak, received=0.):
    return dict(games=16, rear_damage=rear, max_rear_wez_streak=streak,
                clean_rear_wins=16 if rear > 0 else 0,
                rear_damage_games=16 if rear > 0 else 0, damage_received=received)


def panel():
    baseline = dict(per_opponent={'fixed_right': metric(.5848, 1.8094),
                                  'circler': metric(.8154, 3.1531)})
    points = [dict(additional_steps=s, per_opponent={'fixed_right': metric(r, t),
                                                   'circler': metric(.03, .14)})
              for s, r, t in [(2621440, .8993, 3.0187),
                              (3932160, .9959, 3.8625),
                              (5242880, .9468, 3.6)]]
    return baseline, points


def test_real_local_skill_gain_survives_other_opponent_regression_and_caps():
    from experiments.rear_opponent_budget_review import review
    b, p = panel()
    before = copy.deepcopy((b, p))
    d = review(p, b, 5242880, 5242880, 10485760)
    assert d['extend'] and d['qualifying_opponents'] == ['fixed_right']
    assert 'circler' in d['declining_opponents']
    assert (b, p) == before
    assert not review(p, b, 5242879, 5242880, 10485760)['extend']
    assert not review(p, b, 10485760, 5242880, 10485760)['extend']


def test_zero_rear_skill_and_lower_received_damage_do_not_qualify():
    from experiments.rear_opponent_budget_review import review
    b = dict(per_opponent={'ace': metric(0., 0., 1.)})
    p = [dict(additional_steps=i, per_opponent={'ace': metric(0., 0., .8)}) for i in (1, 2, 3)]
    assert not review(p, b, 3, 3, 6)['extend']


def test_mismatched_opponent_panels_are_not_comparable():
    from experiments.rear_opponent_budget_review import review
    b, p = panel()
    p[1]['per_opponent']['fixed_right']['games'] = 8
    with raises(ValueError, match='games'):
        review(p, b, 5242880, 5242880, 10485760)


def test_forgotten_skill_does_not_extend_the_latest_model():
    from experiments.rear_opponent_budget_review import review
    b = dict(per_opponent={'residual': metric(.223, .6906)})
    p = [dict(additional_steps=i, per_opponent={'residual': metric(r, t)})
         for i, r, t in [(1, .2531, .95), (2, .1755, .7125), (3, 0., 0.)]]
    assert not review(p, b, 3, 3, 6)['extend']
