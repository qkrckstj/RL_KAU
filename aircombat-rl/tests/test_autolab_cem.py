import numpy as np
from experiments.plan_a.tactical import Policy, LOW, HIGH, ANCHOR
from tools.autolab_cem import export, rank
from tools.policies import load


def sample_obs(rng):
    x = rng.normal(size=39)
    x[[0,1,15,16]] *= 4000
    x[[3,4,18,19]] *= 150
    x[38] = 120 * rng.random()
    return x


def test_controller_is_translation_rotation_invariant():
    rng = np.random.default_rng(47)
    for _ in range(100):
        policy = Policy(parameters=LOW+rng.random(7)*(HIGH-LOW))
        x = sample_obs(rng)
        y = x.copy()
        # Clockwise 90-degree map rotation, including positions, velocities,
        # and heading. Body rates and the match clock are unchanged.
        for offset in (0,15):
            y[offset],y[offset+1] = x[offset+1]+2700, -x[offset]+7300
            y[offset+3],y[offset+4] = x[offset+4],-x[offset+3]
            y[offset+11] = x[offset+11]+np.pi/2
        assert policy.act(x) == policy.act(y)
        assert 0 <= policy.act(x) < 9


def test_exported_policy_matches_search_through_official_loader(tmp_path):
    export(tmp_path, ANCHOR)
    act, _, mode = load(tmp_path, tmp_path/'policy_net.json')
    policy = Policy(parameters=ANCHOR)
    rng = np.random.default_rng(71)
    assert mode == 'discrete'
    for _ in range(50):
        obs = sample_obs(rng)
        assert act(obs) == policy.act(obs)


def test_health_tiebreak_cannot_override_a_win():
    win = {'summary': {'kills': 1, 'own_health': 0., 'opp_health': 1.}}
    lose = {'summary': {'kills': 0, 'own_health': 1., 'opp_health': 0.}}
    assert rank(win) > rank(lose)


def test_followup_gate_requires_advantage_and_retention():
    from tools.autolab_cem_followup import screening_gate
    summary = dict(records=[dict(best_validation={'rate': .8}, final_validation={'rate': .7}) for _ in range(3)],
                   fixed_baseline={'summary': {'rate': .4}})
    assert screening_gate(summary)['passed']
    summary['records'][0]['final_validation']['rate'] = 0.
    assert not screening_gate(summary)['passed']
    for row in summary['records']:
        row['final_validation']['rate'] = .8
    summary['fixed_baseline']['summary']['rate'] = .7
    assert not screening_gate(summary)['passed']


def test_holdout_uses_initial_condition_clusters():
    from tools.autolab_cem_followup import analyse_holdout
    def result(wins):
        return dict(summary={'rate': wins/100}, episodes=[dict(seed=s,seat=seat,won=s<wins)
                    for seat in ('red','blue') for s in range(100)])
    results = {'fixed':result(40)}
    for seed in (803,804,805):
        results[f's{seed}_selected'] = result(80)
        results[f's{seed}_final'] = result(70)
    verdict = analyse_holdout(results,[803,804,805])
    assert verdict['passed']
    assert verdict['shared_ic_seeds'] == 100
    assert verdict['matches_per_policy'] == 200
    assert verdict['independent_training_runs'] == 3
    assert verdict['paired_ic_bootstrap_95_ci'][0] > 0


def test_replication_preserves_learned_parent_location():
    from tools.autolab_cem_followup import replication_plan
    previous=dict(parent='runs/pilot/s800',parent_sha256='weights-hash',initial_parameters=[1,2,3],training_n=12)
    new=replication_plan(previous,'runs/warm_pilot',warm=True)
    assert new['parent']=='runs/pilot/s800'
    assert new['parent_sha256']=='weights-hash'
    assert new['initial_parameters']==previous['initial_parameters']
    assert new['screening_source']=='runs/warm_pilot'
    assert new['seeds']==[903,904,905]
    assert new['training_n']==12
    assert 'seeds' not in previous
