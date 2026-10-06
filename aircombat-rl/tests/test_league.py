import numpy as np
from experiments.league.controller import Policy, embed
from experiments.plan_a.tactical import Policy as Original
from tools.league_matches import initial_roster, duel, Actor
from tools.autolab_cem import evaluate


def test_disabled_reactive_gates_preserve_original_actions():
    parameters = initial_roster()[0]['parameters']
    old, new = Original(parameters=parameters), Policy(parameters=embed(parameters))
    rng = np.random.default_rng(97)
    for _ in range(500):
        x = rng.normal(size=39)
        x[[0,1,15,16]] *= 10000
        x[[3,4,18,19]] *= 200
        x[38] = rng.uniform(0,120)
        assert old.act(x) == new.act(x)


def test_same_outcomes_as_unchanged_official_ace_evaluation():
    own = initial_roster()[0]
    actual = duel(own,dict(id='ace',kind='bot',name='ace'),30000000,2)
    original = evaluate(own['parameters'],30000000,2)
    assert actual['episodes'] == original['episodes']


def test_policy_match_is_same_physical_fight_from_both_perspectives():
    own, foe = initial_roster()[0], dict(id='fixed',kind='fixed',action=6)
    ab, ba = duel(own,foe,30000001,2),duel(foe,own,30000001,2)
    flipped = {'kill':'died','died':'kill','mutual':'mutual','timeout':'timeout'}
    for a,b in zip(ab['episodes'],reversed(ba['episodes'])):
        assert a['outcome'] == flipped[b['outcome']]
        assert a['steps'] == b['steps']
        assert a['own_health'] == b['opp_health']
        assert a['opp_health'] == b['own_health']


def test_loaded_ddqn_can_coexist_with_other_policy():
    from tools.league_matches import MatchEnv
    roster = initial_roster()
    env = MatchEnv(roster[0],roster[1])
    try:
        obs, _ = env.reset(seed=30000000)
        for _ in range(20):
            obs,_,_,_,_ = env.step(env.own_action(obs))
    finally:
        env.close()


def test_continuous_decoder_does_not_use_learners_discrete_grid():
    actor = Actor(dict(kind='fixed',action=0))
    actor.mode = 'continuous'
    actor.predict = lambda obs: np.array([.5,-.75])
    assert actor.act(obs=np.zeros(39)) == (15.,-15.,0.)


def test_archive_deduplicates_retained_parent_without_forgetting_different_policy():
    from tools.league_train import unique_entrants,entrant
    original=initial_roster()[0]
    equivalent=entrant(embed(original['parameters']),'retained')
    new=entrant(embed(original['parameters']),'changed')
    new['parameters'][0]+=1
    assert [s['id'] for s in unique_entrants([original,equivalent,new])]==['cem_original','changed']


def test_standard_export_matches_reactive_controller(tmp_path):
    from tools.league_train import export,entrant
    from tools.policies import load
    spec=entrant(embed(initial_roster()[0]['parameters']),'test')
    spec['parameters'][11]=1500
    export(tmp_path,spec)
    act,_,mode=load(tmp_path,tmp_path/'policy_net.zip')
    original=Policy(parameters=spec['parameters'])
    rng=np.random.default_rng(101)
    for _ in range(100):
        obs=rng.normal(size=39)
        obs[[0,1,15,16]]*=1000
        obs[38]=rng.uniform(0,120)
        assert act(obs)==original.act(obs)
    assert mode=='discrete'


def test_admission_rejects_large_regression_even_with_average_gain():
    from tools.league_train import admission
    def row(win,loss):
        return dict(summary=dict(rate=win,score=win+.5*(1-win-loss),n=20,own_health=.5,opp_health=.5),episodes=[])
    old=[row(.4,.1)]*4
    candidate=[row(1,0),row(1,0),row(1,0),row(0,1)]
    head=dict(summary={'score':.7})
    assert not admission(candidate,old,head)['promoted']


def test_no_contact_draws_are_less_valuable_than_wins():
    from tools.league_train import metrics
    def row(win,score):
        return dict(summary=dict(rate=win,score=score,n=2,own_health=1,opp_health=1),episodes=[])
    assert metrics([row(.5,.75)])['objective']>metrics([row(0,.5)])['objective']


def test_real_parallel_batch_resumes_without_changing_results(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from tools.league_train import evaluate_candidates,entrant
    from tools.league_matches import initialize_worker
    roster=initial_roster()
    candidate=entrant(embed(roster[0]['parameters']),'reactive')
    candidate['parameters'][9:11]=[3500,600]
    candidate['parameters'][15:17]=[120,7000]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize_worker) as pool:
        first=evaluate_candidates(pool,[roster[0],candidate],[roster[2]],30002000,2,tmp_path)
        resumed=evaluate_candidates(pool,[roster[0],candidate],[roster[2]],30002000,2,tmp_path)
    assert first==resumed
    assert all(r['metrics']['games']==2 for r in first)


def test_escape_response_uses_observation_not_opponent_identity():
    params=embed(initial_roster()[0]['parameters'])
    params[9:11]=[2000,650]
    params[15:17]=[100,1000]
    reactive=Policy(parameters=params)
    obs=np.zeros(39)
    obs[16]=5000
    obs[4]=200
    obs[38]=120
    # Opponent ahead pointing away: end opening and accelerate in pursuit.
    away=reactive.act(obs)
    obs[26]=np.pi
    facing=reactive.act(obs)
    assert away//3!=0
    assert facing//3==0
