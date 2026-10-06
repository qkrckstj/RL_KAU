import os
from tools.league_validate import heldout_analysis,screen_gate,process_live


def record(own,foe,wins,n=20):
    episodes=[dict(seed=40000000+i,seat=seat,won=i<wins,
                   outcome='kill' if i<wins else 'timeout',steps=100)
              for i in range(n) for seat in ('red','blue')]
    rate=wins/n
    return dict(own=own,foe=foe,summary=dict(rate=rate,score=.5+.5*rate,n=2*n,
                 own_health=.7,opp_health=.4),episodes=episodes)


def test_test_statistics_cluster_across_opponents_and_seats():
    selected=['a','b','c']
    foes=['old','held']
    results=[record('old',f,4) for f in foes]
    results += [record(s,f,16) for s in selected for f in foes]
    report=heldout_analysis(results,selected,'old',['held'])
    assert report['passed']
    assert report['shared_ic_seeds']==20
    assert report['matches_per_policy']==80
    assert report['paired_ic_bootstrap_95_ci'][0]>0


def test_test_rejects_loss_of_transfer_performance_despite_training_gain():
    selected=['a','b','c']
    foes=['old','train1','train2','held']
    results=[record('old',f,20 if f=='held' else 0) for f in foes]
    results += [record(s,f,0 if f=='held' else 20) for s in selected for f in foes]
    report=heldout_analysis(results,selected,'old',['held'])
    assert not report['passed']


def test_screen_rejects_draw_only_head_to_head():
    old=[record('old','foe',4)]
    candidate=[record('new','foe',16)]
    head=record('new','old',0)
    assert not screen_gate(candidate,old,head)['passed']


def test_live_controller_check_observes_real_process():
    assert process_live(os.getpid())


def test_pool_ranking_does_not_veto_a_generalist_for_one_direct_match():
    from tools.league_validate import rank_pool_candidates
    specialist=dict(spec={'id':'specialist'},metrics={'objective':.57,'mean_win_rate':.575,'health':.495},
                    head_score=.575)
    generalist=dict(spec={'id':'generalist'},metrics={'objective':.76,'mean_win_rate':.77,'health':.625},
                   head_score=.425)
    assert rank_pool_candidates([specialist,generalist])[0]['spec']['id']=='generalist'
