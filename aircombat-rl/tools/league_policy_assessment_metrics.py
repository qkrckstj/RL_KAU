"""Paired metrics for deterministic policies and two action-RNG replicas."""
import numpy as np
from tools.league_sampled_assess import vectors
from tools.league_tournament_metrics import profile,code_groups,group_weights


def gains(new,old,opponents,band,n,weights):
    a,b=(vectors(rows,opponents,band,n,weights) for rows in (new,old))
    if a.shape[0] not in (1,2) or b.shape[0] not in (1,2):
        raise ValueError('One deterministic policy or two fixed action replicas required')
    differences=a-b
    replicas_count=differences.shape[0]
    rng=np.random.default_rng(815)
    ics=rng.integers(n//2,size=(10000,n//2))
    replicas=rng.integers(replicas_count,size=(10000,replicas_count))
    conditional=differences.mean(axis=0)[ics].mean(axis=1)
    crossed=differences[replicas[:,:,None],ics[:,None,:]].mean(axis=(1,2))
    return dict(mean=float(differences.mean()),per_action_replica_mean=differences.mean(axis=1).tolist(),
        ic_cluster_ci95=np.quantile(conditional,[.025,.975]).tolist(),
        crossed_ic_action_ci95=np.quantile(crossed,[.025,.975]).tolist(),initial_conditions=n//2,
        candidate_action_replicas=a.shape[0],comparator_action_replicas=b.shape[0],
        scope='Conditional on fixed policies/opponents and ICs. Shared IC resampling keeps both seats/all foes together; action replicas are paired when both policies have two. Deterministic policies are evaluated once, not treated as independent duplicated learners. Not uncertainty over future entrants or training seeds.')


def analyze_panel(plan,rows):
    by_own={s['id']:[r for r in rows if r['own']==s['id']] for specs in plan['roles'].values() for s in specs}
    roles={role:[by_own[s['id']] for s in specs] for role,specs in plan['roles'].items()}
    stats={}
    for role,replicas in roles.items():
        ps=[profile(r,plan['groups']) for r in replicas]
        stats[role]=dict(profiles=ps,mean_win_rate=float(np.mean([p['mean_win_rate'] for p in ps])),
            group_balanced_win_rate=float(np.mean([p['group_balanced_win_rate'] for p in ps])),
            conservative_lower_quarter=min(p['lower_quarter_score'] for p in ps),worst_score=min(p['worst_score'] for p in ps),
            opponents={s['id']:dict(wins=sum(r['summary']['wins'] for rs in replicas for r in rs if r['foe']==s['id']),
                draws=sum(r['summary']['draws'] for rs in replicas for r in rs if r['foe']==s['id']),
                losses=sum(r['summary']['losses'] for rs in replicas for r in rs if r['foe']==s['id']),games=plan['n']*len(replicas)) for s in plan['opponents']})
    weights=dict(uniform={s['id']:1/len(plan['opponents']) for s in plan['opponents']},code_group=group_weights(plan['groups']))
    comparisons={role:{label:gains(roles['candidate'],records,plan['opponents'],plan['band'],plan['n'],w) for label,w in weights.items()} for role,records in roles.items() if role!='candidate'}
    return dict(roles=stats,comparisons=comparisons)


def analyze(plan,rows):
    result=analyze_panel(plan,rows)
    result.update(policy_promoted=False,final_opened=False,heldout_opened=bool(plan.get('heldout_opponent_ids')),scope=plan['scope'],partitions={})
    for name,ids in plan['partitions'].items():
        opponents=[s for s in plan['opponents'] if s['id'] in ids]
        if not opponents:raise ValueError('Empty assessment partition')
        local=dict(plan,opponents=opponents,groups=code_groups(opponents))
        result['partitions'][name]=analyze_panel(local,[r for r in rows if r['foe'] in ids])
    return result
