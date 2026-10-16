"""Deduplicate evaluation foes using paired historical combat responses.

This is an empirical development-panel compression, not policy equivalence.
Unknown evidence stays separate; complete-link grouping prevents weak chains.
"""
from itertools import combinations
from experiments.rear_async.storage import read

THRESHOLDS=dict(min_cells=64,min_panels=2,outcome_agreement=.9,
    mean_health_error=.08,mean_duration_error=5.,mean_rear_damage_error=.05,
    mean_head_on_damage_error=.16)


def fingerprints(folders):
    evidence={}
    for folder in folders:
        for path in (folder/'cells').glob('*.json'):
            r=read(path)
            key=(str(folder),r['seed'],r['seat'],r['action_seed'])
            evidence.setdefault(r['opponent'],{})[key]=r
    return evidence


def comparison(a,b):
    keys=a.keys() & b.keys();n=len(keys)
    if a.keys()!=b.keys():return dict(similar=False,matched_cells=n,reason="incomplete paired coverage")
    if n<THRESHOLDS['min_cells'] or len({k[0] for k in keys})<THRESHOLDS['min_panels']:
        return dict(similar=False,matched_cells=n,reason='insufficient paired evidence')
    mean=lambda name:sum(abs(a[k][name]-b[k][name]) for k in keys)/n
    metrics=dict(matched_cells=n,outcome_agreement=sum(a[k]['result']==b[k]['result'] for k in keys)/n,
        mean_health_error=(mean('own_health')+mean('opp_health'))/2,
        mean_duration_error=mean('duration'),mean_rear_damage_error=mean('rear_damage'),
        mean_head_on_damage_error=mean('head_on_damage'))
    return dict(metrics,similar=metrics['outcome_agreement']>=THRESHOLDS['outcome_agreement'] and
        all(metrics[k]<=THRESHOLDS[k] for k in ('mean_health_error','mean_duration_error',
            'mean_rear_damage_error','mean_head_on_damage_error')))


def select_representatives(foes,evidence):
    pairs={(a['id'],b['id']):comparison(evidence.get(a['id'],{}),evidence.get(b['id'],{}))
        for a,b in combinations(foes,2)}
    def similar(a,b):return pairs.get((a,b),pairs.get((b,a),{})).get('similar',False)
    groups=[]
    for foe in foes:
        group=next((g for g in groups if all(similar(foe['id'],f['id']) for f in g)),None)
        if group is None:groups.append([foe])
        else:group.append(foe)
    def difficulty(foe):
        rows=list(evidence.get(foe['id'],{}).values())
        if not rows:return (float('inf'),float('inf'),foe['id'])
        return (sum(r['result']=='win' for r in rows)/len(rows),
            sum(r['own_health']-r['opp_health'] for r in rows)/len(rows),foe['id'])
    selected=[min(g,key=difficulty) for g in groups]
    return selected,dict(original_opponents=len(foes),selected_opponents=len(selected),thresholds=THRESHOLDS,
        scope='Similar paired outcomes, HP exchange, duration and damage geometry across archived learners; not proven behavior equivalence. Preserve all artifacts; freeze roster within the study.',
        representative_rule='Hardest observed member: lowest learner win rate, then health margin; ID breaks ties.',
        groups=[dict(representative=r['id'],members=[f['id'] for f in g],
            pairs=[dict(a=a['id'],b=b['id'],**pairs.get((a['id'],b['id']),pairs.get((b['id'],a['id']),{}))) for a,b in combinations(g,2)])
            for g,r in zip(groups,selected)])
