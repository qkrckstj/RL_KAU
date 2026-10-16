"""Bounded continuation based on combat outcomes, predictive aiming and firing."""

def candidate_status(win_gain):
    return dict(candidate=True,model_retained=True,improvement_confirmed=False,
        status='small_win_gain_candidate' if win_gain is not None and win_gain>0 else 'aim_development_candidate')


def exchange(s):return s['damage_dealt']-s['damage_received']


def budget_decision(points,baseline,steps,minimum,maximum,*,broad,broad_baseline):
    if steps<minimum:return dict(decision='continue_minimum',extend=False)
    if steps>=maximum:return dict(decision='maximum_reached',extend=False)
    if len(points)<3:return dict(decision='await_required_evaluation',extend=False)
    recent=sorted(points,key=lambda p:p['additional_steps'])[-3:]
    win=sum(p['summary']['win_rate']>baseline['win_rate'] for p in recent)
    damage=sum(exchange(p['summary'])>exchange(baseline)+.005 for p in recent)
    firing=sum(p['summary'].get('connected_fire_seconds',0)>baseline.get('connected_fire_seconds',float('inf'))+.05
        and p['summary']['damage_received']<=baseline['damage_received'] for p in recent)
    retained=broad['win_rate']>=broad_baseline['win_rate']-1./broad_baseline.get('games',768) and exchange(broad)>=exchange(broad_baseline)-.005
    broad_gain=broad['win_rate']>broad_baseline['win_rate'] or exchange(broad)>exchange(broad_baseline)+.005
    extend=retained and (max(win,damage,firing)>=2 or broad_gain)
    return dict(decision='extend' if extend else 'review_combat_plateau',extend=extend,
        improving_points=dict(win=win,damage_exchange=damage,connected_firing=firing),
        broad_combat_retained=retained,broad_combat_gain=broad_gain,
        scope='Consumed development trends guide a bounded extension; all checkpoints retained, no automatic promotion. Rear success never gates this branch.')
