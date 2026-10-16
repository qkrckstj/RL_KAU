"""Keep small gains and choose bounded extensions from rear skill trends."""
def candidate_status(win_gain):
    return dict(candidate=True,model_retained=True,improvement_confirmed=False,
        status='small_win_gain_candidate' if win_gain is not None and win_gain>0 else 'rear_development_candidate')

def continuation_limit(minimum,maximum,decision):
    return maximum if decision and decision.get('extend') else minimum

def budget_decision(points,baseline,steps,minimum,maximum):
    if steps<minimum:return dict(decision='continue_minimum',extend=False)
    if steps>=maximum:return dict(decision='maximum_reached',extend=False)
    if len(points)<3:return dict(decision='await_required_evaluation',extend=False)
    recent=sorted(points,key=lambda p:p['additional_steps'])[-3:]
    skills=('rear_damage','clean_rear_wins','max_rear_wez_streak','rear_damage_games')
    improved={m:sum(p['summary'][m]>baseline[m]+1e-8 for p in recent) for m in skills}
    preserved=sum(any(p['summary'][m]>=baseline[m]-1e-8 for m in skills)
                  and p['summary']['damage_received']<baseline['damage_received']-1e-8 for p in recent)
    extend=any(v>=2 for v in improved.values()) or preserved>=2
    return dict(decision='extend' if extend else 'review_plateau',extend=extend,
        improving_points_by_metric=improved,preserved_rear_lower_received_points=preserved,
        scope='Descriptive development trends;not confirmed superiority or frontline promotion.')
