"""Descriptive bounded budget review that retains opponent-specific rear gains.

The frozen v1 trainer considers whole-panel averages only. This independent
review leaves frozen inputs/results unchanged and informs a subsequent resume.
An easy-opponent improvement is skill evidence, not transfer or promotion.
"""

SKILLS = ('rear_damage', 'clean_rear_wins', 'max_rear_wez_streak', 'rear_damage_games')


def review(points, baseline, steps, minimum, maximum):
    if steps < minimum:
        return dict(decision='continue_minimum', extend=False)
    if steps >= maximum:
        return dict(decision='maximum_reached', extend=False)
    unique = {p['additional_steps']: p for p in points}
    if len(unique) < 3:
        return dict(decision='await_required_evaluation', extend=False)
    recent = sorted(unique.values(), key=lambda p: p['additional_steps'])[-3:]
    details = {}
    for foe, parent in baseline['per_opponent'].items():
        if any(foe not in p['per_opponent'] for p in recent):
            raise ValueError('Opponent panel changed: ' + foe)
        rows = [p['per_opponent'][foe] for p in recent]
        if any(r['games'] != parent['games'] for r in rows):
            raise ValueError('Opponent games differ: ' + foe)
        improving = {key: sum(r[key] > parent[key]+1e-8 for r in rows)
                     for key in SKILLS}
        retained_with_less_received = sum(
            any(parent[key] > 1e-8 and r[key] >= parent[key]-1e-8 for key in SKILLS)
            and r['damage_received'] < parent['damage_received']-1e-8
            for r in rows)
        # Older good checkpoints remain candidates. Extending the newest model
        # also requires the qualifying skill to survive at that endpoint.
        latest_gain = any(n >= 2 and rows[-1][key] > parent[key]+1e-8
                          for key, n in improving.items())
        latest_retained = any(parent[key] > 1e-8 and rows[-1][key] >= parent[key]-1e-8
                              for key in SKILLS)
        qualifies = latest_gain or (retained_with_less_received >= 2 and latest_retained)
        declines = sum(r['rear_damage'] < parent['rear_damage']-1e-8 for r in rows) >= 2
        details[foe] = dict(qualifies=qualifies, improving_points_by_metric=improving,
            preserved_positive_rear_less_received_points=retained_with_less_received,
            rear_damage_declines=declines, parent=parent, recent=rows)
    qualifying = [f for f, d in details.items() if d['qualifies']]
    return dict(decision='bounded_local_skill_extension' if qualifying else 'review_plateau',
        extend=bool(qualifying), qualifying_opponents=qualifying,
        declining_opponents=[f for f, d in details.items() if d['rear_damage_declines']],
        opponent_details=details,
        recent_steps=[p['additional_steps'] for p in recent],
        extension_ceiling=maximum,
        scope='Repeated local rear skill on fixed development conditions; not confirmed broad transfer or frontline promotion.')
