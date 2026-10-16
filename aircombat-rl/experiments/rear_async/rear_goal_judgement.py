"""Optional broad rear-skill guard for a fresh bounded training study.

The original monitor judge is retained as evidence. Neither this guard nor the
consumed development panel claims unseen superiority or changes game verdicts.
"""
import math

from experiments.rear_async.judgement import budget_decision as monitor_budget_decision


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a nonnegative integer')
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def budget_decision(points, baseline, steps, minimum, maximum, *, broad,
                    broad_baseline, substantial_clean, source_substantial_clean):
    monitor = monitor_budget_decision(points, baseline, steps, minimum, maximum)
    result = dict(monitor, monitor_decision=monitor)
    if steps < minimum or steps >= maximum or len(points) < 3:
        return result
    games = _count(broad['games'], 'Broad games')
    source_games = _count(broad_baseline['games'], 'Source broad games')
    if not games or games != source_games:
        raise ValueError('Broad panels must have matching positive game counts')
    current_big = _count(substantial_clean, 'Substantial clean cases')
    source_big = _count(source_substantial_clean, 'Source substantial clean cases')
    if current_big > games or source_big > games:
        raise ValueError('Substantial clean cases cannot exceed panel games')
    current_damage = float(broad['rear_damage'])
    source_damage = float(broad_baseline['rear_damage'])
    if any(not math.isfinite(x) or not 0 <= x <= 1 + 1e-9
           for x in (current_damage, source_damage)):
        raise ValueError('Broad mean rear damage must be finite and in0..1')
    retained = current_damage >= source_damage - 1e-8 and current_big >= source_big
    broad_gain = retained and (current_damage > source_damage + 1e-8 or current_big > source_big)
    extend = retained and (monitor['extend'] or broad_gain)
    label = 'extend' if extend else ('review_plateau' if retained else 'review_broad_rear_regression')
    return dict(result, decision=label, extend=extend,
                broad_rear_retained=retained, broad_rear_gain=broad_gain,
                broad_rear_damage=current_damage, source_broad_rear_damage=source_damage,
                substantial_clean=current_big, source_substantial_clean=source_big,
                scope='Consumed broad development rear retention guards a bounded extension;'
                      ' all checkpoints are retained, with no frontline promotion.')
