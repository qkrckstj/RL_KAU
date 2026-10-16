"""Optional training-only rear-dependent win utility; native verdict is intact.

Control floor1 preserves the original reward. A lower floor discounts winning
without rear damage, restoring full win credit at half of the enemy's HP.
This is an experimental objective preference, not an official game rule.
"""
import math

from experiments.league.tail_actual_reward import RearDamageTrainingEnv
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable
from tools.league_matches import Actor


def rear_win_scale(rear_damage, floor=1.):
    rear_damage, floor = float(rear_damage), float(floor)
    if not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('Win utility floor must be finite and in0..1')
    if not math.isfinite(rear_damage) or not 0 <= rear_damage <= 1 + 1e-9:
        raise ValueError('Actual rear HP damage must be finite and in0..1')
    fraction = min(1., rear_damage / .5)
    return floor + (1 - floor) * fraction


class RearWinUtilityTrainingEnv(RearDamageTrainingEnv):
    def __init__(self, *args, rear_win_floor=1., **kwargs):
        self.rear_win_floor = float(rear_win_floor)
        rear_win_scale(0., self.rear_win_floor)
        super().__init__(*args, **kwargs)

    def step(self, action):
        state, reward, term, trunc, info = super().step(action)
        if not (term or trunc):
            return state, reward, term, trunc, info
        result = info['result']
        if result not in ('win', 'loss', 'draw'):
            raise ValueError('Unexpected declared health verdict')
        health_utility = float(info['learning_utility'])
        scale = rear_win_scale(info['actual_rear_damage'], self.rear_win_floor) if result == 'win' else 1.
        utility = health_utility * scale
        adjustment = utility - health_utility
        updated = dict(info, learning_utility=utility, health_learning_utility=health_utility,
                       rear_win_utility_scale=scale, rear_win_utility_adjustment=adjustment)
        # Do not add even a zero adjustment in control/loss/draw branches.
        return state, reward + adjustment if adjustment else reward, term, trunc, updated


def make_worker(plan, index, restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    archive = CurriculumArchive(plan['teacher'], plan['opponents'], plan['probabilities'],
        plan['seed'] * 100 + index, plan['ic_band'], index, plan['workers'],
        'red' if index % 2 == 0 else 'blue', coverage_period=plan['coverage_period'],
        curriculum=plan['curriculum'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    env = RearWinUtilityTrainingEnv(archive, plan['ppo']['gamma'], plan['tail_coefficients'],
        position_per_second=plan['position_per_second'],
        rear_damage_coefficient=plan['rear_damage_coefficient'],
        rear_win_floor=plan.get('rear_win_floor', 1.))
    worker = Restartable(Monitor(env, info_keywords=(
        'verdict_rule', 'result', 'terminal_reason', 'tail_seconds', 'tail_attack_seconds',
        'max_tail_streak', 'damage_dealt', 'damage_received', 'head_on_damage', 'tail_bonus',
        'position_reward_total', 'actual_rear_damage', 'actual_rear_damage_seconds',
        'actual_rear_reward', 'health_learning_utility', 'rear_win_utility_scale',
        'rear_win_utility_adjustment')))
    if restart is not None:
        worker.restore_worker(restart)
    return worker
