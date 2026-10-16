"""Fixed learning-only terminal draw penalty; native combat rules unchanged."""
import math
from experiments.aim_async.heading_rate_worker import HeadingRateGuidanceEnv
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


class DrawUtilityGuidanceEnv(HeadingRateGuidanceEnv):
    def __init__(self, *args, draw_penalty, **kwargs):
        if not math.isfinite(draw_penalty) or not 0. <= draw_penalty <= 1.:
            raise ValueError('Draw penalty must be finite and between0and1')
        super().__init__(*args, **kwargs)
        self.draw_penalty = float(draw_penalty)
        self.draw_utility_total = 0.

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        self.draw_utility_total = 0.
        return state, dict(info, draw_utility_total=0.)

    def step(self, action):
        state, reward, term, trunc, info = super().step(action)
        adjustment = -self.draw_penalty if (term or trunc) and info['result'] == 'draw' else 0.
        self.draw_utility_total += adjustment
        info = dict(info, draw_utility_total=self.draw_utility_total)
        if term or trunc:
            info.update(original_learning_utility=info['learning_utility'],
                        learning_utility=info['learning_utility']+adjustment)
        return state, reward+adjustment, term, trunc, info


def make_worker(plan, index, restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive = CurriculumArchive(plan['teacher'], plan['opponents'], plan['probabilities'],
        plan['seed']*100+index, plan['ic_band'], index, plan['workers'],
        'red' if index%2 == 0 else 'blue', coverage_period=plan['coverage_period'], curriculum=plan['curriculum'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    wrapped = DrawUtilityGuidanceEnv(archive, plan['ppo']['gamma'], plan['aim_coefficients'],
        plan['relative_potential_coefficient'], plan['control_guidance_coefficient'],
        plan['control_guidance_range'], draw_penalty=plan['terminal_draw_penalty'])
    env = Restartable(Monitor(wrapped, info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total','draw_utility_total')))
    if restart is not None:
        env.restore_worker(restart)
    return env
