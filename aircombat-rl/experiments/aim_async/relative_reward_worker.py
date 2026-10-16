"""Fixed original relative combat potential alongside the predictive aim signal."""
import math
from experiments.aim_async.worker import AimTrainingEnv
from experiments.aim_async.policy import original_columns
from experiments.league.residual_env import potential
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


def relative_bonus(old, new, terminal, gamma, coefficient):
    return coefficient * (gamma * (0. if terminal else new) - old)


class RelativeAimTrainingEnv(AimTrainingEnv):
    def __init__(self, env, gamma, coefficients, relative_coefficient):
        super().__init__(env, gamma, coefficients)
        self.relative_coefficient = float(relative_coefficient)
        if not math.isfinite(self.relative_coefficient) or not 0 <= self.relative_coefficient <= 1:
            raise ValueError('Relative coefficient must be finite in0..1')

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        self.relative_phi = potential(state[original_columns()])
        self.relative_shaping_total = 0.
        return state, info

    def step(self, action):
        state, reward, term, trunc, info = super().step(action)
        next_phi = 0. if term or trunc else potential(state[original_columns()])
        bonus = relative_bonus(self.relative_phi, next_phi, term or trunc,
                               self.gamma, self.relative_coefficient)
        self.relative_phi = next_phi
        self.relative_shaping_total += bonus
        info = dict(info, relative_shaping_reward=bonus,
                    relative_shaping_total=self.relative_shaping_total)
        return state, reward + bonus, term, trunc, info


def make_worker(plan, index, restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive = CurriculumArchive(
        plan['teacher'], plan['opponents'], plan['probabilities'],
        plan['seed'] * 100 + index, plan['ic_band'], index, plan['workers'],
        'red' if index % 2 == 0 else 'blue',
        coverage_period=plan['coverage_period'], curriculum=plan['curriculum'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    wrapped = RelativeAimTrainingEnv(archive, plan['ppo']['gamma'],
                                    plan['aim_coefficients'], plan['relative_potential_coefficient'])
    env = Restartable(Monitor(wrapped, info_keywords=(
        'verdict_rule', 'result', 'terminal_reason', 'damage_dealt',
        'damage_received', 'connected_fire_seconds', 'lock_breaks',
        'first_damage_time', 'first_received_time', 'first_shot_advantage',
        'relative_shaping_total')))
    if restart is not None:
        env.restore_worker(restart)
    return env
