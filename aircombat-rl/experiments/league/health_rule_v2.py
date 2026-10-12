"""User-authorized end-health adjudication; flight and termination unchanged."""
import math
from experiments.league.residual_env import PriorTrainingEnv
from experiments.league.strategy_worker import PeriodicArchiveFight

RULE = 'user_terminal_health_round4_v2'


def classify(own, foe):
    own, foe = float(own), float(foe)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (own, foe)):
        raise ValueError('Invalid terminal health')
    a, b = round(own, 4), round(foe, 4)
    return 'win' if a > b else 'loss' if a < b else 'draw'


def terminal_utility(own, foe):
    result = classify(own, foe)
    return 1.+.05*float(own) if result == 'win' else -1. if result == 'loss' else 0.


class HealthTrainingEnv(PriorTrainingEnv):
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        info = dict(info, verdict_rule=RULE)
        if terminated or truncated:
            own, foe = float(info['own_health']), float(info['opp_health'])
            result = classify(own, foe)
            utility = terminal_utility(own, foe)
            info.update(legacy_outcome=info['outcome'], terminal_reason=info['outcome'],
                result=result, outcome=result, won=result == 'win',
                legacy_learning_utility=info['learning_utility'], learning_utility=utility,
                adjudicated_own_health=round(own,4), adjudicated_opp_health=round(foe,4),
                own_health_hex=own.hex(), opp_health_hex=foe.hex())
            reward = utility + info['shaping_reward']
        return state, reward, terminated, truncated, info

    def profile_status(self):
        from tools.league_thread_benchmark import own_memory
        return dict(worker_index=self.unwrapped.worker_index, memory=own_memory(),
                    cached_opponents=len(self.unwrapped.cache), rule=RULE)


def make_health_worker(plan, index):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive = PeriodicArchiveFight(plan['teacher'], plan['opponents'], plan['probabilities'],
        plan['seed']*100+index, plan['ic_band'], index, plan['workers'],
        'red' if index % 2 == 0 else 'blue', coverage_period=plan['coverage_period'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    return Monitor(HealthTrainingEnv(archive, gamma=plan['ppo']['gamma']),
        info_keywords=('verdict_rule','result','terminal_reason','adjudicated_own_health',
                       'adjudicated_opp_health','own_health_hex','opp_health_hex'))
