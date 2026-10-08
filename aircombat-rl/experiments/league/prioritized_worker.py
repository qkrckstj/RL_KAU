"""Report actual opponent sampling probabilities for matched curriculum pilots."""
import gymnasium as gym
import numpy as np
from experiments.league.strategy_worker import make_strategy_worker as original_factory


class CurriculumAudit(gym.Wrapper):
    def curriculum_status(self):
        sampler=self.unwrapped.sampler
        return dict(period=sampler.period,probabilities=sampler.probabilities.tolist(),
                    reward_gamma=self.env.get_wrapper_attr('gamma'))


def make_prioritized_worker(plan,index,instrumented):
    env=CurriculumAudit(original_factory(plan,index,instrumented));s=env.curriculum_status()
    assert s['period']==plan['coverage_period'] and s['reward_gamma']==plan['ppo']['gamma']
    assert np.allclose(s['probabilities'],plan['probabilities'],rtol=0,atol=1e-15)
    return env
