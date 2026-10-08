"""Expose the actual learning reward discount without changing official stepping."""
import gymnasium as gym
from experiments.league.strategy_worker import make_strategy_worker as original_factory


class DiscountAudit(gym.Wrapper):
    def discount_status(self):
        return dict(reward_gamma=self.env.get_wrapper_attr('gamma'),
                    shaping=self.env.get_wrapper_attr('shaping'))


def make_discount_worker(plan, index, instrumented):
    env = original_factory(plan, index, instrumented)
    result = DiscountAudit(env)
    assert result.discount_status() == dict(reward_gamma=plan['ppo']['gamma'], shaping=True)
    return result
