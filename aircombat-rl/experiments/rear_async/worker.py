"""Preserve original rear rewards and capture sampler state for restart."""
import copy
import gymnasium as gym
from experiments.league.tail_actual_reward import make_worker as original_worker

class Restartable(gym.Wrapper):
    def snapshot_worker(self):
        a=self.unwrapped
        return dict(episode_index=a.episode_index,sampler=copy.deepcopy(a.sampler))
    def restore_worker(self,state):
        a=self.unwrapped;a.episode_index=state['episode_index'];a.sampler=copy.deepcopy(state['sampler'])

def make_worker(plan,index,restart=None):
    # The completed geometry focus schedule remains disabled: base48weights.
    env=Restartable(original_worker(plan,index))
    if restart is not None:env.restore_worker(restart)
    return env
