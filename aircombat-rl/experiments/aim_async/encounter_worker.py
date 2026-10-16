"""Add policy history outside the unchanged native reward and simulator wrappers."""
import gymnasium as gym
import numpy as np

from experiments.aim_async.encounter_phase import EncounterState,RULE
from experiments.aim_async.encounter_actor import phase_observation_space
from experiments.aim_async.reengagement import make_worker as original_worker


class EncounterWrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        if env.observation_space.shape!=(43,):
            raise ValueError('Expected original43feature worker')
        self.observation_space=phase_observation_space()
        self.encounter=EncounterState()

    def augment(self,state):
        return np.concatenate((state,np.asarray([self.encounter.observe(state)],dtype=np.float32)))

    def reset(self,**kwargs):
        state,info=self.env.reset(**kwargs)
        prefix=info.get('league_episode',{}).get('prefix',{})
        self.encounter.reset(after=prefix.get('status')=='post_encounter')
        return self.augment(state),info

    def step(self,action):
        state,reward,term,trunc,info=self.env.step(action)
        return self.augment(state),reward,term,trunc,info

    def snapshot_worker(self):
        return dict(self.env.get_wrapper_attr('snapshot_worker')(),encounter_policy_rule=RULE)


def make_worker(plan,index,restart=None):
    if restart is not None and restart.get('encounter_policy_rule')!=RULE:
        raise ValueError('Saved encounter policy contract changed or missing')
    return EncounterWrapper(original_worker(plan,index,restart))
