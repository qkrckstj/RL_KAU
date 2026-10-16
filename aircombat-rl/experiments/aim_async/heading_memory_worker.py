"""Public heading-command history outside the unchanged43reward pipeline."""
import gymnasium as gym
import numpy as np
from experiments.aim_async.heading_memory_features import HeadingCommandMemory, augment, FEATURE_DIM
from experiments.aim_async.heading_rate_worker import HeadingRateGuidanceEnv
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


class PublicHeadingGuidanceEnv(HeadingRateGuidanceEnv):
    def _state(self,raw):
        self.public_raw=np.asarray(raw,dtype=np.float32).copy()
        return super()._state(raw)


class HeadingMemoryObservationEnv(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.heading_memory=HeadingCommandMemory()
        self.observation_space=gym.spaces.Box(-1.,1.,(FEATURE_DIM,),dtype=np.float32)

    def reset(self,**kwargs):
        state,info=self.env.reset(**kwargs)
        self.heading_memory.reset(self.env.public_raw)
        return augment(state,self.heading_memory.error(self.env.public_raw)),info

    def step(self,action):
        self.heading_memory.update(self.env.public_raw,action)
        state,reward,term,trunc,info=self.env.step(action)
        return augment(state,self.heading_memory.error(self.env.public_raw)),reward,term,trunc,info


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=PublicHeadingGuidanceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],plan['control_guidance_range'])
    env=Restartable(Monitor(HeadingMemoryObservationEnv(wrapped),info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total')))
    if restart is not None:env.restore_worker(restart)
    return env
