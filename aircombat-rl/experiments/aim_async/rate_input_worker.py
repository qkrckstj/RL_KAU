"""Add public motion information outside the unchanged43feature reward wrapper."""
import gymnasium as gym
import numpy as np
from experiments.aim_async.rate_input_features import augment, FEATURE_DIM
from experiments.aim_async.heading_rate_worker import HeadingRateGuidanceEnv
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


class RateInputObservationEnv(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space=gym.spaces.Box(-1.,1.,(FEATURE_DIM,),dtype=np.float32)

    def observation(self,state):
        return augment(state,self.env.heading_rate)


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=HeadingRateGuidanceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],plan['control_guidance_range'])
    env=Restartable(Monitor(RateInputObservationEnv(wrapped),info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total')))
    if restart is not None:env.restore_worker(restart)
    return env
