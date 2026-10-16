"""Learning-only public heading feedback; native controls and features unchanged."""
import math
import numpy as np
from experiments.aim_async.guidance_range_worker import RangeGuidanceTrainingEnv, guidance_potential as range_potential
from experiments.aim_async.relative_reward_worker import RelativeAimTrainingEnv, relative_bonus
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


def heading_rate(raw):
    raw=np.asarray(raw,dtype=np.float64)
    if raw.shape!=(39,) or not np.isfinite(raw).all():
        raise ValueError('Expected39finite public observations')
    phi,theta,q,r=(float(raw[i]) for i in (9,10,13,14))
    cosine=math.cos(theta)
    denominator=math.copysign(max(abs(cosine),1e-6),cosine)
    rate=(q*math.sin(phi)+r*math.cos(phi))/denominator
    if not math.isfinite(rate):raise ValueError('Unrepresentable public heading feedback')
    return rate


def guidance_potential(state,rate,range_limit=12000.):
    if not math.isfinite(rate):raise ValueError('Heading feedback must be finite')
    projected=np.array(state,dtype=np.float64,copy=True)
    if projected.shape!=(43,) or not np.isfinite(projected).all():
        raise ValueError('Expected43finite public derived features')
    # Reuse the original inverse-normalized damping slot only inside reward.
    # The actual returned network observation keeps its original bodyr value.
    projected[15]=math.tanh(2.*rate)
    return range_potential(projected,range_limit)


class HeadingRateGuidanceEnv(RangeGuidanceTrainingEnv):
    def _state(self,raw):
        self.heading_rate=heading_rate(raw)
        return super()._state(raw)

    def reset(self,**kwargs):
        state,info=RelativeAimTrainingEnv.reset(self,**kwargs)
        self.control_phi=guidance_potential(state,self.heading_rate,self.range_limit)
        self.control_guidance_total=0.
        return state,dict(info,control_guidance_total=0.)

    def step(self,action):
        state,reward,term,trunc,info=RelativeAimTrainingEnv.step(self,action)
        new=0. if term or trunc else guidance_potential(state,self.heading_rate,self.range_limit)
        bonus=relative_bonus(self.control_phi,new,term or trunc,self.gamma,self.guidance_coefficient)
        self.control_phi=new;self.control_guidance_total+=bonus
        return state,reward+bonus,term,trunc,dict(info,control_guidance_total=self.control_guidance_total)


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
    env=Restartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total')))
    if restart is not None:env.restore_worker(restart)
    return env
