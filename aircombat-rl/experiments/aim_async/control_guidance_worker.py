"""Small public-state control potential; original policy/physics remain intact."""
import math
import numpy as np
from experiments.aim_async.relative_reward_worker import RelativeAimTrainingEnv, relative_bonus
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


def guidance_potential(state):
    state=np.asarray(state,dtype=np.float64)
    if state.shape!=(43,) or not np.isfinite(state).all():
        raise ValueError('Expected43finite public derived features')
    scaled=float(np.clip(state[0],0.,1.-1e-9))
    distance=10000.*scaled/(1.-scaled)
    if distance<150. or distance>=6000.:return 0.
    inverse=lambda v:math.atanh(float(np.clip(v,-.999999,.999999)))
    los_rate=400.*inverse(state[8])/distance
    own_yaw=.5*inverse(state[15])
    lead_error=math.atan2(float(state[31]),float(state[32]))
    demand=lead_error+.5*los_rate-.5*own_yaw
    demand=math.atan2(math.sin(demand),math.cos(demand))
    steering=max(0.,1.-abs(demand)/math.radians(30.))
    # Nominal20kft corner reference; not a claim of actual maximum turn rate.
    target_speed=500. if abs(demand)>math.radians(10.) or abs(los_rate)>.03 else 600.
    speed=300.*inverse(state[9])/.514444
    speed_quality=max(0.,1.-abs(speed-target_speed)/200.)
    gate=max(0.,1.-max(0.,distance-1500.)/4500.)
    return gate*(.75*steering+.25*speed_quality)


def guidance_bonus(old,new,terminal,gamma,coefficient):
    return relative_bonus(old,new,terminal,gamma,coefficient)


class ControlGuidanceTrainingEnv(RelativeAimTrainingEnv):
    def __init__(self,env,gamma,coefficients,relative_coefficient,guidance_coefficient):
        super().__init__(env,gamma,coefficients,relative_coefficient)
        self.guidance_coefficient=float(guidance_coefficient)
        if not math.isfinite(self.guidance_coefficient) or not 0<=self.guidance_coefficient<=.05:
            raise ValueError('Guidance coefficient must be finite in0..0.05')

    def reset(self,**kwargs):
        state,info=super().reset(**kwargs)
        self.control_phi=guidance_potential(state)
        self.control_guidance_total=0.
        return state,dict(info,control_guidance_total=0.)

    def step(self,action):
        state,reward,term,trunc,info=super().step(action)
        new=0. if term or trunc else guidance_potential(state)
        bonus=guidance_bonus(self.control_phi,new,term or trunc,self.gamma,self.guidance_coefficient)
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
    wrapped=ControlGuidanceTrainingEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'])
    env=Restartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total')))
    if restart is not None:env.restore_worker(restart)
    return env
