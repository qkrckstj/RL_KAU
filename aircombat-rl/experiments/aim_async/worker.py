"""Learning-only predictive alignment and actual HP exchange; native guns intact."""
import math
import gymnasium as gym
import numpy as np
from experiments.league.health_rule_v2 import classify, terminal_utility, RULE
from experiments.league.strategy_worker import PeriodicArchiveFight
from experiments.rear_async.worker import Restartable
from experiments.aim_async.features import features, FEATURE_DIM


def aim_potential(info):
    angle=float(info['ata_lead']);r=float(info['range'])
    quality=max(0.,1.-angle/math.radians(30.))
    proximity=min(1.,r/150.) if r<150. else max(0.,1.-max(0.,r-1500.)/2500.)
    track=min(1.,max(0.,float(info['track_time'])))
    return proximity*quality+.25*track


def aim_bonus(old,new,dealt,received,terminal,gamma,potential,dealt_weight,received_weight):
    phi=0. if terminal else new
    return potential*(gamma*phi-old)+dealt_weight*dealt-received_weight*received


class AimTrainingEnv(gym.Wrapper):
    def __init__(self,env,gamma,coefficients):
        super().__init__(env)
        self.gamma=float(gamma)
        if not 0<self.gamma<=1:raise ValueError('Invalid gamma')
        self.observation_space=gym.spaces.Box(-1.,1.,(FEATURE_DIM,),dtype=np.float32)
        self.coefficients=dict(coefficients)

    def _state(self,raw):
        prior=self.env.teacher_action(raw)
        return features(raw,prior),prior

    def reset(self,**kwargs):
        raw,info=self.env.reset(**kwargs)
        state,prior=self._state(raw)
        info=dict(info,prior_action=prior)
        self.previous_aim=aim_potential(info)
        self.previous_health=(float(info['own_health']),float(info['opp_health']))
        self.damage_dealt=self.damage_received=self.connected_fire_seconds=0.
        self.first_damage_time=self.first_received_time=None
        self.lock_breaks=0;self.was_locked=False
        return state,info

    def step(self,action):
        raw,official_reward,term,trunc,info=self.env.step(action)
        state,prior=self._state(raw)
        terminal=bool(term or trunc)
        result=classify(info['own_health'],info['opp_health']) if terminal else None
        reward=terminal_utility(info['own_health'],info['opp_health']) if terminal else 0.
        info=dict(info,prior_action=prior,verdict_rule=RULE,learning_utility=reward,
            official_reward=official_reward,league_episode=dict(self.env.episode_metadata))
        if terminal:
            info=dict(info,terminal_reason=info['outcome'],legacy_outcome=info['outcome'],
                result=result,outcome=result,won=result=='win')
            term,trunc=True,False
        hp,ehp=float(info['own_health']),float(info['opp_health'])
        dealt=max(0.,self.previous_health[1]-ehp);received=max(0.,self.previous_health[0]-hp)
        self.previous_health=(hp,ehp)
        phi=aim_potential(info);c=self.coefficients
        bonus=aim_bonus(self.previous_aim,phi,dealt,received,term or trunc,self.gamma,
            c['potential'],c['damage'],c['received'])
        self.previous_aim=phi
        self.damage_dealt+=dealt;self.damage_received+=received
        self.connected_fire_seconds+=.05*(dealt>0)
        now=float(info['t'])
        if self.first_damage_time is None and dealt>0:self.first_damage_time=now
        if self.first_received_time is None and received>0:self.first_received_time=now
        locked=bool(info['in_wez'] and info['track_time']>=1.)
        self.lock_breaks+=int(self.was_locked and not locked);self.was_locked=locked
        if term or trunc:
            info=dict(info,damage_dealt=self.damage_dealt,damage_received=self.damage_received,
                connected_fire_seconds=self.connected_fire_seconds,lock_breaks=self.lock_breaks,
                first_damage_time=-1. if self.first_damage_time is None else self.first_damage_time,
                first_received_time=-1. if self.first_received_time is None else self.first_received_time,
                first_shot_advantage=int(self.first_damage_time is not None and
                    (self.first_received_time is None or self.first_damage_time<self.first_received_time)))
        return state,reward+bonus,term,trunc,info


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=PeriodicArchiveFight(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=AimTrainingEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'])
    env=Restartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage')))
    if restart is not None:env.restore_worker(restart)
    return env
