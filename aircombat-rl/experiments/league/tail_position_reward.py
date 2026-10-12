"""2D adaptation of Pope et al. relative-position reward, PDF p5 Eq10.

This is a dense objective addition, NOT potential-based invariant shaping.
Distance attenuation and per-second scale are project choices, not copied values.
"""
import math
from aircombat_gym.wvr import obs as O
from experiments.league.tail_reward import TailTrainingEnv
from experiments.league.tail_curriculum import CurriculumArchive


def position_score(raw):
    d=O.unpack(raw);dx=d['opp_x']-d['own_x'];dy=d['opp_y']-d['own_y'];r=math.hypot(dx,dy)
    if r<1e-6:return 0.
    cosine=lambda side:(dx*math.sin(d[side+'_psi'])+dy*math.cos(d[side+'_psi']))/r
    theta=math.acos(max(-1.,min(1.,cosine('own'))))/math.pi
    aspect=math.acos(max(-1.,min(1.,cosine('opp'))))/math.pi
    sigmoid=1/(1+math.exp(-18*(aspect-.5)))
    angular=(theta-2)*sigmoid-theta+1
    return angular/(1+(r/4000.)**2)


class PositionTrainingEnv(TailTrainingEnv):
    def __init__(self,*args,position_per_second,**kwargs):
        super().__init__(*args,**kwargs);self.position_per_second=float(position_per_second)
        if not 0<=self.position_per_second<=.05:raise ValueError('Position coefficient outside declared search range')

    def reset(self,**kwargs):
        self.position_reward_total=0.
        return super().reset(**kwargs)

    def step(self,action):
        state,reward,term,trunc,info=super().step(action)
        bonus=.05*self.position_per_second*position_score(self.env._obs(self.env._last))
        self.position_reward_total+=bonus
        if term or trunc:info=dict(info,position_reward_total=self.position_reward_total)
        return state,reward+bonus,term,trunc,info


def make_worker(plan,index):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    return Monitor(PositionTrainingEnv(archive,plan['ppo']['gamma'],plan['tail_coefficients'],position_per_second=plan['position_per_second']),
        info_keywords=('verdict_rule','result','terminal_reason','tail_seconds','tail_attack_seconds',
                       'max_tail_streak','damage_dealt','damage_received','head_on_damage','tail_bonus','position_reward_total'))
