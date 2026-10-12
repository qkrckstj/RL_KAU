"""Training-only bounded bonus for actual rear damage, after native track lock."""
import math
from experiments.league.tail_position_reward import PositionTrainingEnv
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.league.tail_reward import geometry


def rear_damage_credit(previous_health, health, rear_eligible):
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (previous_health,health)):
        raise ValueError('Invalid health')
    if health > previous_health + 1e-12:
        raise ValueError('Unexpected regeneration')
    return max(0.,previous_health-health) if rear_eligible else 0.


class RearDamageTrainingEnv(PositionTrainingEnv):
    def __init__(self,*args,rear_damage_coefficient,**kwargs):
        super().__init__(*args,**kwargs)
        self.rear_damage_coefficient=float(rear_damage_coefficient)
        if not 0<=self.rear_damage_coefficient<=.5:raise ValueError('Coefficient outside declared range')

    def reset(self,**kwargs):
        state,info=super().reset(**kwargs)
        self.previous_exact_opp_health=float(info['opp_health'])
        self.actual_rear_damage=self.actual_rear_damage_seconds=0.
        return state,info

    def step(self,action):
        state,reward,term,trunc,info=super().step(action)
        # Combat.step moves planes, computes WEZ, then applies locked damage.
        # Its returned health is patched after damage; geometry is same tick.
        now=geometry(self.env._obs(self.env._last))
        health=float(info['opp_health'])
        credit=rear_damage_credit(self.previous_exact_opp_health,health,now['tail_attack'])
        self.previous_exact_opp_health=health
        self.actual_rear_damage+=credit
        self.actual_rear_damage_seconds+=.05*(credit>0)
        assert self.actual_rear_damage<=1.+1e-9
        bonus=self.rear_damage_coefficient*credit
        info=dict(info,actual_rear_damage=self.actual_rear_damage,
                  actual_rear_damage_seconds=self.actual_rear_damage_seconds,
                  actual_rear_reward=self.rear_damage_coefficient*self.actual_rear_damage)
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
    return Monitor(RearDamageTrainingEnv(archive,plan['ppo']['gamma'],plan['tail_coefficients'],
        position_per_second=plan['position_per_second'],rear_damage_coefficient=plan['rear_damage_coefficient']),
        info_keywords=('verdict_rule','result','terminal_reason','tail_seconds','tail_attack_seconds',
                       'max_tail_streak','damage_dealt','damage_received','head_on_damage','tail_bonus','position_reward_total',
                       'actual_rear_damage','actual_rear_damage_seconds','actual_rear_reward'))
