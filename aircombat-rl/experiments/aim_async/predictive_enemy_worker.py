"""Use public native enemy lead in relative reward; combat and actor inputs unchanged."""
import math
import numpy as np
from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.engagement import look
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.aim_async.features import Kinematics
from experiments.aim_async.policy import original_columns
from experiments.aim_async.draw_utility_worker import DrawUtilityGuidanceEnv
from experiments.aim_async.relative_reward_worker import relative_bonus
from experiments.league.residual_env import potential
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


def enemy_lead_cos(raw):
    raw = np.asarray(raw,dtype=np.float32)
    if raw.shape != (39,) or not np.isfinite(raw).all():
        raise ValueError('Expected39finite public observations')
    data = O.unpack(raw)
    enemy = look(Kinematics(data,'opp'),Kinematics(data,'own'),FairFightEnv.wez_cone_deg)
    return math.cos(enemy.ata_lead)


def predictive_enemy_potential(state, enemy_cos):
    state = np.asarray(state)
    if state.shape != (43,) or not np.isfinite(state).all():
        raise ValueError('Expected43finite learner features')
    if not math.isfinite(enemy_cos) or not -1. <= enemy_cos <= 1.:
        raise ValueError('Expected finite enemy lead cosine in[-1,1]')
    old_cos = float(np.clip(-(state[4]*state[6]+state[3]*state[5]),-1.,1.))
    proximity = 1.-float(np.clip(state[0],0.,1.))
    return potential(state[original_columns()])+.125*proximity*(old_cos-enemy_cos)


class PredictiveEnemyGuidanceEnv(DrawUtilityGuidanceEnv):
    def _state(self, raw):
        self.enemy_predictive_cos = enemy_lead_cos(raw)
        return super()._state(raw)

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        self.original_relative_phi = self.relative_phi
        self.relative_phi = predictive_enemy_potential(state,self.enemy_predictive_cos)
        self.predictive_enemy_adjustment_total = 0.
        return state,dict(info,predictive_enemy_adjustment_total=0.)

    def step(self, action):
        previous = self.relative_phi
        original_previous = self.original_relative_phi
        state,reward,term,trunc,info = super().step(action)
        original_next = self.relative_phi
        next_phi = 0. if term or trunc else predictive_enemy_potential(state,self.enemy_predictive_cos)
        corrected = relative_bonus(previous,next_phi,term or trunc,self.gamma,self.relative_coefficient)
        original_counterfactual = relative_bonus(original_previous,original_next,term or trunc,self.gamma,self.relative_coefficient)
        correction = corrected-info['relative_shaping_reward']
        self.relative_phi = next_phi
        self.original_relative_phi = original_next
        self.relative_shaping_total += correction
        self.predictive_enemy_adjustment_total += corrected-original_counterfactual
        return state,reward+correction,term,trunc,dict(info,relative_shaping_reward=corrected,
            relative_shaping_total=self.relative_shaping_total,
            predictive_enemy_adjustment_total=self.predictive_enemy_adjustment_total)


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive = CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    wrapped = PredictiveEnemyGuidanceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],
        plan['control_guidance_range'],draw_penalty=plan['terminal_draw_penalty'])
    env = Restartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total','draw_utility_total','predictive_enemy_adjustment_total')))
    if restart is not None:
        env.restore_worker(restart)
    return env
