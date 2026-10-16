"""Learning-only terminal loss preference; native rules and current gate stay unchanged."""
import math
from experiments.aim_async.terminal_preference_worker import TerminalPreferenceEnv
from experiments.aim_async.threat_metrics import ThreatMetricsEnv,METRIC_NAMES as WEZ_METRIC_NAMES
from experiments.aim_async.wez_or_damaged_metrics import WezOrDamagedMetricsEnv,METRIC_NAMES
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


def validate_loss_penalty(value):
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or not 0.<=value<=4.):
        raise ValueError('loss_penalty must be finite within0..4')
    return float(value)


class TerminalLossPreferenceEnv(TerminalPreferenceEnv):
    def __init__(self,*args,loss_penalty=0.,**kwargs):
        penalty=validate_loss_penalty(loss_penalty)
        super().__init__(*args,**kwargs)
        self.loss_penalty=penalty;self.terminal_loss_total=0.;self.pre_loss_episode_return=0.

    def reset(self,**kwargs):
        state,info=super().reset(**kwargs)
        self.terminal_loss_total=0.;self.pre_loss_episode_return=0.
        return state,dict(info,terminal_loss_total=0.,pre_loss_episode_return=0.)

    def step(self,action):
        state,reward,term,trunc,info=super().step(action)
        adjustment=0.
        if term or trunc:
            adjustment=-self.loss_penalty if info['result']=='loss' else 0.
            info=dict(info,pre_loss_learning_utility=info['learning_utility'],
                      learning_utility=info['learning_utility']+adjustment)
        self.pre_loss_episode_return+=reward
        self.terminal_loss_total+=adjustment
        return state,reward+adjustment,term,trunc,dict(info,terminal_loss_total=self.terminal_loss_total,
            pre_loss_episode_return=self.pre_loss_episode_return)


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=TerminalLossPreferenceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],
        win_bonus=plan['terminal_win_bonus'],draw_penalty=plan['terminal_draw_penalty'],
        loss_penalty=plan.get('terminal_loss_penalty',0.))
    env=Restartable(Monitor(WezOrDamagedMetricsEnv(ThreatMetricsEnv(wrapped)),info_keywords=(
        'verdict_rule','result','terminal_reason','damage_dealt','damage_received',
        'connected_fire_seconds','lock_breaks','first_damage_time','first_received_time',
        'first_shot_advantage','relative_shaping_total','control_guidance_total',
        'terminal_preference_total','terminal_loss_total','pre_loss_episode_return',*WEZ_METRIC_NAMES,*METRIC_NAMES)))
    if restart is not None:env.restore_worker(restart)
    return env
