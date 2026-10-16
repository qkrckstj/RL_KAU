"""Learning-only win/draw utilities for the existing 43-input league family."""
import math
from experiments.aim_async.control_guidance_worker import ControlGuidanceTrainingEnv
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


class TerminalPreferenceEnv(ControlGuidanceTrainingEnv):
    def __init__(self,*args,win_bonus,draw_penalty,**kwargs):
        for name,value,upper in (('win_bonus',win_bonus,4.),('draw_penalty',draw_penalty,1.)):
            if (isinstance(value,bool) or not isinstance(value,(int,float))
                    or not math.isfinite(value) or not 0.<=value<=upper):
                raise ValueError(f'{name} must be finite within0..{upper}')
        super().__init__(*args,**kwargs)
        self.win_bonus=float(win_bonus);self.draw_penalty=float(draw_penalty)
        self.terminal_preference_total=0.

    def reset(self,**kwargs):
        state,info=super().reset(**kwargs)
        self.terminal_preference_total=0.
        return state,dict(info,terminal_preference_total=0.)

    def step(self,action):
        state,reward,term,trunc,info=super().step(action)
        adjustment=0.
        if term or trunc:
            result=info['result']
            if result not in ('win','draw','loss'):raise ValueError('Unknown terminal verdict')
            adjustment=self.win_bonus if result=='win' else -self.draw_penalty if result=='draw' else 0.
            info=dict(info,original_learning_utility=info['learning_utility'],
                      learning_utility=info['learning_utility']+adjustment)
        self.terminal_preference_total+=adjustment
        return state,reward+adjustment,term,trunc,dict(info,terminal_preference_total=self.terminal_preference_total)


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=TerminalPreferenceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],
        win_bonus=plan['terminal_win_bonus'],draw_penalty=plan['terminal_draw_penalty'])
    env=Restartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total','terminal_preference_total')))
    if restart is not None:env.restore_worker(restart)
    return env
