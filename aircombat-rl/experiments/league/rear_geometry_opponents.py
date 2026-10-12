"""Training opponent weighting only; preserve archive coverage and native episodes."""
import numpy as np
from experiments.league.tail_curriculum import CurriculumArchive
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.league.periodic_coverage import PeriodicCoverageSampler
from experiments.league.tail_actual_reward import RearDamageTrainingEnv


def blend_weight(steps, fade_steps, enabled):
    if fade_steps <= 0 or steps < 0:raise ValueError('Invalid schedule')
    return max(0., 1.-steps/fade_steps) if enabled else 0.


class GeometryOpponentArchive(CurriculumArchive):
    def __init__(self,*args,opponent_schedule,**kwargs):
        self.opponent_schedule=dict(opponent_schedule)
        self.physical_steps=0
        super().__init__(*args,**kwargs)
        self.base_probabilities=np.asarray(self.sampler.probabilities).copy()
        ids=[x['id'] for x in self.opponents]
        focus=self.opponent_schedule['focus_ids']
        if not focus or len(set(focus))!=len(focus) or not set(focus)<=set(ids):raise ValueError('Invalid focus roster')
        self.focus_probabilities=np.array([float(x in focus) for x in ids]);self.focus_probabilities/=self.focus_probabilities.sum()

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.sampler=PeriodicCoverageSampler(self.probabilities,int(seed),self.coverage_period)
            self.episode_index=0;self.physical_steps=0
        episode_seed=self.ic_band+self.episode_index*self.workers+self.worker_index
        preview_rng=np.random.default_rng(episode_seed)
        initial,_=FairFightEnv.sample(self,preview_rng)
        quarter=bool(self.curriculum['fraction'] and preview_rng.random()<self.curriculum['fraction'])
        predicted='training_tail_'+self.curriculum['stage'] if quarter else initial.name
        focus=self.opponent_schedule['quarter_focus_ids'] if quarter else self.opponent_schedule['focus_ids']
        ids=[s['id'] for s in self.opponents]
        if not focus or len(set(focus))!=len(focus) or not set(focus)<=set(ids):raise ValueError('Invalid geometry focus')
        self.focus_probabilities=np.array([float(x in focus) for x in ids]);self.focus_probabilities/=self.focus_probabilities.sum()
        weight=blend_weight(self.physical_steps,self.opponent_schedule['fade_steps_per_env'],self.opponent_schedule['enabled'])
        self.sampler.probabilities=(1.-weight)*self.base_probabilities+weight*self.focus_probabilities
        raw,info=super().reset(seed=None,options=options)
        assert self.curriculum_geometry==predicted
        self.episode_metadata.update(opponent_focus_ids=list(focus),opponent_focus_weight=weight,schedule_start_step=self.physical_steps,curriculum_geometry=self.curriculum_geometry)
        info['league_episode']=dict(self.episode_metadata)
        return raw,info

    def step(self,action):
        result=super().step(action)
        self.physical_steps+=1
        return result


def make_worker(plan,index):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=GeometryOpponentArchive(plan['teacher'],plan['opponents'],plan['probabilities'],plan['seed']*100+index,plan['ic_band'],index,plan['workers'],'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'],opponent_schedule=plan['opponent_schedule'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    return Monitor(RearDamageTrainingEnv(archive,plan['ppo']['gamma'],plan['tail_coefficients'],position_per_second=plan['position_per_second'],rear_damage_coefficient=plan['rear_damage_coefficient']),info_keywords=('verdict_rule','result','terminal_reason','tail_seconds','tail_attack_seconds','max_tail_streak','damage_dealt','damage_received','head_on_damage','tail_bonus','position_reward_total','actual_rear_damage','actual_rear_damage_seconds','actual_rear_reward'))
