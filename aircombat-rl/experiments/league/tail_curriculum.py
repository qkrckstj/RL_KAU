"""Training-only initial geometry curriculum; FairFight itself stays unchanged."""
import math
from aircombat_gym.wvr.envs.base import Initial
from experiments.league.strategy_worker import PeriodicArchiveFight
from experiments.league.tail_reward import TailTrainingEnv


def rear_initial(initial,seat,rng,stage):
    """Symmetric physical-seat placement behind an armed, unchanged opponent."""
    if stage not in ('rear','quarter'):raise ValueError(stage)
    own=0 if seat=='red' else 1;foe=1-own
    heading=float(rng.uniform(-180,180))
    bearing=heading+180+float(rng.uniform(-15,15) if stage=='rear' else rng.uniform(-65,65))
    distance=float(rng.uniform(1000,1800) if stage=='rear' else rng.uniform(1800,3200))
    dx=distance*math.sin(math.radians(bearing));dy=distance*math.cos(math.radians(bearing))
    xy=[None,None];xy[own]=(dx/2,dy/2);xy[foe]=(-dx/2,-dy/2)
    psi=[None,None];psi[foe]=heading;psi[own]=heading+float(rng.uniform(-10,10))
    return Initial('training_tail_'+stage,tuple(xy),tuple(psi),initial.v_kt,initial.alt_ft)


class CurriculumArchive(PeriodicArchiveFight):
    def __init__(self,*args,curriculum,**kwargs):
        self.curriculum=curriculum
        if not 0<=curriculum['fraction']<=1:raise ValueError('fraction')
        super().__init__(*args,**kwargs)

    def sample(self,rng):
        initial,foe=super().sample(rng)
        # Fraction0 uses exactly the original sampling stream and IC.
        if self.curriculum['fraction'] and rng.random()<self.curriculum['fraction']:
            initial=rear_initial(initial,self.seat,rng,self.curriculum['stage'])
        self.curriculum_geometry=initial.name
        return initial,foe


def make_worker(plan,index):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    return Monitor(TailTrainingEnv(archive,plan['ppo']['gamma'],plan['tail_coefficients']),
        info_keywords=('verdict_rule','result','terminal_reason','tail_seconds','tail_attack_seconds',
                       'max_tail_streak','damage_dealt','damage_received','head_on_damage','tail_bonus'))
