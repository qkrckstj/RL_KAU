"""Learning-only tail pursuit reward and shared, seat-relative diagnostics."""
import math
from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.league.health_rule_v2 import HealthTrainingEnv
from experiments.league.strategy_worker import PeriodicArchiveFight


def geometry(raw):
    d=O.unpack(raw)
    dx,dy=d['opp_x']-d['own_x'],d['opp_y']-d['own_y']
    r=math.hypot(dx,dy)
    nx,ny=(dx/max(r,1e-9),dy/max(r,1e-9))
    aim=nx*math.sin(d['own_psi'])+ny*math.cos(d['own_psi'])
    tail=nx*math.sin(d['opp_psi'])+ny*math.cos(d['opp_psi'])
    # Tail is +1 behind a target pointing away; head-on gives -1.
    aligned=aim>=math.cos(math.radians(30))
    rear=tail>=math.cos(math.radians(45))
    in_range=FairFightEnv.wez_r_min<=r<=FairFightEnv.wez_r_max
    return dict(tail_position=bool(aligned and rear and in_range),
                tail_attack=bool(rear and d['own_in_wez']>.5),
                head_on=bool(aligned and tail<=-math.cos(math.radians(30)) and r<=FairFightEnv.wez_r_max),
                potential=max(0.,aim)*max(0.,tail)*max(0.,1.-r/4000.),
                own_health=float(d['own_health']),opp_health=float(d['opp_health']))


class TailTrainingEnv(HealthTrainingEnv):
    def __init__(self,env,gamma,coefficients):
        super().__init__(env,gamma=gamma)
        self.coefficients=coefficients

    def reset(self,**kwargs):
        state,info=super().reset(**kwargs)
        # Features preserve the same health channels, while raw geometry comes
        # from the current physical observations, not a guessed feature inverse.
        self.previous=geometry(self.env._obs(self.env._last))
        self.metrics=dict(tail_seconds=0.,tail_attack_seconds=0.,max_tail_streak=0.,
                          damage_dealt=0.,damage_received=0.,head_on_damage=0.,tail_bonus=0.)
        self.streak=0.
        return state,info

    def step(self,action):
        state,reward,term,trunc,info=super().step(action)
        now=geometry(self.env._obs(self.env._last));old=self.previous;dt=.05
        dealt=max(0.,old['opp_health']-now['opp_health'])
        received=max(0.,old['own_health']-now['own_health'])
        self.streak=self.streak+dt if now['tail_position'] else 0.
        m=self.metrics;m['tail_seconds']+=dt*now['tail_position']
        m['tail_attack_seconds']+=dt*now['tail_attack']
        m['max_tail_streak']=max(m['max_tail_streak'],self.streak)
        m['damage_dealt']+=dealt;m['damage_received']+=received
        m['head_on_damage']+=received*old['head_on']
        c=self.coefficients
        phi=0. if term or trunc else now['potential']
        # Persistent-position reward capped per episode; incremental damage
        # rewards cannot be collected repeatedly for an already damaged foe.
        hold=min(dt,max(0.,10.-(m['tail_seconds']-dt))) if now['tail_position'] else 0.
        bonus=c['potential']*(self.gamma*phi-old['potential'])+c['hold']*hold+c['damage']*dealt-c['received']*received
        m['tail_bonus']+=bonus;self.previous=now
        if term or trunc:info=dict(info,**m)
        return state,reward+bonus,term,trunc,info


def make_worker(plan,index):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=PeriodicArchiveFight(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    return Monitor(TailTrainingEnv(archive,plan['ppo']['gamma'],plan['tail_coefficients']),
        info_keywords=('verdict_rule','result','terminal_reason','tail_seconds','tail_attack_seconds',
                       'max_tail_streak','damage_dealt','damage_received','head_on_damage','tail_bonus'))
