"""Frozen CEM pilot plus a learned, public-state defensive break condition."""
import json
import math
import zipfile
from experiments.league.extend_parameter_policy import Policy as BasePolicy

BASE_EXPERT={'chase_bonus':14.980432444467809,'cooldown_seconds':7.246492138983214,
    'cruise_speed':508.3342851556025,'escape_angle':37.023562748213806,
    'escape_seconds':13.399112095780294,'lead_seconds':3.394874618302454,
    'trigger_range':1496.4727986463436}
RISK_NAMES=('enemy_track_threshold','health_margin','break_seconds','break_angle','break_speed','cooldown_seconds')
RISK_LOW=(0.,-.5,0.,-150.,300.,0.)
RISK_HIGH=(2.,.5,5.,150.,650.,12.)
RISK_BASELINE=(.5,0.,0.,0.,600.,4.)


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as z:configuration=json.loads(z.read('parameters.json'))
        self.config=dict(configuration)
        if set(self.config)!=set(RISK_NAMES) or any(not math.isfinite(float(self.config[k])) or not lo<=self.config[k]<=hi for k,lo,hi in zip(RISK_NAMES,RISK_LOW,RISK_HIGH)):
            raise ValueError('Invalid defensive break parameters')
        self.base=BasePolicy(configuration=BASE_EXPERT)
        self.reset()

    def reset(self):
        self.base.reset();self.last_time=None;self.break_until=-1.;self.cooldown_until=-1.;self.break_heading=0.

    def act(self,obs):
        if len(obs)!=39 or not all(math.isfinite(float(x)) for x in obs):raise ValueError('Expected39 finite public channels')
        t=120.-float(obs[38])
        if self.last_time is not None and t<self.last_time-1e-6:self.reset()
        self.last_time=t
        original=self.base.act(obs)  # Preserve the base pilot's actual observation history.
        cfg=self.config
        if cfg['break_seconds']==0.:return original
        if (t>=self.break_until and t>=self.cooldown_until and float(obs[35])>.5
            and float(obs[33])>=cfg['enemy_track_threshold']
            and float(obs[30])-float(obs[31])<=cfg['health_margin']):
            self.break_heading=float(obs[11])+math.radians(cfg['break_angle'])
            self.break_until=t+cfg['break_seconds'];self.cooldown_until=self.break_until+cfg['cooldown_seconds']
        if t>=self.break_until:return original
        error=self.break_heading-float(obs[11]);error=math.atan2(math.sin(error),math.cos(error))
        demand=math.degrees(error-.5*float(obs[14]));turn=2 if demand>3 else 0 if demand< -3 else 1
        speed=math.sqrt(sum(float(v)**2 for v in obs[3:6]))/.514444
        throttle=2 if speed<cfg['break_speed']-5 else 0 if speed>cfg['break_speed']+5 else 1
        return int(3*turn+throttle)
