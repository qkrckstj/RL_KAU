"""Public-state extend controller with learnable tactical parameters."""
import json
import math
import zipfile

NAMES=('trigger_range','escape_seconds','cooldown_seconds','escape_angle','cruise_speed','lead_seconds','chase_bonus')
LOW=(700.,2.,2.,-70.,450.,0.,0.)
HIGH=(2400.,14.,18.,70.,650.,7.,200.)
BASELINE=(1300.,8.,8.,20.,600.,3.,0.)


def wrap(x):
    return math.atan2(math.sin(x),math.cos(x))


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as z:
                configuration=json.loads(z.read('parameters.json'))
        self.config=dict(configuration)
        if set(self.config)!=set(NAMES) or any(not math.isfinite(float(self.config[k])) or not lo<=self.config[k]<=hi for k,lo,hi in zip(NAMES,LOW,HIGH)):
            raise ValueError('Invalid tactical parameters')
        self.reset()

    def reset(self):
        self.last_time=None
        self.escape_until=-1.
        self.escape_heading=0.
        self.cooldown_until=-1.

    def act(self,obs):
        if len(obs)!=39 or not all(math.isfinite(float(x)) for x in obs):
            raise ValueError('Expected39 finite public observation channels')
        t=120.-float(obs[38])
        if self.last_time is not None and t<self.last_time-1e-6:
            self.reset()
        self.last_time=t
        heading=float(obs[11])
        dx,dy=float(obs[15]-obs[0]),float(obs[16]-obs[1])
        distance=math.hypot(dx,dy)
        bearing=math.atan2(dx,dy)
        in_wez=float(obs[34])>.5
        lead=min(self.config['lead_seconds'],distance/1000.)
        desired=math.atan2(dx+lead*float(obs[18]),dy+lead*float(obs[19]))
        speed=self.config['cruise_speed']
        if distance>3000. and not in_wez:
            speed=min(650.,speed+self.config['chase_bonus'])
        if t>=self.escape_until:
            if distance<self.config['trigger_range'] and not in_wez and t>=self.cooldown_until:
                self.escape_heading=bearing+math.pi+math.radians(self.config['escape_angle'])
                self.escape_until=t+self.config['escape_seconds']
                self.cooldown_until=self.escape_until+self.config['cooldown_seconds']
        if t<self.escape_until and not in_wez:
            desired,speed=self.escape_heading,650.
        demand=math.degrees(wrap(desired-heading)-.5*float(obs[14]))
        turn=2 if demand>3. else 0 if demand< -3. else 1
        own_speed=math.sqrt(sum(float(v)**2 for v in obs[3:6]))/.514444
        throttle=2 if own_speed<speed-5 else 0 if own_speed>speed+5 else 1
        return int(3*turn+throttle)
