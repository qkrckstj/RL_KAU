"""Agent-designed public-state break/rejoin pilot for the classroom simulator."""
import json,math,zipfile
from pathlib import Path
import numpy as np

BOUNDS=dict(break_range=(0.,9000.),break_duration=(0.,20.),break_turn_deg=(-150.,150.),
    slot_back=(0.,6000.),slot_side=(-6000.,6000.),lead=(0.,12.),
    entry_speed=(300.,650.),turn_speed=(300.,650.),hold_distance=(200.,1400.),yaw_damping=(0.,3.))
DEFAULTS=dict(break_range=4500.,break_duration=8.,break_turn_deg=-90.,slot_back=1200.,
    slot_side=0.,lead=.4,entry_speed=625.,turn_speed=550.,hold_distance=700.,yaw_damping=.8)

def wrap(x):return math.atan2(math.sin(x),math.cos(x))

def future_displacement(vx,vy,rate,seconds):
    if abs(rate)<1e-7:return vx*seconds,vy*seconds
    angle=rate*seconds;s=math.sin(angle)/rate;c=(1.-math.cos(angle))/rate
    return vx*s+vy*c,vy*s-vx*c


class Policy:
    ACTION_MODE='discrete'
    def __init__(self,weights=None,device='cpu',parameters=None):
        if parameters is None and weights is not None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:parameters=json.loads(z.read('parameters.json'))['parameters']
            else:parameters=json.loads(Path(weights).read_text())['parameters']
        parameters={} if parameters is None else parameters
        if not isinstance(parameters,dict) or set(parameters)-set(DEFAULTS):raise ValueError('Unknown pilot parameters')
        self.parameters=dict(DEFAULTS,**parameters)
        for key,value in self.parameters.items():
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not BOUNDS[key][0]<=value<=BOUNDS[key][1]:raise ValueError('Invalid pilot parameter:'+key)
        self.reset()
    def reset(self):
        self.phase='approach';self.last_time=None;self.last_psi=None;self.break_started=None;self.break_heading=None;self.last_enemy_psi=None;self.enemy_rate=0.
    def act(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        if x.shape!=(39,) or not np.isfinite(x).all():raise ValueError('Expected39finite public observations')
        t=120.-float(x[38]);psi=float(x[11]);enemy_psi=float(x[26]);p=self.parameters
        if self.last_time is not None and t<self.last_time-1e-6:self.reset()
        yaw=0. if self.last_time is None or t<=self.last_time else wrap(psi-self.last_psi)/(t-self.last_time)
        if self.last_time is not None and t>self.last_time:
            measured=float(np.clip(wrap(enemy_psi-self.last_enemy_psi)/(t-self.last_time),-.5,.5))
            self.enemy_rate=.9*self.enemy_rate+.1*measured
        self.last_time=t;self.last_psi=psi;self.last_enemy_psi=enemy_psi
        dx,dy=float(x[15]-x[0]),float(x[16]-x[1]);distance=math.hypot(dx,dy)
        bearing=math.atan2(dx,dy);tail=(dx*math.sin(enemy_psi)+dy*math.cos(enemy_psi))/max(distance,1e-6)
        if self.phase=='approach' and (distance<=p['break_range'] or x[30]<1.):
            self.phase='break';self.break_started=t;self.break_heading=wrap(psi+math.radians(p['break_turn_deg']))
        if self.phase=='break' and t-self.break_started>=p['break_duration']:self.phase='slot'
        if p['break_range']==0.:self.phase='slot'
        enemy_speed=math.sqrt(sum(float(v)**2 for v in x[18:21]))/.514444
        hold=(tail>=math.cos(math.radians(45.)) and distance<2200.) or (x[34]>.5 and x[35]<=.5)
        if self.phase=='break':desired=self.break_heading;speed=p['turn_speed']
        elif hold:
            desired=math.atan2(dx+.4*float(x[18]),dy+.4*float(x[19]))
            speed=float(np.clip(enemy_speed+.08*(distance-p['hold_distance']),400.,650.))
        elif self.phase=='slot':
            px,py=future_displacement(float(x[18]),float(x[19]),self.enemy_rate,p['lead'])
            predicted_heading=enemy_psi+self.enemy_rate*p['lead']
            tx=dx+px-p['slot_back']*math.sin(predicted_heading)+p['slot_side']*math.cos(predicted_heading)
            ty=dy+py-p['slot_back']*math.cos(predicted_heading)-p['slot_side']*math.sin(predicted_heading)
            desired=math.atan2(tx,ty);speed=p['entry_speed']
        else:desired=bearing;speed=p['entry_speed']
        error=wrap(desired-psi)
        if abs(error)>math.radians(70.):speed=min(speed,p['turn_speed'])
        demand=wrap(error-p['yaw_damping']*yaw)
        turn=2 if demand>math.radians(4.) else 0 if demand<math.radians(-4.) else 1
        own_speed=math.sqrt(sum(float(v)**2 for v in x[3:6]))/.514444
        throttle=2 if own_speed<speed-5. else 0 if own_speed>speed+5. else 1
        return int(3*turn+throttle)
