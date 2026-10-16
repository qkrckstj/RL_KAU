"""Public-state predictive pilot with official relative-velocity gun-lead reconstruction."""
import json,math,zipfile
from pathlib import Path
import numpy as np

BOUNDS=dict(break_range=(0.,9000.),break_duration=(0.,20.),break_turn_deg=(-150.,150.),
    slot_back=(0.,6000.),slot_side=(-6000.,6000.),lead=(0.,12.),
    entry_speed=(300.,650.),turn_speed=(300.,650.),hold_distance=(200.,1400.),yaw_damping=(0.,3.),gun_range=(0.,5000.),command_mode=(0.,1.))
DEFAULTS=dict(break_range=4500.,break_duration=8.,break_turn_deg=-90.,slot_back=1200.,
    slot_side=0.,lead=.4,entry_speed=625.,turn_speed=550.,hold_distance=700.,yaw_damping=.8,gun_range=1800.,command_mode=1)

def wrap(x):return math.atan2(math.sin(x),math.cos(x))

def gun_heading(raw):
    dx=float(raw[15])-float(raw[0]);dy=float(raw[16])-float(raw[1]);dz=float(raw[17])-float(raw[2])
    tof=max(math.sqrt(dx*dx+dy*dy+dz*dz),1e-6)/1050.
    return math.atan2(dx+(float(raw[18])-float(raw[3]))*tof,dy+(float(raw[19])-float(raw[4]))*tof)

def choose_heading(current,desired,retained):
    """Select a grid command; zero preserves the estimated prior target.

    The estimate uses public float32 heading, not privileged internal state.
    Ties within numeric roundoff prefer hold.
    """
    candidates=(wrap(current-math.pi/6),retained,wrap(current+math.pi/6))
    action=1;cost=abs(wrap(retained-desired))
    for index in (0,2):
        candidate_cost=abs(wrap(candidates[index]-desired))
        if candidate_cost<cost-1e-12:action=index;cost=candidate_cost
    return action,candidates[action]

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
        if self.parameters['command_mode'] not in (0.,1.):raise ValueError('command_mode must be0or1')
        self.reset()
    def reset(self):
        self.phase='approach';self.last_time=None;self.last_psi=None;self.break_started=None;self.break_heading=None;self.last_enemy_psi=None;self.enemy_rate=0.
        self.command_heading=None;self.command_before_action=None;self.last_observation=None;self.last_action=None
    def synchronize_executed_action(self,obs,action):
        x=np.asarray(obs,dtype=np.float32)
        if type(action) is not int or not 0<=action<9:raise ValueError('Expected one of9integer actions')
        if self.last_observation is None or not np.array_equal(x,self.last_observation):raise ValueError('Expected the most recently queried observation')
        turn=action//3
        self.command_heading=self.command_before_action if turn==1 else wrap(float(x[11])+math.radians(30*(turn-1)))
    def act(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        if x.shape!=(39,) or not np.isfinite(x).all():raise ValueError('Expected39finite public observations')
        t=120.-float(x[38]);psi=float(x[11]);enemy_psi=float(x[26]);p=self.parameters
        if self.last_time is not None and t<self.last_time-1e-6:self.reset()
        if self.last_observation is not None and np.array_equal(x,self.last_observation):return self.last_action
        if self.command_heading is None:self.command_heading=psi
        self.command_before_action=self.command_heading
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
        if p['gun_range']>0. and distance<=p['gun_range'] and self.phase!='break':desired=gun_heading(x)
        error=wrap(desired-psi)
        if abs(error)>math.radians(70.):speed=min(speed,p['turn_speed'])
        demand=wrap(error-p['yaw_damping']*yaw)
        if p['command_mode']==1:
            turn,self.command_heading=choose_heading(psi,wrap(psi+demand),self.command_before_action)
        else:
            turn=2 if demand>math.radians(4.) else 0 if demand<math.radians(-4.) else 1
            self.command_heading=self.command_before_action if turn==1 else wrap(psi+math.radians(30*(turn-1)))
        own_speed=math.sqrt(sum(float(v)**2 for v in x[3:6]))/.514444
        throttle=2 if own_speed<speed-5. else 0 if own_speed>speed+5. else 1
        self.last_observation=x.copy();self.last_action=int(3*turn+throttle)
        return self.last_action
