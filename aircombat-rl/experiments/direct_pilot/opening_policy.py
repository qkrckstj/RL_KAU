"""Public-observation opening turns followed by lead pursuit and closure control."""
import json,math,zipfile
from pathlib import Path
import numpy as np
DEFAULTS=dict(opening_seconds=0.,opening_turn=-1.,opening_speed=350.,tracking_speed=500.,yaw_brake=.8,deadband_deg=4.,closure_range=900.)
BOUNDS=dict(opening_seconds=(0.,60.),opening_turn=(-1.,1.),opening_speed=(300.,650.),tracking_speed=(300.,650.),yaw_brake=(0.,3.),deadband_deg=(.5,15.),closure_range=(400.,1500.))

def wrap(a):return math.atan2(math.sin(a),math.cos(a))
def heading_rate(raw):
    c=math.cos(float(raw[10]));denominator=math.copysign(max(abs(c),1e-3),c)
    rate=(float(raw[13])*math.sin(float(raw[9]))+float(raw[14])*math.cos(float(raw[9])))/denominator
    return max(-2.,min(2.,rate))
def range_rate(raw):
    d=[float(raw[k+15])-float(raw[k]) for k in (0,1,2)]
    v=[float(raw[k+15])-float(raw[k]) for k in (3,4,5)]
    return sum(a*b for a,b in zip(d,v))/max(math.sqrt(sum(a*a for a in d)),1e-6)
def gun_heading(raw):
    dx=float(raw[15])-float(raw[0]);dy=float(raw[16])-float(raw[1]);dz=float(raw[17])-float(raw[2])
    tof=max(math.sqrt(dx*dx+dy*dy+dz*dz),1e-6)/1050.
    return math.atan2(dx+(float(raw[18])-float(raw[3]))*tof,dy+(float(raw[19])-float(raw[4]))*tof)
class Policy:
    ACTION_MODE='discrete'
    def __init__(self,weights=None,device='cpu',parameters=None):
        if parameters is None and weights is not None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:parameters=json.loads(z.read('parameters.json'))['parameters']
            else:parameters=json.loads(Path(weights).read_text())['parameters']
        parameters={} if parameters is None else parameters
        if not isinstance(parameters,dict) or set(parameters)-set(DEFAULTS):raise ValueError('Unknown opening parameters')
        self.parameters=dict(DEFAULTS,**parameters)
        for k,v in self.parameters.items():
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not BOUNDS[k][0]<=v<=BOUNDS[k][1]:raise ValueError('Invalid opening parameter:'+k)
        if self.parameters['opening_turn'] not in (-1.,1.):raise ValueError('Opening turn must be−1or+1')
    def reset(self):pass
    def act(self,obs):
        raw=np.asarray(obs,dtype=np.float32)
        if raw.shape!=(39,) or not np.isfinite(raw).all():raise ValueError('Expected39finite public observations')
        p=self.parameters;speed=math.sqrt(sum(float(raw[k])**2 for k in (3,4,5)))/.514444
        if 120.-float(raw[38])<p['opening_seconds']:
            turn=0 if p['opening_turn']<0 else 2;target=p['opening_speed']
            throttle=2 if speed<target-5. else 0 if speed>target+5. else 1
        else:
            error=wrap(gun_heading(raw)-float(raw[11]))
            # Subtract heading-rate feedback without wrapping the derivative term.
            demand=math.degrees(error-p['yaw_brake']*heading_rate(raw))
            turn=2 if demand>p['deadband_deg'] else 0 if demand<-p['deadband_deg'] else 1
            r=math.sqrt(sum((float(raw[k+15])-float(raw[k]))**2 for k in (0,1,2)))
            if r>p['closure_range']:
                throttle=2 if speed<p['tracking_speed']-5. else 0 if speed>p['tracking_speed']+5. else 1
            elif range_rate(raw)>=-20. and r>400.:throttle=1
            else:throttle=0
        return int(3*turn+throttle)
