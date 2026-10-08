"""Untrained switches between frozen public-state tactical controllers."""
import json
import math
import zipfile
from experiments.league.extend_parameter_policy import Policy as ExpertPolicy


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as z:configuration=json.loads(z.read('parameters.json'))
        self.config=dict(configuration)
        if (set(self.config)!={'experts','mode','threshold','dwell','offset'}
            or len(self.config['experts'])!=2
            or self.config['mode'] not in ['periodic','distance','turn','threat']
            or any(not math.isfinite(float(self.config[k])) for k in ['threshold','dwell','offset'])
            or self.config['threshold']<=0 or not .1<=self.config['dwell']<=5.):
            raise ValueError('Invalid switching configuration')
        self.experts=[ExpertPolicy(configuration=c) for c in self.config['experts']]
        self.reset()

    def reset(self):
        for expert in self.experts:expert.reset()
        self.last_time=None;self.selected=None;self.selected_at=-1e6

    def act(self,obs):
        if len(obs)!=39 or not all(math.isfinite(float(x)) for x in obs):raise ValueError('Finite public39 observation required')
        t=120.-float(obs[38])
        if self.last_time is not None and t<self.last_time-1e-6:self.reset()
        self.last_time=t
        actions=[e.act(obs) for e in self.experts]
        dx,dy=float(obs[15]-obs[0]),float(obs[16]-obs[1]);distance=math.hypot(dx,dy)
        mode,threshold=self.config['mode'],self.config['threshold']
        if mode=='periodic':choice=int((t+self.config['offset'])/threshold)%2
        elif mode=='distance':choice=int(distance<threshold)
        elif mode=='turn':choice=int(abs(float(obs[29]))>threshold)
        else:
            error=math.atan2(dx,dy)+math.pi-float(obs[26]);error=math.atan2(math.sin(error),math.cos(error))
            choice=int(distance<threshold and abs(error)<math.radians(40) and float(obs[34])<.5)
        if self.selected is None or t-self.selected_at>=self.config['dwell']:
            if choice!=self.selected:self.selected_at=t
            self.selected=choice
        return actions[self.selected]
