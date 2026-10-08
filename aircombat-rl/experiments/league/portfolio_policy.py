"""Select a frozen expert using only a short public-motion observation history."""
from pathlib import Path
import importlib.util
import hashlib
import json
import math
import zipfile
import numpy as np


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights,device='cpu'):
        root=Path(weights).resolve().parent
        with zipfile.ZipFile(weights) as z:config=json.loads(z.read('parameters.json'))
        self.probe=float(config['probe_seconds']);self.gate=np.asarray(config['gate'],float)
        self.forced=config.get('forced_expert')
        if not .5<=self.probe<=6 or self.gate.shape!=(3,5) or not np.isfinite(self.gate).all():raise ValueError('Invalid gate')
        if self.forced is not None and self.forced not in range(4):raise ValueError('Invalid forced expert')
        self.experts=[]
        for i in range(4):
            path=root/'experts'/str(i)/'policy.py'
            spec=importlib.util.spec_from_file_location('portfolio_'+hashlib.sha256(str(path).encode()).hexdigest(),path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            expert=module.Policy(str(path.parent/'policy_net.zip'),device=device)
            if getattr(expert,'ACTION_MODE','discrete')!='discrete':raise ValueError('Discrete experts required')
            self.experts.append(expert)
        self.reset()

    def reset(self):
        for expert in self.experts:
            if hasattr(expert,'reset'):expert.reset()
        self.previous=None;self.selected=None;self.integral=0.;self.elapsed=0.;self.features=None

    def act(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        if x.shape!=(39,) or not np.isfinite(x).all():raise ValueError('Expected public39 observations')
        clock=float(x[38])
        if self.previous is not None and clock>self.previous+1e-6:self.reset()
        distance=math.hypot(float(x[15]-x[0]),float(x[16]-x[1]));speed=math.sqrt(sum(float(v)**2 for v in x[18:21]));heading=float(x[26]);turn=float(x[29])
        if self.previous is None:self.start_distance=distance;self.start_speed=speed;self.start_heading=heading;self.previous_turn=turn
        else:
            dt=self.previous-clock
            if dt<0:raise ValueError('Noncausal observation clock')
            self.integral+=.5*(self.previous_turn+turn)*dt;self.elapsed+=dt
        self.previous=clock;self.previous_turn=turn
        if self.selected is None and self.elapsed>=self.probe-1e-5:
            dh=math.atan2(math.sin(heading-self.start_heading),math.cos(heading-self.start_heading))
            self.features=np.array([1.,np.clip(abs(10*self.integral/self.elapsed),0,3),np.clip((speed-self.start_speed)/20.,-3,3),np.clip(abs(dh)/.35,0,3),np.clip((self.start_distance-distance)/self.elapsed/150.,-3,3)])
            self.selected=int(np.argmax(np.r_[0.,self.gate@self.features]))
        # Warm each expert on the actual public prefix, including its own RNG.
        if self.selected is None or self.forced is not None:
            actions=[int(e.act(x)) for e in self.experts]
            return actions[0 if self.forced is None else self.forced]
        return int(self.experts[self.selected].act(x))
