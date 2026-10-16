"""Fixed public-clock replay of agent-selected manual controls, not neural learning.

This reproduces one demonstration. It does not infer a general feedback strategy
or use opponent identity, initial-condition seed, outcome, or hidden state.
"""
from bisect import bisect_right
import json,zipfile
from pathlib import Path
import numpy as np

class Policy:
    ACTION_MODE='discrete'
    def __init__(self,weights=None,device='cpu',parameters=None):
        if parameters is None and weights is not None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:parameters=json.loads(z.read('parameters.json'))['parameters']
            else:parameters=json.loads(Path(weights).read_text())['parameters']
        if not isinstance(parameters,dict) or set(parameters)!={'commands'}:raise ValueError('Expected manual commands only')
        commands=parameters['commands']
        if not isinstance(commands,list) or not commands:raise ValueError('Expected nonempty manual commands')
        self.steps=[];self.actions=[]
        for row in commands:
            if not isinstance(row,dict) or set(row)!={'step','action'}:raise ValueError('Expected step/action pair')
            step,action=row['step'],row['action']
            if type(step)is not int or step<0 or type(action)is not int or not 0<=action<9:raise ValueError('Invalid manual command')
            if (not self.steps and step!=0) or (self.steps and step<=self.steps[-1]):raise ValueError('Expected increasing steps starting at0')
            self.steps.append(step);self.actions.append(action)
    def reset(self):pass
    def act(self,obs):
        raw=np.asarray(obs,dtype=np.float32)
        if raw.shape!=(39,) or not np.isfinite(raw).all():raise ValueError('Expected39finite public observations')
        # Raw clock is float32. Nearest decision tick avoids boundary rounding drift.
        tick=max(0,int(round((120.-float(raw[38]))*20.)))
        return self.actions[bisect_right(self.steps,tick)-1]
