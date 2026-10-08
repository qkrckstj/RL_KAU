"""One frozen expert per episode; public first frame seeds a private RNG."""
import json
import math
import zipfile
import numpy as np
from experiments.league.unseen_temporal_repair_20261007.temporal_extend_right.policy import Policy as TeacherPolicy
from experiments.league.extend_parameter_policy import Policy as CemPolicy


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as z:configuration=json.loads(z.read('parameters.json'))
        self.config=dict(configuration)
        if (set(self.config)!={'teacher','cem','teacher_probability','action_seed'}
            or not isinstance(self.config['action_seed'],int) or isinstance(self.config['action_seed'],bool)
            or self.config['action_seed']<0 or not math.isfinite(float(self.config['teacher_probability']))
            or not 0<=self.config['teacher_probability']<=1):raise ValueError('Invalid episode mixture')
        self.experts=[TeacherPolicy(configuration=self.config['teacher']),CemPolicy(configuration=self.config['cem'])]
        self.reset()

    def reset(self):
        for e in self.experts:e.reset()
        self.selected=None;self.last_time=None

    def act(self,obs):
        raw=np.asarray(obs,dtype='<f4')
        if raw.shape!=(39,) or not np.isfinite(raw).all():raise ValueError('Expected39 finite public channels')
        t=120.-float(raw[38])
        if self.last_time is not None and t<self.last_time-1e-6:self.reset()
        self.last_time=t
        if self.selected is None:
            rng=np.random.default_rng(np.random.SeedSequence([self.config['action_seed'],739,*raw.view('<u4').tolist()]))
            self.selected=int(rng.random()>=self.config['teacher_probability'])
        return self.experts[self.selected].act(raw)
