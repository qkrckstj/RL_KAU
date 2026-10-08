"""Causal public-state changes; current teacher action stays in the last9 slots.

History is local to an episode. No opponent identity or private RNG is read.
This prototype is not installed in any frozen running experiment.
"""
from collections import deque
import math
import numpy as np
from experiments.league.residual_features import FEATURE_DIM as BASE_DIM

PUBLIC_DIM=BASE_DIM-9
LAGS=(1.,5.)
HISTORY_DIM=PUBLIC_DIM*(1+len(LAGS))+9


class HistoryFeatures:
    def __init__(self):self.reset()

    def reset(self):
        self.samples=deque()

    def update(self,features,time_seconds):
        x=np.asarray(features,dtype=np.float32)
        if x.shape!=(BASE_DIM,) or not np.isfinite(x).all():raise ValueError('Finite base features required')
        now=float(time_seconds)
        if not math.isfinite(now):raise ValueError('Finite public game time required')
        if self.samples and now<self.samples[-1][0]:raise ValueError('Reset history before a new episode')
        current=x[:PUBLIC_DIM].copy()
        if self.samples and now==self.samples[-1][0]:self.samples[-1]=(now,current)
        else:self.samples.append((now,current))
        while len(self.samples)>1 and self.samples[1][0]<=now-max(LAGS):self.samples.popleft()
        parts=[current]
        for lag in LAGS:
            past=self.samples[0][1]
            for when,value in self.samples:
                if when>now-lag:break
                past=value
            parts.append((current-past)*np.float32(.5))
        return np.concatenate(parts+[x[-9:]]).astype(np.float32,copy=False)


def numpy_logits(observations,parameters):
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1:]!=(HISTORY_DIM,) or not np.isfinite(x).all():raise ValueError('Finite history features required')
    hidden=np.maximum(0.,x@parameters['w0'].T+parameters['b0'])
    hidden=np.maximum(0.,hidden@parameters['w1'].T+parameters['b1'])
    return hidden@parameters['wa'].T+parameters['ba']+float(parameters['prior_bias'])*x[..., -9:]
