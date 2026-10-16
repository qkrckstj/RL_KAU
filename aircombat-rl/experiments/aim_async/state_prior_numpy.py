"""Public-state teacher-confidence rule shared with standalone inference."""
import math
import numpy as np
from experiments.aim_async.features import numpy_logits as base_numpy_logits


def conflict_gate(observations):
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1:]!=(43,) or not np.isfinite(x).all():
        raise ValueError('Expected finite43public features')
    turn=np.argmax(x[...,-9:],axis=-1)//3-1
    return ((x[...,24]>.5)&(x[...,23]<.5)
            &(x[...,32]<np.float32(math.cos(math.radians(30.))))
            &(turn*x[...,31]<0.))


def numpy_logits(observations,parameters):
    x=np.asarray(observations,dtype=np.float32)
    original=base_numpy_logits(x,parameters)
    correction=np.where(conflict_gate(x),
        parameters['conflict_prior_bias']-parameters['prior_bias'],np.float32(0.))
    return original+correction[...,None]*x[...,-9:]
