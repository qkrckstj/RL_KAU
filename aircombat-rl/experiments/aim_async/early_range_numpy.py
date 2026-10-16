"""Extend the public lower-prior gate to the float32 encoded4km boundary."""
import numpy as np
from experiments.aim_async.features import numpy_logits as base_numpy_logits
from experiments.aim_async.threat_prior_numpy import prior_values

EARLY_RANGE_METERS = 4000.
SCALED_RANGE_LIMIT = np.float32(EARLY_RANGE_METERS/(EARLY_RANGE_METERS+10000.))


def early_range_gate(observations):
    x = np.asarray(observations,dtype=np.float32)
    if x.shape[-1:] != (43,) or not np.isfinite(x).all():
        raise ValueError('Expected finite43public features')
    return (x[...,24]>.5)|(x[...,19]<1.)|(x[...,0]<=SCALED_RANGE_LIMIT)


def numpy_logits(observations,parameters):
    x = np.asarray(observations,dtype=np.float32)
    gate = early_range_gate(x)
    base,lower = prior_values(parameters['prior_bias'],parameters['threat_prior_bias'])
    original = base_numpy_logits(x,parameters)
    correction = np.where(gate,np.float32(lower-base),np.float32(0.))
    return original+correction[...,None]*x[...,-9:]
