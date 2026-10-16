"""Explicit public encoded-range threshold shared by training and standalone use."""
import numpy as np
from experiments.aim_async.features import numpy_logits as base_numpy_logits
from experiments.aim_async.threat_prior_numpy import prior_values


def range_limit(range_threshold_m):
    value=np.asarray(range_threshold_m)
    if (value.shape!=() or value.dtype.kind not in 'fiu' or not np.isfinite(value)
            or not 0.<float(value)<=10000.):
        raise ValueError('Expected finite numeric range within(0,10000]meters')
    distance=float(value)
    return np.float32(distance/(distance+10000.))


def range_gate(observations,range_threshold_m):
    limit=range_limit(range_threshold_m)
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1:]!=(43,) or not np.isfinite(x).all():
        raise ValueError('Expected finite43public features')
    return (x[...,24]>.5)|(x[...,19]<1.)|(x[...,0]<=limit)


def numpy_logits(observations,parameters):
    if 'range_threshold_m' not in parameters:
        raise ValueError('Missing explicit range threshold')
    x=np.asarray(observations,dtype=np.float32)
    gate=range_gate(x,parameters['range_threshold_m'])
    base,lower=prior_values(parameters['prior_bias'],parameters['threat_prior_bias'])
    original=base_numpy_logits(x,parameters)
    correction=np.where(gate,np.float32(lower-base),np.float32(0.))
    return original+correction[...,None]*x[...,-9:]


def experiment_range(settings):
    if 'range_threshold_m' not in settings:
        raise ValueError('Missing explicit experiment range')
    value=settings['range_threshold_m'];range_limit(value)
    if float(value)!=8000.:
        raise ValueError('This experiment requires explicit8000meter range')
    return float(value)
