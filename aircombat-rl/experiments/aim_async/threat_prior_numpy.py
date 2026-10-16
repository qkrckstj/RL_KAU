"""Public opponent-WEZ teacher confidence, shared with standalone inference."""
import numpy as np
from experiments.aim_async.features import numpy_logits as base_numpy_logits


def prior_values(base, threat):
    values=[]
    for item in (base,threat):
        scalar=np.asarray(item)
        if scalar.shape!=() or scalar.dtype.kind not in 'fiu' or not np.isfinite(scalar):
            raise ValueError('Expected finite numeric scalar prior')
        values.append(float(scalar))
    if not 0 < values[1] < values[0]:
        raise ValueError('Threat prior must be positive and below base prior')
    return tuple(values)


def threat_gate(observations):
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1:]!=(43,) or not np.isfinite(x).all():
        raise ValueError('Expected finite43public features')
    return x[...,24]>.5


def numpy_logits(observations,parameters):
    x=np.asarray(observations,dtype=np.float32)
    gate=threat_gate(x)
    base,threat=prior_values(parameters['prior_bias'],parameters['threat_prior_bias'])
    original=base_numpy_logits(x,parameters)
    correction=np.where(gate,np.float32(threat-base),np.float32(0.))
    return original+correction[...,None]*x[...,-9:]
