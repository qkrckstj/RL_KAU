"""Lower additive teacher prior in public WEZ or after observed own damage."""
import numpy as np
from experiments.aim_async.features import numpy_logits as base_numpy_logits
from experiments.aim_async.threat_prior_numpy import prior_values


def wez_or_damaged_gate(observations):
    x = np.asarray(observations, dtype=np.float32)
    if x.shape[-1:] != (43,) or not np.isfinite(x).all():
        raise ValueError('Expected finite43public features')
    return (x[..., 24] > .5) | (x[..., 19] < 1.)


def numpy_logits(observations, parameters):
    x = np.asarray(observations, dtype=np.float32)
    gate = wez_or_damaged_gate(x)
    base, lower = prior_values(parameters['prior_bias'], parameters['threat_prior_bias'])
    original = base_numpy_logits(x, parameters)
    correction = np.where(gate, np.float32(lower-base), np.float32(0.))
    return original+correction[..., None]*x[..., -9:]
