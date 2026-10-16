"""Pure public44features:preserved relative/lead/prior plus heading-rate cue."""
import math
import numpy as np
from experiments.aim_async.features import features as aim_features

FEATURE_DIM=44


def original_columns():
    return list(range(34))+list(range(35,44))


def heading_rate(raw):
    # Same signed singularity floor as the frozen learning reward worker.
    raw=np.asarray(raw,dtype=np.float64)
    if raw.shape!=(39,) or not np.isfinite(raw).all():
        raise ValueError('Expected39finite public observations')
    phi,theta,q,r=(float(raw[i]) for i in (9,10,13,14))
    cosine=math.cos(theta)
    denominator=math.copysign(max(abs(cosine),1e-6),cosine)
    rate=(q*math.sin(phi)+r*math.cos(phi))/denominator
    if not math.isfinite(rate):raise ValueError('Unrepresentable public heading feedback')
    return rate


def augment(state,rate):
    state=np.asarray(state,dtype=np.float32)
    if state.shape!=(43,) or not np.isfinite(state).all() or not math.isfinite(rate):
        raise ValueError('Expected43finite features and finite public rate')
    cue=np.asarray([math.tanh(rate/.5)],dtype=np.float32)
    return np.concatenate((state[:34],cue,state[34:]))


def features(obs,prior_action):
    return augment(aim_features(obs,prior_action),heading_rate(obs))


def numpy_logits(observations,parameters):
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1:]!=(FEATURE_DIM,) or not np.isfinite(x).all():
        raise ValueError('Expected44finite public derived features')
    old=np.ascontiguousarray(np.concatenate((x[...,:31],x[...,35:]),axis=-1))
    weight=np.ascontiguousarray(np.concatenate((parameters['w0'][:,:31],parameters['w0'][:,35:]),axis=1))
    first=old@weight.T+parameters['b0']
    first=first+x[...,31:34]@parameters['w0'][:,31:34].T
    first=first+x[...,34:35]@parameters['w0'][:,34:35].T
    hidden=np.maximum(0.,first)
    hidden=np.maximum(0.,hidden@parameters['w1'].T+parameters['b1'])
    return hidden@parameters['wa'].T+parameters['ba']+float(parameters['prior_bias'])*x[...,-9:]
