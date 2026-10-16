"""Public analytic lead cues inserted before the unchanged nine prior channels."""
import math
import numpy as np
from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.engagement import look
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.league.residual_features import features as relative_features

FEATURE_DIM = 43


class Kinematics:
    def __init__(self, data, side):
        for key in ('x','y','h','psi','theta','vx','vy','vz'):
            setattr(self,key,float(data[side+'_'+key]))


def engagement(obs):
    raw=np.asarray(obs,dtype=np.float32)
    if raw.shape!=(O.STATE_DIM,) or not np.isfinite(raw).all():
        raise ValueError('Expected39 finite public observation channels')
    data=O.unpack(raw)
    return look(Kinematics(data,'own'),Kinematics(data,'opp'),FairFightEnv.wez_cone_deg)


def features(obs, prior_action):
    base=relative_features(obs,prior_action);e=engagement(obs)
    psi=float(O.unpack(obs)['own_psi'])
    signed=(math.atan2(e.aim_dx,e.aim_dy)-psi+math.pi)%(2*math.pi)-math.pi
    aim=np.asarray((math.sin(signed),math.cos(signed),math.cos(e.ata_lead)),dtype=np.float32)
    return np.concatenate((base[:31],aim,base[-9:]))


def numpy_logits(observations, parameters):
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1:]!=(FEATURE_DIM,) or not np.isfinite(x).all():
        raise ValueError('Expected43 finite features')
    # Keep the original dot product intact when the three new weights are zero.
    old=np.ascontiguousarray(np.concatenate((x[...,:31],x[...,34:]),axis=-1))
    weight=np.ascontiguousarray(np.concatenate((parameters['w0'][:,:31],parameters['w0'][:,34:]),axis=1))
    first=old@weight.T+parameters['b0']
    first=first+x[...,31:34]@parameters['w0'][:,31:34].T
    hidden=np.maximum(0.,first)
    hidden=np.maximum(0.,hidden@parameters['w1'].T+parameters['b1'])
    return hidden@parameters['wa'].T+parameters['ba']+float(parameters['prior_bias'])*x[...,-9:]
