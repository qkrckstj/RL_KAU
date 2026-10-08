"""Public relative state plus a preserved controller's current action.

Pure NumPy; no training-library import is required in simulation workers.
The first31 channels match the existing relative_v1 representation.
"""
import math
import operator
import numpy as np
from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.envs.fair import FairFightEnv

FEATURE_DIM = 40


def features(obs, prior_action):
    raw = np.asarray(obs, dtype=np.float32)
    if raw.shape != (O.STATE_DIM,) or not np.isfinite(raw).all():
        raise ValueError('Expected39 finite public observation channels')
    action = operator.index(prior_action)
    if not 0 <= action < 9:
        raise ValueError('Expected a discrete action in0..8')
    d = O.unpack(raw)
    dx, dy, dz = (d[f'opp_{k}']-d[f'own_{k}'] for k in ('x', 'y', 'h'))
    distance = math.sqrt(dx*dx+dy*dy+dz*dz)+1e-9
    horizontal = math.hypot(dx, dy)
    nx, ny = (dx/horizontal, dy/horizontal) if horizontal > 1e-6 else (0., 0.)
    sy, cy = math.sin(d['own_psi']), math.cos(d['own_psi'])
    dvx, dvy = d['opp_vx']-d['own_vx'], d['opp_vy']-d['own_vy']
    difference = d['opp_psi']-d['own_psi']
    speed = lambda side: math.sqrt(sum(d[f'{side}_{axis}']**2 for axis in ('vx', 'vy', 'vz')))
    values = [distance/(distance+10000.), (dx*sy+dy*cy)/(distance+10000.),
        (dx*cy-dy*sy)/(distance+10000.), nx*cy-ny*sy, nx*sy+ny*cy,
        math.sin(difference), math.cos(difference), math.tanh((dvx*nx+dvy*ny)/400.),
        math.tanh((dvx*ny-dvy*nx)/400.), math.tanh(speed('own')/300.),
        math.tanh(speed('opp')/300.), math.sin(d['own_phi']), math.cos(d['own_phi']),
        math.sin(d['opp_phi']), math.cos(d['opp_phi']), math.tanh(d['own_r']/.5),
        math.tanh(d['opp_r']/.5), math.tanh(d['own_nz']/9.), math.tanh(d['opp_nz']/9.),
        d['own_health'], d['opp_health'], min(d['own_track_time']/FairFightEnv.track_lock, 1.),
        min(d['opp_track_time']/FairFightEnv.track_lock, 1.), d['own_in_wez'], d['opp_in_wez'],
        math.tanh(d['own_dist_to_boundary']/50000.), math.tanh(d['opp_dist_to_boundary']/50000.),
        d['t_remaining']/FairFightEnv.t_max, math.tanh(dz/1000.),
        math.tanh(d['own_vz']/100.), math.tanh(d['opp_vz']/100.)]
    prior = np.zeros(9, dtype=np.float32); prior[action] = 1.
    return np.concatenate((np.clip(np.asarray(values, dtype=np.float32), -1., 1.), prior))


def numpy_logits(observations, parameters):
    """Exported actor inference: float32 ReLU MLP plus fixed prior logits."""
    x = np.asarray(observations, dtype=np.float32)
    if x.shape[-1:] != (FEATURE_DIM,) or not np.isfinite(x).all():
        raise ValueError('Expected finite40-channel features')
    hidden = np.maximum(0., x@parameters['w0'].T+parameters['b0'])
    hidden = np.maximum(0., hidden@parameters['w1'].T+parameters['b1'])
    return hidden@parameters['wa'].T+parameters['ba']+float(parameters['prior_bias'])*x[..., -9:]
