"""Untrained temporal opponents using only the official 39-channel observation.

These scripted challengers are an audit panel, not expert flight doctrine.
No imports from the learned controller family or access to opponent identities.
"""
from pathlib import Path
import json
import math
import zipfile


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu', configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as archive:
                configuration = json.loads(archive.read('parameters.json'))
        self.config = dict(configuration)
        if (set(self.config) != {'mode', 'side', 'period', 'speed'}
                or self.config['mode'] not in ('weave', 'extend', 'delayed', 'pulsed')
                or self.config['side'] not in (-1, 1)
                or not 1 <= self.config['period'] <= 20
                or not 300 <= self.config['speed'] <= 650):
            raise ValueError('Invalid temporal opponent configuration')
        self.reset()

    def reset(self):
        self.last_time = None
        self.initial_heading = None
        self.weave_started = None
        self.escape_until = -1.
        self.escape_heading = 0.
        self.cooldown_until = -1.

    def act(self, obs):
        if len(obs) != 39 or not all(math.isfinite(float(x)) for x in obs):
            raise ValueError('Expected 39 finite observation channels')
        t = 120. - float(obs[38])
        # The grader can also call act directly without an explicit reset hook.
        if self.last_time is not None and t < self.last_time - 1e-6:
            self.reset()
        self.last_time = t
        heading = float(obs[11])
        if self.initial_heading is None:
            self.initial_heading = heading
        dx, dy = float(obs[15]-obs[0]), float(obs[16]-obs[1])
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dx, dy)
        enemy_heading = float(obs[26])
        enemy_error = abs(wrap(bearing+math.pi-enemy_heading))
        in_wez = float(obs[34]) > .5
        lead = min(3., distance/1000.)
        desired = math.atan2(dx+lead*float(obs[18]), dy+lead*float(obs[19]))
        speed = self.config['speed']
        period, side, mode = self.config['period'], self.config['side'], self.config['mode']
        if mode == 'weave':
            threatened = distance < 3000. and enemy_error < math.radians(45) and not in_wez
            if threatened:
                if self.weave_started is None:
                    self.weave_started = t
                phase = int((t-self.weave_started)/period)
                desired = enemy_heading+side*((-1)**phase)*math.pi/2
            else:
                self.weave_started = None
        elif mode == 'extend':
            if t >= self.escape_until:
                if distance < 1300. and not in_wez and t >= self.cooldown_until:
                    self.escape_heading = bearing+math.pi+side*math.radians(20)
                    self.escape_until = t+period
                    self.cooldown_until = self.escape_until+8.
            if t < self.escape_until and not in_wez:
                desired, speed = self.escape_heading, 650.
        elif mode == 'delayed':
            if t < period and distance > 1800. and not in_wez:
                desired = self.initial_heading+side*math.radians(70)
        elif distance > 1500. and not in_wez:
            desired += side*math.radians(35)*math.sin(2*math.pi*t/period)
        # Discrete action indices use the official heading/speed ordering.
        demand = math.degrees(wrap(desired-heading)-.5*float(obs[14]))
        turn = 2 if demand > 3. else 0 if demand < -3. else 1
        own_speed = math.sqrt(sum(float(v)**2 for v in obs[3:6]))/.514444
        throttle = 2 if own_speed < speed-5 else 0 if own_speed > speed+5 else 1
        return int(3*turn+throttle)

    def __str__(self):
        return 'Frozen untrained temporal opponent: '+self.config['mode']


TeacherPolicy = Policy

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

"""Public analytic lead cues inserted before the unchanged nine prior channels."""
import math
import numpy as np
from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.engagement import look
from aircombat_gym.wvr.envs.fair import FairFightEnv
relative_features = features

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

"""Public-observation encounter history shared by training and bundled actors."""
import numpy as np

BOUNDARY = np.float32(1500/11500)
RULE = 'public_range1500_first_entry_then_exit_v1'


class EncounterState:
    def __init__(self):
        self.reset()

    def reset(self, after=False):
        if type(after) is not bool:
            raise ValueError('Expected boolean encounter history')
        self.entered = self.after = after

    def observe(self, state):
        x=np.asarray(state,dtype=np.float32)
        if x.shape!=(43,) or not np.isfinite(x).all():
            raise ValueError('Expected43 finite public features')
        self.entered = self.entered or x[0]<=BOUNDARY
        self.after = self.after or (self.entered and x[0]>BOUNDARY)
        return int(self.after)


from io import BytesIO

class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu'):
        if weights is None:
            raise ValueError('A teacher-plus-residual weight archive is required')
        self.weights, self.device = weights, device
        with zipfile.ZipFile(weights) as archive:
            with np.load(BytesIO(archive.read('residual_actor.npz')), allow_pickle=False) as arrays:
                loaded = {k: arrays[k].copy() for k in arrays.files}
                self.parameters = {k:v for k,v in loaded.items() if not k.startswith('opening_')}
                self.opening_parameters = {k[len('opening_'):]:v for k,v in loaded.items() if k.startswith('opening_')}
        p = self.parameters
        if set(p) != {'w0','b0','w1','b1','wa','ba','prior_bias'}:
            raise ValueError('Unexpected residual actor arrays')
        if not all(np.isfinite(v).all() for v in p.values()):
            raise ValueError('Nonfinite actor parameters')
        if (p['w0'].ndim != 2 or p['w0'].shape[1] != FEATURE_DIM
                or p['b0'].shape != (p['w0'].shape[0],)
                or p['w1'].ndim != 2 or p['w1'].shape[1] != p['w0'].shape[0]
                or p['b1'].shape != (p['w1'].shape[0],)
                or p['wa'].shape != (9,p['w1'].shape[0]) or p['ba'].shape != (9,)
                or p['prior_bias'].shape != () or float(p['prior_bias']) <= 0):
            raise ValueError('Incompatible residual actor architecture')
        if set(self.opening_parameters) != set(p) or any(self.opening_parameters[k].shape != p[k].shape or not np.isfinite(self.opening_parameters[k]).all() for k in p):
            raise ValueError('Invalid frozen opening actor arrays')
        self.reset()

    def reset(self):
        self.teacher = TeacherPolicy(self.weights, device=self.device)
        self.last_clock = None
        self.encounter = EncounterState()

    def act(self, obs):
        raw = np.asarray(obs,dtype=np.float32)
        if raw.shape != (39,) or not np.isfinite(raw).all():
            raise ValueError('Expected39 finite public channels')
        clock = float(raw[38])
        if self.last_clock is not None and clock > self.last_clock+1e-6:
            self.reset()
        self.last_clock = clock
        prior = self.teacher.act(raw)
        state = features(raw,prior)
        after = self.encounter.observe(state)
        logits = numpy_logits(state,self.parameters if after else self.opening_parameters)
        if after: logits = numpy_speed_logits(logits,POST_ENCOUNTER_SPEED_TEMPERATURE)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite action logits')
        return int(np.argmax(logits))

    def __str__(self):
        return 'Preserved public-state teacher with learned PPO residual actions'


def validate_temperature(temperature):
    if isinstance(temperature, (bool, np.bool_)):
        raise ValueError('Finite speed temperature >=1 required')
    value=float(temperature)
    if not math.isfinite(value) or value<1:
        raise ValueError('Finite speed temperature >=1 required')
    return value

def numpy_speed_logits(logits, temperature):
    temperature=validate_temperature(temperature)
    x=np.asarray(logits,dtype=np.float32)
    if x.ndim<1 or x.shape[-1]!=9 or not np.isfinite(x).all():
        raise ValueError('Nine finite action logits required')
    if temperature==1:return x
    grid=x.reshape(x.shape[:-1]+(3,3))
    scaled=grid/temperature
    result=scaled+np.logaddexp.reduce(grid,axis=-1,keepdims=True)-np.logaddexp.reduce(scaled,axis=-1,keepdims=True)
    return result.reshape(x.shape)

POST_ENCOUNTER_SPEED_TEMPERATURE = 2.0
