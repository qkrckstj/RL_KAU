"""Fixed .5-second probe and interceptor, with a separate post-probe parent.

Only the same 39 public channels enter the unchanged direction-invariant gate.
Separating the initial actions permits fitting the parent without discarding
the already learned opening. No opponent labels or simulator internals enter.
"""
import json
import zipfile
"""Direction-invariant early-motion strategy selection; no opponent identity.

Two frozen flight policies remain reactive after selection. A small gate reads
the opponent's observed turn, speed change and closing motion during a learned
probe interval. Both the gate and its interval can be fitted by episodic search.
"""
from pathlib import Path
import json
import math
import zipfile
import numpy as np
"""Reactive controller family for population training, using only raw observations.

The seven-parameter Ace specialist is exactly embedded when the new gates are
disabled. Parameters are optimized from complete games; these are not NN weights.
"""
from pathlib import Path
import json
import math
import zipfile
import numpy as np

NAMES = ('opening_seconds', 'opening_speed', 'tracking_speed', 'lead_seconds',
         'aim_bias', 'yaw_brake', 'deadband', 'opening_direction',
         'opening_stop_range', 'far_range', 'far_speed', 'defense_range',
         'defense_angle', 'defense_offset', 'defense_speed',
         'escape_angle', 'escape_range')
EXPERT_LOW = np.array([0,300,300,-2,-90,0,.5,-1,0,0,300,0,0,-180,300,60,0.], float)
EXPERT_HIGH = np.array([120,650,650,6,90,4,15,1,6000,15000,650,3500,90,180,650,180,15000.], float)


def embed(old):
    return np.r_[old, -1., 0., 0., old[2], 0., 60., 90., old[2], 120., 0.]


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu', parameters=None):
        if parameters is None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:
                    parameters = json.loads(z.read('parameters.json'))['parameters']
            else:
                parameters = json.loads(Path(weights).read_text(encoding='utf-8'))['parameters']
        self.parameters = np.asarray(parameters, float)
        if self.parameters.shape != EXPERT_LOW.shape or not np.isfinite(self.parameters).all():
            raise ValueError('Expected 17 finite parameters')
        if np.any(self.parameters < EXPERT_LOW) or np.any(self.parameters > EXPERT_HIGH):
            raise ValueError('Parameters outside controller family bounds')

    def act(self, obs):
        x = obs
        duration, v_open, v_track, lead, bias, brake, dead, direction, stop_r, far_r, v_far, defense_r, defense_a, defense_offset, v_defense, escape_a, escape_r = self.parameters
        dx, dy = float(x[15]-x[0]), float(x[16]-x[1])
        distance = math.hypot(dx, dy)
        enemy_error = abs(math.degrees(wrap(math.atan2(-dx,-dy)-float(x[26]))))
        opening = 120.-float(x[38]) < duration and distance >= stop_r
        if escape_r > 0 and distance > escape_r and enemy_error > escape_a:
            opening = False
        if opening:
            turn, speed = (0 if direction < 0 else 2), v_open
        else:
            bearing = math.atan2(dx+lead*float(x[18]), dy+lead*float(x[19]))
            error = wrap(bearing-float(x[11])+math.radians(bias))
            demand = math.degrees(error-brake*float(x[14]))
            turn = 2 if demand > dead else 0 if demand < -dead else 1
            speed = v_far if far_r > 0 and distance > far_r else v_track
        # Leave an existing own firing solution alone. Otherwise a learned
        # threat gate may interrupt even the opening, without knowing bot ID.
        if distance < defense_r and enemy_error < defense_a and float(x[34]) < .5:
            desired = float(x[26])+math.radians(defense_offset)
            demand = math.degrees(wrap(desired-float(x[11]))-brake*float(x[14]))
            turn = 2 if demand > dead else 0 if demand < -dead else 1
            speed = v_defense
        own_speed = math.sqrt(sum(float(v)**2 for v in x[3:6]))/.514444
        throttle = 2 if own_speed < speed-5 else 0 if own_speed > speed+5 else 1
        return int(3*turn+throttle)

    def __str__(self):
        return 'League-trained reactive 17-parameter controller'

Expert = Policy


GATE_NAMES=('probe_seconds','bias','mean_turn_weight','speed_change_weight',
            'heading_change_weight','closure_weight')
GATE_LOW=np.array([.5,-8,-8,-8,-8,-8.])
GATE_HIGH=np.array([6.,8,8,8,8,8.])
GATE_INITIAL=np.array([.5,1,-8,0,-8,0.])


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:
                    configuration=json.loads(z.read('parameters.json'))
            else:
                configuration=json.loads(Path(weights).read_text(encoding='utf-8'))
        self.gate=np.asarray(configuration['gate'],float)
        if self.gate.shape!=GATE_LOW.shape or not np.isfinite(self.gate).all():
            raise ValueError('Expected six finite gate parameters')
        if np.any(self.gate<GATE_LOW) or np.any(self.gate>GATE_HIGH):
            raise ValueError('Gate parameters outside bounds')
        if len(configuration['experts'])!=2:
            raise ValueError('This gate selects exactly two frozen experts')
        self.experts=[Expert(parameters=e['parameters']) for e in configuration['experts']]
        self.reset()

    def reset(self):
        self.previous_remaining=None
        self.selected=None
        self.features=None
        self.turn_integral=0.
        self.probe_elapsed=0.

    def act(self,obs):
        x=obs
        remaining=float(x[38])
        # The public contract does not require the grader to call reset().
        # Detect a new episode using its clock, including after an early kill.
        if self.previous_remaining is not None and remaining>self.previous_remaining+1e-3:
            self.reset()
        distance=math.hypot(float(x[15]-x[0]),float(x[16]-x[1]))
        speed=math.sqrt(sum(float(v)**2 for v in x[18:21]))
        heading=float(x[26])
        turn=float(x[29])
        if self.previous_remaining is None:
            self.start_remaining=remaining
            self.start_distance=distance
            self.start_speed=speed
            self.start_heading=heading
            self.previous_turn=turn
        else:
            dt=max(0.,self.previous_remaining-remaining)
            self.turn_integral+=.5*(self.previous_turn+turn)*dt
            self.probe_elapsed+=dt
        self.previous_remaining=remaining
        self.previous_turn=turn
        if self.selected is None and self.probe_elapsed>=self.gate[0]-1e-5:
            elapsed=max(self.probe_elapsed,1e-6)
            delta_heading=math.atan2(math.sin(heading-self.start_heading),math.cos(heading-self.start_heading))
            self.features=np.array([1.,np.clip(abs(10*self.turn_integral/elapsed),0,3),
                np.clip((speed-self.start_speed)/20.,-3,3),
                np.clip(abs(delta_heading)/.35,0,3),np.clip((self.start_distance-distance)/elapsed/150.,-3,3)])
            self.selected=int(float(self.gate[1:]@self.features)>=0.)
        # Frozen expert zero performs the initial probe. No opponent label,
        # task side, simulator state, future action or private bot info is read.
        return self.experts[0 if self.selected is None else self.selected].act(obs)

    def __str__(self):
        return 'Direction-invariant two-expert policy with learned early gate'

Gate = Policy


"""Experimental public-observation interceptor; not yet trained or validated.

Constant-velocity interception is a heading heuristic, not a change to flight
dynamics. Actual turning and acceleration still use the official nine actions.
Keep this separate from all frozen reactive and gate experiments.
"""
import math
import json
import zipfile
import numpy as np

NAMES = ('lead_cap', 'far_speed', 'near_speed', 'near_lead',
         'transition_range', 'yaw_brake', 'deadband')
INTERCEPT_LOW = np.array([6., 450., 300., -2., 1000., 0., .5])
INTERCEPT_HIGH = np.array([90., 650., 650., 6., 10000., 4., 15.])
INTERCEPT_INITIAL = np.array([45., 650., 550., 2., 4000., .3, 3.])


def intercept_time(dx, dy, vx, vy, own_speed, cap):
    """Earliest positive solution to |r + v*t| = own_speed*t, capped.

    If no positive solution exists, return the cap as a bounded look-ahead
    heuristic. This does not claim the target can physically be intercepted.
    Inputs use metres, metres/second and seconds in the horizontal plane.
    """
    values = (dx, dy, vx, vy, own_speed, cap)
    if not all(math.isfinite(float(x)) for x in values) or own_speed < 0 or cap < 0:
        raise ValueError('Expected finite geometry, nonnegative speed and cap')
    c = dx*dx + dy*dy
    if c == 0 or cap == 0:
        return 0.
    a = vx*vx + vy*vy - own_speed*own_speed
    b = 2*(dx*vx + dy*vy)
    if abs(a) <= 1e-10 * max(1., vx*vx + vy*vy, own_speed*own_speed):
        return min(cap, -c/b) if b < 0 else cap
    disc = b*b - 4*a*c
    if disc < 0:
        return cap
    # Stable quadratic roots avoid cancellation for nearly equal speeds.
    q = -.5*(b + math.copysign(math.sqrt(disc), b))
    roots = [q/a]
    if q != 0:
        roots.append(c/q)
    positive = [t for t in roots if t >= 0 and math.isfinite(t)]
    return min(cap, min(positive)) if positive else cap


class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu', parameters=None):
        if parameters is None and weights is not None:
            with zipfile.ZipFile(weights) as archive:
                parameters = json.loads(archive.read('parameters.json'))['parameters']
        self.parameters = np.asarray(INTERCEPT_INITIAL if parameters is None else parameters, float).copy()
        if (self.parameters.shape != INTERCEPT_LOW.shape or not np.isfinite(self.parameters).all()
                or np.any(self.parameters < INTERCEPT_LOW) or np.any(self.parameters > INTERCEPT_HIGH)):
            raise ValueError('Expected seven bounded interception parameters')

    def act(self, obs):
        cap, far_speed, near_speed, near_lead, transition, brake, dead = self.parameters
        dx, dy = float(obs[15]-obs[0]), float(obs[16]-obs[1])
        distance = math.hypot(dx, dy)
        horizontal_speed = math.hypot(float(obs[3]), float(obs[4]))
        t = intercept_time(dx, dy, float(obs[18]), float(obs[19]), horizontal_speed, cap)
        # Smoothly move from near-range lead to long-range interception.
        blend = min(1., max(0., (distance-transition)/transition))
        lead = near_lead + blend*(t-near_lead)
        bearing = math.atan2(dx + lead*float(obs[18]), dy + lead*float(obs[19]))
        error = math.atan2(math.sin(bearing-float(obs[11])), math.cos(bearing-float(obs[11])))
        demand = math.degrees(error-brake*float(obs[14]))
        turn = 2 if demand > dead else 0 if demand < -dead else 1
        target_speed = near_speed + blend*(far_speed-near_speed)
        speed = math.sqrt(sum(float(v)**2 for v in obs[3:6]))/.514444
        throttle = 2 if speed < target_speed-5 else 0 if speed > target_speed+5 else 1
        return int(3*turn+throttle)

    def __str__(self):
        return 'Experimental seven-parameter constant-velocity interceptor'

Interceptor = Policy



class ParentAfterProbe:
    def __init__(self, owner, probe, parent):
        self.owner, self.probe, self.parent = owner, Expert(parameters=probe), Expert(parameters=parent)

    def act(self, obs):
        return (self.probe if self.owner.selected is None else self.parent).act(obs)


class Policy(Gate):
    def __init__(self, weights=None, device='cpu', configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as archive:
                configuration = json.loads(archive.read('parameters.json'))
        super().__init__(configuration=dict(experts=[dict(parameters=configuration['probe'])]*2,
            gate=[.5, 1., -8., 0., -8., 0.]))
        self.experts[0] = ParentAfterProbe(self, configuration['probe'], configuration['parent'])
        self.experts[1] = Interceptor(parameters=configuration['interceptor'])

    def __str__(self):
        return 'Fixed public-motion gate with separated probe and post-probe parent'


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


from io import BytesIO

class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu'):
        if weights is None:
            raise ValueError('A teacher-plus-residual weight archive is required')
        self.weights, self.device = weights, device
        with zipfile.ZipFile(weights) as archive:
            with np.load(BytesIO(archive.read('residual_actor.npz')), allow_pickle=False) as arrays:
                self.parameters = {k: arrays[k].copy() for k in arrays.files}
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
        self.reset()

    def reset(self):
        self.teacher = TeacherPolicy(self.weights, device=self.device)
        self.last_clock = None

    def act(self, obs):
        raw = np.asarray(obs,dtype=np.float32)
        if raw.shape != (39,) or not np.isfinite(raw).all():
            raise ValueError('Expected39 finite public channels')
        clock = float(raw[38])
        if self.last_clock is not None and clock > self.last_clock+1e-6:
            self.reset()
        self.last_clock = clock
        prior = self.teacher.act(raw)
        logits = numpy_logits(features(raw,prior),self.parameters)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite action logits')
        return int(np.argmax(logits))

    def __str__(self):
        return 'Preserved public-state teacher with learned PPO residual actions'
