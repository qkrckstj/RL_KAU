"""One public-observation controller derived from manual003/004/005.

An automated state machine, not interactive piloting or neural learning.
Elapsed time is used only to detect episode reset, never to schedule maneuvers.
"""
import json
import math
from pathlib import Path
import zipfile
import numpy as np

DEFAULTS = dict(wide_bearing_deg=-103., align_bearing_deg=-27.5,
    merge_range=240., brake_speed=482., turn_angle_deg=34.5,
    attack_lead_deg=12., attack_brake_health_gap=.05,
    egress_range=1100., egress_foe_bearing_deg=90., egress_bearing_deg=120.)
BOUNDS = dict(wide_bearing_deg=(-179., -1.), align_bearing_deg=(-90., 0.),
    merge_range=(150., 1500.), brake_speed=(300., 650.), turn_angle_deg=(1., 150.),
    attack_lead_deg=(0., 30.), attack_brake_health_gap=(0., 1.),
    egress_range=(150., 10000.), egress_foe_bearing_deg=(90., 180.), egress_bearing_deg=(90., 180.))


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def gun_heading(x):
    dx, dy, dz = (float(x[15+k])-float(x[k]) for k in range(3))
    tof = math.sqrt(dx*dx+dy*dy+dz*dz)/1050.
    return math.atan2(dx+(float(x[18])-float(x[3]))*tof,
                      dy+(float(x[19])-float(x[4]))*tof)


class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu', parameters=None):
        if parameters is None and weights is not None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:
                    parameters = json.loads(z.read('parameters.json'))['parameters']
            else:
                parameters = json.loads(Path(weights).read_text())['parameters']
        parameters = {} if parameters is None else parameters
        if not isinstance(parameters, dict) or set(parameters)-set(DEFAULTS):
            raise ValueError('Unknown controller parameters')
        self.parameters = dict(DEFAULTS, **parameters)
        for k, v in self.parameters.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not BOUNDS[k][0] <= v <= BOUNDS[k][1]:
                raise ValueError('Invalid controller parameter: '+k)
        if self.parameters['wide_bearing_deg'] >= self.parameters['align_bearing_deg']:
            raise ValueError('Wide turn must precede turn-in alignment')
        self.reset()

    def reset(self):
        self.phase = 'wide'
        self.turn_progress = 0.
        self.attack_throttle = 2
        self.last_observation = self.last_action = self.last_remaining = self.last_psi = None
        self.command_heading = None

    def act(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (39,) or not np.isfinite(x).all():
            raise ValueError('Expected39finite public observations')
        remaining, psi, foe_psi = float(x[38]), float(x[11]), float(x[26])
        if self.last_remaining is not None and remaining > self.last_remaining+1e-6:
            self.reset()
        if self.last_observation is not None and np.array_equal(x, self.last_observation):
            return self.last_action
        dx, dy = float(x[15])-float(x[0]), float(x[16])-float(x[1])
        distance = math.hypot(dx, dy)
        bearing = math.degrees(wrap(math.atan2(dx, dy)-psi))
        speed = math.sqrt(sum(float(v)**2 for v in x[3:6]))/.514444
        unilateral = x[34] > .5 and x[35] <= .5
        p = self.parameters
        if self.phase == 'wide':
            if p['wide_bearing_deg'] <= bearing < 0.:
                self.phase = 'turn_in'
        elif self.phase == 'turn_in':
            if bearing >= p['align_bearing_deg']:
                self.phase = 'approach'
        elif self.phase == 'approach':
            if distance <= p['merge_range']:
                self.phase = 'right_brake'
        elif self.phase == 'right_brake':
            if speed <= p['brake_speed']:
                self.phase = 'right_hold'
                self.turn_progress = 0.
        elif self.phase == 'right_hold':
            self.turn_progress += wrap(psi-self.last_psi)
            if math.degrees(self.turn_progress) >= p['turn_angle_deg']:
                self.phase = 'right_intercept'
        elif self.phase == 'right_intercept':
            if unilateral and math.degrees(wrap(gun_heading(x)-psi)) <= p['attack_lead_deg']:
                self.phase = 'attack'
                self.attack_throttle = 0 if float(x[31])-float(x[30]) > p['attack_brake_health_gap'] else 2
        elif self.phase == 'attack':
            if x[34] <= .5:
                self.phase = 'egress_turn'
        elif self.phase == 'egress_turn':
            has_lead = round(float(x[30]), 4) > round(float(x[31]), 4)
            enemy_away = abs(math.degrees(wrap(math.atan2(-dx, -dy)-foe_psi))) >= p['egress_foe_bearing_deg']
            opening = dx*(float(x[18])-float(x[3]))+dy*(float(x[19])-float(x[4])) > 0.
            if has_lead and distance >= p['egress_range'] and enemy_away and opening and x[35] <= .5 and abs(bearing) >= p['egress_bearing_deg']:
                self.phase = 'egress_hold'
            elif not has_lead and distance > 1500.:
                self.phase = 'right_intercept'
        action = dict(wide=1, turn_in=2, approach=5, right_brake=6,
            right_hold=7, right_intercept=8, attack=self.attack_throttle,
            egress_turn=8, egress_hold=5)[self.phase]
        if self.command_heading is None:
            self.command_heading = psi
        turn = action//3
        if turn != 1:
            self.command_heading = wrap(psi+math.radians(30*(turn-1)))
        self.last_observation = x.copy()
        self.last_remaining, self.last_psi, self.last_action = remaining, psi, int(action)
        return self.last_action
