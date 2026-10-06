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
         'defense_angle', 'defense_offset', 'defense_speed')
LOW = np.array([0,300,300,-2,-90,0,.5,-1,0,0,300,0,0,-180,300.], float)
HIGH = np.array([120,650,650,6,90,4,15,1,6000,15000,650,3500,90,180,650.], float)


def embed(old):
    return np.r_[old, -1., 0., 0., old[2], 0., 60., 90., old[2]]


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
        if self.parameters.shape != LOW.shape or not np.isfinite(self.parameters).all():
            raise ValueError('Expected 15 finite parameters')
        if np.any(self.parameters < LOW) or np.any(self.parameters > HIGH):
            raise ValueError('Parameters outside controller family bounds')

    def act(self, obs):
        x = obs
        duration, v_open, v_track, lead, bias, brake, dead, direction, stop_r, far_r, v_far, defense_r, defense_a, defense_offset, v_defense = self.parameters
        dx, dy = float(x[15]-x[0]), float(x[16]-x[1])
        distance = math.hypot(dx, dy)
        opening = 120.-float(x[38]) < duration and distance >= stop_r
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
        enemy_error = abs(math.degrees(wrap(math.atan2(-dx,-dy)-float(x[26]))))
        if distance < defense_r and enemy_error < defense_a and float(x[34]) < .5:
            desired = float(x[26])+math.radians(defense_offset)
            demand = math.degrees(wrap(desired-float(x[11]))-brake*float(x[14]))
            turn = 2 if demand > dead else 0 if demand < -dead else 1
            speed = v_defense
        own_speed = math.sqrt(sum(float(v)**2 for v in x[3:6]))/.514444
        throttle = 2 if own_speed < speed-5 else 0 if own_speed > speed+5 else 1
        return int(3*turn+throttle)

    def __str__(self):
        return 'League-trained reactive 15-parameter controller'
