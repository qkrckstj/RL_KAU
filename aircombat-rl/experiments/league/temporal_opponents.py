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
