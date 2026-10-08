"""Untrained public-observation challengers, not expert combat doctrine."""
import json
import math
import zipfile


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu', configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as z:
                configuration = json.loads(z.read('parameters.json'))
        self.config = dict(configuration)
        if (set(self.config) != {'mode','side','speed','period'}
            or self.config['mode'] not in ['lead_lag','range_escape','brake_pursuit','threat_reversal']
            or self.config['side'] not in [-1,1]
            or not 300 <= self.config['speed'] <= 650
            or not 2 <= self.config['period'] <= 12):
            raise ValueError('Invalid reactive audit configuration')
        self.reset()

    def reset(self):
        self.last_time = None
        self.escape_until = -1.
        self.escape_heading = 0.

    def act(self, obs):
        if len(obs) != 39 or not all(math.isfinite(float(x)) for x in obs):
            raise ValueError('Expected finite public39 observation')
        t = 120.-float(obs[38])
        if self.last_time is not None and t < self.last_time-1e-6:
            self.reset()
        self.last_time = t
        dx,dy = float(obs[15]-obs[0]),float(obs[16]-obs[1])
        distance = math.hypot(dx,dy)
        bearing = math.atan2(dx,dy)
        heading,enemy_heading = float(obs[11]),float(obs[26])
        in_wez = float(obs[34]) > .5
        enemy_aim = abs(wrap(bearing+math.pi-enemy_heading))
        own_speed = math.sqrt(sum(float(v)**2 for v in obs[3:6]))/.514444
        speed = self.config['speed']
        side,period,mode = self.config['side'],self.config['period'],self.config['mode']
        lead = min(4.,distance/800.)
        desired = math.atan2(dx+lead*float(obs[18]),dy+lead*float(obs[19]))
        if mode == 'lead_lag':
            lead *= math.cos(2*math.pi*t/period)
            desired = math.atan2(dx+lead*float(obs[18]),dy+lead*float(obs[19]))
            if not in_wez:
                desired += side*math.radians(15)*math.sin(2*math.pi*t/period)
        elif mode == 'range_escape':
            if distance < 1700. and not in_wez and t >= self.escape_until:
                self.escape_heading = bearing+math.pi+side*math.radians(45)
                self.escape_until = t+period
            if t < self.escape_until and not in_wez:
                desired,speed = self.escape_heading,650.
        elif mode == 'brake_pursuit':
            if distance < 1800. and abs(wrap(bearing-heading)) < math.radians(80):
                speed = 320.
                lead = .4 if in_wez else -1.5
                desired = math.atan2(dx+lead*float(obs[18]),dy+lead*float(obs[19]))
            elif distance > 3000.:
                speed = 650.
        elif distance < 3500. and enemy_aim < math.radians(55) and not in_wez:
            phase = side*(-1 if int(t/period)%2 else 1)
            desired = enemy_heading+phase*math.radians(105)
            speed = 650. if phase > 0 else speed
        demand = math.degrees(wrap(desired-heading)-.5*float(obs[14]))
        turn = 2 if demand > 3. else 0 if demand < -3. else 1
        throttle = 2 if own_speed < speed-5 else 0 if own_speed > speed+5 else 1
        return int(3*turn+throttle)
