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
LOW = np.array([6., 450., 300., -2., 1000., 0., .5])
HIGH = np.array([90., 650., 650., 6., 10000., 4., 15.])
INITIAL = np.array([45., 650., 550., 2., 4000., .3, 3.])


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
        self.parameters = np.asarray(INITIAL if parameters is None else parameters, float).copy()
        if (self.parameters.shape != LOW.shape or not np.isfinite(self.parameters).all()
                or np.any(self.parameters < LOW) or np.any(self.parameters > HIGH)):
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
