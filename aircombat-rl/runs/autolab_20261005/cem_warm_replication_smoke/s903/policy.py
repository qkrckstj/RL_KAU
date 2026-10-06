"""Low-dimensional policy for direct episodic search, using public observations only.

This is a restricted controller family, not a trained neural policy. Neither
physics nor the opponent is changed. Parameters are learned by match returns.
"""
from pathlib import Path
import json
import math
import numpy as np

PARAMETERS = ("opening_seconds", "opening_speed_kt", "tracking_speed_kt",
              "lead_seconds", "aim_bias_deg", "yaw_brake_seconds", "deadband_deg")
LOW = np.array([0., 300., 300., -2., -45., 0., .5])
HIGH = np.array([120., 600., 600., 6., 45., 4., 12.])
INITIAL = np.array([40., 300., 350., .5, 0., 1., 3.75])
ANCHOR = np.array([120., 300., 300., 0., 0., 0., 3.75])


def normalise(values):
    return (np.asarray(values, dtype=float) - LOW) / (HIGH - LOW)


class Policy:
    ACTION_MODE = "discrete"

    def __init__(self, weights=None, device="cpu", parameters=None):
        if parameters is None:
            parameters = json.loads(Path(weights).read_text(encoding="utf-8"))["parameters"]
        self.parameters = np.asarray(parameters, dtype=float)
        if self.parameters.shape != (7,) or not np.isfinite(self.parameters).all():
            raise ValueError("Expected seven finite controller parameters")
        if np.any(self.parameters < LOW) or np.any(self.parameters > HIGH):
            raise ValueError("Controller parameters outside frozen search bounds")

    def act(self, obs):
        # StateSpec v2: own 15 channels, opponent 15, match 9. Stateless across
        # episodes: the public remaining-time channel determines opening phase.
        x = np.asarray(obs)
        duration, opening_speed, tracking_speed, lead, bias, brake, dead = self.parameters
        opening = 120. - float(x[38]) < duration
        if opening:
            turn = 0
            target_speed = opening_speed
        else:
            dx = float(x[15] - x[0]) + lead * float(x[18])
            dy = float(x[16] - x[1]) + lead * float(x[19])
            yaw_error = math.atan2(dx, dy) - float(x[11]) + math.radians(bias)
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            # Body r is only an approximate heading-rate signal while banked.
            # Its contribution is optimized, never assumed to be an exact rate.
            demand = math.degrees(yaw_error - brake * float(x[14]))
            turn = 2 if demand > dead else 0 if demand < -dead else 1
            target_speed = tracking_speed
        speed = math.sqrt(sum(float(v) ** 2 for v in x[3:6])) / .514444
        throttle = 2 if speed < target_speed - 5. else 0 if speed > target_speed + 5. else 1
        return 3 * turn + throttle

    def __str__(self):
        return "CEM-trained 7-parameter opening/lead/speed controller"
