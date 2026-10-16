"""Heading/roll experiments with unchanged lift, altitude and throttle loops."""
import math
from dataclasses import replace

from aircombat_gym.core.control.autopilot import Autopilot, wrap_pi

VARIANTS = ('original', 'anticipation', 'roll_response', 'speed_schedule')


class ProbeAutopilot(Autopilot):
    def __init__(self, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError('Unknown controller variant')
        super().__init__(**kwargs)
        self.variant = variant

    def update(self, s, psi_cmd, v_cmd_kt, dt, alt_cmd_ft=None):
        out = super().update(s, psi_cmd, v_cmd_kt, dt, alt_cmd_ft)
        if self.variant == 'original':
            return out
        phi_cmd = out.phi_cmd
        k_phi, k_p = self.g.k_phi, self.g.k_p
        if self.variant == 'anticipation':
            error = wrap_pi(psi_cmd - s.psi)
            cosine = math.cos(s.theta)
            rate = (s.q * math.sin(s.phi) + s.r * math.cos(s.phi)) / math.copysign(
                max(abs(cosine), 1e-6), cosine)
            # Smoothly activate only near the target. Sustained +/-30-degree
            # policy commands retain the original heading law and bank limit.
            near = max(0., 1. - abs(error) / math.radians(20.))
            horizon = .15 + .65 * min(1., abs(s.phi) / out.max_bank)
            effective = error - near * horizon * rate
            phi_cmd = max(-out.max_bank, min(out.max_bank,
                self.g.k_psi * effective - self.g.k_psi_r * s.r))
        elif self.variant == 'roll_response':
            k_phi = 4.5
        elif self.variant == 'speed_schedule':
            # Dynamic pressure changes roll authority. This is a hypothesis,
            # not a calibrated airframe model: validate in actual JSBSim.
            scale = max(.7, min(1.5, 450. / max(s.v_kt, 1.)))
            k_phi *= scale
            k_p *= scale
        aileron = max(-1., min(1., k_phi * wrap_pi(phi_cmd - s.phi) - k_p * s.p))
        return replace(out, aileron=aileron, phi_cmd=phi_cmd)


def make_controller(variant, **kwargs):
    if variant == 'original':
        return Autopilot(**kwargs)
    return ProbeAutopilot(variant, **kwargs)
