"""Independent feedback and bank-command slew interventions."""
from dataclasses import replace
import math
from experiments.controller_probe.learnability import TurnEffortAutopilot, heading_rate
from aircombat_gym.core.control.autopilot import wrap_pi


class EffortController(TurnEffortAutopilot):
    def __init__(self,feedback=0.,slew_deg=None,**kwargs):
        if not math.isfinite(feedback) or feedback<0.:
            raise ValueError('Feedback must be finite and nonnegative')
        if slew_deg is not None and (not math.isfinite(slew_deg) or slew_deg<=0.):
            raise ValueError('Slew limit must be positive and finite')
        super().__init__(**kwargs)
        self.feedback=feedback
        self.slew_deg=slew_deg
        self.filtered_phi=0.

    def reset(self):
        super().reset()
        self.filtered_phi=0.

    def update(self,s,psi_cmd,v_cmd_kt,dt,alt_cmd_ft=None):
        if not math.isfinite(dt) or dt<=0.:raise ValueError('Positive finite dt required')
        out=super().update(s,psi_cmd,v_cmd_kt,dt,alt_cmd_ft)
        desired=out.phi_cmd
        if self.feedback:
            maximum=9.80665*math.tan(out.max_bank)/max(s.v,1.)
            requested=self.effort*maximum
            corrected=requested+self.feedback*(requested-heading_rate(s))
            desired=math.atan(corrected*max(s.v,1.)/9.80665)
        if self.slew_deg is not None:
            step=math.radians(self.slew_deg)*dt
            desired=max(self.filtered_phi-step,min(self.filtered_phi+step,desired))
        desired=max(-out.max_bank,min(out.max_bank,desired))
        self.filtered_phi=desired
        aileron=max(-1.,min(1.,self.g.k_phi*wrap_pi(desired-s.phi)-self.g.k_p*s.p))
        return replace(out,phi_cmd=desired,aileron=aileron)
