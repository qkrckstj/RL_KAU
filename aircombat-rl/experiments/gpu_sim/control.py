"""Tensor translation of the unchanged 120 Hz aircraft autopilot."""
import math
import torch

from aircombat_gym.core.control.autopilot import AutopilotGains
from aircombat_gym.core.envelope import N_MAX_TABLE, THROTTLE_CAP


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class TensorAutopilot:
    def __init__(self, n, device, dtype):
        self.g = AutopilotGains()
        self.i_nz = torch.zeros(n, device=device, dtype=dtype)
        self.i_v = torch.zeros_like(self.i_nz)
        self.table = torch.tensor(list(N_MAX_TABLE.values()), device=device, dtype=dtype)

    def n_max(self, speed, altitude):
        sx = (speed.clamp(250, 700) - 250) / 50
        sy = (altitude.clamp(5000, 30000) - 5000) / 5000
        ix = sx.long().clamp(max=8)
        iy = sy.long().clamp(max=4)
        tx, ty = sx - ix, sy - iy
        lo = self.table[iy, ix] * (1-tx) + self.table[iy, ix+1] * tx
        hi = self.table[iy+1, ix] * (1-tx) + self.table[iy+1, ix+1] * tx
        return (lo * (1-ty) + hi * ty).clamp(max=9)

    def update(self, raw, heading, speed):
        g = self.g
        h_ft, hdot = raw[:, 2] / .3048, raw[:, 5] / .3048
        v = torch.linalg.vector_norm(raw[:, 3:6], dim=-1) / .514444
        hcmd = (g.k_h * (20000-h_ft)).clamp(-g.hdot_limit, g.hdot_limit)
        climb = (g.k_hd * (hcmd-hdot)).clamp(0, g.n_climb_max)
        available = (self.n_max(v, h_ft)-g.n_reserve).clamp_min(1.05)
        bank = torch.acos(1 / (available-climb).clamp_min(1.02))
        phi = (g.k_psi * wrap(heading-raw[:, 11]) - g.k_psi_r * raw[:, 14]).clamp(-bank, bank)
        aileron = (g.k_phi * wrap(phi-raw[:, 9]) - g.k_p*raw[:, 12]).clamp(-1, 1)
        ff = raw[:, 9].abs().clamp(max=math.radians(g.phi_ff_limit_deg))
        nz = (1 / ff.cos().clamp_min(.10) + g.k_hd * (hcmd-hdot)).clamp(min=-1)
        nz = torch.minimum(nz, available)
        err = nz - raw[:, 8]
        integ = (self.i_nz + err/120).clamp(-g.i_nz_limit, g.i_nz_limit)
        elevator = g.k_nz * err + g.k_nzi * integ
        self.i_nz.copy_(torch.where(elevator.abs() > 1, integ-err/120, integ))
        verr = speed.clamp(300,650) - v
        iv = (self.i_v + verr/120).clamp(-g.i_v_limit, g.i_v_limit)
        throttle = g.thr_bias + g.k_v * verr + g.k_vi * iv
        clipped = throttle.clamp(0, THROTTLE_CAP)
        self.i_v.copy_(torch.where(throttle != clipped, iv-verr/120, iv))
        return torch.stack((aileron, elevator.clamp(-1,1), torch.zeros_like(v), clipped), -1)
