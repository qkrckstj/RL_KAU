"""Public-observation encounter history shared by training and bundled actors."""
import numpy as np

BOUNDARY = np.float32(1500/11500)
RULE = 'public_range1500_first_entry_then_exit_v1'


class EncounterState:
    def __init__(self):
        self.reset()

    def reset(self, after=False):
        if type(after) is not bool:
            raise ValueError('Expected boolean encounter history')
        self.entered = self.after = after

    def observe(self, state):
        x=np.asarray(state,dtype=np.float32)
        if x.shape!=(43,) or not np.isfinite(x).all():
            raise ValueError('Expected43 finite public features')
        self.entered = self.entered or x[0]<=BOUNDARY
        self.after = self.after or (self.entered and x[0]>BOUNDARY)
        return int(self.after)
