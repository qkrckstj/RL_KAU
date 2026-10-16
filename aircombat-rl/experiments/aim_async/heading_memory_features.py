"""One maintained heading-command cue derived from public observations/actions."""
import math
import operator
import numpy as np
from experiments.aim_async.features import features as aim_features
from experiments.aim_async.rate_input_features import augment, original_columns, numpy_logits

FEATURE_DIM=44


def memory_wrap(value):
    return (value+math.pi)%(2*math.pi)-math.pi


def public_heading(raw):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.shape!=(39,) or not np.isfinite(raw).all():
        raise ValueError('Expected39finite public observations')
    return float(raw[11])


class HeadingCommandMemory:
    def __init__(self):
        self.command=None

    def reset(self,raw):
        self.command=memory_wrap(public_heading(raw))

    def error(self,raw):
        heading=public_heading(raw)
        if self.command is None:self.command=memory_wrap(heading)
        return memory_wrap(self.command-heading)

    def update(self,raw,action):
        heading=public_heading(raw);action=operator.index(action)
        if not 0<=action<9:raise ValueError('Expected9action index')
        if self.command is None:self.command=memory_wrap(heading)
        delta=(-30.,0.,30.)[action//3]
        if delta:self.command=memory_wrap(heading+math.radians(delta))


def features(raw,prior_action,heading_error):
    return augment(aim_features(raw,prior_action),heading_error)
