"""Explicit discount-factor intervention after exact learner restoration."""
import math


def configure_gamma(model,value):
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or not .9<=value<=.9999):
        raise ValueError('Gamma must be finite within [.9, .9999]')
    previous=model.gamma
    if model.rollout_buffer.gamma!=previous:
        raise ValueError('Model and rollout buffer gamma differ')
    model.gamma=float(value)
    model.rollout_buffer.gamma=float(value)
    return dict(previous=previous,selected=float(value))
