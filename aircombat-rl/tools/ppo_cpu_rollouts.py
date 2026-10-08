"""Opt-in CPU action collection with GPU PPO optimization.

Use around learn(); ordinary SB3 checkpoint callbacks are supported.
This removes per-action CUDA round trips for CPU simulators. It does not move
JSBSim to the GPU or guarantee a speedup over entirely CPU-based training.
Cross-device arithmetic/RNG means this is not an identical training trajectory.
"""
from contextlib import contextmanager
from functools import wraps

import torch
from stable_baselines3 import PPO


@contextmanager
def cpu_rollouts(model):
    """Retain GPU optimizer state and restore devices/methods even on failure."""
    if not isinstance(model, PPO) or model.use_sde:
        raise ValueError('CPU rollouts support PPO without gSDE only')
    original = model.collect_rollouts
    had_override = 'collect_rollouts' in model.__dict__
    original_excluded = model._excluded_save_params
    had_excluded_override = '_excluded_save_params' in model.__dict__

    def excluded_save_params():
        # Checkpoint callbacks may run while collect_rollouts is on the CPU.
        # Never serialize these temporary closures or their environment references.
        return original_excluded() + ['collect_rollouts', '_excluded_save_params']

    @wraps(original)
    def collect(*args, **kwargs):
        device = model.device
        try:
            model.policy.to('cpu')
            model.device = torch.device('cpu')
            return original(*args, **kwargs)
        finally:
            model.policy.to(device)
            model.device = device

    model.collect_rollouts = collect
    model._excluded_save_params = excluded_save_params
    try:
        yield model
    finally:
        if had_override:
            model.collect_rollouts = original
        else:
            # SB3 serializes instance attributes. Do not leave a bound method
            # (and its model/environment references) in checkpoint data.
            del model.collect_rollouts
        if had_excluded_override:
            model._excluded_save_params = original_excluded
        else:
            del model._excluded_save_params
