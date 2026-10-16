"""Change a fixed additive action prior without resetting PPO state."""
import math

from tools.prepare_rear_cpu_entropy import tensor_digest


def learner_state(model):
    return dict(policy=tensor_digest(model.policy.state_dict()),
                optimizer=tensor_digest(model.policy.optimizer.state_dict()),
                decisions=model.num_timesteps, updates=model._n_updates)


def change_prior(model, target, expected_source=6.):
    target = float(target)
    if not math.isfinite(target) or target <= 0:
        raise ValueError('Target prior must be positive finite')
    if (model.policy.prior_bias != expected_source
            or model.policy_kwargs.get('prior_bias', expected_source) != expected_source):
        raise ValueError('Unexpected source prior')
    if model.observation_space.shape != (43,) or model.action_space.n != 9:
        raise ValueError('Expected43features and9actions')
    before = learner_state(model)
    model.policy.prior_bias = target
    # PPO.save serializes policy_kwargs separately from the state_dict.
    # Changing only the runtime attribute would silently reload the old bias.
    model.policy_kwargs = dict(model.policy_kwargs, prior_bias=target)
    after = learner_state(model)
    if before != after:
        raise ValueError('Actor/critic/Adam/counters changed with the fixed prior')
    return dict(before=before, after=after, source_prior=expected_source,
                target_prior=target, state_exact=True)
