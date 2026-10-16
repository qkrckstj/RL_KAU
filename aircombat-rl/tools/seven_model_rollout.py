"""Configure an empty restored PPO rollout buffer without changing learned state."""
from stable_baselines3.common.buffers import RolloutBuffer
from tools.aim_prior_bias import learner_state


def configure_rollout(model, steps, checkpoint_decisions, total_decisions):
    """Accept the qualified 640/2048 profiles and require exact decision budgets."""
    if type(steps) is not int or steps not in (640, 2048):
        raise ValueError('Rollout ticks must be the integer 640 or 2048')
    if any(type(v) is not int or v <= 0 for v in (checkpoint_decisions, total_decisions)):
        raise ValueError('Decision budgets must be positive integers')
    if type(model.n_envs) is not int or model.n_envs <= 0:
        raise ValueError('Invalid environment count')
    samples = steps * model.n_envs
    if (type(model.batch_size) is not int or model.batch_size <= 0
            or samples % model.batch_size
            or checkpoint_decisions % samples or total_decisions % checkpoint_decisions):
        raise ValueError('Rollout, minibatch, checkpoint and total budgets must divide exactly')
    old = model.rollout_buffer
    if (type(old) is not RolloutBuffer or model.rollout_buffer_class is not RolloutBuffer
            or model.rollout_buffer_kwargs
            or old.buffer_size != model.n_steps or old.n_envs != model.n_envs
            or old.gamma != model.gamma or old.gae_lambda != model.gae_lambda
            or old.pos != 0 or old.full):
        raise ValueError('Expected a consistent empty standard restored rollout buffer')
    previous = model.n_steps
    if previous != steps:
        before = learner_state(model)
        replacement = RolloutBuffer(
            steps, model.observation_space, model.action_space, device=model.device,
            gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs)
        if learner_state(model) != before:
            raise ValueError('Buffer allocation changed actor/critic/Adam/counters')
        model.n_steps = steps
        model.rollout_buffer = replacement
    return dict(previous=previous, selected=steps, rollout_samples=samples)
