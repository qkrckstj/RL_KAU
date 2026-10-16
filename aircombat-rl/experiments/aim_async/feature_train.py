"""Enable existing actor features without replacing parameters or Adam state."""
from experiments.aim_async.actor_head import actor_feature_digest


def enable_actor_features(policy):
    features = list(policy.mlp_extractor.policy_net.parameters())
    head = list(policy.action_net.parameters())
    critic = list(policy.mlp_extractor.value_net.parameters()) + list(policy.value_net.parameters())
    if not features or not head or not critic:
        raise ValueError('Expected separate actor features, action head and critic')
    feature_ids = {id(p) for p in features}
    if feature_ids & {id(p) for p in head + critic}:
        raise ValueError('Shared actor and critic/head parameters are unsupported')
    if list(policy.features_extractor.parameters()):
        raise ValueError('A learned shared feature extractor is unsupported')
    if not all(p.requires_grad for p in head + critic):
        raise ValueError('Action head and critic must remain trainable')
    optimized = [p for group in policy.optimizer.param_groups for p in group['params']]
    expected = features + head + critic
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in expected}:
        raise ValueError('Optimizer must retain every existing parameter exactly once')
    for parameter in features:
        parameter.requires_grad_(True)
        parameter.grad = None
    return dict(
        frozen_parameters=0,
        trainable_feature_parameters=sum(p.numel() for p in features),
        trainable_action_parameters=sum(p.numel() for p in head),
        trainable_critic_parameters=sum(p.numel() for p in critic),
        feature_sha256=actor_feature_digest(policy),
        actor_features_trainable=True,
    )
