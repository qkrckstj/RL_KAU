"""Freeze pretrained actor features without replacing parameters or Adam state."""
import hashlib
import json
import zipfile
from io import BytesIO
import torch


def _digest(state):
    if not state:
        raise ValueError('Actor features are missing')
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        result.update(len(header).to_bytes(8, 'little'))
        result.update(header)
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def actor_feature_digest(policy):
    return _digest(policy.mlp_extractor.policy_net.state_dict())


def archive_actor_feature_digest(archive):
    with zipfile.ZipFile(archive) as source:
        state = torch.load(BytesIO(source.read('policy.pth')), map_location='cpu', weights_only=True)
    prefix = 'mlp_extractor.policy_net.'
    return _digest({name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)})


def freeze_actor_features(policy):
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
    # grad=None also prevents initialized Adam momentum from moving these weights.
    for parameter in features:
        parameter.requires_grad_(False)
        parameter.grad = None
    return dict(
        frozen_parameters=sum(p.numel() for p in features),
        trainable_action_parameters=sum(p.numel() for p in head),
        trainable_critic_parameters=sum(p.numel() for p in critic),
        feature_sha256=actor_feature_digest(policy),
        actor_features_trainable=False,
    )
