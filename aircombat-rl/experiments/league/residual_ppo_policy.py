"""PPO actor with a fixed logit prior from a preserved public-state controller.

Prototype only: no league training driver imports this yet. Zero residual
weights reproduce the prior's greedy action, while all nine actions retain
positive sampling probability and can become preferred through learning.
"""
import math
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor

from experiments.league.residual_features import FEATURE_DIM


class ResidualPriorPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, prior_bias=6., **kwargs):
        if observation_space.shape != (FEATURE_DIM,) or not isinstance(action_space, spaces.Discrete) or action_space.n != 9:
            raise ValueError('Expected40 features and9 discrete actions')
        if not math.isfinite(prior_bias) or prior_bias <= 0:
            raise ValueError('A finite positive prior logit bias is required')
        if kwargs.get('share_features_extractor', True) is not True:
            raise ValueError('Shared flat features are required')
        kwargs.setdefault('net_arch', dict(pi=[64, 64], vf=[64, 64]))
        kwargs.setdefault('activation_fn', torch.nn.ReLU)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if not isinstance(self.features_extractor, FlattenExtractor):
            raise ValueError('Flat public features are required')
        self.prior_bias = float(prior_bias)
        # The baseline is an action prior, not a target for value estimates.
        torch.nn.init.zeros_(self.action_net.weight)
        torch.nn.init.zeros_(self.action_net.bias)

    def _distribution(self, latent_pi, obs):
        logits = self.action_net(latent_pi)+self.prior_bias*obs[..., -9:]
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(latent_vf), log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(latent_pi, obs)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        latent_pi = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self._distribution(latent_pi, obs)

    def _get_constructor_parameters(self):
        return dict(**super()._get_constructor_parameters(), prior_bias=self.prior_bias)


def actor_parameters(policy):
    """Extract a standalone NumPy actor; reject an unsupported architecture."""
    layers = list(policy.mlp_extractor.policy_net)
    if (len(layers) != 4 or not isinstance(layers[0], torch.nn.Linear)
            or not isinstance(layers[1], torch.nn.ReLU)
            or not isinstance(layers[2], torch.nn.Linear)
            or not isinstance(layers[3], torch.nn.ReLU)):
        raise ValueError('Expected two Linear/ReLU actor layers')
    result = {'prior_bias': np.asarray(policy.prior_bias, dtype=np.float32)}
    for suffix, layer in (('0', layers[0]), ('1', layers[2]), ('a', policy.action_net)):
        result['w'+suffix] = layer.weight.detach().cpu().numpy().copy()
        result['b'+suffix] = layer.bias.detach().cpu().numpy().copy()
    return result
