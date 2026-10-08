"""Expand first-layer inputs while retaining the pretrained policy/Adam state."""
from copy import deepcopy
import math
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from experiments.league.residual_ppo_policy import ResidualPriorPolicy
from experiments.league.history_features import PUBLIC_DIM,HISTORY_DIM


class HistoryPriorPolicy(ResidualPriorPolicy):
    def __init__(self,observation_space,action_space,lr_schedule,prior_bias=6.,**kwargs):
        if observation_space.shape!=(HISTORY_DIM,) or not isinstance(action_space,spaces.Discrete) or action_space.n!=9:
            raise ValueError('Expected history features and9 actions')
        if not math.isfinite(prior_bias) or prior_bias<=0:raise ValueError('Positive finite prior required')
        if kwargs.get('share_features_extractor',True) is not True:raise ValueError('Shared flat features required')
        kwargs.setdefault('net_arch',dict(pi=[64,64],vf=[64,64]));kwargs.setdefault('activation_fn',torch.nn.ReLU)
        ActorCriticPolicy.__init__(self,observation_space,action_space,lr_schedule,**kwargs)
        self.prior_bias=float(prior_bias)
        torch.nn.init.zeros_(self.action_net.weight);torch.nn.init.zeros_(self.action_net.bias)


def expand_input(tensor):
    if tensor.ndim!=2 or tensor.shape[1]!=PUBLIC_DIM+9:raise ValueError('Expected original first-layer matrix')
    answer=tensor.new_zeros((tensor.shape[0],HISTORY_DIM))
    answer[:,:PUBLIC_DIM]=tensor[:,:PUBLIC_DIM]
    answer[:,-9:]=tensor[:,-9:]
    return answer


def clone_with_history(original,expanded):
    """No new actor preference: added columns and their Adam moments start zero."""
    first={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    old=dict(original.named_parameters());new=dict(expanded.named_parameters())
    if list(old)!=list(new):raise ValueError('Changed parameter order')
    state=original.state_dict()
    expanded.load_state_dict({k:expand_input(v) if k in first else v.clone() for k,v in state.items()},strict=True)
    optimizer=deepcopy(original.optimizer.state_dict())
    indexes=[i for group in optimizer['param_groups'] for i in group['params']]
    if len(indexes)!=len(old):raise ValueError('Unexpected optimizer parameter layout')
    for name,index in zip(old,indexes,strict=True):
        for key,value in optimizer['state'].get(index,{}).items():
            if name in first and isinstance(value,torch.Tensor) and value.ndim>0:
                optimizer['state'][index][key]=expand_input(value)
    expanded.optimizer.load_state_dict(optimizer)
    expanded.prior_bias=original.prior_bias
    return expanded
