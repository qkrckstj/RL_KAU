"""Preserve the original actor/critic/Adam state while adding analytic lead cues."""
import copy
import math
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from experiments.league.residual_ppo_policy import ResidualPriorPolicy
from experiments.aim_async.features import FEATURE_DIM


def original_columns():
    return list(range(31))+list(range(34,43))


def expand_matrix(value):
    result=value.new_zeros((value.shape[0],43))
    result[:,original_columns()]=value
    return result


class AimLinear(torch.nn.Linear):
    def forward(self,x):
        old=torch.cat((x[...,:31],x[...,34:]),dim=-1)
        weight=torch.cat((self.weight[:,:31],self.weight[:,34:]),dim=1)
        base=torch.nn.functional.linear(old,weight,self.bias)
        return base+torch.nn.functional.linear(x[...,31:34],self.weight[:,31:34])


class AimPriorPolicy(ResidualPriorPolicy):
    def __init__(self,observation_space,action_space,lr_schedule,prior_bias=6.,**kwargs):
        if observation_space.shape!=(FEATURE_DIM,) or not isinstance(action_space,spaces.Discrete) or action_space.n!=9:
            raise ValueError('Expected43 features and9 actions')
        if not math.isfinite(prior_bias) or prior_bias<=0:
            raise ValueError('Expected finite positive prior bias')
        if kwargs.get('share_features_extractor',True) is not True:
            raise ValueError('Shared flat features are required')
        kwargs.setdefault('net_arch',dict(pi=[64,64],vf=[64,64]))
        kwargs.setdefault('activation_fn',torch.nn.ReLU)
        ActorCriticPolicy.__init__(self,observation_space,action_space,lr_schedule,**kwargs)
        if not isinstance(self.features_extractor,FlattenExtractor):
            raise ValueError('Flat features required')
        self.prior_bias=float(prior_bias)
        # Replace layers before rebuilding Adam so its parameter references are current.
        for net in (self.mlp_extractor.policy_net,self.mlp_extractor.value_net):
            layer=net[0]
            replacement=AimLinear(FEATURE_DIM,layer.out_features,bias=layer.bias is not None).to(layer.weight.device)
            replacement.load_state_dict(layer.state_dict())
            net[0]=replacement
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


def expand_policy(source,lr_schedule):
    kwargs=source._get_constructor_parameters()
    kwargs.update(observation_space=spaces.Box(-1.,1.,(FEATURE_DIM,),dtype=np.float32),lr_schedule=lr_schedule)
    policy=AimPriorPolicy(**kwargs).to(source.device)
    state=source.state_dict()
    first={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    for name in first:
        if state[name].shape[1]!=40:raise ValueError('Expected original40-channel first layers')
    expanded={name:expand_matrix(value) if name in first else value.clone() for name,value in state.items()}
    policy.load_state_dict(expanded)
    old_named=list(source.named_parameters());new_named=list(policy.named_parameters())
    if [n for n,_ in old_named]!=[n for n,_ in new_named]:raise ValueError('Parameter names/order changed')
    optimizer=copy.deepcopy(source.optimizer.state_dict())
    ids=[i for group in optimizer['param_groups'] for i in group['params']]
    if len(ids)!=len(old_named):raise ValueError('Unsupported optimizer groups')
    for index,(name,_) in zip(ids,old_named):
        if name in first:
            for key,value in optimizer['state'].get(index,{}).items():
                if isinstance(value,torch.Tensor) and value.ndim==2:
                    optimizer['state'][index][key]=expand_matrix(value)
    policy.optimizer.load_state_dict(optimizer)
    return policy


def expand_learner(source):
    from stable_baselines3.common.buffers import RolloutBuffer
    model=copy.deepcopy(source)
    model.policy=expand_policy(source.policy,source.lr_schedule)
    model.policy_class=AimPriorPolicy
    model.observation_space=model.policy.observation_space
    model.rollout_buffer=RolloutBuffer(model.n_steps,model.observation_space,model.action_space,
        device=model.device,gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=model.n_envs)
    model._last_obs=None
    model._last_original_obs=None
    return model
