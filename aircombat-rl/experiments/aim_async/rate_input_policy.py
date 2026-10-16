"""Add one trainable public cue without resetting original PPO parameters/Adam."""
import copy
import math
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from experiments.league.residual_ppo_policy import ResidualPriorPolicy
from experiments.aim_async.policy import AimPriorPolicy
from experiments.aim_async.rate_input_features import FEATURE_DIM,original_columns
from tools.prepare_rear_cpu_entropy import tensor_digest

FIRST={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}


class RateInputLinear(torch.nn.Linear):
    def forward(self,x):
        old=torch.cat((x[...,:31],x[...,35:]),dim=-1)
        weight=torch.cat((self.weight[:,:31],self.weight[:,35:]),dim=1)
        base=torch.nn.functional.linear(old,weight,self.bias)
        base=base+torch.nn.functional.linear(x[...,31:34],self.weight[:,31:34])
        return base+torch.nn.functional.linear(x[...,34:35],self.weight[:,34:35])


class RateInputPriorPolicy(ResidualPriorPolicy):
    def __init__(self,observation_space,action_space,lr_schedule,prior_bias=6.,**kwargs):
        if observation_space.shape!=(FEATURE_DIM,) or not isinstance(action_space,spaces.Discrete) or action_space.n!=9:
            raise ValueError('Expected44features and9actions')
        if not math.isfinite(prior_bias) or prior_bias<=0:raise ValueError('Expected positive finite prior')
        if kwargs.get('share_features_extractor',True) is not True:raise ValueError('Shared flat features required')
        kwargs.setdefault('net_arch',dict(pi=[64,64],vf=[64,64]));kwargs.setdefault('activation_fn',torch.nn.ReLU)
        ActorCriticPolicy.__init__(self,observation_space,action_space,lr_schedule,**kwargs)
        if not isinstance(self.features_extractor,FlattenExtractor):raise ValueError('Flat features required')
        self.prior_bias=float(prior_bias)
        for net in (self.mlp_extractor.policy_net,self.mlp_extractor.value_net):
            layer=net[0];replacement=RateInputLinear(FEATURE_DIM,layer.out_features,bias=layer.bias is not None).to(layer.weight.device)
            replacement.load_state_dict(layer.state_dict());net[0]=replacement
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


def expand_matrix(value):
    if value.ndim!=2 or value.shape[1]!=43:raise ValueError('Expected43column matrix')
    result=value.new_zeros((value.shape[0],44));result[:,original_columns()]=value
    return result


def expand_policy(source,lr_schedule):
    if type(source) is not AimPriorPolicy:raise ValueError('Expected original43AimPriorPolicy')
    kwargs=source._get_constructor_parameters()
    kwargs.update(observation_space=spaces.Box(-1.,1.,(44,),dtype=np.float32),lr_schedule=lr_schedule)
    policy=RateInputPriorPolicy(**kwargs).to(source.device)
    policy.load_state_dict({name:expand_matrix(value) if name in FIRST else value.clone() for name,value in source.state_dict().items()})
    old_named=list(source.named_parameters());new_named=list(policy.named_parameters())
    if [n for n,_ in old_named]!=[n for n,_ in new_named]:raise ValueError('Parameter names/order differ')
    optimizer=copy.deepcopy(source.optimizer.state_dict());ids=[i for g in optimizer['param_groups'] for i in g['params']]
    if len(ids)!=len(old_named):raise ValueError('Optimizer parameter groups differ')
    for index,(name,_) in zip(ids,old_named):
        if name in FIRST:
            for key,value in optimizer['state'].get(index,{}).items():
                if isinstance(value,torch.Tensor) and value.ndim==2:optimizer['state'][index][key]=expand_matrix(value)
    policy.optimizer.load_state_dict(optimizer)
    return policy


def expand_learner(source):
    from stable_baselines3.common.buffers import RolloutBuffer
    policy=expand_policy(source.policy,source.lr_schedule)
    # Inference may cache a non-leaf tensor in the old action distribution.
    # Replace the policy during cloning rather than copying that transient graph.
    model=copy.deepcopy(source,{id(source.policy):policy})
    model.policy_class=RateInputPriorPolicy;model.observation_space=model.policy.observation_space
    model.rollout_buffer=RolloutBuffer(model.n_steps,model.observation_space,model.action_space,device=model.device,gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=model.n_envs)
    model._last_obs=None;model._last_original_obs=None
    return model


def projected_learner_state(model):
    if type(model.policy) is not RateInputPriorPolicy:raise ValueError('Expected44feature policy')
    state={name:value[:,original_columns()].clone() if name in FIRST else value for name,value in model.policy.state_dict().items()}
    optimizer=copy.deepcopy(model.policy.optimizer.state_dict());named=list(model.policy.named_parameters())
    ids=[i for g in optimizer['param_groups'] for i in g['params']]
    if len(ids)!=len(named):raise ValueError('Optimizer parameter groups differ')
    for index,(name,_) in zip(ids,named):
        if name in FIRST:
            for key,value in optimizer['state'].get(index,{}).items():
                if isinstance(value,torch.Tensor) and value.ndim==2:optimizer['state'][index][key]=value[:,original_columns()].clone()
    return dict(policy=tensor_digest(state),optimizer=tensor_digest(optimizer),decisions=model.num_timesteps,updates=model._n_updates)
