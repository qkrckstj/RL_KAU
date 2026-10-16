"""Preserve original43learner state while adding a public command-memory cue."""
import copy
import torch
from experiments.aim_async.rate_input_policy import RateInputPriorPolicy, expand_policy as split_expand_policy, FIRST
from experiments.aim_async.rate_input_features import original_columns
from tools.prepare_rear_cpu_entropy import tensor_digest


class HeadingMemoryPriorPolicy(RateInputPriorPolicy):
    """Same split44architecture;cue semantics are explicit in the saved class."""


def expand_policy(source,lr_schedule):
    expanded=split_expand_policy(source,lr_schedule)
    kwargs=expanded._get_constructor_parameters();kwargs['lr_schedule']=lr_schedule
    policy=HeadingMemoryPriorPolicy(**kwargs).to(source.device)
    policy.load_state_dict(expanded.state_dict())
    policy.optimizer.load_state_dict(copy.deepcopy(expanded.optimizer.state_dict()))
    return policy


def expand_learner(source):
    from stable_baselines3.common.buffers import RolloutBuffer
    policy=expand_policy(source.policy,source.lr_schedule)
    # Inference may cache a non-leaf tensor in the old action distribution.
    # Replace the policy during cloning rather than copying that transient graph.
    model=copy.deepcopy(source,{id(source.policy):policy})
    model.policy_class=HeadingMemoryPriorPolicy;model.observation_space=model.policy.observation_space
    model.rollout_buffer=RolloutBuffer(model.n_steps,model.observation_space,model.action_space,device=model.device,gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=model.n_envs)
    model._last_obs=None;model._last_original_obs=None
    return model


def projected_learner_state(model):
    if type(model.policy) is not HeadingMemoryPriorPolicy:raise ValueError('Expected44feature policy')
    state={name:value[:,original_columns()].clone() if name in FIRST else value for name,value in model.policy.state_dict().items()}
    optimizer=copy.deepcopy(model.policy.optimizer.state_dict());named=list(model.policy.named_parameters())
    ids=[i for g in optimizer['param_groups'] for i in g['params']]
    if len(ids)!=len(named):raise ValueError('Optimizer parameter groups differ')
    for index,(name,_) in zip(ids,named):
        if name in FIRST:
            for key,value in optimizer['state'].get(index,{}).items():
                if isinstance(value,torch.Tensor) and value.ndim==2:optimizer['state'][index][key]=value[:,original_columns()].clone()
    return dict(policy=tensor_digest(state),optimizer=tensor_digest(optimizer),decisions=model.num_timesteps,updates=model._n_updates)
