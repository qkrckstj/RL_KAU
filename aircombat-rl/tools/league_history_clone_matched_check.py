"""Qualify a history clone with complete preserved PPO settings and Adam state."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from experiments.league.history_features import HISTORY_DIM,PUBLIC_DIM,numpy_logits
from experiments.league.history_policy import HistoryPriorPolicy,clone_with_history,expand_input
from experiments.league.residual_ppo_policy import actor_parameters
from tools.league_sampled_ppo_continue import same_tree


class ShapeEnv(gym.Env):
    observation_space=spaces.Box(-1.,1.,shape=(HISTORY_DIM,),dtype=np.float32)
    action_space=spaces.Discrete(9)
    def reset(self,*,seed=None,options=None):return np.zeros(HISTORY_DIM,dtype=np.float32),{}
    def step(self,action):raise RuntimeError('No flights or learning in clone qualification')


def ppo_configuration(model):
    names=('gamma','gae_lambda','n_epochs','batch_size','ent_coef','target_kl','max_grad_norm','normalize_advantage','vf_coef','use_sde','sde_sample_freq')
    config={k:getattr(model,k) for k in names}
    config['clip_range_samples']=[model.clip_range(x) for x in (1.,.5,0.)]
    config['clip_range_vf_samples']=None if model.clip_range_vf is None else [model.clip_range_vf(x) for x in (1.,.5,0.)]
    config['learning_rate_samples']=[model.lr_schedule(x) for x in (1.,.5,0.)]
    return config


def run(learner,out):
    if out.exists():raise FileExistsError('Preserve prior qualifications')
    torch.set_num_threads(1)
    original=PPO.load(learner,device='cpu')
    kwargs={k:getattr(original,k) for k in ('learning_rate','batch_size','n_epochs','gamma','gae_lambda','clip_range','clip_range_vf','normalize_advantage','ent_coef','vf_coef','max_grad_norm','use_sde','sde_sample_freq','target_kl','rollout_buffer_class','rollout_buffer_kwargs')}
    clone=PPO(HistoryPriorPolicy,ShapeEnv(),n_steps=512,policy_kwargs=original.policy_kwargs,
              stats_window_size=original._stats_window_size,device='cpu',seed=5870,**kwargs)
    clone._n_updates=original._n_updates
    if ppo_configuration(clone)!=ppo_configuration(original):raise AssertionError('PPO configuration changed')
    clone_with_history(original.policy,clone.policy)
    old=dict(original.policy.named_parameters());new=dict(clone.policy.named_parameters())
    first={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    for name in old:
        expected=expand_input(old[name]) if name in first else old[name]
        if not torch.equal(new[name],expected):raise AssertionError('Changed pretrained parameter')
    original_optim=original.policy.optimizer.state_dict();new_optim=clone.policy.optimizer.state_dict()
    if original_optim['param_groups']!=new_optim['param_groups']:raise AssertionError('Changed Adam groups')
    ids=[k for g in original_optim['param_groups'] for k in g['params']]
    for name,index in zip(old,ids,strict=True):
        for key,value in original_optim['state'].get(index,{}).items():
            expected=expand_input(value) if name in first and isinstance(value,torch.Tensor) and value.ndim>0 else value
            if not same_tree(new_optim['state'][index][key],expected):raise AssertionError('Changed old Adam moment/step')
    rng=np.random.default_rng(5871);base=rng.uniform(-1,1,(2048,PUBLIC_DIM+9)).astype(np.float32)
    base[:,-9:]=np.eye(9,dtype=np.float32)[rng.integers(9,size=len(base))]
    expanded=np.concatenate((base[:,:PUBLIC_DIM],rng.uniform(-1,1,(len(base),2*PUBLIC_DIM)).astype(np.float32),base[:,-9:]),axis=1)
    a,b=map(torch.as_tensor,(base,expanded))
    with torch.no_grad():
        old_probs=original.policy.get_distribution(a).distribution.probs
        new_probs=clone.policy.get_distribution(b).distribution.probs
        old_value=original.policy.predict_values(a);new_value=clone.policy.predict_values(b)
        probability_error=float((old_probs-new_probs).abs().max())
        value_error=float((old_value-new_value).abs().max())
        if probability_error>2e-6 or value_error>2e-5:raise AssertionError('Initial actor/critic changed')
        if not torch.equal(old_probs.argmax(1),new_probs.argmax(1)):raise AssertionError('Initial greedy choices changed')
    weights=actor_parameters(clone.policy);np_raw=numpy_logits(expanded,weights)
    np_probs=np.exp(np_raw-np_raw.max(axis=1,keepdims=True));np_probs/=np_probs.sum(axis=1,keepdims=True)
    numpy_probability_error=float(np.max(np.abs(np_probs-new_probs.numpy())))
    if numpy_probability_error>2e-6:raise AssertionError('NumPy actor disagreement')
    out.mkdir(parents=True);clone.num_timesteps=original.num_timesteps;clone.save(out/'learner.zip')
    restored=PPO.load(out/'learner.zip',device='cpu')
    if not same_tree(restored.policy.state_dict(),clone.policy.state_dict()) or not same_tree(restored.policy.optimizer.state_dict(),clone.policy.optimizer.state_dict()):raise AssertionError('History learner/Adam restore mismatch')
    if ppo_configuration(restored)!=ppo_configuration(original):raise AssertionError('Saved PPO configuration mismatch')
    io.write(out/'completion.json',dict(ppo_configuration_preserved=True,ppo_configuration=ppo_configuration(original),status='history_clone_qualified',source=io.relative(learner),source_sha256=io.sha(learner),
        source_checkpoint_steps=original.num_timesteps,prototype=io.relative(out/'learner.zip'),prototype_sha256=io.sha(out/'learner.zip'),
        sources={p:io.sha(io.ROOT/p) for p in ('tools/league_history_clone_matched_check.py','experiments/league/history_policy.py','experiments/league/history_features.py')},
        feature_dim=HISTORY_DIM,observations=len(base),maximum_probability_error=probability_error,maximum_value_error=value_error,
        maximum_numpy_probability_error=numpy_probability_error,old_parameters_and_adam_state_preserved=True,learner_optimizer_restore_exact=True,
        scope='Synthetic input checks with arbitrary history. No new training, flight-trajectory equivalence or improved performance demonstrated.'))
    print(io.read(out/'completion.json'),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--learner',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.learner.resolve(),a.out.resolve())
