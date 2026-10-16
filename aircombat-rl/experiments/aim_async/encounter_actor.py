"""Preserve the opening actor; train the original actor after a range encounter."""
import copy
import hashlib
import math
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from experiments.aim_async.policy import AimPriorPolicy
from experiments.aim_async.opening_anchor import OpeningAnchoredPPO
from experiments.aim_async.reference_ppo import categorical_reference_kl
from experiments.aim_async.feature_train import enable_actor_features as enable_original_features


def phase_observation_space():
    low=np.full(44,-1.,np.float32);low[-1]=0.
    return spaces.Box(low,np.ones(44,np.float32),dtype=np.float32)


class EncounterActorPolicy(AimPriorPolicy):
    def __init__(self,observation_space,action_space,lr_schedule,**kwargs):
        if observation_space.shape!=(44,):
            raise ValueError('Expected44 features including a binary phase')
        super().__init__(spaces.Box(-1.,1.,(43,),dtype=np.float32),action_space,lr_schedule,**kwargs)
        self.observation_space=phase_observation_space()
        self.protect_opening()

    def protect_opening(self):
        if hasattr(self,'opening_actor_features'):
            raise ValueError('Never replace a preserved opening actor')
        # Copies are created AFTER Adam, so all original optimizer identities stay.
        self.opening_actor_features=copy.deepcopy(self.mlp_extractor.policy_net)
        self.opening_action_head=copy.deepcopy(self.action_net)
        for module in (self.opening_actor_features,self.opening_action_head):
            module.requires_grad_(False)
            module.eval()

    @staticmethod
    def base_observation(obs):
        if obs.shape[-1]!=44 or not torch.isfinite(obs).all() or not torch.all((obs[...,-1]==0)|(obs[...,-1]==1)):
            raise ValueError('Expected44 finite features and binary phase')
        return obs[...,:43]

    def extract_features(self,obs,features_extractor=None):
        return super().extract_features(self.base_observation(obs),features_extractor)

    def predict_values(self,obs):
        # SB3's value method calls BaseModel.extract_features directly.
        return super().predict_values(self.base_observation(obs))

    def _distribution(self,latent_pi,obs):
        base=self.base_observation(obs)
        current=self.action_net(latent_pi)+self.prior_bias*base[...,-9:]
        with torch.no_grad():
            opening=self.opening_action_head(self.opening_actor_features(base))+self.prior_bias*base[...,-9:]
        logits=torch.where(obs[...,-1:]==1,current,opening)
        return self.action_dist.proba_distribution(action_logits=logits)


def enable_actor_features(policy):
    if type(policy) is not EncounterActorPolicy:
        raise ValueError('Expected encounter actor')
    result=enable_original_features(policy)
    frozen=[p for n,p in policy.named_parameters() if n.startswith('opening_')]
    if not frozen or any(p.requires_grad for p in frozen):
        raise ValueError('Opening actor must remain frozen')
    return dict(result,frozen_parameters=sum(p.numel() for p in frozen))


class EncounterPPO(OpeningAnchoredPPO):
    def train(self):
        self.logger.record('train/after_encounter_fraction',float(np.mean(self.rollout_buffer.observations[...,-1])))
        super().train()

    def attach_reference(self,archive,coefficient):
        coefficient=float(coefficient)
        if not math.isfinite(coefficient) or coefficient<=0:
            raise ValueError('Positive original reference coefficient required')
        archive=Path(archive).resolve()
        with zipfile.ZipFile(archive) as saved:payload=saved.read('policy.pth')
        digest=hashlib.sha256(payload).hexdigest()
        if self.reference_policy_hash not in (None,digest) or self.reference_coefficient not in (0.,coefficient):
            raise ValueError('Saved frozen reference changed')
        kwargs=self.policy._get_constructor_parameters()
        kwargs['observation_space']=spaces.Box(-1.,1.,(43,),dtype=np.float32)
        reference=AimPriorPolicy(**kwargs).to(self.device)
        reference.load_state_dict(torch.load(BytesIO(payload),map_location=self.device,weights_only=True),strict=True)
        reference.set_training_mode(False);reference.requires_grad_(False)
        self.reference_policy=reference
        self.reference_archive=str(archive);self.reference_policy_hash=digest;self.reference_coefficient=coefficient

    def attach_opening_anchor(self,bank,coefficient,batch_size,seed):
        super().attach_opening_anchor(bank,coefficient,batch_size,seed)
        obs=self.opening_observations
        self.opening_observations=torch.cat((obs,obs.new_zeros((len(obs),1))),dim=1)

    def reference_kl(self,observations):
        if self.reference_policy is None:
            raise RuntimeError('Restore frozen reference before training')
        base=self.policy.base_observation(observations)
        with torch.no_grad():frozen=self.reference_policy.get_distribution(base).distribution.logits
        current=self.policy.get_distribution(observations).distribution.logits
        original=categorical_reference_kl(frozen,current)
        if self.opening_coefficient==0:return original
        if self.opening_observations is None or self.reference_coefficient<=0:
            raise RuntimeError('Restore opening bank before training')
        indices=self.opening_rng.integers(len(self.opening_observations),size=self.opening_batch_size)
        current=self.policy.get_distribution(self.opening_observations[indices]).distribution.logits
        opening=categorical_reference_kl(self.opening_logits[indices],current)
        self._opening_metrics.append((original.detach().item(),opening.detach().item()))
        return original+(self.opening_coefficient/self.reference_coefficient)*opening


def load_with_rate(path,*,env,seed,learning_rate):
    rate=float(learning_rate)
    if not math.isfinite(rate) or rate<=0:
        raise ValueError('Positive learning rate required')
    # Old checkpoints must be migrated before SB3 checks the new environment space.
    model=EncounterPPO.load(path,env=None,device='cpu',seed=seed,force_reset=True,
                            n_steps=640,learning_rate=rate)
    if type(model.policy) is AimPriorPolicy:
        model.policy.__class__=EncounterActorPolicy
        model.policy.observation_space=phase_observation_space()
        model.policy.protect_opening()
        model.policy_class=EncounterActorPolicy
        model.observation_space=model.policy.observation_space
    elif type(model.policy) is not EncounterActorPolicy:
        raise ValueError('Unsupported source policy for encounter migration')
    for group in model.policy.optimizer.param_groups:group['lr']=rate
    enable_actor_features(model.policy)
    if env is not None:
        model.n_envs=env.num_envs
        model.set_env(env,force_reset=True)
    model.rollout_buffer=RolloutBuffer(model.n_steps,model.observation_space,model.action_space,
        device=model.device,gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=model.n_envs)
    model._last_obs=None;model._last_original_obs=None
    return model
