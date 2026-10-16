"""Explore speed choices after an encounter while preserving heading marginals."""
import hashlib
import math
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from experiments.aim_async.encounter_actor import EncounterActorPolicy, EncounterPPO
from experiments.aim_async.feature_train import enable_actor_features as original_features
from experiments.aim_async.policy import AimPriorPolicy


def validate_temperature(temperature):
    if isinstance(temperature, (bool, np.bool_)):
        raise ValueError('Finite speed temperature >=1 required')
    value=float(temperature)
    if not math.isfinite(value) or value<1:
        raise ValueError('Finite speed temperature >=1 required')
    return value


def numpy_speed_logits(logits, temperature):
    temperature=validate_temperature(temperature)
    x=np.asarray(logits,dtype=np.float32)
    if x.ndim<1 or x.shape[-1]!=9 or not np.isfinite(x).all():
        raise ValueError('Nine finite action logits required')
    if temperature==1:return x
    grid=x.reshape(x.shape[:-1]+(3,3))
    scaled=grid/temperature
    result=scaled+np.logaddexp.reduce(grid,axis=-1,keepdims=True)-np.logaddexp.reduce(scaled,axis=-1,keepdims=True)
    return result.reshape(x.shape)


def speed_logits(logits, temperature):
    temperature=validate_temperature(temperature)
    if logits.ndim<1 or logits.shape[-1]!=9 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError('Nine finite floating action logits required')
    if temperature==1:return logits
    grid=logits.reshape(logits.shape[:-1]+(3,3))
    scaled=grid/temperature
    result=scaled+torch.logsumexp(grid,-1,keepdim=True)-torch.logsumexp(scaled,-1,keepdim=True)
    return result.reshape(logits.shape)


class SpeedEncounterActorPolicy(EncounterActorPolicy):
    def __init__(self,*args,speed_temperature=2.,**kwargs):
        self.speed_temperature=validate_temperature(speed_temperature)
        super().__init__(*args,**kwargs)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),speed_temperature=self.speed_temperature)

    def _distribution(self,latent_pi,obs):
        base=self.base_observation(obs)
        current=self.action_net(latent_pi)+self.prior_bias*base[...,-9:]
        current=speed_logits(current,self.speed_temperature)
        with torch.no_grad():
            opening=self.opening_action_head(self.opening_actor_features(base))+self.prior_bias*base[...,-9:]
        logits=torch.where(obs[...,-1:]==1,current,opening)
        return self.action_dist.proba_distribution(action_logits=logits)


def enable_actor_features(policy):
    if type(policy) is not SpeedEncounterActorPolicy:raise ValueError('Expected conditional-speed encounter actor')
    result=original_features(policy)
    frozen=[p for n,p in policy.named_parameters() if n.startswith('opening_')]
    if not frozen or any(p.requires_grad for p in frozen):raise ValueError('Opening actor must remain frozen')
    return dict(result,frozen_parameters=sum(p.numel() for p in frozen))


class SpeedEncounterPPO(EncounterPPO):
    def attach_reference(self,archive,coefficient):
        # The immutable reference remains the original43-feature policy.
        # Strip only our new constructor option; keep inherited KL semantics.
        coefficient=float(coefficient)
        if not math.isfinite(coefficient) or coefficient<=0:raise ValueError('Positive original reference coefficient required')
        archive=Path(archive).resolve()
        with zipfile.ZipFile(archive) as saved:payload=saved.read('policy.pth')
        digest=hashlib.sha256(payload).hexdigest()
        if self.reference_policy_hash not in (None,digest) or self.reference_coefficient not in (0.,coefficient):
            raise ValueError('Saved frozen reference changed')
        kwargs=self.policy._get_constructor_parameters();kwargs.pop('speed_temperature')
        kwargs['observation_space']=spaces.Box(-1.,1.,(43,),dtype=np.float32)
        reference=AimPriorPolicy(**kwargs).to(self.device)
        reference.load_state_dict(torch.load(BytesIO(payload),map_location=self.device,weights_only=True),strict=True)
        reference.set_training_mode(False);reference.requires_grad_(False)
        self.reference_policy=reference
        self.reference_archive=str(archive);self.reference_policy_hash=digest;self.reference_coefficient=coefficient


def load_with_rate(path,*,env,seed,learning_rate,speed_temperature=2.):
    temperature=validate_temperature(speed_temperature);rate=float(learning_rate)
    if not math.isfinite(rate) or rate<=0:raise ValueError('Positive learning rate required')
    model=SpeedEncounterPPO.load(path,env=None,device='cpu',seed=seed,force_reset=True,n_steps=640,learning_rate=rate)
    if type(model.policy) is EncounterActorPolicy:
        model.policy.__class__=SpeedEncounterActorPolicy
        model.policy.speed_temperature=temperature
    elif type(model.policy) is SpeedEncounterActorPolicy:
        if model.policy.speed_temperature!=temperature:raise ValueError('Saved speed temperature differs')
    else:raise ValueError('Expected an authenticated encounter checkpoint')
    model.policy_class=SpeedEncounterActorPolicy
    model.policy_kwargs=dict(model.policy_kwargs,speed_temperature=temperature)
    for group in model.policy.optimizer.param_groups:group['lr']=rate
    enable_actor_features(model.policy)
    if env is not None:
        model.n_envs=env.num_envs;model.set_env(env,force_reset=True)
    model.rollout_buffer=RolloutBuffer(model.n_steps,model.observation_space,model.action_space,
        device=model.device,gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=model.n_envs)
    model._last_obs=None;model._last_original_obs=None
    return model
