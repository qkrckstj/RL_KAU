"""Reduce a fixed teacher logit prior only during a public aiming conflict."""
import copy
import math
import torch
from experiments.aim_async.policy import AimPriorPolicy
from tools.aim_prior_bias import learner_state


class StatePriorPolicy(AimPriorPolicy):
    def __init__(self,*args,prior_bias=6.,conflict_prior_bias=3.,**kwargs):
        if not math.isfinite(conflict_prior_bias) or not 0<conflict_prior_bias<prior_bias:
            raise ValueError('Conflict prior must be positive and below base prior')
        super().__init__(*args,prior_bias=prior_bias,**kwargs)
        self.conflict_prior_bias=float(conflict_prior_bias)

    def _distribution(self,latent_pi,obs):
        logits=self.action_net(latent_pi)+self.prior_bias*obs[...,-9:]
        turn=torch.argmax(obs[...,-9:],dim=-1)//3-1
        threshold=torch.as_tensor(math.cos(math.radians(30.)),dtype=obs.dtype,device=obs.device)
        conflict=((obs[...,24]>.5)&(obs[...,23]<.5)&(obs[...,32]<threshold)&(turn*obs[...,31]<0.))
        correction=torch.where(conflict,self.conflict_prior_bias-self.prior_bias,0.)
        logits=logits+correction[...,None]*obs[...,-9:]
        return self.action_dist.proba_distribution(action_logits=logits)

    def _get_constructor_parameters(self):
        return dict(**super()._get_constructor_parameters(),conflict_prior_bias=self.conflict_prior_bias)


def install_state_prior(model):
    old=model.policy
    if (type(old) is not AimPriorPolicy or old.prior_bias!=6.
            or model.policy_kwargs.get('prior_bias',6.)!=6.):
        raise ValueError('Expected unchanged Aim prior6source')
    before=learner_state(model)
    kwargs=old._get_constructor_parameters()
    kwargs.update(lr_schedule=model.lr_schedule,conflict_prior_bias=3.)
    policy=StatePriorPolicy(**kwargs).to(model.device)
    if [(n,tuple(p.shape)) for n,p in old.named_parameters()]!=[(n,tuple(p.shape)) for n,p in policy.named_parameters()]:
        raise ValueError('Actor/critic parameter order or shapes changed')
    policy.load_state_dict(old.state_dict())
    policy.optimizer.load_state_dict(copy.deepcopy(old.optimizer.state_dict()))
    model.policy=policy;model.policy_class=StatePriorPolicy
    model.policy_kwargs=dict(model.policy_kwargs,prior_bias=6.,conflict_prior_bias=3.)
    after=learner_state(model)
    if before!=after:raise ValueError('Actor/critic/Adam/counters changed')
    return dict(before=before,after=after,state_exact=True,base_prior=6.,conflict_prior=3.,
                policy_behavior_unchanged=False,gate='enemy_solution_and_teacher_opposes_lead_v1')
