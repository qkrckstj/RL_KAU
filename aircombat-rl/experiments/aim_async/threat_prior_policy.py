"""Lower teacher logit bias during public opponent WEZ; native rules unchanged."""
import copy
import torch
from experiments.aim_async.policy import AimPriorPolicy
from experiments.aim_async.threat_prior_numpy import prior_values
from tools.aim_prior_bias import learner_state
from tools.prepare_rear_cpu_entropy import tensor_digest


class ThreatPriorPolicy(AimPriorPolicy):
    def __init__(self,*args,prior_bias=6.,threat_prior_bias=3.,**kwargs):
        base,threat=prior_values(prior_bias,threat_prior_bias)
        super().__init__(*args,prior_bias=base,**kwargs)
        self.threat_prior_bias=threat

    def _distribution(self,latent_pi,obs):
        if obs.shape[-1:]!=(43,) or not torch.isfinite(obs).all():
            raise ValueError('Expected finite43public features')
        logits=self.action_net(latent_pi)+self.prior_bias*obs[...,-9:]
        correction=torch.where(obs[...,24]>.5,self.threat_prior_bias-self.prior_bias,0.)
        logits=logits+correction[...,None]*obs[...,-9:]
        return self.action_dist.proba_distribution(action_logits=logits)

    def _get_constructor_parameters(self):
        return dict(**super()._get_constructor_parameters(),threat_prior_bias=self.threat_prior_bias)


def install_threat_prior(model):
    old=model.policy
    if (type(old) is not AimPriorPolicy or old.prior_bias!=6.
            or model.policy_kwargs.get('prior_bias',6.)!=6.):
        raise ValueError('Expected unchanged Aim prior6source')
    if getattr(model,'reference_policy',None) is not None:
        raise ValueError('Install before independently reconstructing the fixed reference')
    before=learner_state(model)
    kwargs=old._get_constructor_parameters()
    kwargs.update(lr_schedule=model.lr_schedule,threat_prior_bias=3.)
    # Construction initializes temporary weights before loading the exact source.
    # Do not consume the learner's CPU random stream during that construction.
    with torch.random.fork_rng(devices=[]):
        policy=ThreatPriorPolicy(**kwargs).to(model.device)
    if [(n,tuple(p.shape)) for n,p in old.named_parameters()]!=[(n,tuple(p.shape)) for n,p in policy.named_parameters()]:
        raise ValueError('Actor/critic parameter names/order/shapes changed')
    policy.load_state_dict(old.state_dict(),strict=True)
    policy.optimizer.load_state_dict(copy.deepcopy(old.optimizer.state_dict()))
    policy.set_training_mode(old.training)
    if (tensor_digest(policy.state_dict())!=before['policy'] or
            tensor_digest(policy.optimizer.state_dict())!=before['optimizer']):
        raise ValueError('Actor/critic/Adam changed during installation')
    model.policy=policy
    model.policy_class=ThreatPriorPolicy
    model.policy_kwargs=dict(model.policy_kwargs,prior_bias=6.,threat_prior_bias=3.)
    after=learner_state(model)
    if before!=after:
        raise ValueError('Learned state/counters changed during installation')
    return dict(before=before,after=after,state_exact=True,base_prior=6.,threat_prior=3.,
                policy_behavior_unchanged=False,gate='public_opp_in_wez_v1')
