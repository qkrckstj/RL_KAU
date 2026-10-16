"""Extend an existing publicWEZ-or-damaged prior with earlier public range."""
import copy
import torch
from experiments.aim_async.wez_or_damaged_policy import WezOrDamagedPriorPolicy
from experiments.aim_async.early_range_numpy import SCALED_RANGE_LIMIT
from tools.aim_prior_bias import learner_state
from tools.prepare_rear_cpu_entropy import tensor_digest


class EarlyRangePriorPolicy(WezOrDamagedPriorPolicy):
    def _distribution(self, latent_pi, obs):
        if obs.shape[-1:] != (43,) or not torch.isfinite(obs).all():
            raise ValueError('Expected finite43public features')
        logits = self.action_net(latent_pi)+self.prior_bias*obs[..., -9:]
        gate = (obs[..., 24] > .5) | (obs[..., 19] < 1.) | (obs[..., 0] <= float(SCALED_RANGE_LIMIT))
        correction = torch.where(gate, self.threat_prior_bias-self.prior_bias, 0.)
        logits = logits+correction[..., None]*obs[..., -9:]
        return self.action_dist.proba_distribution(action_logits=logits)


def install_early_range_prior(model):
    old = model.policy
    if (type(old) is not WezOrDamagedPriorPolicy or old.prior_bias != 6.
            or old.threat_prior_bias != 3.
            or model.policy_kwargs.get('prior_bias', 6.) != 6.
            or model.policy_kwargs.get('threat_prior_bias', 3.) != 3.):
        raise ValueError('Expected exact WezOrDamagedPriorPolicy prior6/lower3 source')
    if (model.device.type != 'cpu' or model.observation_space.shape != (43,)
            or model.action_space.n != 9):
        raise ValueError('Expected CPU43features and9actions')
    if getattr(model, 'reference_policy', None) is not None:
        raise ValueError('Install before independently reconstructing the fixed reference')
    before = learner_state(model)
    kwargs = old._get_constructor_parameters()
    kwargs.update(lr_schedule=model.lr_schedule)
    with torch.random.fork_rng(devices=[]):
        policy = EarlyRangePriorPolicy(**kwargs).to(model.device)
    if [(n, tuple(p.shape)) for n, p in old.named_parameters()] != [
            (n, tuple(p.shape)) for n, p in policy.named_parameters()]:
        raise ValueError('Actor/critic parameter names/order/shapes changed')
    policy.load_state_dict(old.state_dict(), strict=True)
    policy.optimizer.load_state_dict(copy.deepcopy(old.optimizer.state_dict()))
    policy.set_training_mode(old.training)
    if (tensor_digest(policy.state_dict()) != before['policy'] or
            tensor_digest(policy.optimizer.state_dict()) != before['optimizer']):
        raise ValueError('Actor/critic/Adam changed during installation')
    model.policy = policy
    model.policy_class = EarlyRangePriorPolicy
    model.policy_kwargs = dict(model.policy_kwargs, prior_bias=6., threat_prior_bias=3.)
    if learner_state(model) != before:
        raise ValueError('Learned state/counters changed during installation')
    return dict(before=before, after=learner_state(model), state_exact=True,
        base_prior=6., lower_prior=3., policy_behavior_unchanged=False,
        gate='public_wez_or_damaged_or_float32_range_le4000_v1',
        range_threshold_m=4000.,scaled_range_limit=float(SCALED_RANGE_LIMIT))
