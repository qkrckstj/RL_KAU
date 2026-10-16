"""Training-only stratified rehearsal; inference and native dynamics are unchanged."""
import math
import numpy as np
from experiments.aim_async.opening_anchor import OpeningAnchoredPPO
from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO, categorical_reference_kl


SAMPLING = dict(mode='half_uniform_half_healthy_closing_contact',
                range_min_m=1000,range_max_m=2500,health_min_exclusive=.999,
                uniform_fraction=.5,with_replacement=True)


def contact_indices(observations):
    """43-feature schema: distance, radial velocity, own and opponent health."""
    x=np.asarray(observations)
    if x.ndim!=2 or x.shape[1]!=43 or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError('Require finite float32 43-feature observations')
    return np.flatnonzero((x[:,0]>=np.float32(1000/11000)) &
                         (x[:,0]<=np.float32(2500/12500)) & (x[:,7]<0) &
                         (x[:,19]>np.float32(.999)) & (x[:,20]>np.float32(.999)))


class ContactAnchoredPPO(OpeningAnchoredPPO):
    def __init__(self,*args,**kwargs):
        self.contact_bank_indices=None
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['contact_bank_indices']

    def attach_opening_anchor(self,bank,coefficient,batch_size,seed):
        if type(batch_size) is not int or batch_size<=0 or batch_size%2:
            raise ValueError('Require a positive even opening batch size')
        super().attach_opening_anchor(bank,coefficient,batch_size,seed)
        self.contact_bank_indices=contact_indices(self.opening_observations.detach().cpu().numpy())
        if not len(self.contact_bank_indices):raise ValueError('Opening bank has no contact observations')

    def sample_opening_indices(self):
        if self.contact_bank_indices is None or not len(self.contact_bank_indices):
            raise RuntimeError('Restore contact bank before training')
        half=self.opening_batch_size//2
        uniform=self.opening_rng.integers(len(self.opening_observations),size=half)
        contact=self.contact_bank_indices[self.opening_rng.integers(len(self.contact_bank_indices),size=half)]
        return np.concatenate([uniform,contact])

    def reference_kl(self,observations):
        # Calling ReferenceAnchoredPPO directly avoids sampling the opening bank twice.
        original=ReferenceAnchoredPPO.reference_kl(self,observations)
        if self.opening_coefficient==0:return original
        if self.opening_observations is None or self.reference_coefficient<=0:
            raise RuntimeError('Restore opening bank and original reference before training')
        indices=self.sample_opening_indices()
        current=self.policy.get_distribution(self.opening_observations[indices]).distribution.logits
        opening=categorical_reference_kl(self.opening_logits[indices],current)
        self._opening_metrics.append((original.detach().item(),opening.detach().item()))
        return original+(self.opening_coefficient/self.reference_coefficient)*opening


def load_with_rate(path,*,env,seed,learning_rate):
    rate=float(learning_rate)
    if not math.isfinite(rate) or rate<=0:raise ValueError('Positive learning rate required')
    model=ContactAnchoredPPO.load(path,env=env,device='cpu',seed=seed,force_reset=True,
                                n_steps=640,learning_rate=rate)
    for group in model.policy.optimizer.param_groups:group['lr']=rate
    return model
