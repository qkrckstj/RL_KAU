"""Training-only rehearsal of the selected parent's native opening observations.

The existing PPO train body applies reference_coefficient * reference_kl.
This subclass adds an independently weighted KL on a frozen training bank at
that boundary, before the existing joint backward and gradient clipping. The
old reference term, rollout advantages and inference policy stay unchanged.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from experiments.aim_async.features import features
from experiments.aim_async.post_encounter import PostEncounterArchive
from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO, categorical_reference_kl


class OpeningCaptureArchive(PostEncounterArchive):
    """Observe teacher calls already required by a native prefix; add no calls."""
    def __init__(self, *args, **kwargs):
        self.opening_features=[]
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        self.opening_features=[]
        return super().reset(**kwargs)

    def teacher_action(self, raw):
        action=super().teacher_action(raw)
        self.opening_features.append(features(raw,action))
        return action


class OpeningAnchoredPPO(ReferenceAnchoredPPO):
    def __init__(self, *args, **kwargs):
        self.opening_coefficient=0.
        self.opening_bank=None
        self.opening_bank_sha256=None
        self.opening_batch_size=None
        self.opening_rng=None
        self.opening_observations=None
        self.opening_logits=None
        self._opening_metrics=[]
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['opening_observations','opening_logits','_opening_metrics']

    def attach_opening_anchor(self, bank, coefficient, batch_size, seed):
        coefficient=float(coefficient)
        if not math.isfinite(coefficient) or coefficient<=0 or self.reference_coefficient<=0:
            raise ValueError('Positive opening and original reference coefficients required')
        if type(batch_size) is not int or batch_size<=0:
            raise ValueError('Positive integer opening batch size required')
        bank=Path(bank).resolve();digest=hashlib.sha256(bank.read_bytes()).hexdigest()
        if self.opening_bank_sha256 not in (None,digest):
            raise ValueError('Frozen opening bank changed')
        if self.opening_coefficient not in (0.,coefficient) or self.opening_batch_size not in (None,batch_size):
            raise ValueError('Saved opening configuration changed')
        with np.load(bank,allow_pickle=False) as data:
            observations=data['observations'];logits=data['logits']
        if (observations.ndim!=2 or observations.shape[1:]!=(43,) or not len(observations)
                or logits.shape!=(len(observations),9) or observations.dtype!=np.float32
                or logits.dtype!=np.float32 or not np.isfinite(observations).all()
                or not np.isfinite(logits).all() or np.max(np.abs(observations))>1.):
            raise ValueError('Invalid opening bank arrays')
        self.opening_observations=torch.as_tensor(observations,device=self.device)
        self.opening_logits=torch.as_tensor(logits,device=self.device)
        self.opening_bank=str(bank);self.opening_bank_sha256=digest
        self.opening_coefficient=coefficient;self.opening_batch_size=batch_size
        if self.opening_rng is None:self.opening_rng=np.random.default_rng(seed)

    def opening_kl_all(self):
        """Measured bank divergence for checkpoint reporting; preserves RNG."""
        if self.opening_observations is None:
            raise RuntimeError('Restore the frozen opening bank before training')
        current=self.policy.get_distribution(self.opening_observations).distribution.logits
        return categorical_reference_kl(self.opening_logits,current)

    def reference_kl(self, observations):
        original=super().reference_kl(observations)
        if self.opening_coefficient==0:return original
        if self.opening_observations is None or self.reference_coefficient<=0:
            raise RuntimeError('Restore opening bank and original reference before training')
        indices=self.opening_rng.integers(len(self.opening_observations),size=self.opening_batch_size)
        current=self.policy.get_distribution(self.opening_observations[indices]).distribution.logits
        opening=categorical_reference_kl(self.opening_logits[indices],current)
        self._opening_metrics.append((original.detach().item(),opening.detach().item()))
        return original+(self.opening_coefficient/self.reference_coefficient)*opening

    def train(self):
        self._opening_metrics=[]
        super().train()
        if self.opening_coefficient:
            if not self._opening_metrics:raise RuntimeError('Opening loss was not used by PPO')
            original,opening=np.mean(self._opening_metrics,axis=0)
            # Preserve the original reference-KL meaning in the CSV.
            self.logger.record('train/reference_kl',original)
            self.logger.record('train/opening_kl',opening)
            self.logger.record('train/opening_coefficient',self.opening_coefficient)
            self.logger.record('train/reference_and_opening_penalty',
                               self.reference_coefficient*original+self.opening_coefficient*opening)


def load_with_rate(path, *, env, seed, learning_rate):
    rate=float(learning_rate)
    if not math.isfinite(rate) or rate<=0:raise ValueError('Positive learning rate required')
    model=OpeningAnchoredPPO.load(path,env=env,device='cpu',seed=seed,force_reset=True,
                                n_steps=640,learning_rate=rate)
    for group in model.policy.optimizer.param_groups:group['lr']=rate
    return model
