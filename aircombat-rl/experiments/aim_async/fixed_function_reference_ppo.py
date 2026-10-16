"""Freeze the archive's complete original Aim policy, independently of learner gates."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import math
import numbers
import random
import zipfile
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from experiments.aim_async.policy import AimPriorPolicy
from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO


@contextmanager
def preserved_random_streams():
    python_state=random.getstate();numpy_state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)


class FixedFunctionReferencePPO(ReferenceAnchoredPPO):
    def __init__(self,*args,**kwargs):
        self.reference_function_identity=None
        super().__init__(*args,**kwargs)

    def attach_reference(self,archive,coefficient):
        if (isinstance(coefficient,bool) or not isinstance(coefficient,numbers.Real)
                or not math.isfinite(coefficient) or not 0<=coefficient<=1):
            raise ValueError('Invalid reference coefficient')
        coefficient=float(coefficient)
        if self.reference_coefficient not in (0.,coefficient):
            raise ValueError('Saved reference coefficient differs')
        if (self.device.type!='cpu' or self.observation_space.shape!=(43,)
                or not isinstance(self.action_space,spaces.Discrete) or self.action_space.n!=9):
            raise ValueError('Fixed reference requires CPU43features and9actions')
        archive=Path(archive).resolve()
        archive_digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as source:
            policy_digest=hashlib.sha256(source.read('policy.pth')).hexdigest()
        if self.reference_policy_hash is not None and self.reference_policy_hash!=policy_digest:
            raise ValueError('Saved frozen reference tensor identity differs')
        existing=self.reference_function_identity
        if existing is not None and existing['archive_sha256']!=archive_digest:
            raise ValueError('Saved complete reference archive identity differs')
        # PPO.load rebuilds policy objects and applies the archive's seed. It must
        # not reseed the learner's Python/NumPy/Torch sampling streams.
        with preserved_random_streams():
            loaded=PPO.load(archive,device='cpu')
        reference=loaded.policy
        if (type(reference) is not AimPriorPolicy or reference.prior_bias!=6.
                or loaded.policy_kwargs.get('prior_bias',6.)!=6.
                or loaded.observation_space.shape!=(43,)
                or not isinstance(loaded.action_space,spaces.Discrete) or loaded.action_space.n!=9):
            raise ValueError('Reference must be the original AimPriorPolicy43/9prior6 function')
        function_identity=dict(schema='original_aim_prior6_function_v1',
            archive_sha256=archive_digest,policy_sha256=policy_digest,
            policy_class='experiments.aim_async.policy.AimPriorPolicy',
            prior_bias=6.,observation_shape=[43],action_count=9)
        if existing is not None and existing!=function_identity:
            raise ValueError('Saved complete reference function identity differs')
        reference.set_training_mode(False)
        for parameter in reference.parameters():parameter.requires_grad_(False)
        # Commit only after all validation; any rejected archive leaves the
        # previously attached reference and learner state untouched.
        self.reference_policy=reference
        self.reference_archive=str(archive)
        self.reference_policy_hash=policy_digest
        self.reference_coefficient=coefficient
        self.reference_function_identity=function_identity
