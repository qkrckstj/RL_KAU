"""Retain original heading hold; isolate observation and useful fine actions."""
from experiments.controller_study.tracking import TrackingEnv as Base

VARIANTS=('original3_obs','fine7','fine7_obs')


class TrackingEnv(Base):
    def __init__(self,variant,stabilizer,horizon_steps=600):
        if variant not in VARIANTS:raise ValueError('Unknown follow-up variant')
        base='original3' if variant=='original3_obs' else 'original7'
        super().__init__(base,stabilizer,horizon_steps)
        self.variant=variant
        self.enriched=variant.endswith('_obs')
        if variant.startswith('fine7'):
            self.commands=(-30.,-2.,-1.,0.,1.,2.,30.)


def make_env(variant,stabilizer):
    return TrackingEnv(variant,stabilizer)
