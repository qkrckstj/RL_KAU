from utils import PolicyBase, make_learner as _make_learner
from wrappers import SPEC
class Policy(PolicyBase):
    SPEC = SPEC
    TOTAL_STEPS = 4096000
    @staticmethod
    def make_learner(make_env, seed, device):
        return _make_learner(SPEC, seed, device)
