from utils import State as _State, make_env as _make_env
SPEC = {'variant': 'observed_dqn', 'task': 'fair', 'algorithm': 'dqn', 'observation': 'relative_v1', 'gamma': 0.999, 'finite_match': True, 'action_mode': 'discrete', 'feature_version': 1, 'learning_rate': 0.0001}
ACTION_MODE = 'discrete'
class State(_State):
    def __init__(self):
        super().__init__(SPEC)
def make_env(seed=None, shaped=True):
    return _make_env(SPEC, seed, shaped)
