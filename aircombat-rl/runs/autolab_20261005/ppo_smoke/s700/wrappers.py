from utils import State as _State, make_env as _make_env
SPEC = {'variant': 'observed_ppo', 'task': 'fair', 'algorithm': 'ppo', 'observation': 'relative_v1', 'gamma': 0.999, 'finite_match': True, 'action_mode': 'discrete', 'feature_version': 1, 'learning_rate': 0.0001, 'n_steps': 16, 'batch_size': 8, 'n_epochs': 2, 'gae_lambda': 1.0, 'ent_coef': 0.0, 'target_kl': 0.01}
ACTION_MODE = 'discrete'
class State(_State):
    def __init__(self):
        super().__init__(SPEC)
def make_env(seed=None, shaped=True):
    return _make_env(SPEC, seed, shaped)
