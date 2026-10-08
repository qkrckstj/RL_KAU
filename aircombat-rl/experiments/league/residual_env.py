"""Mixed archived opponents and public teacher actions for a future PPO run.

FairFight sampling, stepping and verdicts are inherited unchanged. Training
utility/shaping and finite-horizon bootstrap handling are explicit wrappers.
Opponent choice uses a separate RNG from official initial-condition sampling.
"""
from copy import deepcopy
import operator
import gymnasium as gym
import numpy as np
from aircombat_gym.wvr.envs.fair import FairFightEnv
from tools.league_matches import Actor
from experiments.league.residual_features import features, FEATURE_DIM


class CoverageSampler:
    """Alternate full shuffled archive coverage with frozen weighted draws."""
    def __init__(self, probabilities, seed):
        probabilities = np.asarray(probabilities, dtype=float)
        if (probabilities.ndim != 1 or not len(probabilities) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or probabilities.sum() <= 0):
            raise ValueError('Expected a nonempty finite nonnegative probability vector')
        self.probabilities = probabilities/probabilities.sum()
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 739]))
        self.coverage = []
        self.draws = 0

    def next(self):
        if self.draws % 2 == 0:
            if not self.coverage:
                self.coverage = self.rng.permutation(len(self.probabilities)).tolist()
            chosen = self.coverage.pop()
        else:
            chosen = int(self.rng.choice(len(self.probabilities), p=self.probabilities))
        self.draws += 1
        return chosen


class ArchiveFight(FairFightEnv):
    def __init__(self, teacher, opponents, probabilities, stream_seed, ic_band,
                 worker_index, workers, seat, ic_count=1_000_000):
        if not opponents or len(probabilities) != len(opponents):
            raise ValueError('Opponent probabilities must match a nonempty roster')
        if len({s['id'] for s in opponents}) != len(opponents):
            raise ValueError('Opponent IDs must be unique')
        if not 0 <= worker_index < workers or ic_band < 0 or ic_count < workers:
            raise ValueError('Invalid disjoint initial-condition stream')
        self.opponents = deepcopy(opponents)
        self.probabilities = tuple(probabilities)
        self.teacher = Actor(deepcopy(teacher))
        if self.teacher.mode != 'discrete':
            raise ValueError('The action prior requires a9-action discrete teacher')
        self.cache = {}
        self.stream_seed = int(stream_seed)
        self.ic_band, self.ic_count = int(ic_band), int(ic_count)
        self.worker_index, self.workers = int(worker_index), int(workers)
        self.sampler = CoverageSampler(self.probabilities, self.stream_seed)
        self.episode_index = 0
        self.current_foe = None
        self.episode_metadata = None
        super().__init__(action_mode='discrete', seat=seat)

    def sample(self, rng):
        initial, _ = super().sample(rng)
        if self.current_foe is None:
            raise RuntimeError('Choose the archived opponent before reset')
        return initial, self.current_foe

    def reset(self, *, seed=None, options=None):
        # Gym's explicit seed restarts opponent selection and the disjoint IC
        # sequence. It never consumes the official IC RNG to select a policy.
        if seed is not None:
            self.sampler = CoverageSampler(self.probabilities, int(seed))
            self.episode_index = 0
        offset = self.episode_index*self.workers+self.worker_index
        if offset >= self.ic_count:
            raise RuntimeError('Reserved training initial-condition band exhausted')
        episode_seed = self.ic_band+offset
        spec = self.opponents[self.sampler.next()]
        if spec['id'] not in self.cache:
            self.cache[spec['id']] = Actor(spec)
        self.current_foe = self.cache[spec['id']]
        self.teacher.seed(episode_seed, 0 if self.seat == 'red' else 1)
        self.current_foe.seed(episode_seed, 1 if self.seat == 'red' else 0)
        self.teacher.reset()
        raw, info = super().reset(seed=episode_seed, options=options)
        self.episode_metadata = dict(episode_index=self.episode_index, episode_seed=episode_seed,
                                     opponent_id=spec['id'], seat=self.seat)
        self.episode_index += 1
        return raw, dict(info, league_episode=dict(self.episode_metadata))

    def teacher_action(self, raw):
        return int(self.encode_action(self.teacher.act(self._last[self.seat], raw)))


def potential(state):
    """Same bounded public-geometry potential as the existing relative_v1 setup."""
    f = np.asarray(state)
    if f.shape != (FEATURE_DIM,) or not np.isfinite(f).all():
        raise ValueError('Expected finite public features plus action prior')
    own_cos = float(np.clip(f[4], -1., 1.))
    opp_cos = float(np.clip(-(f[4]*f[6]+f[3]*f[5]), -1., 1.))
    proximity = 1.-float(np.clip(f[0], 0., 1.))
    health = float(np.clip(f[19], 0., 1.)-np.clip(f[20], 0., 1.))
    track = float(np.clip(f[21], 0., 1.)-np.clip(f[22], 0., 1.))
    return .25*proximity*(own_cos-opp_cos)/2+.125*(health+track)


class PriorTrainingEnv(gym.Wrapper):
    """Learning-only utility: kill1+.05health, loss-1, draw0, optional Phi."""
    def __init__(self, env, gamma=.999, shaping=True):
        super().__init__(env)
        if not 0 < gamma <= 1:
            raise ValueError('Discount must lie in(0,1]')
        self.gamma, self.shaping = float(gamma), bool(shaping)
        self.observation_space = gym.spaces.Box(-1., 1., (FEATURE_DIM,), dtype=np.float32)
        self.previous_potential = None

    def _state(self, raw):
        action = self.env.teacher_action(raw)
        return features(raw, action), action

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        state, prior = self._state(raw)
        self.previous_potential = potential(state)
        return state, dict(info, prior_action=prior)

    def step(self, action):
        if self.previous_potential is None:
            raise RuntimeError('reset() is required before stepping')
        action = operator.index(action)
        if not 0 <= action < 9:
            raise ValueError('Expected an official discrete action in0..8')
        raw, official_reward, terminated, truncated, info = self.env.step(action)
        state, prior = self._state(raw)
        official_flags = dict(terminated=bool(terminated), truncated=bool(truncated))
        outcome = info.get('outcome')
        true_end = bool(terminated or (truncated and outcome == 'timeout'))
        utility = 0.
        if true_end:
            if outcome == 'kill': utility = 1.+.05*float(state[19])
            elif outcome == 'died': utility = -1.
            elif outcome not in ('mutual', 'timeout'):
                raise ValueError('Unexpected official terminal outcome')
        next_potential = 0. if true_end else potential(state)
        bonus = self.gamma*next_potential-self.previous_potential if self.shaping else 0.
        if true_end:
            terminated, truncated = True, False
        self.previous_potential = None if terminated or truncated else next_potential
        info = dict(info, league_episode=dict(self.env.episode_metadata), prior_action=prior,
                    learning_utility=utility, shaping_reward=bonus, official_reward=official_reward,
                    official_episode_flags=official_flags)
        return state, utility+bonus, bool(terminated), bool(truncated), info
