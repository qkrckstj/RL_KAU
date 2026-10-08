"""Reuse a paused PPO environment process for isolated official evaluation."""
from contextlib import contextmanager
import random
import numpy as np
import torch

from experiments.league.residual_env import ArchiveFight, PriorTrainingEnv
from tools.league_matches import duel


@contextmanager
def isolated_rng():
    python = random.getstate()
    numpy = np.random.get_state()
    tensor = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(tensor)


class LeagueWorker(PriorTrainingEnv):
    def evaluate_partition(self, own, opponents, band, n):
        """Called while vector stepping is paused; no training env is reset."""
        with isolated_rng():
            return [duel(own, foe, band, n) for foe in
                    opponents[self.env.worker_index::self.env.workers]]

    def worker_status(self):
        from tools.league_thread_benchmark import own_memory
        return dict(**own_memory(), worker_index=self.env.worker_index,
                    episodes_started=self.env.episode_index,
                    opponent_draws=self.env.sampler.draws,
                    episode=self.env.episode_metadata)


def make_worker(teacher, opponents, probabilities, stream_seed, ic_band,
                worker_index, workers, seat, gamma=.999):
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    env = ArchiveFight(teacher, opponents, probabilities, stream_seed, ic_band,
                       worker_index, workers, seat)
    return Monitor(LeagueWorker(env, gamma=gamma))
