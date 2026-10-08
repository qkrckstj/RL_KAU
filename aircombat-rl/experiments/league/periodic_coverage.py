"""Explicit archive-coverage frequency for a future controlled curriculum.

Does not change observations, rewards, initial conditions, or flight physics.
Period2 reproduces the original sampler exactly; period4 leaves three of four
episode choices available to the frozen weighted opponent distribution.
"""
import operator
from experiments.league.residual_env import CoverageSampler


class PeriodicCoverageSampler(CoverageSampler):
    def __init__(self, probabilities, seed, period):
        self.period=operator.index(period)
        if self.period not in (2,4):raise ValueError('Qualified periods are2 or4')
        super().__init__(probabilities,seed)

    def next(self):
        if self.draws%self.period==0:
            if not self.coverage:
                self.coverage=self.rng.permutation(len(self.probabilities)).tolist()
            chosen=self.coverage.pop()
        else:
            chosen=int(self.rng.choice(len(self.probabilities),p=self.probabilities))
        self.draws+=1
        return chosen
