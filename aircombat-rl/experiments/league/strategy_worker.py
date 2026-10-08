"""Matched curriculum workers; official flight/observation/reward are inherited."""
import gymnasium as gym
from experiments.league.residual_env import ArchiveFight
from experiments.league.periodic_coverage import PeriodicCoverageSampler
from experiments.league.residual_worker import LeagueWorker
from tools.league_matches import Actor
from tools import league_thread_benchmark as io


class PeriodicArchiveFight(ArchiveFight):
    def __init__(self,*args,coverage_period,**kwargs):
        self.coverage_period=coverage_period
        super().__init__(*args,**kwargs)
        self.sampler=PeriodicCoverageSampler(self.probabilities,self.stream_seed,self.coverage_period)

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.sampler=PeriodicCoverageSampler(self.probabilities,int(seed),self.coverage_period)
            self.episode_index=0
        return super().reset(seed=None,options=options)


def make_strategy_worker(plan,index,instrumented):
    if instrumented:raise ValueError('This worker is not a timing study')
    import torch
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    archive=PeriodicArchiveFight(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],'red' if index%2==0 else 'blue',
        coverage_period=plan['coverage_period'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    worker=Monitor(LeagueWorker(archive,gamma=plan['ppo']['gamma']))
    class Profile(gym.Wrapper):
        def profile_status(self):
            return dict(worker_index=index,cached_opponents=len(archive.cache),memory=io.own_memory(),coverage_period=archive.sampler.period)
    return Profile(worker)
