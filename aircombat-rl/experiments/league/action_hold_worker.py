"""Semi-Markov action repetition over unchanged official physical steps."""
import operator
import gymnasium as gym
from experiments.league.strategy_worker import PeriodicArchiveFight
from experiments.league.residual_worker import LeagueWorker
from tools.league_matches import Actor
from tools import league_thread_benchmark as io


class HoldDecisions(gym.Wrapper):
    def __init__(self,env,hold_steps,physical_gamma):
        super().__init__(env)
        self.hold_steps=operator.index(hold_steps);self.physical_gamma=float(physical_gamma)
        if self.hold_steps<1 or not 0<self.physical_gamma<=1:raise ValueError('Invalid action duration/discount')
        self.physical_steps=0;self.decision_steps=0;self.episode_physical_steps=0

    def reset(self,**kwargs):
        self.episode_physical_steps=0
        return self.env.reset(**kwargs)

    def step(self,action):
        total=0.;steps=0
        for i in range(self.hold_steps):
            obs,reward,terminated,truncated,info=self.env.step(action)
            total+=self.physical_gamma**i*float(reward);steps+=1
            if terminated or truncated:break
        self.physical_steps+=steps;self.decision_steps+=1;self.episode_physical_steps+=steps
        return obs,total,terminated,truncated,dict(info,physical_steps=steps,episode_physical_steps=self.episode_physical_steps,decision_hold_steps=self.hold_steps)

    def physical_status(self):
        return dict(hold_steps=self.hold_steps,physical_steps=self.physical_steps,decision_steps=self.decision_steps,episode_physical_steps=self.episode_physical_steps,physical_gamma=self.physical_gamma)


def make_hold_worker(plan,index,instrumented):
    if instrumented:raise ValueError('Not a timing instrumentation worker')
    import torch
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    archive=PeriodicArchiveFight(plan['teacher'],plan['opponents'],plan['probabilities'],plan['seed']*100+index,plan['ic_band'],index,plan['workers'],'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    held=HoldDecisions(LeagueWorker(archive,gamma=plan['physical_gamma']),plan['hold_steps'],plan['physical_gamma'])
    class Profile(gym.Wrapper):
        def profile_status(self):
            return dict(worker_index=index,cached_opponents=len(archive.cache),memory=io.own_memory(),coverage_period=archive.sampler.period,hold_steps=held.hold_steps)
    return Profile(Monitor(held))
