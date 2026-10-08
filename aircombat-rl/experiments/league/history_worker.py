"""Public-clock history wrapper; the existing official fight/reward is unchanged."""
import hashlib
import numpy as np
import gymnasium as gym
from experiments.league.history_features import HistoryFeatures,HISTORY_DIM
from experiments.league.strategy_worker import make_strategy_worker as make_base_worker


class HistoryOrPlain(gym.Wrapper):
    def __init__(self,env,enabled):
        super().__init__(env);self.enabled=enabled;self.history=HistoryFeatures()
        if enabled:self.observation_space=gym.spaces.Box(-1.,1.,shape=(HISTORY_DIM,),dtype=np.float32)

    def transform(self,state,info):
        # This is the exact float32 clock channel in the public39-channel obs,
        # not inferred geometry or an opponent/environment identifier.
        return self.history.update(state,-float(np.float32(info['t_remaining']))) if self.enabled else state

    def reset(self,**kwargs):
        self.history.reset();state,info=self.env.reset(**kwargs)
        return self.transform(state,info),info

    def step(self,action):
        state,reward,terminated,truncated,info=self.env.step(action)
        return self.transform(state,info),reward,terminated,truncated,info

    def history_status(self):
        payload=b''.join(np.float64(t).tobytes()+x.tobytes() for t,x in self.history.samples)
        return dict(enabled=self.enabled,samples=len(self.history.samples),sha256=hashlib.sha256(payload).hexdigest())

    def profile_status(self):
        return dict(self.env.get_wrapper_attr('profile_status')(),feature_history=self.enabled,observation_dim=self.observation_space.shape[0])


def make_history_worker(plan,index,instrumented):
    return HistoryOrPlain(make_base_worker(plan,index,instrumented),plan['feature_history'])
