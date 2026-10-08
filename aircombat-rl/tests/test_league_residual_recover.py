import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.league.residual_ppo_policy import ResidualPriorPolicy
from tools.league_residual_recover import load_continuation


class Tiny(gym.Env):
    observation_space=gym.spaces.Box(-1.,1.,(40,),dtype=np.float32)
    action_space=gym.spaces.Discrete(9)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        x=np.zeros(40,np.float32);x[35]=1.
        return x,{}
    def step(self,action):
        self.steps+=1;x=np.zeros(40,np.float32);x[35]=1.
        return x,float(action==4),self.steps==7,False,{}


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_preserves_policy_and_optimizer_but_resets_stream_and_counter(tmp_path):
    torch.set_num_threads(1)
    oldenv=DummyVecEnv([Tiny,Tiny]);newenv=DummyVecEnv([Tiny])
    try:
        old=PPO(ResidualPriorPolicy,oldenv,seed=11,device='cpu',n_steps=8,batch_size=8,n_epochs=1)
        old.learn(64);old.save(tmp_path/'old.zip')
        new,prefix=load_continuation(tmp_path/'old.zip',newenv,22,16)
        assert prefix==64 and new.num_timesteps==0 and new._last_obs is None
        assert new.n_envs==1 and new.n_steps==16
        assert equal(old.policy.state_dict(),new.policy.state_dict())
        assert equal(old.policy.optimizer.state_dict(),new.policy.optimizer.state_dict())
        assert newenv._seeds==[22]
        new.learn(32,reset_num_timesteps=False)
        assert new.num_timesteps==32
        assert not equal(old.policy.state_dict(),new.policy.state_dict())
    finally:oldenv.close();newenv.close()
