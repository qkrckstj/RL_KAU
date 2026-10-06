from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer
from tools.resume_dqn import resize_replay


@pytest.mark.parametrize("count", [2,7])
def test_resize_preserves_chronology_and_transition_fields(count):
    buf = ReplayBuffer(5,gym.spaces.Box(-100,100,(1,),dtype=np.float32),gym.spaces.Discrete(2))
    for i in range(count):
        buf.add(np.array([[i]]),np.array([[i+1]]),np.array([i%2]),np.array([i]),
                np.array([i%2]),[{"TimeLimit.truncated":bool(i%2)}])
    start = max(0,count-5)
    small, big = SimpleNamespace(replay_buffer=buf), SimpleNamespace(replay_buffer=buf)
    resize_replay(small,5)
    resize_replay(big,10)
    for name in ("observations","next_observations","actions","rewards","dones","timeouts"):
        np.testing.assert_array_equal(getattr(small.replay_buffer,name)[:min(count,5)],
                                      getattr(big.replay_buffer,name)[:min(count,5)])
    np.testing.assert_array_equal(big.replay_buffer.rewards[:min(count,5),0],np.arange(start,count))
    assert buf.size() == min(count,5) and big.replay_buffer.size() == min(count,5)
    with pytest.raises(ValueError):
        resize_replay(big,1)
