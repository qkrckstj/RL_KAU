import gymnasium as gym
import numpy as np
import pytest

from experiments.league import residual_env as module
from experiments.league.residual_features import features
from experiments.plan_a.core import potential as old_potential


def raw(clock=120., health=1.):
    x = np.zeros(39, np.float32)
    x[2] = x[17] = 6000.; x[4] = 240.; x[18] = 280.; x[16] = 4000.
    x[30:32] = [health, .7]; x[36:38] = 10000.; x[38] = clock
    return x


def test_coverage_reproducibility_and_independent_ic_stream(monkeypatch):
    a = module.CoverageSampler([.8, .1, .1], 81)
    b = module.CoverageSampler([.8, .1, .1], 81)
    first = [a.next() for _ in range(60)]
    assert first == [b.next() for _ in range(60)]
    for start in range(0, 60, 6): assert set(first[start:start+6:2]) == {0, 1, 2}
    with pytest.raises(ValueError): module.CoverageSampler([1., np.nan], 1)

    class FakeActor:
        mode = 'discrete'
        def __init__(self, spec): self.spec=spec; self.seeds=[]; self.resets=0
        def seed(self, seed, seat): self.seeds.append((seed, seat))
        def reset(self): self.resets += 1
    monkeypatch.setattr(module, 'Actor', FakeActor)
    def init(env, action_mode, seat): env.seat=seat
    def reset(env, seed, options):
        env.initial, env.foe = env.sample(np.random.default_rng(seed))
        env.foe.reset()
        return raw(), {}
    monkeypatch.setattr(module.FairFightEnv, '__init__', init)
    monkeypatch.setattr(module.FairFightEnv, 'reset', reset)
    foes = [{'id':str(i),'kind':'fixed','action':i} for i in range(3)]
    env = module.ArchiveFight(foes[0], foes, [.8,.1,.1], 81, 1000, 1, 4, 'blue', ic_count=9)
    env.reset(); initial_a=env.initial
    expected,_ = module.FairFightEnv.sample(env, np.random.default_rng(1001))
    assert initial_a == expected
    assert env.episode_metadata['episode_seed'] == 1001
    assert env.teacher.seeds[-1] == (1001,1) and env.foe.seeds[-1] == (1001,0)
    env.reset(); assert env.episode_metadata['episode_seed'] == 1005
    with pytest.raises(RuntimeError, match='band exhausted'): env.reset()
    env.reset(seed=999)
    assert env.initial == initial_a  # Opponent RNG reseeding does not perturb ICs.
    assert env.teacher.resets == 3


class FakeFight(gym.Env):
    action_space = gym.spaces.Discrete(9)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (39,), dtype=np.float32)
    def __init__(self, ending='kill'):
        self.ending=ending; self.actions=[]; self.prior_calls=0
        self.episode_metadata=dict(episode_seed=1000,opponent_id='example',seat='red')
    def teacher_action(self, obs): self.prior_calls+=1; return 4
    def reset(self, **kwargs): self.actions=[]; self.prior_calls=0; return raw(), {}
    def step(self, action):
        self.actions.append(action); end=len(self.actions)==2
        truncated=end and self.ending in ('timeout','collector')
        terminated=end and not truncated
        return raw(120.-len(self.actions),.5), 0., terminated, truncated, ({'outcome':self.ending,'won':self.ending=='kill'} if end else {})


def test_reward_shaping_telescope_and_official_actions_flags_preserved():
    base=FakeFight(); env=module.PriorTrainingEnv(base,gamma=.9,shaping=True)
    state,_=env.reset(); initial_phi=module.potential(state)
    assert initial_phi == old_potential(state[:31])
    a=env.step(7); b=env.step(1)
    assert base.actions == [7,1] and base.prior_calls == 3
    assert b[2:4] == (True,False) and b[4]['outcome']=='kill'
    assert b[4]['official_episode_flags']=={'terminated':True,'truncated':False}
    assert b[4]['learning_utility']==pytest.approx(1.025)
    assert a[4]['shaping_reward']+.9*b[4]['shaping_reward']==pytest.approx(-initial_phi)
    with pytest.raises(RuntimeError): env.step(1)


@pytest.mark.parametrize('ending,utility,flags',[('died',-1.,(True,False)),('mutual',0.,(True,False)),('timeout',0.,(True,False)),('collector',0.,(False,True))])
def test_terminal_utility_and_timeout_bootstrap_are_explicit(ending,utility,flags):
    env=module.PriorTrainingEnv(FakeFight(ending),shaping=False)
    env.reset(); env.step(2); _,reward,terminated,truncated,info=env.step(3)
    assert reward==utility and (terminated,truncated)==flags
    assert info['outcome']==ending
    if ending=='timeout': assert info['official_episode_flags']=={'terminated':False,'truncated':True}
