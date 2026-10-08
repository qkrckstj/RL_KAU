"""Independent scalar contracts for the optional GPU simulator."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aircombat_gym.core.control.autopilot import Autopilot
from aircombat_gym.wvr.engagement import look
from aircombat_gym.wvr.obs import mirror
from experiments.plan_a.core import State, specification
from experiments.league.controller import Policy, LOW, HIGH
from experiments.gpu_sim.control import TensorAutopilot
from experiments.gpu_sim.geometry import engagement, observe, features, reactive_actions, ace_actions
from aircombat_gym.wvr.baselines import Ace


def aircraft(n=32):
    g = torch.Generator().manual_seed(71)
    x = torch.randn(n, 2, 15, generator=g, dtype=torch.float64)
    x[..., :2] *= 1000
    x[..., 2] = 6096 + x[..., 2] * 20
    x[..., 3:5] *= 150
    x[..., 5] *= 3
    x[..., 8] = x[..., 8].abs() + 1
    x[..., 9:12] *= .4
    x[..., 12:] *= .1
    return x


def test_autopilot_matches_original_including_integrators():
    raw = aircraft().reshape(-1, 15)
    tensor = TensorAutopilot(len(raw), 'cpu', torch.float64)
    scalar = [Autopilot() for _ in raw]
    heading = torch.linspace(-4, 4, len(raw), dtype=torch.float64)
    speed = torch.linspace(250, 700, len(raw), dtype=torch.float64)
    for _ in range(15):
        actual = tensor.update(raw, heading, speed)
        expected = []
        for x, pilot, psi, v in zip(raw.tolist(), scalar, heading.tolist(), speed.tolist()):
            s = SimpleNamespace(h_ft=x[2]/.3048, h_dot=x[5],
                v_kt=math.sqrt(sum(a*a for a in x[3:6]))/.514444,
                phi=x[9], psi=x[11], p=x[12], r=x[14], nz=x[8])
            out = pilot.update(s, psi, v, 1/120)
            expected.append([out.aileron, out.elevator, out.rudder, out.throttle])
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(tensor.i_nz, torch.tensor([p._i_nz for p in scalar], dtype=torch.float64))
    torch.testing.assert_close(tensor.i_v, torch.tensor([p._i_v for p in scalar], dtype=torch.float64))


def test_geometry_and_public_features_match_original():
    raw = aircraft()
    actual = engagement(raw)
    for i, pair in enumerate(raw.tolist()):
        for side in range(2):
            def kin(x):
                return SimpleNamespace(x=x[0], y=x[1], h=x[2], vx=x[3], vy=x[4], vz=x[5], psi=x[11], theta=x[10])
            ref = look(kin(pair[side]), kin(pair[1-side]), 30.)
            for name in ('ata', 'ata_signed', 'ata_lead', 'aa', 'r', 'r_dot', 'in_wez', 'damage_rate'):
                assert float(actual[name][i, side]) == pytest.approx(float(getattr(ref, name)), abs=1e-9)
    health = torch.ones(len(raw), 2, dtype=torch.float64)
    track = torch.zeros_like(health)
    obs = observe(raw, health, track, actual['in_wez'], torch.full((len(raw),), 120.))
    state = State(specification('observed_ppo'))
    for pair in obs:
        np.testing.assert_array_equal(mirror(pair[0].float().numpy()), pair[1].float().numpy())
        for x in pair:
            np.testing.assert_allclose(features(x.float()).numpy(), state(x.float().numpy()), atol=2e-7)


def test_tensor_reactive_policy_matches_archived_scalar_controller():
    raw = aircraft(100)
    health = torch.ones(100, 2, dtype=torch.float64)
    obs = observe(raw, health, health*.3, engagement(raw)['in_wez'], torch.linspace(0,120,100))
    rng = np.random.default_rng(43)
    for _ in range(8):
        p = rng.uniform(LOW, HIGH)
        actual = reactive_actions(obs, torch.tensor(p, dtype=torch.float64))
        scalar = Policy(parameters=p)
        expected = [[scalar.act(x.numpy()) for x in pair] for pair in obs]
        assert actual.tolist() == expected


def test_ace_snap_threshold_matches_original():
    raw = aircraft(100)
    e = engagement(raw)
    actual = ace_actions(raw)
    for i in range(len(raw)):
        for side in range(2):
            info = dict(range=float(e['r'][i,side]), range_rate=float(e['r_dot'][i,side]),
                        lead_signed=float(e['lead_signed'][i,side]), alt_diff=0.)
            dh,dv,_ = Ace().act(info)
            assert actual[i,side] == int(dh/30+1)*3+int(dv/20+1)


def test_weapon_lock_and_mutual_verdict_match_scalar():
    from experiments.gpu_sim.env import score_step
    h = torch.ones(4,2,dtype=torch.float64)
    tr = torch.zeros_like(h)
    # Match 0: red tracks; match 1: both track; match 2: lock breaks.
    inside = torch.tensor([[True,False],[True,True],[True,False],[False,False]])
    t = torch.zeros(4,dtype=torch.float64)
    for k in range(100):
        if k == 18: inside[2,0] = False
        h,tr,t,done,outcome = score_step(h,tr,t,inside,torch.zeros(4,dtype=torch.bool))
        if k == 18: assert h[0,1] == 1 and tr[2,0] == 0
        if k == 19: assert h[0,1] == pytest.approx(1-.33*.05)
    assert outcome.tolist()[:3] == [1,2,0]
    h = torch.tensor([[.019,0.],[.021,0.]],dtype=torch.float64)
    _,_,_,done,outcome = score_step(h,torch.zeros_like(h),torch.zeros(2),torch.zeros(2,2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    assert outcome.tolist() == [2,1]
    # Timeout at 120 seconds; a simultaneous kill and invalid state take precedence.
    h = torch.tensor([[1.,1.],[1.,0.],[1.,0.]],dtype=torch.float64)
    _,_,_,done,outcome = score_step(h,torch.zeros_like(h),torch.full((3,),119.96,dtype=torch.float64),
        torch.zeros(3,2,dtype=torch.bool),torch.tensor([False,False,True]))
    assert done.all() and outcome.tolist() == [3,1,4]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_env_reset_target_persistence_and_capture():
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.env import BatchedFairFight
    env = BatchedFairFight(4,seed=123)
    before = env.raw.clone()
    action = torch.tensor([[8,0]]*4,device='cuda')
    env.step(action)
    assert torch.isfinite(env.raw).all() and not torch.equal(before,env.raw)
    targets = env.heading.clone(),env.speed.clone()
    env.step(torch.full_like(action,4))
    torch.testing.assert_close(env.heading,targets[0],rtol=0,atol=0)
    torch.testing.assert_close(env.speed,targets[1],rtol=0,atol=0)
    watched = {k:v.clone() for k,v in env.dyn.watched_tensors().items()}
    control = env.control.i_nz.clone()
    env.reset(mask=torch.tensor([True,False,False,False],device='cuda'))
    for k,v in env.dyn.watched_tensors().items():
        torch.testing.assert_close(v[2:],watched[k][2:],rtol=0,atol=0)
    torch.testing.assert_close(env.control.i_nz[2:],control[2:],rtol=0,atol=0)
    initial=env.initial.clone()
    env.reset(initial=initial)
    env.step(action)
    expected=env.obs.clone()
    env.reset(initial=initial)
    saved=[x.clone() for x in env._buffers()]
    env.capture()
    for actual,before in zip(env._buffers(),saved):
        torch.testing.assert_close(actual,before,rtol=0,atol=0)
    env.reset(initial=initial)
    env.step(action)
    torch.testing.assert_close(env.obs,expected,rtol=1e-5,atol=1e-4)
    assert env.obs.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_invalid_flight_is_flagged_and_terminal_observation_is_frozen():
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.env import BatchedFairFight
    env=BatchedFairFight(2)
    env.dyn.rb.uvw[0,0]=float('nan')
    env.step(torch.full((2,2),4,device='cuda'))
    assert env.invalid[0] and env.done[0] and env.outcome[0] == 4
    assert not env.invalid[1]
    assert torch.isfinite(env.obs).all()
    before=env.obs[0].clone()
    env.step(torch.full((2,2),4,device='cuda'))
    torch.testing.assert_close(env.obs[0],before)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_reset_accepts_done_mask_without_aliasing_and_graph_reset_agrees():
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.env import BatchedFairFight
    env=BatchedFairFight(2)
    env.step(torch.tensor([[8,0],[0,8]],device='cuda'))
    env.done[0]=True
    env.reset(mask=env.done)
    assert env.obs[0,0,38] == 120
    assert env.obs[1,0,38] < 120
    env.capture()
    env.step(torch.tensor([[8,0],[0,8]],device='cuda'))
    old=env.obs[1].clone()
    env.done[0]=True
    env.reset_done()
    assert not env.done.any() and env.obs[0,0,38] == 120
    torch.testing.assert_close(env.obs[1],old,rtol=0,atol=0)
