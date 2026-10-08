import subprocess
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tools.league_residual_profile import Timings,tensor_digest,analyze
from tests.test_league_residual_recover import Tiny
from experiments.league.residual_ppo_policy import ResidualPriorPolicy


def test_real_ppo_instrumentation_preserves_updates_actions_and_method_lookup():
    torch.set_num_threads(1)
    runs=[]
    for instrumented in (False,True):
        env=DummyVecEnv([Tiny,Tiny])
        try:
            model=PPO(ResidualPriorPolicy,env,seed=43,device='cpu',n_steps=8,batch_size=8,n_epochs=2)
            timer=Timings();initial=tensor_digest(model.policy.state_dict())
            if instrumented:
                with timer.instrument(model,env):model.learn(128)
                assert 'forward' not in vars(model.policy) and 'train' not in vars(model)
            else:model.learn(128)
            runs.append((initial,tensor_digest(model.policy.state_dict()),model.rollout_buffer.actions.copy()))
            if instrumented:
                assert timer.rows['optimizer_inclusive']['calls']==8
                assert timer.rows['action_value_forward']['calls']==64
                assert timer.rows['wait_receive_stack']['calls']==64
                assert all(r['wall_seconds']>=0 for r in timer.rows.values())
        finally:env.close()
    assert runs[0][:2]==runs[1][:2]
    np.testing.assert_array_equal(runs[0][2],runs[1][2])


def test_waiting_profile_module_does_not_import_learning_stack():
    code="import sys;import tools.league_residual_profile;assert not any(n in sys.modules for n in ('numpy','torch','jsbsim'))"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_profile_rejects_a_changed_trajectory():
    import pytest
    rows=[dict(steps=64,initial_parameters_sha256='same',final_parameters_sha256='same',trajectory_sha256='same') for _ in range(4)]
    rows[-1]['trajectory_sha256']='different'
    with pytest.raises(AssertionError):analyze(rows,64)
