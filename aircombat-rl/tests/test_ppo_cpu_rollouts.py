"""GPU PPO updates must survive temporary CPU policy inference."""
import pytest
import torch
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file

from tools.ppo_cpu_rollouts import cpu_rollouts


def test_dqn_is_not_silently_moved_every_four_steps():
    model = DQN('MlpPolicy', 'CartPole-v1', device='cpu', buffer_size=32)
    try:
        with pytest.raises(ValueError, match='PPO'):
            with cpu_rollouts(model):
                pass
    finally:
        model.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_callback_checkpoint_does_not_capture_temporary_methods(tmp_path):
    model = PPO('MlpPolicy', 'CartPole-v1', device='cuda', seed=4,
                n_steps=16, batch_size=8, n_epochs=1)
    checkpoint = tmp_path / 'during_rollout.zip'

    class SaveCollection(BaseCallback):
        def _on_step(self):
            if self.num_timesteps == 33:
                self.expected = {k: v.detach().cpu().clone()
                                 for k, v in self.model.policy.state_dict().items()}
                self.model.save(checkpoint)
            return True

    restored = None
    try:
        model.learn(32)
        callback = SaveCollection()
        with cpu_rollouts(model):
            model.learn(32, reset_num_timesteps=False, callback=callback)
        data, _, _ = load_from_zip_file(checkpoint)
        assert 'collect_rollouts' not in data
        assert '_excluded_save_params' not in data
        assert '_excluded_save_params' not in model.__dict__
        restored = PPO.load(checkpoint, env=gym.make('CartPole-v1'), device='cuda')
        for key, actual in restored.policy.state_dict().items():
            torch.testing.assert_close(actual.cpu(), callback.expected[key], rtol=0, atol=0)
        restored.learn(16)
        assert all(torch.isfinite(p).all() for p in restored.policy.parameters())
        assert restored.policy.device.type == 'cuda'
    finally:
        model.env.close()
        if restored is not None:
            restored.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cpu_collection_gpu_updates_and_failure_restoration(tmp_path):
    model = PPO('MlpPolicy', 'CartPole-v1', device='cuda', seed=3,
                n_steps=16, batch_size=8, n_epochs=1)
    class CheckCollection(BaseCallback):
        def _on_step(self):
            assert self.model.device.type == 'cpu'
            assert self.model.policy.device.type == 'cpu'
            return True
    class FailCollection(CheckCollection):
        def _on_step(self):
            super()._on_step()
            raise RuntimeError('interrupted rollout')
    original = model.collect_rollouts
    try:
        model.learn(32)  # populate GPU Adam state before the transfers
        params = list(model.policy.parameters())
        before = [p.detach().clone() for p in params]
        with cpu_rollouts(model):
            model.learn(32, reset_num_timesteps=False, callback=CheckCollection())
            assert model.policy.device.type == model.device.type == 'cuda'
        assert model.collect_rollouts == original
        assert 'collect_rollouts' not in model.__dict__  # keep SB3 save data free of bound methods
        assert all(a is b for a, b in zip(params, model.policy.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(before, params))
        for value in model.policy.optimizer.state.values():
            assert value['exp_avg'].device.type == 'cuda'
        with pytest.raises(RuntimeError, match='interrupted rollout'):
            with cpu_rollouts(model):
                model.learn(16, reset_num_timesteps=False, callback=FailCollection())
        assert model.collect_rollouts == original
        assert model.policy.device.type == model.device.type == 'cuda'
        model.learn(16)  # normal training remains usable after context exit
        model.save(tmp_path/'policy')
        restored = PPO.load(tmp_path/'policy.zip', device='cuda')
        assert restored.num_timesteps == 16
        for left, right in zip(model.policy.parameters(), restored.policy.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        model.env.close()
