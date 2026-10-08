"""The performance audit must measure real learning without touching archives."""
import pytest
import torch

from tools import runtime_audit


def test_cuda_unavailable_is_not_a_cpu_benchmark(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        runtime_audit.neural_case('dqn', 'cuda', 2048, 0)


def test_neural_audit_counts_real_updates_and_preserves_budget():
    result = runtime_audit.neural_case('dqn', 'cpu', 20480, 0)
    assert result['actual_steps'] == 20480
    assert result['optimizer_steps'] == 4870
    assert result['initial_weights_sha256'] != result['final_weights_sha256']
    assert result['train_seconds'] >= result['rollout_seconds'] >= result['environment_seconds'] > 0
    assert result['update_seconds'] > 0
    assert result['all_parameters_on_requested_device']


def test_parallelism_requires_explicit_ppo_experiment():
    with pytest.raises(ValueError, match='PPO'):
        runtime_audit.neural_case('ddqn', 'cpu', 2048, 0, n_envs=2)


def test_existing_results_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        runtime_audit.main(['--out', str(tmp_path), '--algorithms', 'dqn',
                            '--devices', 'cpu', '--steps', '2048', '--repeats', '1'])
