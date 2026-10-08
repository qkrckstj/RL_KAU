"""Contracts for GPU replay learning and paired population search."""
import json

import pytest
import torch


def test_replay_wrap_copies_transitions_and_samples_only_live_rows():
    from experiments.gpu_sim.dqn.train import Replay
    replay = Replay(5, device='cpu')
    for first in (0, 3):
        obs = torch.arange(first, first + 3).float()[:, None].expand(-1, 31).clone()
        next_obs = obs + 10
        done = torch.tensor([True, False, True])
        replay.add(obs, torch.arange(3), obs[:, 0], next_obs, done)
        obs.fill_(-999)
        next_obs.fill_(-999)
        done.fill_(False)
    assert replay.size == 5 and replay.position == 1
    assert sorted(replay.obs[:, 0].tolist()) == [1, 2, 3, 4, 5]
    torch.testing.assert_close(replay.next_obs, replay.obs + 10)
    assert replay.done.tolist() == [True, False, True, True, False]
    batch = replay.sample(100)
    assert set(batch[0][:, 0].tolist()) == {1., 2., 3., 4., 5.}
    torch.testing.assert_close(batch[3], batch[0] + 10)
    with pytest.raises(ValueError):
        replay.add(torch.zeros(6, 31), torch.zeros(6, dtype=torch.long),
                   torch.zeros(6), torch.zeros(6, 31), torch.zeros(6, dtype=torch.bool))
    with pytest.raises(ValueError):
        Replay(4, device='cpu').sample(1)


def test_dqn_targets_stop_bootstrap_at_terminal_transition():
    from experiments.gpu_sim.dqn.train import td_targets
    result = td_targets(torch.tensor([1., 2.]), torch.tensor([True, False]),
                        torch.tensor([[10., 3.], [4., 20.]]), gamma=.9)
    torch.testing.assert_close(result, torch.tensor([1., 20.]))


def test_cem_elites_fit_smoothed_distribution_with_floor():
    from experiments.gpu_sim.cem.train import fit_elites
    population = torch.tensor([[.1], [.3], [.9]])
    mean, std, order = fit_elites(population, torch.tensor([0., 2., 1.]),
                                  torch.tensor([.5]), torch.tensor([.2]), 2, .05)
    torch.testing.assert_close(mean, torch.tensor([.55]))
    torch.testing.assert_close(std, torch.tensor([.25]))
    assert order.tolist() == [1, 2]
    _, std, _ = fit_elites(population[:1], torch.ones(1), torch.tensor([.1]),
                           torch.zeros(1), 1, .05)
    assert std.item() == pytest.approx(.05)


def test_two_candidate_cem_keeps_a_random_challenger():
    from experiments.gpu_sim.cem.train import sample_population
    torch.manual_seed(71)
    mean = torch.full((17,), .5)
    incumbent = torch.full((17,), .4)
    population = sample_population(mean, torch.full_like(mean, .1), incumbent, 2)
    torch.testing.assert_close(population[0], incumbent)
    assert not torch.equal(population[1], mean)
    assert not torch.equal(population[1], incumbent)
    assert ((population >= 0) & (population <= 1)).all()


def test_cem_layout_pairs_each_candidate_opponent_and_ic_in_both_seats():
    from experiments.gpu_sim.cem.train import paired_layout
    candidate, opponent, ic, seat = paired_layout(3, 2, 4, device='cpu')
    rows = list(zip(candidate.tolist(), opponent.tolist(), ic.tolist(), seat.tolist()))
    assert len(rows) == 48
    assert set(rows) == {(c, o, i, s) for c in range(3) for o in range(2)
                         for i in range(4) for s in range(2)}
    assert rows[:4] == [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1)]


def test_cem_scores_seat_relative_verdicts_and_rejects_incomplete_or_invalid():
    from experiments.gpu_sim.cem.train import game_scores
    result = game_scores(torch.tensor([1, 1, -1, -1, 2, 3]), torch.tensor([0, 1, 0, 1, 0, 1]))
    torch.testing.assert_close(result, torch.tensor([1., 0., 0., 1., .5, .5]))
    for bad in (0, 4):
        with pytest.raises(RuntimeError):
            game_scores(torch.tensor([bad]), torch.tensor([0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_dqn_updates_target_and_exports_official_policy(tmp_path):
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.dqn.train import main
    from experiments.gpu_sim.dqn.policy import Policy
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    out = tmp_path / 'dqn'
    metrics = main(['--out', str(out), '--envs', '8', '--steps', '8',
                    '--buffer-size', '32', '--learning-starts', '16', '--train-freq', '2',
                    '--batch-size', '16', '--target-updates', '2'])
    assert metrics['samples'] == 64 and metrics['optimizer_steps'] == 4
    assert metrics['target_copies'] == 2 and metrics['weights_changed']
    assert metrics['invalid_episodes'] == 0 and metrics['all_parameters_cuda']
    saved = torch.load(out / 'policy.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(saved['model'][k], saved['target'][k]) for k in saved['model'])
    env = FairFightEnv()
    try:
        raw, _ = env.reset(seed=33030000)
        policy = Policy(out / 'policy.pt')
        assert 0 <= policy.act(raw) < 9
    finally:
        env.close()
    with pytest.raises(FileExistsError):
        main(['--out', str(out)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_dqn_collector_preserves_real_timeout_transition_before_reset(tmp_path, monkeypatch):
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.dqn import train
    real_env, real_replay = train.BatchedFairFight, train.Replay
    retained = {}

    def near_timeout(*args, **kwargs):
        env = real_env(*args, **kwargs)
        env.time.fill_(119.96)
        env.obs[:, :, 38] = .04
        retained['env'] = env
        return env

    def keep_replay(*args, **kwargs):
        retained['replay'] = real_replay(*args, **kwargs)
        return retained['replay']

    monkeypatch.setattr(train, 'BatchedFairFight', near_timeout)
    monkeypatch.setattr(train, 'Replay', keep_replay)
    result = train.main(['--out', str(tmp_path/'timeout'), '--envs', '2', '--steps', '4',
                         '--buffer-size', '8', '--learning-starts', '2', '--train-freq', '1',
                         '--batch-size', '4'])
    replay = retained['replay']
    assert result['episodes'] == 2 and replay.done[:2].all()
    assert not replay.done[2:].any()
    assert (replay.obs[:2, 27] > 0).all()
    assert (replay.next_obs[:2, 27] <= 0).all()
    assert (retained['env'].obs[:, :, 38] > 119).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_cem_completes_paired_games_and_updates_distribution(tmp_path):
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.cem.train import main
    from experiments.gpu_sim.cem.policy import Policy
    from experiments.league.controller import LOW, HIGH
    out = tmp_path / 'cem'
    metrics = main(['--out', str(out), '--population', '3', '--elites', '2',
                    '--ics', '1', '--generations', '1'])
    assert metrics['games'] == 12 and metrics['distribution_updates'] == 1
    assert metrics['invalid_games'] == 0 and metrics['distribution_changed']
    generation = json.loads((out / 'generation_000.json').read_text())
    assert len(generation['outcomes']) == 3
    assert torch.tensor(generation['outcomes']).shape == (3, 2, 1, 2)
    assert (torch.tensor(generation['outcomes']) != 0).all()
    policy = Policy(out / 'policy_net.json')
    assert ((policy.parameters >= LOW) & (policy.parameters <= HIGH)).all()
    saved = torch.load(out / 'search_state.pt', map_location='cpu', weights_only=True)
    assert saved['mean'].shape == (17,) and saved['std'].min() >= .035
    assert saved['mean'].device.type == 'cpu'
    with pytest.raises(FileExistsError):
        main(['--out', str(out)])
