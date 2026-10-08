"""Vanilla DQN on experimental GPU flight batches; not an archived SB3 resume."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from ..env import BatchedFairFight, BACKEND_COMMIT
from ..geometry import features, ace_actions
from ..run_utils import run_record, write_json
from ..train import digest


class QNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(31, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 9))


class Replay:
    """GPU ring with copied transitions; capacity and position count individual samples."""
    def __init__(self, capacity, device='cuda'):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.capacity, self.position, self.size = capacity, 0, 0
        self.obs = torch.empty(capacity, 31, device=device)
        self.next_obs = torch.empty_like(self.obs)
        self.actions = torch.empty(capacity, device=device, dtype=torch.long)
        self.rewards = torch.empty(capacity, device=device)
        self.done = torch.empty(capacity, device=device, dtype=torch.bool)

    def add(self, obs, actions, rewards, next_obs, done):
        n = len(obs)
        if not 1 <= n <= self.capacity:
            raise ValueError('An insertion must fit within the replay capacity')
        index = (torch.arange(n, device=self.obs.device) + self.position) % self.capacity
        for target, value in zip((self.obs, self.actions, self.rewards, self.next_obs, self.done),
                                 (obs, actions, rewards, next_obs, done)):
            target[index] = value
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, n):
        if self.size == 0 or n < 1:
            raise ValueError('Sampling requires positive batch size and nonempty replay')
        index = torch.randint(self.size, (n,), device=self.obs.device)
        return tuple(x[index] for x in (self.obs, self.actions, self.rewards, self.next_obs, self.done))


def td_targets(reward, done, next_q, gamma=.999):
    return reward + gamma * torch.where(done, 0., next_q.max(-1).values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--envs', type=int, default=1024)
    p.add_argument('--steps', type=int, default=4096, help='Vector decisions; total samples = steps * envs')
    p.add_argument('--buffer-size', type=int, default=262144)
    p.add_argument('--learning-starts', type=int, default=32768, help='Collected transitions before updates')
    p.add_argument('--train-freq', type=int, default=4, help='Vector decisions per optimizer step')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--target-updates', type=int, default=100, help='Optimizer steps between target copies')
    p.add_argument('--exploration-samples', type=int, default=1048576)
    p.add_argument('--seed', type=int, default=701)
    a = p.parse_args(argv)
    if min(a.envs, a.steps, a.buffer_size, a.learning_starts, a.train_freq,
           a.batch_size, a.target_updates, a.exploration_samples) < 1:
        p.error('All sizes must be positive')
    if a.envs < 2 or a.buffer_size < a.envs:
        p.error('Use at least two environments and enough replay for one vector step')
    if a.learning_starts > a.envs * (a.steps // a.train_freq) * a.train_freq:
        p.error('Run is too short to perform an optimizer update')
    with run_record(a, 'dqn', dict(backend_commit=BACKEND_COMMIT, gamma=.999,
                    learning_rate=3e-4, epsilon_initial=1., epsilon_final=.05,
                    opponent='tensor Ace, closed vertical', reward='win +1, other terminal -0.2, otherwise 0')) as config:
        setup = time.perf_counter()
        env = BatchedFairFight(a.envs, seed=a.seed)
        env.capture()
        model = QNetwork().cuda()
        target = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        replay = Replay(a.buffer_size)
        initial_hash = digest(model)
        row = torch.arange(a.envs, device='cuda')
        seat = row % 2
        invalid = env.invalid.any().clone()
        episodes = torch.zeros((), device='cuda', dtype=torch.long)
        wins = torch.zeros_like(episodes)
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup
        start = time.perf_counter()
        updates = copies = 0
        history = []
        for step in range(1, a.steps + 1):
            epsilon = 1. - .95 * min(1., ((step - 1) * a.envs) / a.exploration_samples)
            with torch.no_grad():
                obs = features(env.obs[row, seat]).float()
                greedy = model(obs).argmax(-1)
                action = torch.where(torch.rand(a.envs, device='cuda') < epsilon,
                                     torch.randint(9, (a.envs,), device='cuda'), greedy)
                both = ace_actions(env.raw)
                both[row, seat] = action
                env.step(both)
                won = ((seat == 0) & (env.outcome == 1)) | ((seat == 1) & (env.outcome == -1))
                rewards = torch.where(env.done, torch.where(won, 1., -.2), 0.)
                # Copy terminal next states and done flags before reset mutates live buffers.
                replay.add(obs, action, rewards, features(env.obs[row, seat]).float(), env.done)
                episodes.add_(env.done.sum())
                wins.add_(won.sum())
                invalid.logical_or_(env.invalid.any())
                env.reset_done()
                invalid.logical_or_(env.invalid.any())
            if step % a.train_freq == 0 and step * a.envs >= a.learning_starts:
                if bool(invalid):
                    raise RuntimeError('Invalid simulation; DQN run rejected')
                x, action, reward, nx, done = replay.sample(a.batch_size)
                with torch.no_grad():
                    y = td_targets(reward, done, target(nx))
                predicted = model(x).gather(1, action[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(predicted, y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite DQN loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                if updates % a.target_updates == 0:
                    target.load_state_dict(model.state_dict())
                    copies += 1
            if step % 256 == 0 or step == a.steps:
                torch.cuda.synchronize()
                record = dict(vector_steps=step, samples=step*a.envs, seconds=time.perf_counter()-start,
                              optimizer_steps=updates, target_copies=copies, epsilon=epsilon,
                              replay_size=replay.size, episodes=int(episodes), wins=int(wins))
                history.append(record)
                write_json(a.out/'progress.json', record)
                print(json.dumps(record), flush=True)
        if bool(invalid) or updates == 0 or not all(bool(torch.isfinite(x).all()) for x in model.parameters()):
            raise RuntimeError('Finite valid DQN learning was not verified')
        final_hash = digest(model)
        if initial_hash == final_hash:
            raise RuntimeError('DQN weights did not change')
        metrics = dict(history[-1], setup_seconds=setup_seconds, invalid_episodes=0,
                       weights_changed=True, initial_weights_sha256=initial_hash,
                       final_weights_sha256=final_hash, all_parameters_cuda=all(x.is_cuda for x in model.parameters()),
                       peak_gpu_bytes=torch.cuda.max_memory_allocated())
        metrics['samples_per_second'] = metrics['samples']/metrics['seconds']
        torch.save(dict(model=model.state_dict(), target=target.state_dict(), optimizer=optimizer.state_dict(),
                        config=config, metrics=metrics, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), a.out/'policy.pt')
        write_json(a.out/'results.json', dict(metrics=metrics, history=history))
        return metrics
