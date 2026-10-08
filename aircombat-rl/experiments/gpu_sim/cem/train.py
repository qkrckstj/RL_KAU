"""GPU CEM search of reactive coefficients, separate from the archived league search."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.league.controller import LOW, HIGH, NAMES, Policy, embed
from ..env import BatchedFairFight, BACKEND_COMMIT
from ..geometry import reactive_actions, ace_actions
from ..run_utils import APP, run_record, write_json

DEFAULT_PARENT = APP / 'experiments/league/bundle/models/final/policy_net.json'
ORIGINAL = APP / 'experiments/league/bundle/models/cem_original/policy_net.json'


def paired_layout(candidates, opponents, initial_conditions, device='cuda'):
    axes = torch.meshgrid(*(torch.arange(n, device=device) for n in
                           (candidates, opponents, initial_conditions, 2)), indexing='ij')
    return tuple(x.flatten() for x in axes)


def game_scores(outcomes, seat):
    valid = (outcomes == 1) | (outcomes == -1) | (outcomes == 2) | (outcomes == 3)
    if not bool(valid.all()):
        raise RuntimeError('Incomplete or invalid CEM games cannot be scored')
    won = ((seat == 0) & (outcomes == 1)) | ((seat == 1) & (outcomes == -1))
    draw = (outcomes == 2) | (outcomes == 3)
    return torch.where(won, 1., torch.where(draw, .5, 0.))


def fit_elites(population, scores, mean, std, elites, minimum_std):
    if not 1 <= elites <= len(population) or not bool(torch.isfinite(scores).all()):
        raise ValueError('Valid elite count and finite scores are required')
    indices = torch.argsort(scores, descending=True, stable=True)[:elites]
    selected = population[indices]
    return (.5*mean + .5*selected.mean(0),
            (.5*std + .5*selected.std(0, unbiased=False)).clamp_min(minimum_std), indices)


def sample_population(mean, std, incumbent, size):
    population = (mean + std*torch.randn(size, 17, device=mean.device)).clamp(0, 1)
    population[0] = incumbent
    if size > 2:
        population[1] = mean
    return population


@torch.no_grad()
def evaluate(env, parameters, original, initial, layout):
    """Each candidate gets each opponent/IC/seat once; no reset until evaluation ends."""
    candidate, opponent, ic, seat = layout
    row = torch.arange(env.n, device=env.device)
    env.reset(initial=initial[ic])
    if bool(env.invalid.any()):
        raise RuntimeError('Invalid initial CEM simulation')
    both = original.expand(env.n, 2, 17).clone()
    both[row, seat] = parameters[candidate]
    effective_steps = torch.zeros(env.n, device=env.device, dtype=torch.long)
    for step in range(1, 2402):
        actions = reactive_actions(env.obs, both)
        ace = ace_actions(env.raw)
        actions[row, 1-seat] = torch.where(opponent == 0, ace[row, 1-seat], actions[row, 1-seat])
        effective_steps.add_(~env.done)
        env.step(actions)
        # Periodic synchronization avoids a CPU round-trip after every decision.
        if step % 64 == 0 and bool(env.done.all()):
            break
    if bool(env.invalid.any()):
        raise RuntimeError('Invalid simulation; CEM generation rejected')
    scores = game_scores(env.outcome, seat)
    health = env.health[row, seat] - env.health[row, 1-seat]
    return dict(scores=scores, health=health, outcomes=env.outcome.clone(),
                effective_steps=effective_steps, dispatched_steps=step*env.n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--elites', type=int, default=8)
    p.add_argument('--ics', type=int, default=4, help='Shared training ICs; games per generation = population * ics * 4')
    p.add_argument('--generations', type=int, default=3)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--initial-std', type=float, default=.14)
    p.add_argument('--minimum-std', type=float, default=.035)
    p.add_argument('--seed', type=int, default=702)
    a = p.parse_args(argv)
    if a.population < 2 or not 1 <= a.elites <= a.population or min(a.ics, a.generations) < 1:
        p.error('Use population >= 2, 1 <= elites <= population and positive IC/generation counts')
    if not 0 < a.minimum_std <= a.initial_std <= 1:
        p.error('Require 0 < minimum-std <= initial-std <= 1 in normalized parameter space')
    parent = Policy(a.parent).parameters.tolist()
    original = json.loads(ORIGINAL.read_text())['parameters']
    if len(original) == 7:
        original = embed(original).tolist()
    with run_record(a, 'cem', dict(backend_commit=BACKEND_COMMIT,
                    parent_parameters=parent, parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
                    original_opponent_parameters=original,
                    opponents=['tensor Ace, closed vertical', 'embedded original CEM'],
                    objective='mean game score + 0.001 * mean final own-minus-foe health',
                    smoothing=.5, final_test_opened=False)) as config:
        setup = time.perf_counter()
        layout = paired_layout(a.population, 2, a.ics)
        env = BatchedFairFight(len(layout[0]), seed=a.seed)
        initial = env.sample_initial()[:a.ics].clone()
        env.capture()
        low = torch.as_tensor(LOW, device='cuda', dtype=torch.float32)
        span = torch.as_tensor(HIGH-LOW, device='cuda', dtype=torch.float32)
        mean = ((torch.tensor(parent, device='cuda')-low)/span).clamp(0, 1)
        std = torch.full_like(mean, a.initial_std)
        incumbent = mean.clone()
        initial_mean, initial_std = mean.clone(), std.clone()
        original = torch.tensor(original, device='cuda')
        write_json(a.out/'training_initial_conditions.json', dict(
            seed=a.seed, fields=['east_m', 'north_m', 'heading_rad', 'speed_kt'],
            scope='Fixed open training ICs, reused across generations; not final evaluation',
            initial=initial.tolist()))
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter()-setup
        start = time.perf_counter()
        total_steps = total_dispatched = 0
        history = []
        for generation in range(a.generations):
            population = sample_population(mean, std, incumbent, a.population)
            parameters = low + span*population
            result = evaluate(env, parameters, original, initial, layout)
            shape = (a.population, 2, a.ics, 2)
            game_score = result['scores'].reshape(shape).mean((1, 2, 3))
            health = result['health'].reshape(shape).mean((1, 2, 3))
            objective = game_score + .001*health
            mean, std, elites = fit_elites(population, objective, mean, std, a.elites, a.minimum_std)
            incumbent = population[elites[0]].clone()
            total_steps += int(result['effective_steps'].sum())
            total_dispatched += result['dispatched_steps']
            torch.cuda.synchronize()
            summary = dict(generation=generation+1, games=(generation+1)*env.n,
                           effective_steps=total_steps, dispatched_steps=total_dispatched,
                           seconds=time.perf_counter()-start, best_objective=float(objective[elites[0]]),
                           best_game_score=float(game_score[elites[0]]), incumbent_objective=float(objective[0]))
            history.append(summary)
            write_json(a.out/f'generation_{generation:03d}.json', dict(
                summary=summary, parameters=parameters.tolist(), objective=objective.tolist(),
                outcomes=result['outcomes'].reshape(shape).tolist(),
                scores=result['scores'].reshape(shape).tolist(),
                final_health_difference=result['health'].reshape(shape).tolist(),
                effective_steps=result['effective_steps'].reshape(shape).tolist(),
                elite_indices=elites.tolist(), mean=mean.tolist(), std=std.tolist()))
            write_json(a.out/'progress.json', summary)
            print(json.dumps(summary), flush=True)
        changed = not (torch.equal(mean, initial_mean) and torch.equal(std, initial_std))
        metrics = dict(history[-1], setup_seconds=setup_seconds, distribution_updates=a.generations,
                       distribution_changed=changed, invalid_games=0, environments=env.n,
                       all_search_tensors_cuda=mean.is_cuda and std.is_cuda,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated())
        metrics['effective_steps_per_second'] = total_steps/metrics['seconds']
        selected = (low+span*incumbent).tolist()
        # Original loader enforces bounds and finiteness too.
        Policy(parameters=selected)
        write_json(a.out/'policy_net.json', dict(parameter_names=NAMES, parameters=selected,
                   format='Reactive controller coefficients, not neural weights',
                   experimental=True, selection='Training-only CEM incumbent; not a promoted champion'))
        torch.save(dict(mean=mean, std=std, incumbent=incumbent, initial=initial,
                        config=config, metrics=metrics, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), a.out/'search_state.pt')
        write_json(a.out/'results.json', dict(metrics=metrics, history=history))
        return metrics
