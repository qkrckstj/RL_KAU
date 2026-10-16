"""Training-only native first-pass prefixes; deployed policies are unchanged."""
import copy
import math
import time

import numpy as np

from experiments.league.periodic_coverage import PeriodicCoverageSampler
from experiments.league.strategy_worker import PeriodicArchiveFight
from experiments.rear_async.worker import Restartable
from tools.league_matches import Actor


class PostEncounterArchive(PeriodicArchiveFight):
    """Continue the same live simulator after its first surviving range pass.

    Prefix actions come from one frozen sampled parent. They are not PPO data.
    An unsuccessful prefix rolls sampler/index back and starts the same original
    episode normally. No HP, integrator, clock, weapon or controller state is
    constructed or transplanted. Native termination precedes range-exit success.
    """
    def __init__(self, *args, curriculum, prefix_spec, **kwargs):
        if (set(curriculum) != {'stage', 'fraction'} or curriculum['stage'] != 'post_encounter'
                or not math.isfinite(curriculum['fraction']) or not 0 <= curriculum['fraction'] <= 1):
            raise ValueError('Require a finite post_encounter attempt fraction')
        self.curriculum = copy.deepcopy(curriculum)
        self.prefix_spec = copy.deepcopy(prefix_spec)
        self.prefix_actor = Actor(prefix_spec)
        if self.prefix_actor.mode != 'discrete':
            raise ValueError('Require the frozen discrete parent')
        self.prefix_totals = dict(resets=0, attempted=0, post_encounter=0,
            fallback_no_entry=0, fallback_no_exit=0, native_steps=0, native_seconds=0.)
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        # Snapshot after an explicit seed has initialized selection, before a draw.
        if seed is not None:
            self.sampler = PeriodicCoverageSampler(self.probabilities, int(seed), self.coverage_period)
            self.episode_index = 0
        before = (self.episode_index, copy.deepcopy(self.sampler))
        raw, info = super().reset(seed=None, options=options)
        episode_seed = self.episode_metadata['episode_seed']
        selector = np.random.default_rng(np.random.SeedSequence([episode_seed, 47228]))
        attempted = bool(selector.random() < self.curriculum['fraction'])
        status, steps, elapsed = 'normal', 0, 0.
        initial = dict(self.episode_metadata)
        ended = None
        if attempted:
            self.prefix_actor.seed(episode_seed, 0 if self.seat == 'red' else 1)
            self.prefix_actor.reset()
            entered = float(info['range']) <= 1500.
            started = time.perf_counter()
            while True:
                if not np.isfinite(raw).all() or not math.isfinite(float(info['range'])):
                    raise ValueError('Nonfinite native prefix observation')
                # Separate teachers: archive prior history and exported actor history.
                # Do not consume the final handoff observation here.
                self.teacher_action(raw)
                action = self.encode_action(self.prefix_actor.act(self._last[self.seat], raw))
                raw, _, term, trunc, info = super().step(action)
                steps += 1
                if not np.isfinite(raw).all() or not math.isfinite(float(info['range'])):
                    raise ValueError('Nonfinite native prefix observation')
                entered |= float(info['range']) <= 1500.
                if term or trunc:
                    status = 'fallback_no_exit' if entered else 'fallback_no_entry'
                    ended = dict(t=float(info['t']), outcome=info.get('outcome'),
                                 own_health=float(info['own_health']), opp_health=float(info['opp_health']))
                    break
                if entered and float(info['range']) > 1500.:
                    status = 'post_encounter'
                    break
                if steps > math.ceil(self.t_max/.05)+2:
                    raise RuntimeError('Native prefix did not terminate by its original horizon')
            elapsed = time.perf_counter()-started
            if status != 'post_encounter':
                self.episode_index, self.sampler = before
                raw, info = super().reset(seed=None, options=options)
                if self.episode_metadata != initial:
                    raise RuntimeError('Fallback changed the selected native episode')
        record = dict(attempted=attempted, status=status, native_steps=steps,
            native_seconds=elapsed, handoff_time=float(info['t']),
            handoff_own_health=float(info['own_health']), handoff_opp_health=float(info['opp_health']),
            prefix_terminal=ended)
        self.episode_metadata = dict(self.episode_metadata, prefix=record)
        self.prefix_totals['resets'] += 1
        self.prefix_totals['attempted'] += int(attempted)
        if attempted: self.prefix_totals[status] += 1
        self.prefix_totals['native_steps'] += steps
        self.prefix_totals['native_seconds'] += elapsed
        return raw, dict(info, league_episode=copy.deepcopy(self.episode_metadata))


class PrefixRestartable(Restartable):
    """Preserve cumulative prefix work including unfinished PPO episodes."""
    def snapshot_worker(self):
        return dict(super().snapshot_worker(), prefix_totals=copy.deepcopy(self.unwrapped.prefix_totals))

    def restore_worker(self, state):
        if 'prefix_totals' not in state:
            raise ValueError('Cannot resume a prefix run without its accounting')
        super().restore_worker(state)
        self.unwrapped.prefix_totals = copy.deepcopy(state['prefix_totals'])


def make_worker(plan, index, restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from experiments.aim_async.draw_utility_worker import DrawUtilityGuidanceEnv
    torch.set_num_threads(1)
    archive = PostEncounterArchive(plan['teacher'], plan['opponents'], plan['probabilities'],
        plan['seed']*100+index, plan['ic_band'], index, plan['workers'],
        'red' if index%2 == 0 else 'blue', coverage_period=plan['coverage_period'],
        curriculum=plan['curriculum'], prefix_spec=plan['prefix_spec'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    wrapped = DrawUtilityGuidanceEnv(archive, plan['ppo']['gamma'], plan['aim_coefficients'],
        plan['relative_potential_coefficient'], plan['control_guidance_coefficient'],
        plan['control_guidance_range'], draw_penalty=plan['terminal_draw_penalty'])
    env = PrefixRestartable(Monitor(wrapped, info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total','draw_utility_total')))
    if restart is not None: env.restore_worker(restart)
    return env
