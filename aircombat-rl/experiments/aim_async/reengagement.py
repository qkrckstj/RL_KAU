"""Training-only native closing reengagement starts; inference is unchanged."""
import copy
import math
import time

import numpy as np

from experiments.aim_async.prefix_diversity import DiversePrefixArchive, DiversePrefixRestartable
from experiments.league.periodic_coverage import PeriodicCoverageSampler
from experiments.league.strategy_worker import PeriodicArchiveFight
from tools.league_matches import Actor


class ReengagementBoundary:
    def __init__(self, initial_range, rule):
        if (set(rule) != {'stage','max_range_m'} or rule['stage'] != 'closing_reengagement'
                or not math.isfinite(rule['max_range_m']) or rule['max_range_m'] <= 1500.):
            raise ValueError('Expected a finite closing reengagement boundary beyond1500m')
        self.limit = float(rule['max_range_m'])
        self.entered = initial_range <= 1500.
        self.exited = False
        self.first_exit_time = None

    def observe(self, previous_range, info, terminal):
        distance = float(info['range'])
        if not math.isfinite(distance) or not math.isfinite(previous_range):
            raise ValueError('Nonfinite native reengagement range')
        self.entered |= distance <= 1500.
        if terminal:
            return ('fallback_no_reengagement' if self.exited else
                    'fallback_no_exit' if self.entered else 'fallback_no_entry')
        if self.exited and distance <= self.limit and distance < previous_range:
            return 'post_encounter'
        if self.entered and not self.exited and distance > 1500.:
            self.exited = True
            self.first_exit_time = float(info['t'])
        return None


class ReengagementArchive(DiversePrefixArchive):
    def __init__(self, *args, handoff_rule, **kwargs):
        ReengagementBoundary(10000., handoff_rule)
        self.handoff_rule = copy.deepcopy(handoff_rule)
        super().__init__(*args, **kwargs)
        self.prefix_totals['fallback_no_reengagement'] = 0
        for total in self.prefix_source_totals.values():
            total['fallback_no_reengagement'] = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.sampler = PeriodicCoverageSampler(self.probabilities, int(seed), self.coverage_period)
            self.episode_index = 0
        before = (self.episode_index, copy.deepcopy(self.sampler))
        raw, info = PeriodicArchiveFight.reset(self, seed=None, options=options)
        episode_seed = self.episode_metadata['episode_seed']
        selector = np.random.default_rng(np.random.SeedSequence([episode_seed,47228]))
        attempted = bool(selector.random() < self.curriculum['fraction'])
        initial = dict(self.episode_metadata)
        status, steps, elapsed = 'normal', 0, 0.
        ended = role = boundary = range_delta = None
        if attempted:
            self.prefix_actor.seed(episode_seed, 0 if self.seat == 'red' else 1)
            self.prefix_actor.reset()
            role = self.prefix_actor.role
            boundary = ReengagementBoundary(float(info['range']), self.handoff_rule)
            started = time.perf_counter()
            while True:
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite native prefix observation')
                previous_range = float(info['range'])
                self.teacher_action(raw)
                action = self.encode_action(self.prefix_actor.act(self._last[self.seat], raw))
                raw, _, term, trunc, info = PeriodicArchiveFight.step(self, action)
                steps += 1
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite native prefix observation')
                range_delta = float(info['range'])-previous_range
                status = boundary.observe(previous_range, info, bool(term or trunc))
                if term or trunc:
                    ended = dict(t=float(info['t']),outcome=info.get('outcome'),
                                 own_health=float(info['own_health']),opp_health=float(info['opp_health']))
                if status is not None:
                    break
                if steps > math.ceil(self.t_max/.05)+2:
                    raise RuntimeError('Native prefix exceeded its original horizon')
            elapsed = time.perf_counter()-started
            if status != 'post_encounter':
                self.episode_index, self.sampler = before
                raw, info = PeriodicArchiveFight.reset(self, seed=None, options=options)
                if self.episode_metadata != initial:
                    raise RuntimeError('Fallback changed the selected native episode')
                range_delta = None
        record = dict(attempted=attempted,status=status,native_steps=steps,native_seconds=elapsed,
            handoff_time=float(info['t']),handoff_own_health=float(info['own_health']),
            handoff_opp_health=float(info['opp_health']),prefix_terminal=ended,source_role=role,
            first_exit_time=boundary.first_exit_time if boundary is not None else None,
            handoff_range=float(info['range']),handoff_range_delta=range_delta,
            handoff_rule=copy.deepcopy(self.handoff_rule))
        self.episode_metadata = dict(self.episode_metadata,prefix=record)
        self.prefix_totals['resets'] += 1
        self.prefix_totals['attempted'] += int(attempted)
        self.prefix_totals['native_steps'] += steps
        self.prefix_totals['native_seconds'] += elapsed
        if attempted:
            self.prefix_totals[status] += 1
            total = self.prefix_source_totals[role]
            total['attempted'] += 1
            total[status] += 1
            total['native_steps'] += steps
            total['native_seconds'] += elapsed
        return raw, dict(info,league_episode=copy.deepcopy(self.episode_metadata))


class ReengagementRestartable(DiversePrefixRestartable):
    def snapshot_worker(self):
        return dict(super().snapshot_worker(),handoff_rule=copy.deepcopy(self.unwrapped.handoff_rule))

    def restore_worker(self,state):
        if state.get('handoff_rule') != self.unwrapped.handoff_rule:
            raise ValueError('Saved native handoff boundary changed or missing')
        super().restore_worker(state)


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from experiments.aim_async.draw_utility_worker import DrawUtilityGuidanceEnv
    torch.set_num_threads(1)
    archive = ReengagementArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],
        curriculum=plan['curriculum'],prefix_spec=plan['prefix_spec'],prefix_pool=plan['prefix_pool'],
        handoff_rule=plan['handoff_rule'])
    for spec in plan['opponents']:
        archive.cache[spec['id']] = Actor(spec)
    wrapped = DrawUtilityGuidanceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],
        plan['control_guidance_range'],draw_penalty=plan['terminal_draw_penalty'])
    env = ReengagementRestartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total','draw_utility_total')))
    if restart is not None:
        env.restore_worker(restart)
    return env
