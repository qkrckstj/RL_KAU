"""Official FairFight with replaceable pilots; physics and verdict are inherited.

Both actors receive the same-tick seat-relative observations. Draw utility is
an explicit development convention (1/.5/0), not a claimed tournament rule.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import time
import numpy as np
from aircombat_gym.wvr import baselines
from aircombat_gym.wvr.actions import DELTA_HEADING_DEG, DELTA_SPEED_KT
from aircombat_gym.wvr.envs.fair import FairFightEnv
from experiments.plan_a.tactical import Policy as CemPolicy
from experiments.league.controller import Policy as ReactivePolicy
from tools.autolab_cem import ROOT, read, write, sha
from tools.grade import play, summarise


def initialize_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # Import only when a neural entrant needs it, avoiding startup cost otherwise.


class Actor:
    def __init__(self, spec):
        self.spec = spec
        self.mode = 'discrete'
        self.bot = None
        kind = spec['kind']
        if kind == 'bot':
            self.bot = baselines.BY_NAME[spec['name']]()
        elif kind == 'fixed':
            self.predict = lambda obs: spec['action']
        elif kind in ('cem', 'reactive'):
            self.predict = (CemPolicy if kind == 'cem' else ReactivePolicy)(parameters=spec['parameters']).act
        elif kind == 'submission':
            import torch
            torch.set_num_threads(1)
            from tools.policies import load
            self.predict, _, self.mode = load(ROOT/spec['design'], ROOT/spec['weights'])
        else:
            raise ValueError(f'Unknown entrant: {kind}')

    def seed(self, seed, physical_seat):
        if self.bot is not None and hasattr(self.bot, 'rng'):
            self.bot.rng = np.random.default_rng(np.random.SeedSequence([seed, 71, physical_seat]))

    def reset(self):
        if self.bot is not None:
            self.bot.reset()
        else:
            owner = getattr(self.predict, '__self__', None)
            if hasattr(owner, 'reset'):
                owner.reset()

    def act(self, info=None, obs=None):
        if self.bot is not None:
            return self.bot.act(info, obs)
        action = self.predict(obs)
        if self.mode == 'continuous':
            a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            if a.shape != (2,) or not np.isfinite(a).all():
                raise ValueError('Invalid continuous action')
            return float(a[0]*30), float(a[1]*20), 0.
        a = int(action)
        if not 0 <= a < 9:
            raise ValueError(f'Invalid discrete action {action}')
        return DELTA_HEADING_DEG[a//3], DELTA_SPEED_KT[a%3], 0.


class MatchEnv(FairFightEnv):
    def __init__(self, own, foe, seat='red'):
        self.own_actor, self.foe_actor = Actor(own), Actor(foe)
        super().__init__(action_mode=self.own_actor.mode, seat=seat)

    def sample(self, rng):
        ic, _ = super().sample(rng)
        return ic, self.foe_actor

    def reset(self, *, seed=None, options=None):
        if seed is None:
            raise ValueError('League games require an explicit seed')
        self.own_actor.seed(seed, 0 if self.seat == 'red' else 1)
        self.foe_actor.seed(seed, 1 if self.seat == 'red' else 0)
        self.own_actor.reset()
        return super().reset(seed=seed, options=options)

    def own_action(self, obs):
        return self.encode_action(self.own_actor.act(self._last[self.seat], obs))


def summary(rows):
    out = summarise(rows)
    out['wins'] = sum(r['outcome'] == 'kill' for r in rows)
    out['losses'] = sum(r['outcome'] == 'died' for r in rows)
    out['draws'] = len(rows)-out['wins']-out['losses']
    out['score'] = (out['wins']+.5*out['draws'])/len(rows)
    return out


def duel(own, foe, band, n):
    if n <= 0 or n % 2:
        raise ValueError('A match batch must contain complete seat pairs')
    rows = []
    started = time.perf_counter()
    for seat in ('red', 'blue'):
        env = MatchEnv(own, foe, seat)
        try:
            part = play(env, env.own_action, band, n//2, seat)
            for row in part:
                row['seat'] = seat
            rows.extend(part)
        finally:
            env.close()
    return dict(own=own['id'], foe=foe['id'], band=band, summary=summary(rows), episodes=rows,
                elapsed_seconds=time.perf_counter()-started)


def run_job(job):
    return duel(job['own'], job['foe'], job['band'], job['n'])


def evaluate_jobs(jobs, out, workers=3):
    """Append-only results with resumable jobs; one process per game batch."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    results = {}
    for i, job in enumerate(jobs):
        path = out/f'match_{i:04d}.json'
        if path.exists():
            record = read(path)
            if record['job'] != job:
                raise ValueError('Cannot resume changed match schedule')
            results[i] = record['result']
        else:
            pending.append((i,job))
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = {pool.submit(run_job, job):(i,job) for i,job in pending}
        for future in as_completed(futures):
            i,job = futures[future]
            results[i] = future.result()
            write(out/f'match_{i:04d}.json',dict(job=job,result=results[i]))
            write(out/'progress.json',dict(done=len(results),total=len(jobs)))
    return [results[i] for i in range(len(jobs))]


def initial_roster():
    current = read(ROOT/'experiments/plan_a/cem_policy/policy_net.json')['parameters']
    design = 'runs/plan_a_20261005/double/ddqn/s1'
    return [dict(id='cem_original',kind='cem',parameters=current),
            dict(id='ddqn_s1',kind='submission',design=design,weights=f'{design}/checkpoints/step_1024000/policy_net.zip'),
            *[dict(id=name,kind='bot',name=name) for name in ('ace','pursuit','evader')],
            dict(id='fixed_left',kind='fixed',action=0),dict(id='fixed_right',kind='fixed',action=6)]


def baseline(out, n):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    roster = initial_roster()
    # All pairs, including scripted baselines; reserve Lead and Circler as
    # not-trained-against transfer opponents, rather than peeking now.
    jobs = [dict(own=a,foe=b,band=30000000,n=n)
            for i,a in enumerate(roster) for b in roster[i+1:]]
    write(out/'plan.json',dict(roster=roster,jobs=jobs,
        purpose='Development cross-play; not held-out evaluation',
        score_convention='win=1, mutual/timeout=.5, loss=0; report raw outcomes too'))
    results = evaluate_jobs(jobs,out/'matches')
    write(out/'results.json',results)
    write(out/'completion.json',dict(status='complete'))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=12)
    a = p.parse_args()
    baseline(a.out.resolve(), a.n)
