"""Measure public early-motion gate features after an existing search exits.

This is an open-development observation diagnostic, not a performance test.
Observers always select the frozen parent and must match its actions exactly.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time
import traceback
import numpy as np
from experiments.league.adaptive import Policy
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_matches import MatchEnv, initialize_worker
from tools.league_train import verify
from tools.league_validate import process_live


def observe(parent, experts, foe, band, n):
    rows = []
    for seat in ('red', 'blue'):
        env = MatchEnv(parent, foe, seat)
        try:
            for seed in range(band, band+n//2):
                obs, _ = env.reset(seed=seed)
                probes = [Policy(configuration=dict(experts=experts, gate=[t, -8, 0, 0, 0, 0]))
                          for t in (.5, 1.5, 3., 6.)]
                for step in range(161):
                    native = int(env.own_actor.predict(obs))
                    for probe in probes:
                        if probe.act(obs) != native:
                            raise ValueError('Diagnostic observer changed parent action')
                    if all(p.features is not None for p in probes):
                        break
                    obs, _, terminal, truncated, _ = env.step(native)
                    if terminal or truncated:
                        raise ValueError('Episode ended before observation probe completed')
                else:
                    raise ValueError('Probe clock did not advance')
                rows.append(dict(seed=seed, seat=seat, probes=[dict(seconds=float(p.gate[0]),
                    features=p.features.tolist()) for p in probes], steps=step))
        finally:
            env.close()
    return dict(foe=foe['id'], observations=rows)


def run(search, out, after_pid):
    old = read(search/'plan.json')
    verify(old)
    plan = dict(search=search.relative_to(ROOT).as_posix(), band=34020000, n=4, workers=3,
        parent=old['parent'], experts=old['experts'], opponents=old['opponents'],
        search_plan_sha256=sha(search/'plan.json'), source_sha256=sha(Path(__file__)),
        scope='Fresh open-development public-motion observations; no full-episode performance claim')
    if (out/'plan.json').exists() and read(out/'plan.json') != plan:
        raise ValueError('Changed diagnostic configuration')
    if (out/'completion.json').exists():
        return
    if (out/'runtime.json').exists():
        pid = read(out/'runtime.json')['pid']
        if pid != os.getpid() and process_live(pid):
            raise ValueError('Diagnostic already running')
    write(out/'plan.json', plan)
    write(out/'runtime.json', dict(pid=os.getpid(), waiting_for_pid=after_pid))
    while after_pid and process_live(after_pid):
        write(out/'progress.json', dict(stage='waiting_for_existing_training', pid=after_pid))
        time.sleep(45)
    if not (search/'completion.json').exists():
        raise ValueError('Prior controller exited without a completed experiment; inspect its failure first')
    verify(old)
    write(out/'progress.json', dict(stage='measuring_public_motion', workers=3))
    results = {}
    with ProcessPoolExecutor(max_workers=3, initializer=initialize_worker) as pool:
        pending = {}
        for i, foe in enumerate(plan['opponents']):
            path = out/f'foe_{i:03d}.json'
            if path.exists():
                results[i] = read(path)
            else:
                pending[pool.submit(observe, plan['parent'], plan['experts'], foe, plan['band'], plan['n'])] = (i, path)
        for future in as_completed(pending):
            i, path = pending[future]
            results[i] = future.result()
            write(path, results[i])
            write(out/'progress.json', dict(stage='measuring_public_motion', done=len(results), total=len(plan['opponents'])))
    summary = []
    for i in range(len(plan['opponents'])):
        r = results[i]
        values = np.asarray([[p['features'] for p in row['probes']] for row in r['observations']])
        summary.append(dict(foe=r['foe'], probe_seconds=[.5, 1.5, 3., 6.],
                            minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist()))
    verify(old)
    if sha(Path(__file__)) != plan['source_sha256']:
        raise ValueError('Diagnostic source changed')
    write(out/'summary.json', summary)
    write(out/'completion.json', dict(status='complete', games_observed=len(results)*plan['n'],
        scope=plan['scope'], action_equivalence='Every observer action matched native parent throughout the prefix'))
    write(out/'progress.json', dict(stage='complete'))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--search', type=Path, default=Path('runs/league_adaptive_20261006'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--after-pid', type=int)
    a = p.parse_args()
    try:
        run(a.search.resolve(), a.out.resolve(), a.after_pid)
    except Exception as error:
        write(a.out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
