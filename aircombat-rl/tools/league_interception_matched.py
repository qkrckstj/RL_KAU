"""Matched-speed interception ablation; open development, no champion promotion."""
from argparse import ArgumentParser
from pathlib import Path
import os
import shutil
import time
import traceback
from experiments.league.interception import INITIAL
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_interception_export import export
from tools.league_matches import evaluate_jobs
from tools.league_train import verify
from tools.league_validate import process_live


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def run(out, prior):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'runtime.json').exists():
        pid = read(out/'runtime.json')['pid']
        if pid != os.getpid() and process_live(pid):
            raise ValueError('Probe already running')
    if (out/'completion.json').exists():
        return
    if (out/'plan.json').exists():
        plan = read(out/'plan.json')
        if plan['prior'] != relative(prior):
            raise ValueError('Changed prior run')
        verify(plan)
    else:
        diagnostic = ROOT/'runs/league_interception_probe_20261006/completion.json'
        if read(diagnostic)['status'] != 'diagnostic_complete':
            raise ValueError('Initial diagnostic must complete first')
        base = read(prior/'plan.json')
        verify(base)
        models = [base['parent'], dict(id='pursuit_650_lead6', kind='reactive',
                                       parameters=base['experts'][1]['parameters'])]
        for cap in (15., 45., 90.):
            parameters = INITIAL.copy()
            parameters[0] = cap
            parameters[2] = 650.
            parameters[3] = 6.
            models.append(export(out/f'models/cap{int(cap)}', parameters, f'intercept_matched_cap{int(cap)}'))
        paths = ['experiments/league/interception.py', 'tools/league_interception_export.py',
                 'tools/league_interception_matched.py']
        sources = dict(base['source_sha256'])
        sources.update({path:sha(ROOT/path) for path in paths})
        inputs = dict(base['input_sha256'])
        inputs[relative(prior/'plan.json')] = sha(prior/'plan.json')
        inputs[relative(diagnostic)] = sha(diagnostic)
        for model in models[2:]:
            folder = ROOT/model['design']
            inputs.update({relative(folder/name):digest for name,digest in
                           read(folder/'artifact_sha256.json').items()})
        plan = dict(prior=relative(prior), candidates=models,
            opponents=[dict(id=name, kind='bot', name=name) for name in ('evader', 'circler')],
            band=36001000, n=40, workers=3,
            source_sha256=sources, input_sha256=inputs,
            scope='Open development; manually specified candidates, not learned policies. Same twenty initial conditions in both seats. Interceptor far/near speeds both 650 kt and near lead 6 s match the pursuit baseline; vary long-range interception cap only. No promotion or unseen-test claim.',
            next='Inspect paired outcomes, firing time and approach metrics. Any tuning uses this as development; confirm on a fresh band and full archive before promotion.')
        write(out/'plan.json', plan)
        for name in sources:
            target = out/'source_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
    write(out/'runtime.json', dict(pid=os.getpid(), workers=3))
    while True:
        verify(plan)
        live = []
        for folder in (prior, prior/'novel_audit'):
            if (folder/'runtime.json').exists():
                pid = read(folder/'runtime.json')['pid']
                if process_live(pid):
                    live.append(pid)
        if not live:
            break
        write(out/'progress.json', dict(stage='waiting_for_prior_controllers', pids=live))
        time.sleep(45)
    # A crashed/missing predecessor is not permission to silently proceed.
    for folder in (prior, prior/'novel_audit'):
        if not (folder/'completion.json').exists():
            raise ValueError(f'Missing predecessor completion: {folder}')
    write(out/'progress.json', dict(stage='open_development_matches'))
    jobs = [dict(own=model, foe=foe, band=plan['band'], n=plan['n'])
            for model in plan['candidates'] for foe in plan['opponents']]
    results = evaluate_jobs(jobs, out/'matches', workers=plan['workers'])
    verify(plan)
    write(out/'completion.json', dict(status='diagnostic_complete', results=results,
        scope=plan['scope'], next=plan['next'], champion_changed=False))
    write(out/'progress.json', dict(stage='complete'))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, default=Path('runs/league_symmetric_20261006'))
    args = parser.parse_args()
    try:
        run(args.out.resolve(), args.prior.resolve())
    except Exception:
        write(args.out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
