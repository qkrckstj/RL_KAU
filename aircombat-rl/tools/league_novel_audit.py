"""Audit a previously selected gate against a frozen, untrained opponent panel.

This never selects candidates or launches training. Run after the prior search
and its motion diagnostic have finished to respect the three-worker limit.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import shutil
import sys
import time
import traceback

from tools.autolab_cem import ROOT, read, write, sha
from tools.league_adaptive_train import immutable_json, paired_ci, relative
from tools.league_matches import initialize_worker
from tools.league_train import evaluate_candidates, verify
from tools.league_validate import process_live


def assert_idle(folder):
    runtime = folder/'runtime.json'
    if runtime.exists():
        pid = read(runtime)['pid']
        if pid != os.getpid() and process_live(pid):
            raise ValueError(f'Prior process is still live: {pid}')


def freeze(out, search, panel_path):
    if (out/'plan.json').exists():
        plan = read(out/'plan.json')
        if plan['search'] != relative(search) or plan['panel'] != relative(panel_path):
            raise ValueError('Cannot change audit inputs')
        verify(plan)
        return plan
    prior = read(search/'plan.json')
    verify(prior)
    completed = read(search/'completion.json')
    selected = read(search/'frozen_selection.json')
    if completed['status'] != 'evaluation_passed':
        raise ValueError('Prior selected policy has not passed its final evaluation')
    if completed['selected'] != selected['chosen']:
        raise ValueError('Final policy differs from pre-test selection')
    panel = read(panel_path)
    for path, digest in panel['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Panel controller source changed')
    # A new parameter vector is a novel family member, not a novel architecture.
    seen = {tuple(x['parameters']) for x in prior['opponents'] if 'parameters' in x}
    if any(tuple(x['parameters']) in seen for x in panel['opponents']):
        raise ValueError('Audit opponent was already present in training')
    original = next(x for x in prior['opponents'] if x['id'] == 'cem_original')
    candidates = [original, prior['parent'], selected['chosen']]
    inputs = dict(prior['input_sha256'])
    for path in (search/'plan.json', search/'completion.json',
                 search/'frozen_selection.json', panel_path):
        inputs[relative(path)] = sha(path)
    model = ROOT/selected['chosen']['design']
    for name, digest in read(model/'artifact_sha256.json').items():
        if sha(model/name) != digest:
            raise ValueError('Selected artifact changed')
        inputs[relative(model/name)] = digest
    sources = dict(prior['source_sha256'])
    sources[relative(Path(__file__))] = sha(Path(__file__))
    plan = dict(search=relative(search), panel=relative(panel_path),
        candidates=candidates, opponents=panel['opponents'], band=panel['band'],
        n=panel['n'], workers=3, source_sha256=sources, input_sha256=inputs,
        scope=panel['limits'], selection_forbidden=True,
        interpretation='Audit all opponents and paired uncertainty; do not select another model from these results.')
    # This central claim prevents silently reopening the same panel elsewhere.
    claim = dict(out=relative(out), panel_sha256=sha(panel_path),
                 selection_sha256=sha(search/'frozen_selection.json'))
    immutable_json(ROOT/f"runs/holdout_claim_{panel['band']}.json", claim)
    immutable_json(out/'plan.json', plan)
    for name in sources:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return plan


def run(out, search, panel):
    assert_idle(out)
    assert_idle(search)
    assert_idle(search/'routing_probe')
    plan = freeze(out, search, panel)
    if (out/'completion.json').exists():
        return read(out/'completion.json')
    write(out/'runtime.json', dict(pid=os.getpid(), executable=sys.executable, workers=3))
    write(out/'progress.json', dict(stage='novel_opponent_audit'))
    with ProcessPoolExecutor(max_workers=3, initializer=initialize_worker) as pool:
        records = evaluate_candidates(pool, plan['candidates'], plan['opponents'],
            plan['band'], plan['n'], out/'matches')
    original, parent, chosen = records
    old = {r['foe']:r['summary'] for r in parent['results']}
    new = {r['foe']:r['summary'] for r in chosen['results']}
    result = dict(status='audit_complete', selected=plan['candidates'][2],
        selected_before_test=True, model_selection_performed=False,
        metrics=[r['metrics'] for r in records],
        gain_vs_parent=paired_ci(chosen['results'], parent['results']),
        gain_vs_original=paired_ci(chosen['results'], original['results']),
        per_opponent={foe:dict(parent=old[foe], chosen=new[foe],
            win_gain=new[foe]['rate']-old[foe]['rate']) for foe in old},
        scope=plan['scope'], next='Review all raw outcomes; audit completion alone does not establish improvement.')
    verify(plan)
    write(out/'completion.json', result)
    write(out/'progress.json', dict(stage='complete'))
    return result


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--search', type=Path, default=Path('runs/league_adaptive_20261006'))
    p.add_argument('--panel', type=Path, default=Path('experiments/league/novel_panel_20261006.json'))
    p.add_argument('--after-pid', type=int, help='Queue behind an existing controller without adding workers')
    a = p.parse_args()
    try:
        if a.after_pid:
            assert_idle(a.out.resolve())
            script_sha = sha(Path(__file__))
            write(a.out/'runtime.json', dict(pid=os.getpid(), executable=sys.executable,
                                            waiting_for_pid=a.after_pid))
            while process_live(a.after_pid):
                write(a.out/'progress.json', dict(stage='waiting_for_prior_controller', pid=a.after_pid))
                time.sleep(45)
            if sha(Path(__file__)) != script_sha:
                raise ValueError('Queued audit source changed')
            if (a.search/'routing_probe/plan.json').exists():
                diagnostic = read(a.search/'routing_probe/completion.json')
                if diagnostic['status'] != 'complete':
                    raise ValueError('Scheduled motion diagnostic did not complete')
            if read(a.search/'completion.json')['status'] != 'evaluation_passed':
                write(a.out/'completion.json', dict(status='not_eligible', panel_opened=False,
                    reason='Prior selected model did not pass its final evaluation'))
                write(a.out/'progress.json', dict(stage='complete', status='not_eligible'))
                sys.exit(0)
        run(a.out.resolve(), a.search.resolve(), a.panel.resolve())
    except Exception as error:
        write(a.out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
