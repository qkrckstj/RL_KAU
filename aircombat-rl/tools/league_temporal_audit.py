"""Queue an unopened temporal-opponent audit after a successful frozen search."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import sys
import time
import traceback

from tools.autolab_cem import ROOT, read, write, sha
from tools.league_condition_train import environment
from tools.league_novel_audit import assert_idle
from tools.league_symmetric_train import immutable_json, paired_ci
from tools.league_temporal_panel import relative
from tools.league_train import evaluate_candidates, verify
from tools.league_matches import initialize_worker
from tools.league_validate import process_live


def validate_selection(completed, selected):
    if completed['status'] != 'evaluation_passed':
        raise ValueError('Search did not pass its final evaluation')
    if completed['selected'] != selected['chosen'] or not completed.get('selected_before_test'):
        raise ValueError('Policy must match the selection frozen before final testing')


def freeze(out, search, panel_path, workers):
    if (out/'plan.json').exists():
        plan=read(out/'plan.json')
        if (plan['search'],plan['panel'],plan['workers']) != (relative(search),relative(panel_path),workers):
            raise ValueError('Cannot change frozen audit settings')
        verify(plan)
        return plan
    if workers < 1:
        raise ValueError('Positive worker count required')
    prior=read(search/'plan.json'); panel=read(panel_path)
    verify(prior); verify(panel)
    if panel['training_plan'] != relative(search/'plan.json'):
        raise ValueError('Panel was frozen for another training archive')
    completed=read(search/'completion.json'); selected=read(search/'frozen_selection.json')
    validate_selection(completed,selected)
    if environment() != prior['environment']:
        raise ValueError('Search environment changed before the audit')
    known_ids={x['id'] for x in prior['opponents']}
    if any(x['id'] in known_ids for x in panel['opponents']):
        raise ValueError('Audit foe already present in training')
    original=next(x for x in prior['opponents'] if x['id']=='cem_original')
    sources=dict(prior['source_sha256']); sources.update(panel['source_sha256'])
    for name in ('tools/league_temporal_audit.py','tools/league_novel_audit.py'):
        sources[name]=sha(ROOT/name)
    inputs=dict(prior['input_sha256']); inputs.update(panel['input_sha256'])
    for path in (search/'plan.json',search/'completion.json',search/'frozen_selection.json',panel_path):
        inputs[relative(path)]=sha(path)
    model=ROOT/selected['chosen']['design']
    for name,digest in read(model/'artifact_sha256.json').items():
        if sha(model/name) != digest:
            raise ValueError('Chosen model artifact changed')
        inputs[relative(model/name)]=digest
    inputs[relative(model/'artifact_sha256.json')]=sha(model/'artifact_sha256.json')
    plan=dict(search=relative(search),panel=relative(panel_path),workers=workers,
        candidates=[original,prior['anchor'],selected['chosen']],opponents=panel['opponents'],
        band=panel['band'],n=panel['n'],source_sha256=sources,input_sha256=inputs,
        environment=environment(),scope=panel['limits'],model_selection_forbidden=True)
    verify(plan)
    immutable_json(ROOT/f"runs/holdout_claim_{plan['band']}.json",dict(out=relative(out),
        panel_sha256=sha(panel_path),selection_sha256=sha(search/'frozen_selection.json')))
    immutable_json(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return plan


def run(out,search,panel,workers):
    assert_idle(out); assert_idle(search)
    plan=freeze(out,search,panel,workers)
    if (out/'completion.json').exists():
        return read(out/'completion.json')
    write(out/'runtime.json',dict(pid=os.getpid(),executable=sys.executable,workers=workers))
    write(out/'progress.json',dict(stage='unseen_temporal_opponents'))
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        records=evaluate_candidates(pool,plan['candidates'],plan['opponents'],
            plan['band'],plan['n'],out/'matches')
    original,anchor,chosen=records
    per_opponent=[]
    for a,b in zip(anchor['results'],chosen['results']):
        if a['foe'] != b['foe']:
            raise ValueError('Opponent order mismatch')
        per_opponent.append(dict(foe=a['foe'],anchor=a['summary'],chosen=b['summary'],
            paired_gain=paired_ci([b],[a])))
    result=dict(status='audit_complete',selected=plan['candidates'][2],selected_before_audit=True,
        model_selection_performed=False,metrics=[r['metrics'] for r in records],
        gain_vs_anchor=paired_ci(chosen['results'],anchor['results']),
        gain_vs_original=paired_ci(chosen['results'],original['results']),
        per_opponent=per_opponent,scope=plan['scope'],
        interpretation='Report improvements and regressions. Audit completion does not itself establish superiority; no model is selected from these outcomes.')
    verify(plan)
    write(out/'completion.json',result); write(out/'progress.json',dict(stage='complete'))
    return result


def queued(out,search,panel,workers,after_pid):
    assert_idle(out)
    prior=read(search/'plan.json'); suite=read(panel)
    verify(prior); verify(suite)
    sources=dict(prior['source_sha256']); sources.update(suite['source_sha256'])
    for name in ('tools/league_temporal_audit.py','tools/league_novel_audit.py'):
        sources[name]=sha(ROOT/name)
    inputs=dict(prior['input_sha256']); inputs.update(suite['input_sha256'])
    inputs[relative(panel)]=sha(panel)
    queue=dict(source_sha256=sources,input_sha256=inputs,environment=environment(),
        pid=os.getpid(),waiting_for_pid=after_pid,workers_when_active=workers,
        simulation_workers_while_waiting=0,started_at=datetime.now(timezone.utc).isoformat())
    write(out/'runtime.json',queue)
    while process_live(after_pid):
        write(out/'progress.json',dict(stage='waiting_for_search',pid=after_pid))
        time.sleep(45)
    verify(queue)
    if environment() != queue['environment']:
        raise ValueError('Queued audit environment changed')
    completed=read(search/'completion.json')
    if completed['status'] != 'evaluation_passed':
        result=dict(status='not_eligible',panel_opened=False,
            reason='Preceding search did not pass its final evaluation')
        write(out/'completion.json',result); write(out/'progress.json',dict(stage='complete'))
        return result
    return run(out,search,panel,workers)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--search',type=Path,default=Path('runs/league_conditions_20261006'))
    p.add_argument('--panel',type=Path,default=Path('experiments/league/temporal_panel_20261006.json'))
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--after-pid',type=int)
    args=p.parse_args()
    try:
        if args.after_pid:
            queued(args.out.resolve(),args.search.resolve(),args.panel.resolve(),args.workers,args.after_pid)
        else:
            run(args.out.resolve(),args.search.resolve(),args.panel.resolve(),args.workers)
    except Exception:
        write(args.out/'failure.json',dict(traceback=traceback.format_exc()))
        raise
