"""Freeze a development-selected composition, then test fresh ICs and opponents."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import traceback
from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_train import verify
from tools.league_matches import initialize_worker
from tools.league_repair_train import Evaluation
from tools.league_thread_benchmark import environment, process_live, resources, THREAD_KEYS
from tools.league_tournament_metrics import compare, profile, ranking
from tools.league_tournament_assess import temporal_audit
from tools.league_damage_analysis import episode_rows, summarize


def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def freeze(out, development, workers):
    if (development/'runtime.json').exists() and process_live(read(development/'runtime.json')['pid']):
        raise ValueError('Development probe still live')
    if read(development/'completion.json')['status']!='development_probe_complete': raise ValueError('Incomplete development')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS): raise ValueError('Set thread limits before Python')
    if (out/'plan.json').exists():
        plan=read(out/'plan.json');verify(plan)
        if (plan['previous'],plan['workers'],plan['environment'])!=(relative(development),workers,environment()):
            raise ValueError('Changed validation configuration')
        return plan
    prior=read(development/'plan.json');verify(prior)
    source=ROOT/prior['source'];old=read(source/'plan.json');verify(old)
    records=[read(development/f'development/candidate_{i:03d}.json') for i in range(len(prior['models']))]
    baseline=[read(source/f'selection/candidate_{i:03d}.json') for i in (0,1)]
    comparisons=[[compare(r['results'],b['results'],prior['groups']) for b in baseline] for r in records]
    eligible=[i for i,cs in enumerate(comparisons) if all(c['profile_passed'] for c in cs)]
    if not eligible: raise ValueError('No development candidate satisfies the unchanged profile')
    choice=max(eligible,key=lambda i:ranking(profile(records[i]['results'],prior['groups'])))
    chosen=prior['models'][choice]
    sources=dict(prior['source_sha256']);sources['tools/league_separated_validate.py']=sha(__file__)
    inputs=dict(prior['input_sha256'])
    for p in [development/'plan.json',development/'completion.json',development/'compatibility.json']+[
            development/f'development/candidate_{i:03d}.json' for i in range(len(records))]:
        inputs[relative(p)]=sha(p)
    plan=dict(previous=relative(development),training_source='runs/league_repair_20261007',workers=workers,
        environment=environment(),opponents=prior['opponents'],groups=prior['groups'],
        anchor=old['anchor'],parent=old['parent'],warm=prior['warm_start'],chosen=chosen,
        candidates=prior['models'],development_comparisons=comparisons,eligible=eligible,chosen_index=choice,
        final_band=58000000,final_n=80,temporal=old['temporal'],profile_name=old['profile_name'],
        final_models=[old['anchor'],old['parent'],prior['warm_start'],chosen],
        final_rule=old['final_rule'],
        warm_comparison_rule='Report paired uniform/group win gains, lower-tail metrics and damage against the prior strongest development model. The unchanged qualification gates compare the same validated anchor and preserved parent; no retroactive extra gate or model selection from final data.',
        selection_rule='Choose once from the completed two-composition development experiment using its unchanged profile against both references. Those conditions were consumed; no new selection simulation is needed.',
        scope='Fresh final ICs on the fixed 45-opponent archive; conditional unopened temporal panel. Composition uses existing learned policies, not new CEM RNG training. Legacy policies and failed runs preserved; no universal or actual-student-policy claim.',
        source_sha256=sources,input_sha256=inputs)
    for band in (plan['final_band'],plan['temporal']['band']):
        if (ROOT/f'runs/holdout_claim_{band}.json').exists(): raise ValueError(f'Band already consumed: {band}')
    immutable_json(out/'plan.json',plan)
    immutable_json(out/'frozen_selection.json',dict(chosen=chosen,chosen_index=choice,eligible=eligible,
        development_only=True,plan_sha256=sha(out/'plan.json')))
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return plan


def run(out, development, workers):
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']): raise ValueError('Validation already live')
    plan=freeze(out,development,workers)
    if (out/'completion.json').exists(): return read(out/'completion.json')
    if read(out/'frozen_selection.json')['chosen']!=plan['chosen']: raise ValueError('Changed pre-final choice')
    before=resources()
    if before['commit_headroom_gib']<3+.42*workers+.3: raise ValueError('Insufficient commit headroom')
    write(out/'runtime.json',dict(pid=os.getpid(),workers=workers,started_at=datetime.now(timezone.utc).isoformat(),resources=before))
    immutable_json(ROOT/f"runs/holdout_claim_{plan['final_band']}.json",dict(out=relative(out),
        plan_sha256=sha(out/'plan.json'),selection_sha256=sha(out/'frozen_selection.json')))
    evaluate=Evaluation()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        write(out/'progress.json',dict(stage='fresh_final',chosen=plan['chosen']['id']))
        records=evaluate(pool,plan['final_models'],plan['opponents'],plan['final_band'],plan['final_n'],out/'final',traced=True)
        comparisons=[compare(records[-1]['results'],r['results'],plan['groups'],final=True) for r in records[:2]]
        passed=all(c['profile_passed'] for c in comparisons)
        result=dict(status='tournament_profile_passed' if passed else 'tournament_profile_failed',
            profile_name=plan['profile_name'],selected=plan['chosen'],selected_before_test=True,
            final_opened=True,temporal_opened=False,comparisons=comparisons,
            warm_comparison=compare(records[-1]['results'],records[2]['results'],plan['groups'],final=True),
            legacy_safeguards_passed=all(c['legacy_safeguards_passed'] for c in comparisons),scope=plan['scope'])
        write(out/'final/damage_summary.json',[dict(model=r['request']['spec']['id'],foe=f['foe'],
            summary=summarize(episode_rows(f,plan['final_band'],plan['final_n']))) for r in records for f in r['results']])
        immutable_json(out/'final_decision.json',result)
        if passed:
            result['temporal_audit']=temporal_audit(pool,evaluate,plan,plan['chosen'],out)
            result['temporal_opened']=True
        verify(plan);write(out/'completion.json',result);write(out/'progress.json',dict(stage='complete',status=result['status']))
    return result


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=Path('runs/league_separated_probe_20261007'));p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    try:run(a.out.resolve(),a.previous.resolve(),a.workers)
    except Exception:write(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
