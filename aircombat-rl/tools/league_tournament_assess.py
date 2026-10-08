"""Fresh assessment of preserved repair candidates under a new explicit profile.

Does not relabel the old experiment or retrain/select on final/unseen outcomes.
Launch only after the prior search, conditional audit and thread benchmark end.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import sys
import traceback

from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_train import verify
from tools.league_matches import initialize_worker
from tools.league_repair_train import Evaluation, append_distinct_hybrids, configuration_of
from tools.league_thread_benchmark import environment, process_live, resources, THREAD_KEYS
from tools.league_tournament_metrics import code_groups, profile, compare, ranking, paired_gain
from tools.league_damage_analysis import episode_rows, summarize


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def terminal(folder, allowed):
    if (folder / 'runtime.json').exists() and process_live(read(folder / 'runtime.json')['pid']):
        raise ValueError(f'Predecessor still live: {folder}')
    if read(folder / 'completion.json')['status'] not in allowed:
        raise ValueError(f'Unexpected predecessor result: {folder}')


def freeze(out, previous, audit, benchmark, panel_path):
    terminal(previous, {'no_development_improvement', 'evaluation_failed'})
    terminal(audit, {'not_eligible'})
    terminal(benchmark, {'benchmark_complete'})
    decision = read(benchmark / 'completion.json')
    workers = decision['recommended_workers']
    if not workers or not decision['exact_episode_equality']:
        raise ValueError('No verified execution setting')
    if any(os.environ.get(k) != '1' for k in THREAD_KEYS):
        raise ValueError('Set all three thread limits BEFORE Python starts')
    if (out / 'plan.json').exists():
        plan = read(out / 'plan.json')
        if any(plan[k] != relative(v) for k, v in dict(previous=previous, audit=audit,
                benchmark=benchmark, panel=panel_path).items()) or plan['environment'] != environment():
            raise ValueError('Changed assessment configuration')
        verify(plan)
        return plan
    old = read(previous / 'plan.json'); verify(old)
    panel = read(panel_path); verify(panel)
    if panel['training_plan'] != relative(previous / 'plan.json'):
        raise ValueError('Temporal panel was not frozen for this search')
    if (ROOT / f"runs/holdout_claim_{panel['band']}.json").exists():
        raise ValueError('Temporal panel already consumed')
    development = sorted(previous.glob('s*/g*/development/candidate_000.json'))
    old_groups = code_groups(old['opponents'])
    candidates = []
    nominations = []
    for seed in old['seeds']:
        choices = [(p, read(p)) for p in development if p.parents[2].name == f's{seed}']
        if not choices:
            raise ValueError('Missing independent search development records')
        path, record = max(choices, key=lambda pair: ranking(profile(pair[1]['results'], old_groups)))
        spec = record['request']['spec']; configuration_of(spec)
        if spec not in candidates:
            candidates.append(spec)
        nominations.append(dict(seed=seed, spec=spec, evidence=relative(path),
            development_profile=profile(record['results'], old_groups)))
    additions = [read(p)['request']['spec'] for p in development]
    opponents, aliases = append_distinct_hybrids(old['opponents'], additions, old['anchor'])
    if set(code_groups(opponents)) & {s['id'] for s in panel['opponents']}:
        raise ValueError('Unseen panel appears in assessment archive')
    sources = dict(old['source_sha256']); sources.update(panel['source_sha256'])
    sources.update(read(benchmark / 'plan.json')['source_sha256'])
    for name in ('tools/league_tournament_metrics.py', 'tools/league_tournament_assess.py',
                 'tools/league_thread_benchmark.py'):
        sources[name] = sha(ROOT / name)
    inputs = dict(old['input_sha256']); inputs.update(panel['input_sha256'])
    extra = [previous / 'plan.json', previous / 'completion.json', audit / 'completion.json',
        benchmark / 'plan.json', benchmark / 'completion.json', panel_path] + development
    extra += [previous / f's{s}/result.json' for s in old['seeds']]
    for spec in opponents + candidates + [old['anchor'], old['parent']]:
        if spec['kind'] == 'submission':
            folder = ROOT / spec['design']
            extra += list(folder.glob('*.py')) + [ROOT / spec['weights']]
            extra += [folder / n for n in ('policy_net.json', 'entrant.json', 'artifact_sha256.json') if (folder / n).exists()]
    for path in extra:
        inputs[relative(path)] = sha(path)
    plan = dict(previous=relative(previous), audit=relative(audit), benchmark=relative(benchmark),
        panel=relative(panel_path), workers=workers, thread_environment={k: '1' for k in THREAD_KEYS},
        environment=environment(), anchor=old['anchor'], parent=old['parent'],
        candidates=candidates, nominations=nominations, opponents=opponents, opponent_aliases=aliases,
        groups=code_groups(opponents), selection_band=55010000, selection_n=40,
        final_band=56000000, final_n=80, temporal=panel,
        profile_name='aggregate_and_absolute_weakness_v1',
        nomination_rule='One preserved candidate per repair RNG, by mean of uniform and code-group-balanced development win rates; ties lower-quarter score, worst score, fewer losing matchups. Not independent retraining.',
        selection_rule='Against BOTH validated anchor and preserved parent: >=0.5 percentage point gain in both uniform and code-group-balanced win rates; no decline in worst or lower-quarter score; no increase in losing matchups or Evader losses. Choose highest mean of the two win rates among eligible candidates.',
        final_rule='Exactly one choice frozen before fresh final data; same absolute safeguards and >=0.5 point gains; BOTH IC-cluster win-gain 95% lower bounds >0 against BOTH references. Report every relative regression and legacy gate failure separately.',
        unseen_rule='Only after a final profile pass; evaluate the same frozen model plus original CEM and both references on the previously unopened temporal panel. Descriptive audit, no model selection from its outcomes.',
        score_definition='Local analysis: win=1, draw=.5, loss=0; not asserted official tournament points.',
        limits='Profile chosen using development evidence. Code groups are a sensitivity analysis, not independent architectures or actual student-policy probabilities. Passing does not mean beating every foe or passing legacy per-foe caps. Existing validated policy remains preserved.',
        source_sha256=sources, input_sha256=inputs)
    for band in (plan['selection_band'], plan['final_band']):
        immutable_json(ROOT / f'runs/assessment_reservation_{band}.json', dict(out=relative(out), band=band))
    immutable_json(out / 'plan.json', plan)
    for name in sources:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    return plan


def select(records, plan):
    comparisons = [[compare(r['results'], b['results'], plan['groups']) for b in records[:2]] for r in records[2:]]
    eligible = [i for i, cs in enumerate(comparisons) if all(c['profile_passed'] for c in cs)]
    choice = max(eligible, key=lambda i: ranking(profile(records[i+2]['results'], plan['groups']))) if eligible else None
    return dict(chosen_index=choice, chosen=None if choice is None else plan['candidates'][choice],
        eligible=eligible, comparisons=comparisons, selection_only=True)


def temporal_audit(pool, evaluate, plan, chosen, out):
    panel = plan['temporal']
    original = next(s for s in plan['opponents'] if s['id'] == 'cem_original')
    models = [original, plan['anchor'], plan['parent'], chosen]
    immutable_json(ROOT / f"runs/holdout_claim_{panel['band']}.json", dict(out=relative(out),
        plan_sha256=sha(out / 'plan.json'), selection_sha256=sha(out / 'frozen_selection.json')))
    write(out / 'progress.json', dict(stage='unseen_temporal_audit'))
    records = evaluate(pool, models, panel['opponents'], panel['band'], panel['n'], out / 'temporal')
    weights = {s['id']: 1 / len(panel['opponents']) for s in panel['opponents']}
    report = dict(status='audit_complete', selected=chosen, model_selection_performed=False,
        comparisons=[dict(reference=b['request']['spec']['id'],
            paired_gain=paired_gain(records[-1]['results'], b['results'], weights)) for b in records[:-1]],
        per_opponent=[dict(model=r['request']['spec']['id'], foe=f['foe'], summary=f['summary'])
            for r in records for f in r['results']], scope=panel['limits'],
        interpretation='Audit completion is measurement, not a claim of generalized superiority.')
    write(out / 'temporal/completion.json', report)
    return report


def run(out, previous, audit, benchmark, panel):
    out = out.resolve()
    if (out / 'runtime.json').exists() and process_live(read(out / 'runtime.json')['pid']):
        raise ValueError('Assessment already running')
    plan = freeze(out, previous.resolve(), audit.resolve(), benchmark.resolve(), panel.resolve())
    if (out / 'completion.json').exists():
        return read(out / 'completion.json')
    memory = resources()
    decision = read(benchmark / 'completion.json')
    worker_private = decision['median_initialized_worker_private_gib'][f"single_w{plan['workers']}"]
    if memory['commit_headroom_gib'] < 3 + (worker_private + .05) * plan['workers'] + .3:
        raise ValueError('Insufficient commit headroom for measured worker setting')
    write(out / 'runtime.json', dict(pid=os.getpid(), executable=sys.executable, workers=plan['workers'],
        started_at=datetime.now(timezone.utc).isoformat(), resources_before=memory))
    evaluate = Evaluation()
    with ProcessPoolExecutor(max_workers=plan['workers'], initializer=initialize_worker) as pool:
        write(out / 'progress.json', dict(stage='fresh_selection'))
        records = evaluate(pool, [plan['anchor'], plan['parent']] + plan['candidates'], plan['opponents'],
            plan['selection_band'], plan['selection_n'], out / 'selection')
        selected = select(records, plan)
        immutable_json(out / 'frozen_selection.json', dict(**selected, plan_sha256=sha(out / 'plan.json')))
        result = dict(status='no_profile_selection', final_opened=False, temporal_opened=False,
            profile_name=plan['profile_name'], original_results_unchanged=True)
        if selected['chosen'] is not None:
            chosen = selected['chosen']; verify(plan)
            immutable_json(ROOT / f"runs/holdout_claim_{plan['final_band']}.json", dict(out=relative(out),
                selection_sha256=sha(out / 'frozen_selection.json'), plan_sha256=sha(out / 'plan.json')))
            write(out / 'progress.json', dict(stage='fresh_final', chosen=chosen['id']))
            final = evaluate(pool, [plan['anchor'], plan['parent'], chosen], plan['opponents'],
                plan['final_band'], plan['final_n'], out / 'final', traced=True)
            comparisons = [compare(final[2]['results'], r['results'], plan['groups'], final=True) for r in final[:2]]
            passed = all(c['profile_passed'] for c in comparisons)
            result.update(status='tournament_profile_passed' if passed else 'tournament_profile_failed',
                final_opened=True, selected_before_test=True, selected=chosen, comparisons=comparisons,
                legacy_safeguards_passed=all(c['legacy_safeguards_passed'] for c in comparisons))
            damage = [dict(model=r['request']['spec']['id'], foe=f['foe'],
                summary=summarize(episode_rows(f, plan['final_band'], plan['final_n']))) for r in final for f in r['results']]
            write(out / 'final/damage_summary.json', damage)
            # Persist final outcome before the audit; it cannot change selection.
            immutable_json(out / 'final_decision.json', result)
            if passed:
                result['temporal_audit'] = temporal_audit(pool, evaluate, plan, chosen, out)
                result['temporal_opened'] = True
        verify(plan)
        write(out / 'completion.json', result)
        write(out / 'progress.json', dict(stage='complete', status=result['status']))
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('runs/league_repair_20261007'))
    parser.add_argument('--audit', type=Path, default=Path('runs/league_temporal_audit_v3_20261007'))
    parser.add_argument('--benchmark', type=Path, default=Path('runs/league_thread_benchmark_v2_20261007'))
    parser.add_argument('--panel', type=Path, default=Path('experiments/league/temporal_panel_repair_20261007.json'))
    args = parser.parse_args()
    try:
        run(args.out, args.previous, args.audit, args.benchmark, args.panel)
    except Exception:
        write(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise
