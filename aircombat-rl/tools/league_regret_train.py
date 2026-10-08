"""Learn maneuvers with paired per-opponent regression costs and guarded checkpoints.

Conditional successor to a completed unsuccessful maneuver experiment. Keeps
official simulation, prior models, full archive, and separate final conditions.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np

from experiments.league.controller import LOW, HIGH
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_cem_population import sample_population, distinct_elites
from tools.league_cached_evaluation import validate as validate_record
from tools.league_damage_analysis import episode_rows, summarize, paired_change
from tools.league_damage_trace import observe
from tools.league_interception_gate_export import export
from tools.league_matches import duel, initialize_worker
from tools.league_symmetric_train import immutable_json, comparison, paired_ci
from tools.league_train import evaluate_candidates, ordering, unique_entrants, verify
from tools.league_validate import process_live


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment():
    return dict(python=sys.version, platform=platform.platform(),
        packages=[list(x) for x in sorted((d.metadata['Name'], d.version)
                  for d in distributions() if d.metadata['Name'])])


def healthy_wins(record, weights=None):
    results = record['results']
    w = np.ones(len(results)) if weights is None else np.asarray(weights, float)
    values = [np.mean([e['own_health'] if e['outcome'] == 'kill' else 0.
                       for e in r['episodes']]) for r in results]
    return float(w @ values / w.sum())


def regression_cost(record, reference):
    for key in ('opponents', 'band', 'n', 'weights'):
        if record['request'][key] != reference['request'][key]:
            raise ValueError('Regression cost requires the same match conditions')
    excess = []; evader_loss = 0.
    if len(record['results']) != len(reference['results']):
        raise ValueError('Different opponent counts')
    for new, old in zip(record['results'], reference['results']):
        if new['foe'] != old['foe']:
            raise ValueError('Different opponent order')
        keys = lambda r: [(e['seed'], e['seat']) for e in r['episodes']]
        a, b = keys(new), keys(old)
        if len(set(a)) != len(a) or sorted(a) != sorted(b):
            raise ValueError('Unpaired regression comparison')
        rate = lambda r, outcome: sum(e['outcome'] == outcome for e in r['episodes']) / len(r['episodes'])
        tolerance = .05 if new['foe'] in ('evader', 'circler') else .10
        excess.append(max(0., rate(old, 'kill') - rate(new, 'kill') - tolerance))
        if new['foe'] == 'evader':
            evader_loss = max(0., rate(new, 'died') - rate(old, 'died'))
    return .25 * max(excess) + .25 * float(np.mean(excess)) + .25 * evader_loss


def rank(record, reference, weights=None):
    return (record['metrics']['objective'] + .05 * healthy_wins(record, weights)
            - regression_cost(record, reference),
            *ordering(record['metrics']))


def checkpoint_rank(record):
    m = record['metrics']
    return m['objective'], m['mean_win_rate'], healthy_wins(record), m['health']


def extension_warranted(previous, latest):
    """Decide at generation six from development only, before final selection."""
    a, b = previous['metrics'], latest['metrics']
    return bool(b['mean_win_rate'] >= a['mean_win_rate'] - 1e-12 and
        (b['objective'] - a['objective'] >= .0025 - 1e-12 or
         (b['objective'] >= a['objective'] - 1e-12 and
          healthy_wins(latest) - healthy_wins(previous) >= .01 - 1e-12)))


def timed_observe(own, foe, band, n):
    started = time.perf_counter()
    result = observe(own, foe, band, n)
    result['band'] = band
    result['elapsed_seconds'] = time.perf_counter() - started
    return result


class TracePool:
    def __init__(self, pool):
        self.pool = pool

    def submit(self, function, *args):
        if function is not duel:
            raise ValueError('Unexpected evaluation job')
        return self.pool.submit(timed_observe, *args)


class Evaluation:
    """One controller's exact-request memo; files preserve normal result layout.

    No cross-experiment cache. Caller verifies the frozen plan and installed
    package versions on resume. Reused records are never new independent games.
    """
    def __init__(self):
        self.memo = {}

    def __call__(self, pool, specs, opponents, band, n, out, weights=None, traced=False):
        if n <= 0 or n % 2 or not specs or not opponents:
            raise ValueError('Nonempty panel and complete seat pairs required')
        requests = [dict(spec=s, opponents=opponents, band=band, n=n, weights=weights) for s in specs]
        keys = [digest(dict(request=r, traced=traced)) for r in requests]
        immutable_json(out / 'schedule.json', dict(requests=requests, traced=traced))
        missing = {}
        for i, (key, request) in enumerate(zip(keys, requests)):
            path = out / f'candidate_{i:03d}.json'
            if path.exists():
                record = read(path)
                if record['request'] != request or any(('damage_traces' in x) != traced for x in record['results']):
                    raise ValueError('Changed saved evaluation')
                validate_record(record, request)
                self.memo[key] = record
            elif key not in self.memo:
                missing.setdefault(key, request)
        if missing:
            fresh = evaluate_candidates(TracePool(pool) if traced else pool,
                [r['spec'] for r in missing.values()], opponents, band, n,
                out / ('fresh_' + digest(list(missing))[:16]), weights)
            for key, record in zip(missing, fresh):
                validate_record(record, missing[key])
                self.memo[key] = record
        records = [deepcopy(self.memo[k]) for k in keys]
        for i, record in enumerate(records):
            immutable_json(out / f'candidate_{i:03d}.json', record)
        write(out / 'reuse.json', dict(unique_requests=len(set(keys)),
            executed_or_resumed_requests=len(missing), memo_or_saved_requests=len(set(keys))-len(missing),
            scope='Reused evidence, not independent replication; underlying fresh evaluator resumes existing jobs.'))
        return records


def candidate(out, plan, parameters):
    config = dict(parent=np.asarray(parameters).tolist(), interceptor=plan['initial_configuration']['interceptor'])
    if config == plan['initial_configuration']:
        return plan['anchor']
    key = digest(config)
    folder = out / 'models' / key[:20]
    if (folder / 'entrant.json').exists():
        if read(folder / 'policy_net.json') != config:
            raise ValueError('Candidate collision or changed configuration')
        for name, value in read(folder / 'artifact_sha256.json').items():
            if sha(folder / name) != value:
                raise ValueError('Changed candidate artifact')
        return read(folder / 'entrant.json')
    return export(folder, config, f'{out.name}_{key[:12]}')


def decode(vector, initial):
    # Preserve the explicit unchanged anchor exactly across normalize/decode.
    if np.array_equal(vector, (initial - LOW) / (HIGH - LOW)):
        return initial.copy()
    return LOW + vector * (HIGH - LOW)


def configuration_of(spec):
    folder = ROOT / spec['design']
    config = read(folder / 'policy_net.json')
    with zipfile.ZipFile(ROOT / spec['weights']) as archive:
        if json.loads(archive.read('parameters.json')) != config:
            raise ValueError('Warm-start JSON and loaded weights differ')
    if read(folder / 'entrant.json') != spec:
        raise ValueError('Warm-start entrant identity differs')
    for name, value in read(folder / 'artifact_sha256.json').items():
        if sha(folder / name) != value:
            raise ValueError('Warm-start artifact changed')
    return config


def freeze(out, previous, damage, workers=3):
    if not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError('Invalid simulation worker count')
    predecessors = [(previous, {'evaluation_failed', 'no_development_improvement'}),
        (damage, {'diagnostic_complete'}), (damage / 'analysis', {'analysis_complete'})]
    for folder, allowed in predecessors:
        if not (folder / 'completion.json').exists():
            if (folder / 'runtime.json').exists() and process_live(read(folder / 'runtime.json')['pid']):
                raise ValueError(f'Predecessor still live: {folder}')
            raise ValueError(f'Predecessor incomplete: {folder}')
        if read(folder / 'completion.json')['status'] not in allowed:
            raise ValueError(f'Unexpected predecessor terminal status: {folder}')
    if read(damage / 'analysis/completion.json')['status'] != 'analysis_complete':
        raise ValueError('Damage analysis is required')
    if read(damage / 'compatibility.json')['status'] != 'passed':
        raise ValueError('Actual damage logger compatibility is required')
    if (out / 'plan.json').exists():
        plan = read(out / 'plan.json')
        if (plan['previous'] != relative(previous) or plan['damage'] != relative(damage)
                or plan['environment'] != environment() or plan.get('workers', 3) != workers):
            raise ValueError('Changed predecessor or execution environment; use a new run')
        verify(plan)
        return plan
    old = read(previous / 'plan.json'); verify(old)
    finalists = [read(previous / f's{s}/result.json')['selected'] for s in old['seeds']]
    chosen = old['anchor']
    config = configuration_of(chosen)
    if set(config) != {'parent', 'interceptor'}:
        raise ValueError('Expected an exported interceptor configuration')
    opponents = old['opponents'] + [x['spec'] for x in finalists] + [chosen]
    development_files = sorted(previous.glob('s*/g*/development/candidate_000.json'))
    opponents += [read(path)['request']['spec'] for path in development_files]
    opponents = unique_entrants(opponents)
    if len({x['id'] for x in opponents}) != len(opponents):
        raise ValueError('Duplicate opponent identities')
    sources = dict(old['source_sha256'])
    for name in ('tools/league_regret_train.py', 'tools/league_damage_trace.py',
                 'tools/league_damage_analysis.py', 'tools/league_cached_evaluation.py'):
        sources[name] = sha(ROOT / name)
    inputs = dict(old['input_sha256'])
    extra = [previous / n for n in ('plan.json', 'completion.json', 'frozen_selection.json')]
    extra += [previous / f's{s}/result.json' for s in old['seeds']]
    extra += [damage / n for n in ('plan.json', 'completion.json', 'compatibility.json', 'analysis/report.json', 'analysis/completion.json')]
    extra += development_files
    for spec in opponents + [chosen]:
        if spec['kind'] == 'submission':
            model_sources = list((ROOT / spec['design']).glob('*.py'))
            if not model_sources or not (ROOT / spec['weights']).is_file():
                raise ValueError('Missing archived model sources or weights')
            extra += model_sources + [ROOT / spec['weights']]
            config_path = ROOT / spec['design'] / 'policy_net.json'
            if config_path.exists():
                extra.append(config_path)
    for path in extra:
        inputs[relative(path)] = sha(path)
    plan = dict(previous=relative(previous), damage=relative(damage), anchor=chosen,
        parent=old['parent'], reference=chosen,
        initial_configuration=config, opponents=opponents, seeds=[2700, 2701], workers=workers,
        base_generations=6, maximum_generations=8, population=12, elites=3,
        screen_n=4, confirmation_n=8, development_n=20, selection_n=40, final_n=80,
        training_band=180000000, development_band=47000000, selection_band=47010000, final_band=48000000,
        initial_std=.08, minimum_std=.035, environment=environment(),
        source_sha256=sources, input_sha256=inputs,
        method='17 parent coefficients; fixed interceptor and fixed public-motion gate. Two search RNGs share warm start and development conditions.',
        training_rank='Existing weighted objective + .05 * healthy wins - .25 * maximum excess per-foe win regression - .25 * mean excess regression - .25 * extra Evader loss rate. Paired unchanged anchor in every screening/confirmation batch. Excess tolerances: .10 normally, .05 Evader/Circler. No opponent reweighting change beyond the existing weakness curriculum.',
        checkpoint_rule='Only retain development candidates with nonnegative objective/win change and existing per-foe/lower-quarter/Evader-loss safeguards against both anchor and preserved champion; retain a separate record of rejected challengers.',
        budget_rule='At generation 6 compare retained checkpoints after generations 4 and 6. Extend to 8 if no mean-win regression and objective gain >= .0025, or objective nonregression and healthy-win gain >= .01. No final data used.',
        selection_rule='Versus both warm-start anchor and preserved champion: overall win gain >= .005 and objective gain >0; lower-quarter regression <=.02; no opponent win regression >.10; Evader/Circler regression <=.05; no extra Evader losses.',
        final_rule='Versus both warm-start anchor and preserved champion: paired IC-cluster overall win gain 95% lower >0 plus positive objective gain and the same regression guards. Individual weakness and damage findings reported separately, not additional significance gates.',
        initialization='Preserved warm start plus engineered left/right defense anchors (1200 m, 30 degrees, +/-90 degree offset, 450 kt); do not describe engineered initialization as a learned improvement.',
        scope='Official physics/actions/verdicts untouched. Full archive throughout parent search. Final includes damage traces without extra diagnostic matches. No GitHub publication.')
    immutable_json(out / 'plan.json', plan)
    for name in sources:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    return plan


def safeguards(c):
    return (c['lower_quarter_gain'] >= -.02 - 1e-12 and
        c['worst_opponent_win_gain'] >= -.10 - 1e-12 and
        all(c['per_opponent_win_gain'][k] >= -.05 - 1e-12 for k in ('evader', 'circler')) and
        c['evader_loss_increase'] <= 0)


def checkpoint_admissible(record, baselines):
    comparisons = [comparison(record['results'], base['results']) for base in baselines]
    return all(c['win_gain'] >= -1e-12 and c['objective_gain'] >= -1e-12 and safeguards(c)
               for c in comparisons)


def search(out, model_root, plan, seed, ordinal, base, evaluate, pool):
    if (out / 'result.json').exists():
        return read(out / 'result.json')
    baselines = base
    base = baselines[0]
    initial = np.asarray(plan['initial_configuration']['parent'])
    unit = lambda p: (p - LOW) / (HIGH - LOW)
    mean = unit(initial); incumbent = mean.copy(); std = np.full(len(mean), plan['initial_std'])
    rng = np.random.default_rng(seed)
    weakness = np.array([1 - x['summary']['rate'] for x in base['results']]) + .1
    weights = (.5 / len(weakness) + .5 * weakness / weakness.sum()).tolist()
    defense = []
    for direction in (-90., 90.):
        p = initial.copy(); p[[11, 12, 13, 14]] = [1200., 30., direction, 450.]
        defense.append(unit(p))
    best = deepcopy(base); checkpoints = []; history = []; budget = None
    for g in range(plan['maximum_generations']):
        verify(plan)
        population, anchors = sample_population(rng, mean, std,
            [unit(initial), incumbent, mean] + defense, plan['population'])
        specs = [candidate(model_root, plan, decode(v, initial)) for v in population]
        folder = out / f'g{g}'; band = plan['training_band'] + ordinal * 10000 + g * 100
        screened = evaluate(pool, specs, plan['opponents'], band, plan['screen_n'], folder / 'screen', weights)
        reference = screened[anchors[0]]
        order = sorted(range(len(specs)), key=lambda i: rank(screened[i], reference, weights), reverse=True)
        indices = list(dict.fromkeys(distinct_elites(order, population, plan['elites']) + [anchors[0], anchors[1]]))
        confirmed = evaluate(pool, [specs[i] for i in indices], plan['opponents'], band+20,
                             plan['confirmation_n'], folder / 'confirm', weights)
        reference = confirmed[indices.index(anchors[0])]
        order = sorted(range(len(indices)), key=lambda i: rank(confirmed[i], reference, weights), reverse=True)
        elites = distinct_elites([indices[i] for i in order], population, plan['elites'])
        incumbent = population[elites[0]].copy()
        mean = .5 * mean + .5 * population[elites].mean(axis=0)
        std = np.maximum(plan['minimum_std'], .5 * std + .5 * population[elites].std(axis=0))
        admitted = None
        if (g + 1) % 2 == 0:
            checked = evaluate(pool, [specs[elites[0]]], plan['opponents'], plan['development_band'],
                               plan['development_n'], folder / 'development')[0]
            admitted = checkpoint_admissible(checked, baselines)
            if admitted and checkpoint_rank(checked) > checkpoint_rank(best):
                best = checked
            checkpoints.append(deepcopy(best))
        state = dict(generation=g, best=dict(spec=best['request']['spec'], metrics=best['metrics'],
            healthy_wins=healthy_wins(best)), challenger=specs[elites[0]], elite_indices=elites,
            population=population.tolist(), anchor_indices=anchors, mean=mean.tolist(), std=std.tolist(),
            rng=rng.bit_generator.state, development_admissible=admitted,
            training_games=sum(r['metrics']['games'] for r in screened+confirmed),
            confirmation_regression_costs=[regression_cost(r, reference) for r in confirmed])
        immutable_json(folder / 'state.json', state); history.append(state)
        write(out / 'progress.json', dict(completed_generations=g+1, best=state['best']))
        if g + 1 == plan['base_generations']:
            extend = extension_warranted(checkpoints[-2], checkpoints[-1])
            budget = dict(after_generations=g+1, extend=extend, next_limit=8 if extend else 6,
                          rule=plan['budget_rule'], final_data_used=False)
            immutable_json(out / 'budget_decision.json', budget)
            if not extend:
                break
    result = dict(status='complete', seed=seed, selected=history[-1]['best'], history=history,
                  budget=budget, training_games=sum(h['training_games'] for h in history))
    write(out / 'result.json', result)
    return result


def run(out, previous, damage, workers=3):
    out, previous, damage = out.resolve(), previous.resolve(), damage.resolve()
    queue = out / 'queue_runtime.json'
    if not (out / 'completion.json').exists() and queue.exists():
        owner = read(queue)['pid']
        if owner != os.getpid() and process_live(owner):
            raise ValueError('Another controller owns the queued regret run')
    if (not (out / 'completion.json').exists() and (out / 'runtime.json').exists()
            and process_live(read(out / 'runtime.json')['pid'])):
        raise ValueError('Regret training already running')
    if (not (out / 'plan.json').exists() and (previous / 'completion.json').exists()
            and read(previous / 'completion.json')['status'] == 'evaluation_passed'):
        write(out / 'completion.json', dict(status='not_eligible_prior_passed', final_opened=False))
        write(out / 'progress.json', dict(stage='complete', status='not_eligible_prior_passed'))
        return
    plan = freeze(out, previous, damage, workers=workers)
    if (out / 'completion.json').exists():
        return
    write(out / 'runtime.json', dict(pid=os.getpid(), workers=plan['workers'], started_at=datetime.now(timezone.utc).isoformat()))
    evaluate = Evaluation()
    with ProcessPoolExecutor(max_workers=plan['workers'], initializer=initialize_worker) as pool:
        write(out / 'progress.json', dict(stage='shared_baseline'))
        base = evaluate(pool, [plan['anchor'], plan['parent']], plan['opponents'], plan['development_band'],
                        plan['development_n'], out / 'baseline')
        results = []
        for ordinal, seed in enumerate(plan['seeds']):
            write(out / 'progress.json', dict(stage='training', seed=seed, search=ordinal+1, searches=2))
            results.append(search(out/f's{seed}', out, plan, seed, ordinal, base, evaluate, pool))
        challengers = [r['selected']['spec'] for r in results]
        write(out / 'progress.json', dict(stage='selection'))
        records = evaluate(pool, [plan['anchor'], plan['parent']]+challengers, plan['opponents'],
                           plan['selection_band'], plan['selection_n'], out/'selection')
        comparisons = [comparison(r['results'], records[0]['results']) for r in records[2:]]
        champion_comparisons = [comparison(r['results'], records[1]['results']) for r in records[2:]]
        eligible = [i for i in range(len(challengers)) if all(c['win_gain'] >= .005-1e-12
            and c['objective_gain'] > 1e-12 and safeguards(c)
            for c in (comparisons[i], champion_comparisons[i]))]
        choice = max(eligible, key=lambda i:(checkpoint_rank(records[i+2]), -plan['seeds'][i])) if eligible else None
        chosen = challengers[choice] if choice is not None else plan['anchor']
        immutable_json(out/'frozen_selection.json', dict(chosen=chosen, eligible=eligible,
            chosen_index=choice, comparisons=comparisons, champion_comparisons=champion_comparisons,
            plan_sha256=sha(out/'plan.json')))
        completion = dict(status='no_development_improvement', final_opened=False, anchor_preserved=True)
        if choice is not None:
            claim = dict(run=relative(out), band=plan['final_band'],
                selection_sha256=sha(out/'frozen_selection.json'), plan_sha256=sha(out/'plan.json'))
            immutable_json(ROOT/f"runs/holdout_claim_{plan['final_band']}.json", claim)
            immutable_json(out/'final_claim.json', claim)
            write(out/'progress.json', dict(stage='final_evaluation', chosen=chosen['id']))
            final = evaluate(pool, [plan['anchor'], plan['parent'], chosen], plan['opponents'], plan['final_band'],
                             plan['final_n'], out/'final', traced=True)
            final_comparisons = []
            for baseline in final[:2]:
                cmp = comparison(final[2]['results'], baseline['results'])
                cmp['paired_gain'] = paired_ci(final[2]['results'], baseline['results'])
                final_comparisons.append(cmp)
            passed = all(c['paired_gain']['ci95'][0] > 0 and c['objective_gain'] > 0
                         and safeguards(c) for c in final_comparisons)
            damage_results = []
            for old, champion, new in zip(final[0]['results'], final[1]['results'], final[2]['results']):
                if old['foe'] != new['foe'] or champion['foe'] != new['foe']:
                    raise ValueError('Damage comparison opponent order mismatch')
                a = episode_rows(old, plan['final_band'], plan['final_n'])
                c = episode_rows(champion, plan['final_band'], plan['final_n'])
                b = episode_rows(new, plan['final_band'], plan['final_n'])
                damage_results.append(dict(foe=old['foe'], anchor=summarize(a), candidate=summarize(b),
                    champion=summarize(c), change_vs_anchor=paired_change(b, a), change_vs_champion=paired_change(b, c)))
            write(out/'damage_comparison.json', dict(results=damage_results,
                scope='Descriptive temporal damage from the final matches themselves; no extra matches or model selection from this report.'))
            completion = dict(status='evaluation_passed' if passed else 'evaluation_failed',
                final_opened=True, selected=chosen, selected_before_test=True,
                anchor_preserved=True, comparison=final_comparisons[1], anchor_comparison=final_comparisons[0])
    verify(plan)
    write(out/'completion.json', completion); write(out/'progress.json', dict(stage='complete', status=completion['status']))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=Path('runs/league_maneuver_parallel_20261006'))
    p.add_argument('--damage', type=Path, default=Path('runs/league_damage_trace_20261006'))
    p.add_argument('--after-pid', type=int, help='Wait for the prior maneuver controller, then run only after failure')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    try:
        if args.after_pid and not (args.out/'completion.json').exists():
            for name in ('queue_runtime.json', 'runtime.json'):
                path = args.out/name
                if path.exists() and process_live(read(path)['pid']):
                    raise ValueError('Regret controller already queued or running')
            prior = read(args.previous/'plan.json'); verify(prior)
            queued_sources = dict(prior['source_sha256'])
            for name in ('tools/league_regret_train.py', 'tools/league_damage_trace.py',
                         'tools/league_damage_analysis.py', 'tools/league_cached_evaluation.py'):
                queued_sources[name] = sha(ROOT/name)
            queued_environment = environment()
            write(args.out/'queue_runtime.json', dict(pid=os.getpid(), waiting_for_pid=args.after_pid,
                simulation_workers_while_waiting=0, workers_when_active=args.workers, source_sha256=queued_sources,
                environment=queued_environment, started_at=datetime.now(timezone.utc).isoformat()))
            while not (args.previous/'completion.json').exists() and process_live(args.after_pid):
                write(args.out/'progress.json', dict(stage='waiting_for_previous_experiment', pid=args.after_pid))
                time.sleep(45)
            if any(sha(ROOT/name) != value for name, value in queued_sources.items()):
                raise ValueError('Queued successor source changed')
            if environment() != queued_environment:
                raise ValueError('Queued successor environment changed')
        run(args.out, args.previous, args.damage, workers=args.workers)
    except Exception:
        write(args.out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
