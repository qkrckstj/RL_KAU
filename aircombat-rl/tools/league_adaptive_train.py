"""Learn an observation-only gate between frozen league and pursuit experts.

Three independent gate searches share the same frozen expert pair and archive.
Training, checkpoint development, selection confirmation and final ICs differ.
Existing league sources/results are inputs, never overwritten by this driver.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys
import time
import traceback
import numpy as np
from experiments.league.adaptive import GATE_INITIAL, GATE_LOW, GATE_HIGH
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_adaptive_export import export
from tools.league_matches import duel, initialize_worker
from tools.league_train import (evaluate_candidates, metrics, ordering,
                               source_paths, unique_entrants, verify)

PARENT_GATE = [.5, -8., 0., 0., 0., 0.]
PURSUIT_GATE = [.5, 8., 0., 0., 0., 0.]


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def immutable_json(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f'Cannot change frozen record: {path}')
    else:
        write(path, data)


def candidate(out, experts, gate, identity):
    configuration = dict(experts=experts, gate=np.asarray(gate, float).tolist())
    if (out/'entrant.json').exists():
        if read(out/'policy_net.json') != configuration:
            raise ValueError('Cannot change exported candidate')
        record = read(out/'artifact_sha256.json')
        for name, digest in record.items():
            if sha(out/name) != digest:
                raise ValueError(f'Candidate artifact changed: {out/name}')
        spec = read(out/'entrant.json')
        if spec['id'] != identity:
            raise ValueError('Candidate identity changed')
        return spec
    spec = export(out, configuration, identity)
    write(out/'artifact_sha256.json', {
        name: sha(out/name) for name in ('policy.py', 'wrappers.py',
                                        'policy_net.json', 'policy_net.zip', 'entrant.json')})
    return spec


def gate_of(spec):
    return np.asarray(read(ROOT/spec['design']/'policy_net.json')['gate'])


def freeze(out, previous):
    old = read(previous/'main/plan.json')
    verify(old)
    if (out/'plan.json').exists():
        plan = read(out/'plan.json')
        if plan['previous'] != relative(previous):
            raise ValueError('Changed previous experiment')
        verify(plan)
        return plan
    selection = read(previous/'followup_pool/frozen_selection.json')
    probe = read(previous/'runner_probe/completion.json')['best']
    if not read(previous/'runner_confirm/completion.json')['conditional_training_warranted']:
        raise ValueError('Runner confirmation did not support this branch')
    experts = [dict(parameters=selection['chosen']['parameters']),
               dict(parameters=probe['parameters'])]
    opponents = unique_entrants(selection['opponents']+selection['selected']+[probe])
    paths = source_paths()+['experiments/league/adaptive.py',
                           'tools/league_adaptive_export.py', 'tools/league_adaptive_train.py']
    inputs = set(old['input_sha256'])
    for foe in opponents:
        if foe['kind'] == 'submission':
            inputs.add(foe['weights'])
            inputs.update(relative(p) for p in (ROOT/foe['design']).glob('*.py'))
    inputs.update(relative(previous/p) for p in (
        'main/plan.json', 'followup_pool/frozen_selection.json',
        'runner_probe/completion.json', 'runner_confirm/completion.json'))
    plan = dict(previous=relative(previous), parent=selection['chosen'], experts=experts,
        opponents=opponents, seeds=[2300, 2301, 2302], workers=3,
        generations=3, population=10, elites=3, screen_n=4, confirmation_n=8,
        development_n=20, initial_std=.22, minimum_std=.04,
        smoke_band=33900000, training_band=130000000,
        development_band=34000000, selection_band=34010000, selection_n=40,
        final_band=41000000, final_n=80,
        selection_rule='Runner win gain >= .05; overall win regression <= .005; objective improves; lower-quarter score regression <= .02; no opponent win regression > .15; no extra Evader losses. Then highest objective/wins/health, then smaller seed.',
        final_rule='Chosen policy: paired-IC overall win gain 95% CI lower > 0, Evader gain CI lower > 0, overall win gain > 0, lower-quarter score regression <= .02, no opponent win regression > .15, no extra Evader losses.',
        scope='Public 39-value observations only; 6 learned gate coefficients, 2 frozen 17-coefficient experts. All opponents are observed local policies, not unseen student submissions. Independent gate RNG searches share both experts.',
        source_sha256={p:sha(ROOT/p) for p in paths},
        input_sha256={p:sha(ROOT/p) for p in sorted(inputs)},
        created_at=datetime.now(timezone.utc).isoformat())
    immutable_json(out/'plan.json', plan)
    for name in plan['source_sha256']:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return plan


def smoke(out, plan, anchor):
    """Real flights through exported loader must equal native frozen parent."""
    path = out/'smoke.json'
    if path.exists():
        if read(path)['status'] != 'passed':
            raise ValueError('Prior compatibility check failed')
        return
    foe = dict(id='evader', kind='bot', name='evader')
    native = duel(plan['parent'], foe, plan['smoke_band'], 2)
    exported = duel(anchor, foe, plan['smoke_band'], 2)
    same = native['episodes'] == exported['episodes']
    write(path, dict(status='passed' if same else 'failed',
        native=native, exported=exported, exact_episode_records_equal=same,
        scope='Actual full flights in both seats; compatibility only, not performance evidence'))
    if not same:
        raise ValueError('Constant gate changed official flight outcomes')


def search(out, plan, seed, ordinal, anchor):
    if (out/'result.json').exists():
        return read(out/'result.json')
    rng = np.random.default_rng(seed)
    normal = lambda g: (np.asarray(g)-GATE_LOW)/(GATE_HIGH-GATE_LOW)
    mean = normal(GATE_INITIAL)
    incumbent = mean.copy()
    std = np.full(len(mean), plan['initial_std'])
    spec = dict(seed=seed, training_band=plan['training_band']+ordinal*10000,
                development_band=plan['development_band'], plan_sha256=sha(out.parent/'plan.json'))
    immutable_json(out/'search_plan.json', spec)
    with ProcessPoolExecutor(max_workers=plan['workers'], initializer=initialize_worker) as pool:
        first = evaluate_candidates(pool, [anchor], plan['opponents'], plan['development_band'],
                                    plan['development_n'], out/'initial')[0]
        weakness = np.array([1-r['summary']['rate'] for r in first['results']])+.1
        weights = (.5/len(weakness)+.5*weakness/weakness.sum()).tolist()
        best = dict(spec=anchor, metrics=metrics(first['results'], weights), generation=-1)
        history = []
        for generation in range(plan['generations']):
            verify(plan)
            population = np.clip(rng.normal(mean, std, (plan['population'], len(mean))), 0, 1)
            population[0] = normal(PARENT_GATE)
            population[1] = incumbent
            population[2] = mean
            population[3] = normal(GATE_INITIAL)
            population[4] = normal(PURSUIT_GATE)
            population[-2:] = rng.random((2, len(mean)))
            gates = GATE_LOW+population*(GATE_HIGH-GATE_LOW)
            specs = [candidate(out/f'g{generation}/models/c{i}', plan['experts'], gate,
                               f'gate_s{seed}_g{generation}_c{i}') for i, gate in enumerate(gates)]
            band = spec['training_band']+generation*100
            screen = evaluate_candidates(pool, specs, plan['opponents'], band,
                plan['screen_n'], out/f'g{generation}/screen', weights)
            order = sorted(range(len(specs)), key=lambda i:ordering(screen[i]['metrics']), reverse=True)
            indices = list(dict.fromkeys(order[:plan['elites']]+[0]))
            confirmed = evaluate_candidates(pool, [specs[i] for i in indices], plan['opponents'],
                band+20, plan['confirmation_n'], out/f'g{generation}/confirm', weights)
            order = sorted(range(len(confirmed)), key=lambda i:ordering(confirmed[i]['metrics']), reverse=True)
            elites = [indices[i] for i in order[:plan['elites']]]
            incumbent = population[elites[0]].copy()
            mean = .5*mean+.5*population[elites].mean(axis=0)
            std = np.maximum(plan['minimum_std'], .5*std+.5*population[elites].std(axis=0))
            challenger = specs[elites[0]]
            checked = evaluate_candidates(pool, [challenger], plan['opponents'], plan['development_band'],
                plan['development_n'], out/f'g{generation}/development', weights)[0]
            if ordering(checked['metrics']) > ordering(best['metrics']):
                best = dict(spec=challenger, metrics=checked['metrics'], generation=generation)
            record = dict(generation=generation, challenger=challenger, metrics=checked['metrics'],
                best=best, training_games=sum(r['metrics']['games'] for r in screen+confirmed),
                mean=mean.tolist(), std=std.tolist(), rng=rng.bit_generator.state, elite_indices=elites)
            immutable_json(out/f'g{generation}/state.json', record)
            history.append(record)
            write(out/'progress.json', dict(stage='training', generation=generation+1,
                generations=plan['generations'], selected=best))
    result = dict(status='complete', seed=seed, selected=best, weights=weights,
                  history=history, training_games=sum(r['training_games'] for r in history))
    write(out/'result.json', result)
    return result


def comparison(candidate_results, parent_results):
    cm, pm = metrics(candidate_results), metrics(parent_results)
    by_foe = {r['foe']:r for r in candidate_results}
    parent_by_foe = {r['foe']:r for r in parent_results}
    if set(by_foe) != set(parent_by_foe):
        raise ValueError('Opponent panels differ')
    gain = {foe:by_foe[foe]['summary']['rate']-parent_by_foe[foe]['summary']['rate'] for foe in by_foe}
    return dict(candidate=cm, parent=pm, win_gain=cm['mean_win_rate']-pm['mean_win_rate'],
        objective_gain=cm['objective']-pm['objective'],
        lower_quarter_gain=cm['lower_quarter_score']-pm['lower_quarter_score'],
        per_opponent_win_gain=gain, worst_opponent_win_gain=min(gain.values()),
        evader_loss_increase=by_foe['evader']['summary']['losses']-parent_by_foe['evader']['summary']['losses'])


def development_pass(c):
    return bool(c['per_opponent_win_gain']['evader'] >= .05-1e-12 and
        c['win_gain'] >= -.005-1e-12 and c['objective_gain'] > 1e-12 and
        c['lower_quarter_gain'] >= -.02-1e-12 and c['worst_opponent_win_gain'] >= -.15-1e-12 and
        c['evader_loss_increase'] <= 0)


def paired_ci(new_results, old_results):
    def matrix(results):
        return {r['foe']:{(e['seed'], e['seat']):int(e['won']) for e in r['episodes']} for r in results}
    new, old = matrix(new_results), matrix(old_results)
    if set(new) != set(old):
        raise ValueError('Paired CI opponent mismatch')
    keys = set(next(iter(old.values())))
    for foe in old:
        if set(new[foe]) != keys or set(old[foe]) != keys:
            raise ValueError('Paired CI seed/seat mismatch')
    seeds = sorted({k[0] for k in keys})
    if keys != {(s, seat) for s in seeds for seat in ('red', 'blue')}:
        raise ValueError('Incomplete seat pairs')
    values = np.array([np.mean([new[f][s, seat]-old[f][s, seat]
                                for f in old for seat in ('red', 'blue')]) for s in seeds])
    rng = np.random.default_rng(723)
    limits = np.quantile(values[rng.integers(len(seeds), size=(10000, len(seeds)))].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), ci95=limits.tolist(), shared_initial_conditions=len(seeds),
                scope='IC-cluster bootstrap, both seats and all fixed opponents kept together; conditional on learned policies')


def final_analysis(new_results, old_results):
    c = comparison(new_results, old_results)
    c['paired_gain'] = paired_ci(new_results, old_results)
    c['evader_paired_gain'] = paired_ci([r for r in new_results if r['foe']=='evader'],
                                      [r for r in old_results if r['foe']=='evader'])
    c['passed'] = bool(c['paired_gain']['ci95'][0] > 0 and c['evader_paired_gain']['ci95'][0] > 0
        and c['win_gain'] > 0 and c['lower_quarter_gain'] >= -.02-1e-12
        and c['worst_opponent_win_gain'] >= -.15-1e-12 and c['evader_loss_increase'] <= 0)
    return c


def run(out, previous, smoke_only=False):
    out = out.resolve()
    plan = freeze(out, previous.resolve())
    if (out/'completion.json').exists():
        return read(out/'completion.json')
    anchor = candidate(out/'models/parent_gate', plan['experts'], PARENT_GATE, 'gate_parent')
    write(out/'runtime.json', dict(pid=os.getpid(), executable=sys.executable,
        workers=plan['workers'], started_at=datetime.now(timezone.utc).isoformat()))
    write(out/'progress.json', dict(stage='actual_flight_compatibility'))
    smoke(out, plan, anchor)
    if smoke_only:
        return dict(status='smoke_passed')
    results = []
    for i, seed in enumerate(plan['seeds']):
        write(out/'progress.json', dict(stage='training', seed=seed, replication=i+1,
            replications=len(plan['seeds']), workers=plan['workers']))
        results.append(search(out/f's{seed}', plan, seed, i, anchor))
    challengers = [r['selected']['spec'] for r in results]
    write(out/'progress.json', dict(stage='development_selection'))
    with ProcessPoolExecutor(max_workers=plan['workers'], initializer=initialize_worker) as pool:
        records = evaluate_candidates(pool, [anchor]+challengers, plan['opponents'],
            plan['selection_band'], plan['selection_n'], out/'selection')
        comparisons = [comparison(r['results'], records[0]['results']) for r in records[1:]]
        eligible = [i for i,c in enumerate(comparisons) if development_pass(c)]
        choice = max(eligible, key=lambda i:(*ordering(comparisons[i]['candidate']), -plan['seeds'][i])) if eligible else None
        selection = dict(challengers=challengers, comparisons=comparisons, eligible=eligible,
            chosen_index=choice, chosen=challengers[choice] if choice is not None else anchor,
            plan_sha256=sha(out/'plan.json'), final_opened=False)
        immutable_json(out/'frozen_selection.json', selection)
        if choice is None:
            completion = dict(status='no_development_improvement', final_opened=False,
                parent_preserved=True, comparisons=comparisons,
                next='Inspect gate routing and expert tradeoffs; preserve all candidates and use a new plan for a changed policy family.')
        else:
            verify(plan)
            claim = dict(band=plan['final_band'], selection_sha256=sha(out/'frozen_selection.json'),
                         plan_sha256=sha(out/'plan.json'))
            immutable_json(out/'final_claim.json', claim)
            write(out/'progress.json', dict(stage='final_evaluation', chosen=challengers[choice]['id']))
            final = evaluate_candidates(pool, [anchor]+challengers, plan['opponents'],
                plan['final_band'], plan['final_n'], out/'final')
            analysis = [final_analysis(r['results'], final[0]['results']) for r in final[1:]]
            completion = dict(status='evaluation_passed' if analysis[choice]['passed'] else 'evaluation_failed',
                final_opened=True, selected=challengers[choice], selected_before_test=True,
                chosen_index=choice, comparisons=analysis, parent_preserved=True,
                next='Audit real-loader compatibility, routing, all artifacts and replication results before promotion/publication.')
    verify(plan)
    write(out/'completion.json', completion)
    write(out/'progress.json', dict(stage='complete', status=completion['status']))
    return completion


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('runs/league_20261006'))
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.out, args.previous, args.smoke_only), ensure_ascii=False))
    except Exception as error:
        write(args.out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                          at=datetime.now(timezone.utc).isoformat()))
        raise
