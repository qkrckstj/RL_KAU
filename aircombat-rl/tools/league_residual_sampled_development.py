"""Choose a continuation checkpoint using existing development conditions.

The action-mode diagnostic motivates inspecting already-trained checkpoints
under sampled deployment before spending another long training budget. Two
fixed action replicas are averaged equally; no best action seed is selected.
This consumes no new test panel and does not promote any policy.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from tools import league_thread_benchmark as io


def freeze(source, diagnostic, benchmark, out):
    from tools.league_residual_action_mode import build_variant
    from tools.league_tournament_metrics import code_groups
    if (out / 'plan.json').exists():
        raise FileExistsError('Use a fresh sampled-development directory')
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS):
        raise ValueError('Set thread limits before imports')
    old = io.read(source / 'plan.json')
    io.verify(old)
    if io.read(source / 'completion.json')['status'] != 'selection_failed':
        raise ValueError('Expected completed continuation')
    if not io.read(diagnostic / 'completion.json')['hypothesis_supported_on_this_panel']:
        raise ValueError('Require the completed action-mode diagnosis')
    if io.read(benchmark / 'completion.json')['status'] != 'benchmark_complete':
        raise ValueError('Grouped benchmark must finish first')
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for name in ('tools/league_residual_sampled_development.py', 'tools/league_residual_action_mode.py'):
        sources[name] = io.sha(io.ROOT / name)
    candidates, models, paths = [], [], []
    for seed in old['seeds']:
        search = io.read(source / f's{seed}/result.json')
        for chunk in search['history']:
            spec = chunk['candidate']['spec']
            folder = (io.ROOT / spec['design']).parent
            checkpoint = io.read(folder / 'checkpoint.json')
            if io.sha(folder / 'learner.zip') != checkpoint['learner_sha256']:
                raise ValueError('Changed continuation learner')
            candidate = dict(training_seed=seed, step=checkpoint['step'], learner=io.relative(folder / 'learner.zip'),
                             greedy=spec, replicas=[])
            for action_seed in (4900, 4901):
                identity = f"sampled_s{seed}_t{checkpoint['step']}_a{action_seed}"
                variant = build_variant(spec, out / 'models' / identity, action_seed, identity)
                candidate['replicas'].append(dict(action_seed=action_seed, spec=variant))
                models.append(variant)
            candidates.append(candidate)
            paths.extend([folder / 'learner.zip', folder / 'checkpoint.json', io.ROOT / spec['weights'], io.ROOT / spec['design'] / 'policy.py'])
    if len(candidates) != 8:
        raise ValueError('Expected all eight completed continuation checkpoints')
    opponents = old['development_opponents']
    for spec in models:
        paths.extend((io.ROOT / spec['design']).glob('*'))
    paths.extend([source / 'plan.json', source / 'completion.json', diagnostic / 'completion.json',
                  diagnostic / 'analysis.json', benchmark / 'completion.json', benchmark / 'analysis.json'])
    for path in paths:
        if path.is_file(): inputs[io.relative(path)] = io.sha(path)
    plan = dict(source=io.relative(source), diagnostic=io.relative(diagnostic), benchmark=io.relative(benchmark),
        candidates=candidates, opponents=opponents, groups=code_groups(opponents),
        band=old['development_band'], n=old['development_n'], workers=4,
        action_replicas=[4900, 4901], games=len(models) * len(opponents) * old['development_n'],
        rank_rule='Mean across both fixed action replicas of the average uniform/code-group win rate; then minimum replica lower-quarter score; then minimum worst score; then negative maximum losing-matchup count. Earlier checkpoint wins exact ties. Choose training warm start only; do not select an action RNG.',
        source_sha256=sources, input_sha256=inputs, environment=io.environment(),
        scope='Exploratory continuation-source selection on already-used sparse development conditions. No fresh final/holdout, promotion, new policy training or GitHub. Original weights and all replicas retained.')
    io.verify(plan)
    io.write(out / 'plan.json', plan)
    for name in sources:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT / name, target)
    return plan


def run(source, diagnostic, benchmark, out):
    plan = freeze(source, diagnostic, benchmark, out)
    if min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) < 4.5:
        raise MemoryError('Insufficient evaluation startup headroom')
    from tools.league_matches import evaluate_jobs
    from tools.league_tournament_metrics import profile
    from tools.league_residual_train import validate_records
    io.write(out / 'runtime.json', dict(pid=os.getpid(), workers=4, started_at=datetime.now(timezone.utc).isoformat()))
    jobs = [dict(own=replica['spec'], foe=foe, band=plan['band'], n=plan['n'])
            for candidate in plan['candidates'] for replica in candidate['replicas'] for foe in plan['opponents']]
    io.write(out / 'jobs.json', jobs)
    started = time.perf_counter()
    print(f"Inspecting all8 checkpoints x2 fixed action replicas on reused development: {plan['games']} games.", flush=True)
    with io.Sampler() as sampler:
        results = evaluate_jobs(jobs, out / 'matches', workers=4)
    reports = []
    for candidate in plan['candidates']:
        profiles = []
        for replica in candidate['replicas']:
            rows = [r for r in results if r['own'] == replica['spec']['id']]
            validate_records(rows, plan['opponents'], plan['band'], plan['n'])
            profiles.append(profile(rows, plan['groups']))
        rank = [sum(.5 * (p['mean_win_rate'] + p['group_balanced_win_rate']) for p in profiles) / len(profiles),
                min(p['lower_quarter_score'] for p in profiles), min(p['worst_score'] for p in profiles),
                -max(p['losing_matchups'] for p in profiles)]
        reports.append(dict(**candidate, replica_profiles=profiles, rank=rank))
    chosen = max(reports, key=lambda r: (*r['rank'], -r['step'], -r['training_seed']))
    io.write(out / 'warm_start_choice.json', dict(chosen=chosen, candidates=reports, rank_rule=plan['rank_rule'],
        scope='Selected learner/optimizer initialization for a future independent continuation; both action replicas averaged, neither action seed promoted. All prior failed selection outcomes remain valid.'))
    io.verify(plan)
    io.write(out / 'completion.json', dict(status='sampled_development_complete', games=plan['games'],
        additional_training_steps=0, elapsed_seconds=time.perf_counter() - started, resources=sampler.result,
        chosen_training_seed=chosen['training_seed'], chosen_step=chosen['step'], chosen_rank=chosen['rank'],
        promotion=False, completed_at=datetime.now(timezone.utc).isoformat(), scope=plan['scope']))
    print(json.dumps(io.read(out / 'completion.json')), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('source', 'diagnostic', 'benchmark', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.source.resolve(), args.diagnostic.resolve(), args.benchmark.resolve(), args.out.resolve())
    except BaseException:
        io.write(args.out / 'failure.json', dict(error=traceback.format_exc(), pid=os.getpid(), at=datetime.now(timezone.utc).isoformat()))
        raise
