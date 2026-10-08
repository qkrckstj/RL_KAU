"""Recompute completed matched pilots from official raw development games."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_tournament_metrics import code_groups, profile
from tools.league_sampled_ppo_continue import pair_rank


def analyze(source, destination):
    if destination.exists():
        raise FileExistsError('Preserve existing analyses')
    plan = io.read(source/'plan.json')
    completion = io.read(source/'completion.json')
    if completion['status'] not in ('matched_strategy_pilots_complete', 'matched_strategy_integration_passed'):
        raise ValueError('Completed matched pilot batch required')
    io.verify(plan)
    groups = code_groups(plan['development_opponents'])
    inputs = {io.relative(p): io.sha(p) for p in (source/'plan.json', source/'completion.json')}
    inputs[io.relative(Path(__file__))] = io.sha(Path(__file__))

    def read_pair(paths):
        profiles, records = [], []
        for path in paths:
            data = io.read(path)
            inputs[io.relative(path)] = io.sha(path)
            request = data['request']
            if (request['opponents'] != plan['development_opponents'] or
                    request['band'] != plan['development_band'] or request['n'] != plan['development_n']):
                raise ValueError('Changed development conditions')
            validate_records(data['results'], plan['development_opponents'], plan['development_band'], plan['development_n'])
            actual = profile(data['results'], groups)
            if actual != data['profile']:
                raise ValueError('Saved profile differs from raw games')
            profiles.append(actual)
            records.append(data['results'])
        wins = sum(r['summary']['wins'] for rs in records for r in rs)
        games = sum(len(r['episodes']) for rs in records for r in rs)
        foes = {foe: dict(wins=sum(next(r['summary']['wins'] for r in rs if r['foe'] == foe) for rs in records),
                         games=2*plan['development_n']) for foe in groups}
        return dict(wins=wins, games=games, uniform_win_rate=wins/games,
                    group_win_rate=sum(p['group_balanced_win_rate'] for p in profiles)/2,
                    rank=pair_rank(profiles), opponents=foes, replica_profiles=profiles)

    baseline = {arm: read_pair([source/arm/f'baseline_action_{k}.json' for k in plan['action_replicas']])
                for arm in ('control', 'lower_prior')}
    rows = []
    for arm in plan['arms']:
        for seed in plan['seeds']:
            folder = source/arm['name']/f's{seed}'
            result_path = folder/'result.json'
            result = io.read(result_path)
            inputs[io.relative(result_path)] = io.sha(result_path)
            matching = [r for r in completion['searches'] if r['arm'] == arm['name'] and r['seed'] == seed]
            if len(matching) != 1 or any(matching[0][k] != v for k, v in result.items()):
                raise ValueError('Completion/result mismatch')
            last = result['history'][-1]
            checkpoint = io.ROOT/last['candidate']['learner']
            candidate = read_pair([checkpoint.parent/f'development_{k}.json' for k in plan['action_replicas']])
            if candidate['rank'] != last['candidate']['rank']:
                raise ValueError('Candidate rank mismatch')
            reference = baseline['lower_prior' if arm['prior_bias'] == 3 else 'control']
            rows.append(dict(arm=arm['name'], seed=seed, additional_steps=result['additional_steps'],
                candidate=candidate, baseline=reference,
                win_gain=candidate['uniform_win_rate']-reference['uniform_win_rate'],
                group_win_gain=candidate['group_win_rate']-reference['group_win_rate'],
                selected=result['selected'], learning_seconds=last['learning_seconds'],
                evaluation_seconds=last['evaluation_seconds']))
    expected_steps=sum(r['additional_steps'] for r in rows)
    if expected_steps != completion['additional_training_steps']:
        raise ValueError('Training budget mismatch')
    arms={arm['name']: dict(mean_win_rate=sum(r['candidate']['uniform_win_rate'] for r in rows if r['arm']==arm['name'])/len(plan['seeds']),
                           mean_group_win_rate=sum(r['candidate']['group_win_rate'] for r in rows if r['arm']==arm['name'])/len(plan['seeds']),
                           seed_win_rates=[r['candidate']['uniform_win_rate'] for r in rows if r['arm']==arm['name']]) for arm in plan['arms']}
    answer=dict(status='raw_matched_pilots_verified', searches=rows, arms=arms,
                additional_training_steps=expected_steps, baseline_bias6=baseline['control'],
                input_sha256=inputs, final_opened=False, heldout_opened=False, promoted=False,
                scope='Reused sparse development panel with two fixed action replicas. Learner repeats share a pretrained prefix. No confidence or independent final-generalization claim.')
    io.write(destination,answer)
    print({k:v for k,v in answer.items() if k in ('status','arms','additional_training_steps')},flush=True)
    return answer


if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.source.resolve(),args.out.resolve())
