"""Post-completion paired comparisons for every rate nominee versus its source.

Consumes only the completed, raw-verified assessment. No simulations, policy
selection, promotion or training launch is performed by this report.
"""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_policy_assessment_metrics import gains
from tools.league_tournament_metrics import code_groups, group_weights


def analyze(out):
    raw_path = out/'raw_rate_comparison.json'
    raw = io.read(raw_path)
    assert raw['status'] == 'rate_comparison_raw_verified'
    assert raw['games'] == 15760
    for name, digest in raw['input_sha256'].items():
        assert io.sha(io.ROOT/name) == digest, name
    plan = io.read(out/'plan.json')
    io.verify(plan)
    rows = [io.read(out/f'matches/match_{i:04d}.json')['result']
            for i in range(len(plan['jobs']))]
    analysis = raw['analysis']
    panels = dict(all=[s['id'] for s in plan['opponents']], **plan['partitions'])
    result = {}
    fields = ('mean_win_rate', 'group_balanced_win_rate',
              'conservative_lower_quarter', 'worst_score')
    for name, ids in panels.items():
        stats = analysis if name == 'all' else analysis['partitions'][name]
        foes = [s for s in plan['opponents'] if s['id'] in ids]
        grouped = code_groups(foes)
        weights = dict(uniform={s['id']: 1/len(foes) for s in foes},
                       code_group=group_weights(grouped))
        def records(role):
            return [[r for r in rows if r['own'] == s['id'] and r['foe'] in ids]
                    for s in plan['roles'][role]]
        source = stats['roles']['baseline']
        result[name] = {}
        for role in ('candidate', 'repeat', 'low_rate', 'teacher', 'cem'):
            current = stats['roles'][role]
            paired = {key: gains(records(role), records('baseline'), foes,
                                plan['band'], plan['n'], w)
                      for key, w in weights.items()}
            if role == 'candidate':
                assert paired == stats['comparisons']['baseline']
            deltas = {key: current[key]-source[key] for key in fields}
            per_foe = []
            for foe in foes:
                k = foe['id']; a = current['opponents'][k]; b = source['opponents'][k]
                per_foe.append(dict(opponent=k,
                    win_gain=a['wins']/a['games']-b['wins']/b['games'],
                    loss_gain=a['losses']/a['games']-b['losses']/b['games'],
                    current=a, source=b))
            result[name][role] = dict(
                metrics={key: current[key] for key in fields}, deltas=deltas,
                source_metrics={key: source[key] for key in fields}, paired=paired,
                worst_win_changes=sorted(per_foe, key=lambda r: (r['win_gain'], r['opponent']))[:8],
                largest_loss_increases=sorted(per_foe, key=lambda r: (-r['loss_gain'], r['opponent']))[:8])
    artifact = dict(status='rate_transfer_comparisons_verified', panels=result,
        policy_promoted=False,
        scope='Descriptive paired source comparisons, not a new significance or extension gate. Four ICs and fixed action replicas; shared learned ancestors. Known-family parameter panel is consumed after this assessment. No inference over arbitrary tournament entrants. Continuation requires interpreting mean, group, tail and repeat evidence together.',
        input_sha256={io.relative(raw_path): io.sha(raw_path),
                      io.relative(Path(__file__).resolve()): io.sha(Path(__file__).resolve())})
    io.write(out/'transfer_source_comparisons.json', artifact)
    return artifact


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = analyze(args.folder.resolve())
    print(result['status'])
    for panel, roles in result['panels'].items():
        print(panel)
        for role, r in roles.items():
            print(role, r['metrics'], 'delta', r['deltas'],
                  'paired_ci', r['paired']['uniform']['crossed_ic_action_ci95'])
