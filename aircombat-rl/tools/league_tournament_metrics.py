"""A separately declared aggregate/absolute-weakness assessment profile.

Code groups are a sensitivity analysis, not a tournament population estimate.
The former per-opponent relative-regression gates remain reported separately.
"""
from collections import Counter
import hashlib
import json
import numpy as np

from tools.autolab_cem import ROOT, sha
from tools.league_train import metrics
from tools.league_symmetric_train import comparison
from tools.league_repair_train import safeguards


def code_groups(opponents):
    groups = {}
    for foe in opponents:
        kind = foe['kind']
        if kind == 'submission':
            files = sorted((ROOT / foe['design']).glob('*.py'))
            if not files:
                raise ValueError('Submission has no Python sources')
            signature = [(p.name, sha(p)) for p in files]
            group = 'submission:' + hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        elif kind == 'bot':
            group = 'bot:' + foe['name']
        else:
            group = kind
        if foe['id'] in groups:
            raise ValueError('Duplicate opponent')
        groups[foe['id']] = group
    return groups


def group_weights(groups):
    counts = Counter(groups.values())
    return {foe: 1 / (len(counts) * counts[group]) for foe, group in groups.items()}


def profile(results, groups):
    if len(results) != len(groups) or {r['foe'] for r in results} != set(groups):
        raise ValueError('Profile opponent mismatch')
    weights = group_weights(groups)
    return dict(**metrics(results),
        group_balanced_win_rate=sum(weights[r['foe']] * r['summary']['rate'] for r in results),
        losing_matchups=sum(r['summary']['wins'] < r['summary']['losses'] for r in results),
        groups=len(set(groups.values())))


def paired_gain(new_results, old_results, weights):
    def matrix(results):
        answer = {}
        for row in results:
            episodes = {(e['seed'], e['seat']): int(e['won']) for e in row['episodes']}
            if row['foe'] in answer or len(episodes) != len(row['episodes']):
                raise ValueError('Duplicate opponent or episode')
            answer[row['foe']] = episodes
        return answer
    new, old = matrix(new_results), matrix(old_results)
    if not old or set(new) != set(old) or set(weights) != set(old):
        raise ValueError('Paired opponent mismatch')
    if any(w <= 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1.):
        raise ValueError('Positive normalized weights required')
    keys = set(next(iter(old.values())))
    seeds = sorted({seed for seed, _ in keys})
    if not seeds or keys != {(s, seat) for s in seeds for seat in ('red', 'blue')}:
        raise ValueError('Incomplete seat pairs')
    if any(set(rows[f]) != keys for rows in (new, old) for f in old):
        raise ValueError('Unpaired initial conditions')
    values = np.array([sum(weights[f] * sum(new[f][s, seat] - old[f][s, seat]
        for seat in ('red', 'blue')) / 2 for f in old) for s in seeds])
    rng = np.random.default_rng(723)
    limits = np.quantile(values[rng.integers(len(seeds), size=(10000, len(seeds)))].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), ci95=limits.tolist(), shared_initial_conditions=len(seeds),
        scope='IC-cluster bootstrap; both seats/all fixed foes kept together; conditional on these policies and this panel, not uncertainty over future opponents.')


def compare(new_results, old_results, groups, final=False):
    old, new = profile(old_results, groups), profile(new_results, groups)
    legacy = comparison(new_results, old_results)
    result = dict(candidate=new, reference=old,
        gains={k: new[k] - old[k] for k in ('mean_win_rate', 'group_balanced_win_rate',
            'lower_quarter_score', 'worst_score', 'losing_matchups')},
        evader_loss_increase=legacy['evader_loss_increase'],
        per_opponent_win_gain=legacy['per_opponent_win_gain'],
        legacy_safeguards_passed=bool(safeguards(legacy)),
        legacy_checkpoint_passed=bool(safeguards(legacy) and legacy['win_gain'] >= -1e-12
            and legacy['objective_gain'] >= -1e-12))
    if final:
        result['uniform_paired_gain'] = paired_gain(new_results, old_results, {f: 1 / len(groups) for f in groups})
        result['group_paired_gain'] = paired_gain(new_results, old_results, group_weights(groups))
    result['profile_passed'] = passed(result, final)
    return result


def passed(result, final=False):
    g = result['gains']
    ok = (g['mean_win_rate'] >= .005 - 1e-12 and g['group_balanced_win_rate'] >= .005 - 1e-12
        and g['lower_quarter_score'] >= -1e-12 and g['worst_score'] >= -1e-12
        and g['losing_matchups'] <= 0 and result['evader_loss_increase'] <= 0)
    if final:
        ok = ok and all(result[k]['ci95'][0] > 0 for k in ('uniform_paired_gain', 'group_paired_gain'))
    return bool(ok)


def ranking(p):
    return (.5 * (p['mean_win_rate'] + p['group_balanced_win_rate']),
        p['lower_quarter_score'], p['worst_score'], -p['losing_matchups'])
