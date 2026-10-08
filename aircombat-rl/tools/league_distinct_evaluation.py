"""Evaluate exact duplicate entrant specifications once in future experiments.

No behavioural equivalence is inferred from IDs, parameters or checkpoint names.
The immutable index map preserves which independent searches chose each model.
"""
from copy import deepcopy
import json
from tools.league_adaptive_train import immutable_json
from tools.league_train import evaluate_candidates


def evaluate_distinct_candidates(pool, candidates, opponents, band, n, out, weights=None):
    unique, indices, keys = [], [], {}
    for spec in candidates:
        key = json.dumps(spec, sort_keys=True, allow_nan=False)
        if key not in keys:
            keys[key] = len(unique)
            unique.append(spec)
        indices.append(keys[key])
    immutable_json(out/'index_map.json', dict(candidates=candidates,
        unique_candidates=unique, indices=indices, opponents=opponents,
        band=band, n=n, weights=weights,
        scope='Exact identical specifications only; search identities remain in index order.'))
    records = evaluate_candidates(pool, unique, opponents, band, n, out/'unique', weights)
    return [deepcopy(records[i]) for i in indices]
