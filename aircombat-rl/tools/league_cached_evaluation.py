"""Exact-request reuse for a future frozen, deterministic experiment.

Prepared only: not imported by any currently running training driver. The
caller must freeze the complete code/input closure and runtime fingerprint,
and verify those artifacts before calling. No equivalence is inferred between
different IDs, paths or parameter files. One controller owns a cache directory.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
from tools.autolab_cem import read, write
from tools.league_adaptive_train import immutable_json
from tools.league_train import evaluate_candidates


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def validate(record, request):
    if record['request'] != request:
        raise ValueError('Cached request mismatch')
    results=record['results']
    if len(results)!=len(request['opponents']):
        raise ValueError('Incomplete opponent panel')
    expected={(request['band']+i,seat) for i in range(request['n']//2) for seat in ('red','blue')}
    for result,foe in zip(results,request['opponents']):
        rows=result['episodes']
        if (result['own']!=request['spec']['id'] or result['foe']!=foe['id']
                or len(rows)!=request['n'] or result['summary']['n']!=request['n']
                or {(r['seed'],r['seat']) for r in rows}!=expected):
            raise ValueError('Incomplete or mismatched paired episodes')


def evaluate_cached(pool,candidates,opponents,band,n,out,cache,context,weights=None):
    """Reuse exact records across phases/search RNGs with unchanged signatures.

    `plan_sha256` covers all frozen code and inputs. `runtime_sha256` covers
    the checked interpreter, simulator/data, numerical packages and execution
    settings. Cached games are reused evidence, never new independent samples.
    """
    if n<=0 or n%2 or not candidates or not opponents:
        raise ValueError('Nonempty candidates/panel and complete seat pairs required')
    if (set(context)!={'plan_sha256','runtime_sha256','deterministic'}
            or context['deterministic'] is not True
            or any(not isinstance(context[k],str) or not re.fullmatch('[0-9a-f]{64}',context[k])
                   for k in ('plan_sha256','runtime_sha256'))):
        raise ValueError('Verified deterministic plan and runtime fingerprints required')
    out,cache=Path(out),Path(cache)
    requests=[dict(spec=s,opponents=opponents,band=band,n=n,weights=weights) for s in candidates]
    keys=[digest(dict(context=context,request=r)) for r in requests]
    immutable_json(out/'request_map.json',dict(context=context,requests=requests,keys=keys))
    records={};missing=[];seen=set()
    for key,request in zip(keys,requests):
        if key in seen:continue
        seen.add(key)
        path=cache/f'{key}.json'
        if path.exists():
            saved=read(path)
            if saved['context']!=context or saved['record_sha256']!=digest(saved['record']):
                raise ValueError('Cached record integrity mismatch')
            validate(saved['record'],request);records[key]=saved['record']
        else:missing.append((key,request))
    if missing:
        # Stable layout allows a partially completed batch to resume even if
        # some cache entries were published before an interruption.
        batch=out/('uncached_'+digest([k for k,_ in missing])[:16])
        fresh=evaluate_candidates(pool,[r['spec'] for _,r in missing],opponents,band,n,batch,weights)
        if len(fresh)!=len(missing):raise ValueError('Incomplete evaluator return')
        for (key,request),record in zip(missing,fresh):
            validate(record,request)
            write(cache/f'{key}.json',dict(context=context,record=record,record_sha256=digest(record)))
            records[key]=record
    write(out/'cache_usage.json',dict(unique_requests=len(seen),executed_requests=len(missing),
        reused_requests=len(seen)-len(missing),
        scope='Reuse only; no new independent evidence from cached records.'))
    return [deepcopy(records[key]) for key in keys]
