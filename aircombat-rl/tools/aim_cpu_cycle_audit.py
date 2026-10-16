"""Read every completed episode, PPO CSV row and paired evaluation cell offline."""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from experiments.rear_async.storage import verify, write


def load(path):
    return json.loads(path.read_text())


def mean(values):
    return statistics.mean(values) if values else None


def episode_summary(rows):
    n=len(rows)
    result=dict(games=n,outcomes=dict(collections.Counter(r['result'] for r in rows)),
                terminal_reasons=dict(collections.Counter(r['terminal_reason'] for r in rows)))
    for key in ('r','l','damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
                'first_shot_advantage','relative_shaping_total'):
        values=[float(r[key]) for r in rows if key in r]
        result[key]=dict(mean=mean(values),coverage=len(values))
    result['win_rate']=sum(r['result']=='win' for r in rows)/n if n else None
    return result


def indexed_cells(folder):
    paths=sorted((folder/'cells').glob('*.json'))
    rows=[load(p) for p in paths]
    indexed={tuple(r[k] for k in ('opponent','seed','seat','action_seed')):r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate paired cell')
    return indexed,paths


def pending_evaluations(out, seed=None):
    pending=[]
    for path in sorted((out/'queue').glob('*.json')):
        job=load(path);meta=verify(Path(job['checkpoint']))
        if meta['baseline'] or (seed is not None and meta['seed']!=seed):continue
        folder=out/'evaluations'/job['job_id']
        if (folder/'blocked.json').exists():
            raise RuntimeError('Required evaluation blocked: '+job['job_id'])
        completion=folder/'completion.json'
        if not completion.exists():pending.append(job['job_id']);continue
        entry=load(completion)
        if (entry['seed'],entry['checkpoint_id'],entry['kind'])!=(meta['seed'],job['checkpoint_id'],job['kind']):
            raise ValueError('Queued evaluation completion identity differs')
    return pending


def wait_for_seed_evaluations(out, seed):
    import psutil
    while True:
        if (out/'STOP').exists():raise InterruptedError('STOP while awaiting completed-repeat audit')
        pending=pending_evaluations(out,seed)
        if not pending:break
        pid=load(out/'processes.json')['evaluator']
        try:alive=psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:alive=False
        if not alive:raise RuntimeError('Evaluator exited before completed-repeat audit')
        write(out/'training_waiting.json',dict(seed=seed,pending_jobs=pending,
            reason='Awaiting all target-seed evaluations before completed-repeat log audit'))
        time.sleep(2)
    waiting=out/'training_waiting.json'
    if waiting.exists():waiting.unlink()


def audit(out, seed=None):
    pending=pending_evaluations(out,seed)
    if pending:raise RuntimeError('Incomplete queued evaluations; refuse partial repeat audit: '+','.join(pending))
    reports=dict(training={},evaluations=[],read_files={},extra_simulated_games=0,
                 limits=['Training quarters have different episode/opponent draws; not paired comparisons.',
                         'Endpoint/episode summaries do not contain full flight or action trajectories.',
                         'Low entropy, reward changes and outcome correlations are hypotheses, not established causes.',
                         'Consumed evaluation conditions are not an unseen final test.'])
    def remember(path):
        reports['read_files'][str(path.relative_to(out))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for result_path in sorted(out.glob('s*_result.json')):
        result_seed=int(result_path.name.split('_')[0][1:])
        if seed is not None and result_seed!=seed:continue
        result=load(result_path);remember(result_path)
        current_seed=int(Path(result['checkpoint']).name.split('_')[0][1:])
        episodes_path=out/f's{current_seed}_episodes.jsonl'
        rows=[json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        remember(episodes_path)
        if len(rows)!=result['training_games']:raise ValueError('Episode count differs from saved result')
        identities=[(r['episode_metadata']['episode_seed'],r['episode_metadata']['seat']) for r in rows]
        if len(set(identities))!=len(rows):raise ValueError('Duplicate training episode identity')
        if any(r['seed']!=current_seed or r['additional_step']>result['additional_steps'] for r in rows):
            raise ValueError('Episode outside completed training snapshot')
        groups=collections.defaultdict(list)
        for r in rows:
            if r['result'] not in ('win','draw','loss'):raise ValueError('Unknown episode verdict')
            for key in ('r','l','damage_dealt','damage_received','connected_fire_seconds','lock_breaks'):
                if not math.isfinite(float(r[key])):raise ValueError('Nonfinite episode data')
            groups[r['episode_metadata']['opponent_id']].append(r)
        ordered=sorted(rows,key=lambda r:r['additional_step']);quarter=max(1,len(rows)//4)
        csv_path=out/f's{current_seed}_metrics/progress.csv';remember(csv_path)
        csv_rows=list(csv.DictReader(csv_path.open()));metrics={};nonfinite=[]
        for key in csv_rows[0].keys() if csv_rows else ():
            values=[]
            for i,row in enumerate(csv_rows):
                if not row[key]:continue
                v=float(row[key])
                if math.isfinite(v):values.append(v)
                else:nonfinite.append(dict(row=i,column=key))
            metrics[key]=dict(records=len(values),last=values[-1] if values else None,
                              median=statistics.median(values) if values else None)
        reports['training'][str(current_seed)]=dict(saved_result=result,all_episodes=episode_summary(rows),
            first_quarter=episode_summary(ordered[:quarter]),last_quarter=episode_summary(ordered[-quarter:]),
            by_opponent={k:episode_summary(v) for k,v in sorted(groups.items())},
            ppo_rows=len(csv_rows),ppo_metrics=metrics,nonfinite_ppo_fields=nonfinite)
    entries=[(p,load(p)) for p in sorted((out/'evaluations').glob('*/completion.json'))]
    baselines={e['kind']:(p,e) for p,e in entries if e['baseline']}
    for path,e in entries:
        if seed is not None and not e['baseline'] and e['seed']!=seed:continue
        remember(path);rows,files=indexed_cells(path.parent)
        if len(rows)!=e['summary']['games'] or sum(r['result']=='win' for r in rows.values())!=e['summary']['wins']:
            raise ValueError('Evaluation count/verdict differs from summary')
        for f in files:remember(f)
        if e['baseline']:continue
        bp,b=baselines[e['kind']];base,_=indexed_cells(bp.parent)
        if rows.keys()!=base.keys():raise ValueError('Candidate/baseline paired conditions differ')
        foes=sorted({r['opponent'] for r in rows.values()})
        foe_changes=[]
        for foe in foes:
            keys=[k for k in rows if k[0]==foe]
            foe_changes.append(dict(opponent=foe,games=len(keys),
                parent_wins=sum(base[k]['result']=='win' for k in keys),
                candidate_wins=sum(rows[k]['result']=='win' for k in keys),
                dealt_delta=mean([rows[k]['damage_dealt']-base[k]['damage_dealt'] for k in keys]),
                received_delta=mean([rows[k]['damage_received']-base[k]['damage_received'] for k in keys])))
        reports['evaluations'].append(dict(checkpoint=e['checkpoint_id'],kind=e['kind'],seed=e['seed'],
            summary=e['summary'],baseline=b['summary'],
            paired_transitions=dict(collections.Counter(base[k]['result']+' -> '+r['result'] for k,r in rows.items())),
            per_opponent=foe_changes))
    reports['read_file_count']=len(reports['read_files'])
    name='CYCLE_LOG_AUDIT' if seed is None else f'S{seed}_LOG_AUDIT'
    (out/(name+'.json')).write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    lines=['# Completed cycle log audit','',
           'Every completed episode JSONL row, PPO CSV row and cached completed evaluation cell was read. No new simulations.',
           '', '| Seed | Episodes | PPO rows | Training win rate | First-shot rate | Mean reward | Nonfinite PPO fields |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for seed_label,r in reports['training'].items():
        s=r['all_episodes'];lines.append(f"| {seed_label} | {s['games']} | {r['ppo_rows']} | {100*s['win_rate']:.3f}% | {100*s['first_shot_advantage']['mean']:.3f}% | {s['r']['mean']:.6f} | {len(r['nonfinite_ppo_fields'])} |")
    lines+=['','| Broad checkpoint | Candidate /parent wins | Largest opponent win changes |','|---|---:|---|']
    for r in reports['evaluations']:
        if r['kind']!='development':continue
        changes=sorted(r['per_opponent'],key=lambda f:abs(f['candidate_wins']-f['parent_wins']),reverse=True)[:5]
        text='; '.join(f"{f['opponent']}:{f['parent_wins']}->{f['candidate_wins']}" for f in changes)
        lines.append(f"| {r['checkpoint']} | {r['summary']['wins']}/{r['baseline']['wins']} | {text} |")
    lines+=['',*reports['limits'],'',f"Read{reports['read_file_count']}files. Full per-opponent episode, metric and paired-outcome tables are in{name}.json."]
    (out/(name+'.md')).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();audit(args.out.resolve(),args.seed)
