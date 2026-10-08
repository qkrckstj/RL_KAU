"""Queued matched diagnostic of sampled PPO actions versus argmax actions.

This is an exploratory execution-mode diagnosis, not an official greedy score
or promotion gate. The waiting process imports only the standard library.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import shutil
import sys
import time
import traceback
import zipfile

from tools import league_thread_benchmark as io


SAMPLER_SOURCE = '''

GreedyPolicy = Policy

def sampling_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError('Expected nine finite logits')
    weights = np.exp(values-values.max())
    return weights/weights.sum()

def sampling_rng(first_observation, action_seed):
    raw = np.asarray(first_observation, dtype='<f4')
    if raw.shape != (39,) or not np.isfinite(raw).all():
        raise ValueError('Expected39 finite public channels')
    if not isinstance(action_seed, int) or action_seed < 0:
        raise ValueError('Nonnegative integer action seed required')
    # Only public first-frame bits and a preregistered replicate seed. No
    # environment seed, seat label, opponent ID, or hidden state is consulted.
    words = raw.view('<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence([action_seed, 739, *words]))

class Policy(GreedyPolicy):
    def __init__(self, weights=None, device='cpu'):
        with zipfile.ZipFile(weights) as archive:
            metadata = json.loads(archive.read('sampling_metadata.json'))
        if metadata['temperature'] != 1.0:
            raise ValueError('This diagnostic fixes temperature at1')
        self.action_seed = metadata['action_seed']
        super().__init__(weights, device)

    def reset(self):
        super().reset()
        self.action_rng = None

    def act(self, obs):
        raw = np.asarray(obs, dtype=np.float32)
        if raw.shape != (39,) or not np.isfinite(raw).all():
            raise ValueError('Expected39 finite public channels')
        clock = float(raw[38])
        if self.last_clock is not None and clock > self.last_clock+1e-6:
            self.reset()
        self.last_clock = clock
        if self.action_rng is None:
            self.action_rng = sampling_rng(raw, self.action_seed)
        prior = self.teacher.act(raw)
        logits = numpy_logits(features(raw, prior), self.parameters)
        return int(self.action_rng.choice(9, p=sampling_probabilities(logits)))

    def __str__(self):
        return 'PPO residual sampled at temperature1; exploratory diagnostic'
'''


def build_variant(spec, out, seed, identity):
    if out.exists():
        raise FileExistsError('Use a fresh sampled submission folder')
    original = io.ROOT/spec['design']
    source = (original/'policy.py').read_text(encoding='utf-8')
    if source.count('return int(np.argmax(logits))') != 1 or 'class Policy:' not in source:
        raise ValueError('Expected the preserved residual argmax export')
    metadata = dict(action_seed=seed, temperature=1.0, source=spec,
        source_weights_sha256=io.sha(io.ROOT/spec['weights']),
        rng_protocol='Independent NumPy generator seeded by fixed replicate seed and39 public float32 first-frame words; reset per episode. No global RNG consumption or private environment input.',
        scope='Execution-mode diagnostic only; original teacher and learned actor arrays unchanged.')
    out.mkdir(parents=True)
    (out/'policy.py').write_text(source+SAMPLER_SOURCE, encoding='utf-8')
    shutil.copyfile(original/'wrappers.py', out/'wrappers.py')
    with zipfile.ZipFile(io.ROOT/spec['weights']) as old, zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as new:
        if set(old.namelist()) != {'parameters.json','residual_actor.npz','residual_metadata.json'}:
            raise ValueError('Unexpected source archive members')
        for name in old.namelist():
            new.writestr(name, old.read(name))
        new.writestr('sampling_metadata.json', json.dumps(metadata))
    io.write(out/'policy_net.json', metadata)
    variant = dict(id=identity, kind='submission', design=io.relative(out), weights=io.relative(out/'policy_net.zip'))
    io.write(out/'entrant.json', variant)
    return variant


def process_creation(pid):
    k = ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x1000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        values=[wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(handle,*(ctypes.byref(v) for v in values)):raise ctypes.WinError(ctypes.get_last_error())
        return (values[0].dwHighDateTime<<32)+values[0].dwLowDateTime
    finally:k.CloseHandle(handle)


def prepare(source, after, pid, out):
    if (out/'queue_plan.json').exists():raise FileExistsError('Use a fresh diagnostic output')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set pre-import thread limits')
    old=io.read(source/'plan.json');io.verify(old)
    preceding=io.read(after/'queue_plan.json');io.verify(preceding)
    if io.read(after/'queue_runtime.json')['pid']!=pid:raise ValueError('Wrong predecessor PID')
    candidates=[];paths=[source/'plan.json',after/'queue_plan.json',after/'queue_runtime.json']
    for step in (2097152,6291456):
        path=source/f's3300/checkpoints/step_{step}/checkpoint.json'
        checkpoint=io.read(path);spec=checkpoint['entrant']
        if checkpoint['step']!=step:raise ValueError('Wrong fixed diagnostic checkpoint')
        if io.sha(path.parent/'learner.zip')!=checkpoint['learner_sha256']:raise ValueError('Changed learner')
        paths += [path,path.parent/'learner.zip']
        candidates.append(spec)
    # Candidate identities are fixed before fresh diagnostic games. Later
    # main/repair selection outcomes never choose which checkpoint is diagnosed.
    action_seeds=[4100,4101,4102,4103]
    modes=[dict(role='early_greedy_reference',spec=candidates[0]),
           dict(role='late_greedy',spec=candidates[1])]
    for seed in action_seeds:
        spec=build_variant(candidates[1],out/f'models/sampled_{seed}',seed,f'{out.name}_late_sampled_{seed}')
        modes.append(dict(role='late_sampled',action_seed=seed,spec=spec))
    opponents=old['development_opponents']
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for name in ('tools/league_residual_action_mode.py','tests/test_league_residual_action_mode.py',
                 'tools/league_residual_repair_followup.py'):
        sources[name]=io.sha(io.ROOT/name)
    for spec in opponents+[m['spec'] for m in modes]:
        if spec['kind']=='submission':
            paths+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    for path in paths:inputs[io.relative(path)]=io.sha(path)
    plan=dict(source=io.relative(source),after_run=io.relative(after),after_pid=pid,
        created_filetime=process_creation(pid),workers=4,band=72000000,n=32,
        opponents=opponents,groups={s['id']:old['groups'][s['id']] for s in opponents},
        modes=modes,action_seeds=action_seeds,source_sha256=sources,input_sha256=inputs,
        environment=io.environment(),additional_training_steps=0,diagnostic_games=len(modes)*len(opponents)*32,
        qualification_games=4,qualification_band=170000000,
        analysis_rule='Compare all four sampled replicas averaged equally against identical late weights under argmax; report each replica and early greedy reference too. Bootstrap paired IC clusters and separately crossed IC/action-replicate clusters; no best action seed selection.',
        next_action_rule='An execution-mode hypothesis is supported on this panel only if sampled mean uniform and code-group win gains versus late greedy are at least .05 and both crossed-bootstrap lower95% bounds exceed zero. Otherwise report weak/inconclusive evidence. No promotion or automatic change of training from this diagnostic alone.',
        scope='Exploratory diagnosis on known24-opponent panel and fresh16 ICs, both seats, temperature1. Exact learned weights fixed; no independent training replication or official greedy score claim. Main/queued repair run finishes first. No GitHub publication.')
    reservation=io.ROOT/f"runs/residual_action_mode_reservation_{plan['band']}.json"
    if reservation.exists():raise FileExistsError('Diagnostic band already reserved')
    io.verify(plan);io.write(out/'queue_plan.json',plan)
    io.write(reservation,dict(out=io.relative(out),band=plan['band'],n=plan['n'],queue_plan_sha256=io.sha(out/'queue_plan.json')))
    for name in sources:
        destination=out/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(io.ROOT/name,destination)
    return plan


def paired_mode_gain(sampled, greedy, weights, band, n, bootstrap_seed=743):
    import numpy as np
    expected={(band+i,seat) for i in range(n//2) for seat in ('red','blue')}
    def vector(results):
        if len(results)!=len(weights) or {r['foe'] for r in results}!=set(weights):
            raise ValueError('Opponent panel mismatch')
        value=np.zeros(n//2)
        for result in results:
            rows={(r['seed'],r['seat']):r for r in result['episodes']}
            if len(rows)!=len(result['episodes']) or set(rows)!=expected:raise ValueError('Unpaired IC/seat records')
            for i in range(n//2):
                value[i]+=weights[result['foe']]*sum(int(rows[band+i,seat]['won']) for seat in ('red','blue'))/2
        return value
    if not sampled or n<=0 or n%2 or any(w<=0 for w in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError('Require sampled replicas, full seat pairs and normalized weights')
    base=vector(greedy);differences=np.array([vector(rows)-base for rows in sampled])
    rng=np.random.default_rng(bootstrap_seed)
    ic=rng.integers(n//2,size=(10000,n//2))
    replicas=rng.integers(len(sampled),size=(10000,len(sampled)))
    conditional=differences.mean(axis=0)[ic].mean(axis=1)
    crossed=differences[replicas[:,:,None],ic[:,None,:]].mean(axis=(1,2))
    return dict(mean=float(differences.mean()),per_action_replica_mean=differences.mean(axis=1).tolist(),
        ic_cluster_ci95=np.quantile(conditional,[.025,.975]).tolist(),
        crossed_ic_action_replica_ci95=np.quantile(crossed,[.025,.975]).tolist(),
        initial_conditions=n//2,action_replicas=len(sampled),
        scope='Exploratory bootstrap on fixed known opponents; both seats/all opponents stay together per IC. Four action replicas do not establish generalization to arbitrary action RNGs or tournament opponents.')


def analyze(plan, results):
    from tools.league_matches import summary
    from tools.league_tournament_metrics import group_weights
    grouped={m['spec']['id']:[] for m in plan['modes']}
    for record in results:
        if record['own'] not in grouped or record['band']!=plan['band']:raise ValueError('Unexpected match result')
        if record['summary']!=summary(record['episodes']):raise ValueError('Wrong saved summary')
        grouped[record['own']].append(record)
    descriptions=[]
    weights=group_weights(plan['groups']);uniform={foe:1/len(weights) for foe in weights}
    for mode in plan['modes']:
        rows=grouped[mode['spec']['id']]
        paired_mode_gain([rows],rows,uniform,plan['band'],plan['n'])
        outcomes=Counter(e['outcome'] for r in rows for e in r['episodes'])
        descriptions.append(dict(**mode,outcomes=dict(outcomes),games=sum(outcomes.values()),
            uniform_win_rate=sum(sum(int(e['won']) for e in r['episodes'])/len(r['episodes'])/len(rows) for r in rows),
            group_win_rate=sum(weights[r['foe']]*sum(int(e['won']) for e in r['episodes'])/len(r['episodes']) for r in rows),
            per_opponent={r['foe']:r['summary'] for r in rows}))
    late=grouped[next(m['spec']['id'] for m in plan['modes'] if m['role']=='late_greedy')]
    sampled=[grouped[m['spec']['id']] for m in plan['modes'] if m['role']=='late_sampled']
    gains={label:paired_mode_gain(sampled,late,w,plan['band'],plan['n']) for label,w in [('uniform',uniform),('code_group',weights)]}
    supported=all(v['mean']>=.05-1e-12 and v['crossed_ic_action_replica_ci95'][0]>0 for v in gains.values())
    return dict(modes=descriptions,sampled_vs_same_weights_greedy=gains,
        hypothesis_supported_on_this_panel=supported,
        action='Investigate stochastic deployment/training-objective alignment on broader fresh conditions.' if supported else 'Do not attribute the decline to argmax alone; investigate policy drift, reward and opponent mixture.',
        promotion=False,scope=plan['scope'])


def execute(plan,out):
    after=io.ROOT/plan['after_run'];source=io.ROOT/plan['source']
    for folder in (after,source):
        if (folder/'failure.json').exists() or not (folder/'completion.json').exists():
            raise ValueError('Require clean completed predecessors')
    resource=io.resources()
    if min(resource['commit_headroom_gib'],resource['available_memory_gib'])<4.5:
        raise MemoryError('Insufficient diagnostic startup headroom')
    from tools.league_matches import duel,evaluate_jobs
    io.verify(plan)
    io.write(out/'runtime.json',dict(pid=os.getpid(),workers=plan['workers'],started_at=datetime.now(timezone.utc).isoformat(),resources=resource))
    sampled=next(m['spec'] for m in plan['modes'] if m['role']=='late_sampled')
    ace=next(s for s in plan['opponents'] if s['id']=='ace')
    first=duel(sampled,ace,plan['qualification_band'],2)
    second=duel(sampled,ace,plan['qualification_band'],2)
    if first['episodes']!=second['episodes']:raise AssertionError('Sampled policy reset/replay is not reproducible')
    io.write(out/'qualification.json',dict(first=first,second=second,official_records_equal=True,games=4,
        scope='Reused compatibility IC only, replay/reset check; not independent quality evidence.'))
    jobs=[dict(own=mode['spec'],foe=foe,band=plan['band'],n=plan['n']) for foe in plan['opponents'] for mode in plan['modes']]
    io.write(out/'jobs.json',jobs);started=time.perf_counter()
    print(f"Starting{plan['diagnostic_games']} paired diagnostic games after both predecessors completed.",flush=True)
    with io.Sampler() as sampler:
        results=evaluate_jobs(jobs,out/'matches',workers=plan['workers'])
    result=analyze(plan,results)
    io.write(out/'analysis.json',result);io.verify(plan)
    io.write(out/'completion.json',dict(status='diagnostic_complete',additional_training_steps=0,
        games=len(jobs)*plan['n'],qualification_games=4,elapsed_seconds=time.perf_counter()-started,
        resources=sampler.result,hypothesis_supported_on_this_panel=result['hypothesis_supported_on_this_panel'],
        promotion=False,completed_at=datetime.now(timezone.utc).isoformat(),scope=plan['scope']))
    print(json.dumps(io.read(out/'completion.json')),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--after-run',type=Path,required=True);parser.add_argument('--after-pid',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:
        from tools.league_residual_repair_followup import wait_for_source
        out=args.out.resolve();plan=prepare(args.source.resolve(),args.after_run.resolve(),args.after_pid,out)
        wait_for_source(plan,out);execute(plan,out)
    except BaseException:
        io.write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
