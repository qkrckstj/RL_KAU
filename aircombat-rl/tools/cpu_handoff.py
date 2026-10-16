"""Portable CPU reproduction. Uses original native evaluator and strict v15 scorer.

No training, fresh confirmation, pool admission or version assignment is automated.
"""
import argparse
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time

from experiments.rear_async.storage import read, write, sha, identity
from tools.seven_model_league_v15 import score_panel, paired_jobs

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = 'artifacts/20261016/direct/feedback_002/candidate'


def verify_bundle(root):
    count = 0
    for name in ('artifacts/20261015/manifest.json', 'artifacts/20261016/manifest.json'):
        manifest = read(root/name)
        for relative, digest in manifest['sha256'].items():
            path = (root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != digest:
                raise ValueError('Missing or changed published input: '+relative)
            count += 1
    return count


def candidate_files(candidate):
    for name in ('policy.py','wrappers.py','policy_net.zip','policy_net.json'):
        if not (candidate/name).is_file(): raise ValueError('Missing candidate file: '+name)
    return sorted(p for p in candidate.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))


def jobs_and_protocol(root, candidate):
    root, candidate = root.resolve(), candidate.resolve()
    models = read(root/'artifacts/20261015/manifest.json')['models']
    pool = []
    for model in models:
        specs = {}
        for action in (4900, 4901):
            folder = root/model['checkpoint']/f'sampled_{action}'
            specs[str(action)] = dict(id=model['id'],kind='submission',design=str(folder),
                                      weights=str(folder/'policy_net.zip'))
        pool.append(dict(id=model['id'], sampled_specs=specs))
    own = dict(id=candidate.parent.name,kind='submission',design=str(candidate),
               weights=str(candidate/'policy_net.zip'))
    files = {str(p.relative_to(candidate)):sha(p) for p in candidate_files(candidate)}
    protocol = dict(stage='screen',opponents=[m['id'] for m in models],
                    seeds=list(range(3401500000,3401500008)),seats=['red','blue'],
                    action_seeds=[4900,4901],model_id=identity(files),
                    pool_id=identity(read(root/'artifacts/20261015/manifest.json')['sha256']),smoke=False)
    return list(paired_jobs({str(a):own for a in (4900,4901)},pool,protocol)),protocol


def play(job):
    from experiments.direct_pilot.evaluate import cell
    own,foe,seed,seat,action = job
    return dict(cell(own,foe,seed,seat),action_seed=action)


def evaluate(root,candidate,output,workers,smoke):
    if output.exists():
        raise FileExistsError('Output already exists; inspect it and choose a new directory')
    if workers not in (1,2,3): raise ValueError('Use 1, 2 or 3 CPU workers')
    import psutil
    if psutil.virtual_memory().available < (4+2*(workers-1))*2**30:
        raise MemoryError('Insufficient available RAM for the requested worker count')
    verify_bundle(root)
    jobs,protocol = jobs_and_protocol(root,candidate)
    if smoke:
        jobs = [j for j in jobs if j[2]==3401500000 and j[4]==4900]
        protocol = dict(protocol,stage='reproduction_smoke',seeds=[3401500000],action_seeds=[4900],smoke=True)
    # Freeze actual local source and candidate, including an optional new candidate.
    paths = [p for folder in ('aircombat-rl/aircombat_gym','aircombat-rl/tools','aircombat-rl/experiments')
             for p in (root/folder).rglob('*.py') if '__pycache__' not in p.parts]
    paths += candidate_files(candidate)
    paths += [root/'artifacts/20261015/manifest.json',root/'artifacts/20261016/manifest.json']
    frozen = {str(p):sha(p) for p in paths}
    lockfile = root/'aircombat-rl/runs/cpu_handoff.controller.lock'
    lockfile.parent.mkdir(parents=True,exist_ok=True)
    with lockfile.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'plan.json',dict(protocol=protocol,jobs=jobs,input_sha256=frozen,
            workers=workers,new_neural_training_decisions=0,
            scope='Consumed-condition CPU reproduction. Not unseen quality evidence.'))
        write(output/'runtime.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
        started = time.monotonic()
        try:
            rows = []
            with multiprocessing.get_context('spawn').Pool(workers) as pool:
                for row in pool.imap(play,jobs):
                    write(output/'cells'/f'{len(rows):03d}.json',row)
                    rows.append(row)
                    write(output/'progress.json',dict(completed=len(rows),expected=len(jobs)))
            for name,digest in frozen.items():
                if sha(Path(name)) != digest: raise ValueError('Frozen local input changed: '+name)
            verify_bundle(root)
            from experiments.direct_pilot.evaluate import profile
            result = dict(protocol=protocol,profile=profile(rows),games=len(rows),
                          elapsed_seconds=time.monotonic()-started,new_neural_training_decisions=0,
                          admitted=False,qualified_version=None,scope='Consumed reproduction, not fresh confirmation')
            result['score'] = None if smoke else score_panel(rows,protocol)
            result['game_wdl'] = [sum(r['result']==label for r in rows) for label in ('win','draw','loss')]
            write(output/'completion.json',result)
            return result
        except BaseException as exc:
            write(output/'failure.json',dict(error=repr(exc),elapsed_seconds=time.monotonic()-started))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('verify','smoke','screen'))
    parser.add_argument('--candidate',default=DEFAULT_CANDIDATE)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    if args.command=='verify':
        print(json.dumps(dict(verified_files=verify_bundle(REPO))))
        return
    if args.output is None: parser.error('--output must name a new output directory')
    candidate=Path(args.candidate)
    if not candidate.is_absolute(): candidate=REPO/candidate
    print(json.dumps(evaluate(REPO,candidate,args.output.resolve(),args.workers,args.command=='smoke'),indent=2))


if __name__=='__main__': main()
