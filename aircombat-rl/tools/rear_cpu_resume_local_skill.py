"""Prepare a bounded resume from independent opponent-specific Goal review.

The frozen trainer stays unchanged. Preserve its original automatic judgments
and terminal records, then write operative extension decisions as reviewed
outputs. Checkpoint counters/RNG/Adam/samplers are restored by the existing
resume path; native flights restart and exact mid-flight restoration is not
claimed. This tool prepares only; it never launches another live trainer.
"""
import argparse
import fcntl
import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

from experiments.rear_async.storage import read, write, verify, sha
from experiments.rear_opponent_budget_review import review
from tools.rear_async_cpu import check_plan


def prepare(out):
    import psutil
    for name in ('training_complete.json', 'evaluation_complete.json'):
        if not (out/name).exists():
            raise RuntimeError('Current run is not terminal: ' + name)
    if read(out/'evaluation_complete.json')['status'] != 'queue_drained':
        raise RuntimeError('Evaluation queue has not drained')
    for pid in read(out/'processes.json').values():
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            if p.status() != psutil.STATUS_ZOMBIE and 'tools.rear_async_cpu' in ' '.join(p.cmdline()):
                raise RuntimeError('Current trainer/evaluator remains live')
    plan = read(out/'plan.json')
    check_plan(plan)
    entries = [read(p) for p in (out/'evaluations').glob('*/completion.json')]
    baseline = next(e for e in entries if e['kind'] == 'monitor' and e['baseline'])
    decisions = {}
    for seed in plan['seeds']:
        points = [e for e in entries if e['kind'] == 'monitor' and not e['baseline'] and e['seed'] == seed]
        ready = [p for p in (out/'checkpoints').glob(f's{seed}_step_*') if (p/'READY.json').exists()]
        latest = max(ready, key=lambda p: verify(p)['additional_steps'])
        m = verify(latest)
        steps = m['additional_steps']
        if not any(e['kind'] == 'development' and e['seed'] == seed
                   and not e['baseline'] and e['additional_steps'] == steps for e in entries):
            raise RuntimeError('Latest checkpoint lacks completed broader assessment')
        d = review(points, baseline, steps, plan['minimum_steps'], plan['maximum_steps'])
        if d['extend']:
            d.update(reviewed_by='Active CPU rear Goal; per-opponent skill review',
                     source_checkpoint=str(latest), learner_sha256=sha(latest/'learner.zip'),
                     resume_scope='Adam/RNG/sampler/counters restored; native flight episodes restart.')
            decisions[seed] = d
    if not decisions:
        print(json.dumps(dict(status='no_qualifying_extension')))
        return
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Available host memory below 10 GiB')
    with ExitStack() as locks:
        for name in ('training.lock', 'evaluation.lock'):
            lock = locks.enter_context((out/name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # If process state changed since inspection, lock acquisition prevents
        # mutation of a live trainer/evaluator's decisions and terminal flags.
        segment = out/'segments'/f'completed_before_local_extension_{time.time_ns()}'
        segment.mkdir(parents=True, exist_ok=False)
        snapshots = list(out.glob('s*_budget_decision.json'))
        snapshots += [out/name for name in ('training_complete.json', 'evaluation_complete.json',
                                            'SUMMARY.md', 'PROGRESS.md', 'PROGRESS.json', 'processes.json')]
        for path in snapshots:
            if path.exists():
                shutil.copyfile(path, segment/path.name)
        evidence = dict(status='prepared_not_launched', segment=str(segment),
            reviewed_seeds=list(decisions), maximum_steps=plan['maximum_steps'],
            review_source_sha256=sha(Path(__file__)),
            budget_review_sha256=sha(Path(review.__code__.co_filename)),
            original_source_and_plan_unchanged=True,
            launch_argv=['python', '-m', 'tools.rear_async_cpu', 'launch', '--out', str(out),
                         '--workers', str(plan['processes']), '--resume'])
        write(segment/'local_skill_extension_review.json', dict(evidence, decisions=decisions))
        for seed, decision in decisions.items():
            write(out/f's{seed}_budget_decision.json', dict(decision, preserved_automatic_record=str(segment/f's{seed}_budget_decision.json')))
        # Remove terminal markers only after immutable snapshots and operative
        # review decisions exist. The existing resume engine retains step caps.
        for name in ('training_complete.json', 'evaluation_complete.json'):
            (out/name).unlink()
        write(out/'local_skill_resume.json', evidence)
    print(json.dumps(evidence))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.out.resolve())
