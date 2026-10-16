"""Explicit resume and evaluator lifecycle for isolated aiming studies."""
import time
from experiments.rear_async.storage import read,sha,verify


def latest_training_checkpoint(out,seed):
    checkpoints=[]
    for path in (out/'checkpoints').glob(f's{seed}_step_*'):
        if not (path/'READY.json').exists():continue
        meta=verify(path)
        if not meta.get('baseline',False) and meta['additional_steps']>0:
            checkpoints.append((meta['additional_steps'],path))
    return max(checkpoints,key=lambda item:item[0])[1] if checkpoints else None


def evaluation_status(out):
    if (out/'STOP').exists():return 'stopped'
    if list((out/'evaluations').glob('*/blocked.json')):return 'blocked'
    if (out/'training_failure.json').exists():return 'training_failed'
    for path in (out/'queue').glob('*.json'):
        job=read(path)
        if not (out/'evaluations'/job['job_id']/'completion.json').exists():return 'incomplete'
    return 'queue_drained'


def prepare_resume(out):
    """Retire only a specifically diagnosed failure; never blindly retry it."""
    if (out/'STOP').exists():raise RuntimeError('STOP marker is active')
    if list((out/'evaluations').glob('*/blocked.json')):
        raise RuntimeError('Evaluation blocked: diagnose and archive blocked evidence before resume')
    failure=out/'training_failure.json';resolution=out/'resume_resolution.json'
    stamp=str(time.time_ns())
    history=out/'failure_history'
    if failure.exists():
        fix=read(resolution) if resolution.exists() else {}
        if not fix.get('resolved') or fix.get('training_failure_sha256')!=sha(failure):
            raise RuntimeError('A matching diagnosed resume_resolution.json is required')
        history.mkdir(exist_ok=True)
        failure.rename(history/f'{stamp}_training_failure.json')
        resolution.rename(history/f'{stamp}_resume_resolution.json')
    if not (out/'training_complete.json').exists():
        for name in ('evaluation_complete.json','training_waiting.json'):
            path=out/name
            if path.exists():
                history.mkdir(exist_ok=True);path.rename(history/f'{stamp}_{name}')


def repeat_elapsed(out,seed,now=None):
    began=out/f's{seed}_started.json';finished=out/f's{seed}_learning_complete.json'
    if not began.exists():return None
    now=time.time() if now is None else now
    completions=[];pending=False
    for path in (out/'queue').glob('*.json'):
        job=read(path);name=job['checkpoint_id']
        if not name.startswith(f's{seed}_step_') or name.endswith('_step_000000000'):continue
        complete=out/'evaluations'/job['job_id']/'completion.json'
        if complete.exists():completions.append(complete.stat().st_mtime)
        else:pending=True
    end=max([finished.stat().st_mtime,*completions]) if finished.exists() and not pending else now
    return end-read(began)['time']
