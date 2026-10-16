"""Report fixed-reference alignment while preserving feature-learning accounting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from tools.aim_cpu_selected510_reporting import update as base_update
from tools.rear_cpu_reporting import atomic, read


def update(out):
    data = base_update(out)
    plan = read(out / 'plan.json')
    data['entropy_coefficient'] = plan['ppo']['ent_coef']
    data['previous_reference_policy_sha256'] = plan['previous_reference_policy_sha256']
    data['reference_policy_sha256'] = plan['reference_policy_sha256']
    data['comparison_control_run'] = plan['source_run']
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace(
        'original500 fixed-reference KL0.1', 'current510 fixed-reference KL0.1').replace(
        'current510 fixed-reference KL0.1, gradient roles and all rewards are unchanged from the preceding feature-learning study.',
        'fixed-reference actor original500→current510 is the only intervention; coefficient0.1, LR/GAE, gradient roles and rewards stay fixed relative to stable510 control.').replace(
        'original500 reference', 'current510 reference').replace(
        'Initial ZIP is byte-identical to the selected nativequalified510checkpoint learner;',
        'Initial non-data ZIP payloads exactly match selected510; only reference hash/path metadata change;')
    text = ('Fixed training reference and adaptive comparator use the same preserved510 model; coefficient0.1 unchanged. '
            'Initial policy/Adam/counters/native inference remain exact. Original500 and all candidate models are preserved.\n\n' + text)
    atomic(out / 'PROGRESS.md', text); atomic(out / 'SUMMARY.md', text)
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--watch', action='store_true'); a = p.parse_args(); out = a.out.resolve()
    with (out / 'reporter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(out / 'reporter_runtime.json', json.dumps(dict(pid=os.getpid(), status='running')) + '\n')
        while True:
            data = update(out)
            if not a.watch or data['training_failed'] or (data['training_complete'] and data['evaluation_complete']):
                break
            time.sleep(30)
        atomic(out / 'reporter_runtime.json', json.dumps(dict(pid=os.getpid(), status='stopped')) + '\n')
