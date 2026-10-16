"""Keep native counts and timing while recording effective fixed GAE settings."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from tools.aim_cpu_reference_reporting import update as reference_update
from tools.rear_cpu_reporting import atomic, read


def update(out):
    data = reference_update(out)
    plan = read(out / 'plan.json')
    data['gae_lambda'] = plan['ppo']['gae_lambda']
    data['gamma'] = plan['ppo']['gamma']
    data['credit_runtime'] = {}
    for seed in plan['seeds']:
        file = out / f's{seed}_credit_runtime.json'
        if file.exists():
            runtime = read(file)
            if (runtime['model_gae_lambda'], runtime['buffer_gae_lambda']) != (.999, .999):
                raise ValueError('Effective GAE settings differ from the planned trial')
            data['credit_runtime'][str(seed)] = runtime
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace(
        '# Fixed-reference CPU PPO continuation from qualified500',
        '# Long-credit CPU PPO continuation from qualified500', 1)
    description = (
        '\n\nPlanned fixed GAE lambda0.999 with gamma0.999; frozen reference KL0.1 remains unchanged. '
        'The byte-identical parent archive retains lambda0.99 metadata; only GAE lambda is overridden '
        'when loading the learner. Actual model and rollout-buffer settings are recorded in each '
        'seed credit_runtime.json. Initial actor/value/Adam/counters and native inference are preserved; '
        'the advantage estimator intentionally changes. This is a credit-assignment hypothesis, '
        'not an established explanation of regression or a promised improvement.\n\n')
    heading, body = text.split('\n', 1)
    text = heading + description + body.lstrip('\n')
    atomic(out / 'PROGRESS.md', text)
    atomic(out / 'SUMMARY.md', text)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    with (out / 'reporter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(out / 'reporter_runtime.json', json.dumps(dict(pid=os.getpid(), status='running')) + '\n')
        while True:
            data = update(out)
            if not args.watch or data['training_failed'] or (data['training_complete'] and data['evaluation_complete']):
                break
            time.sleep(30)
        atomic(out / 'reporter_runtime.json', json.dumps(dict(pid=os.getpid(), status='stopped')) + '\n')
