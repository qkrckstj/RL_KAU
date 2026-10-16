"""Report native PPO action-head adaptation and actual frozen feature roles."""
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
    data['actor_features_trainable'] = plan['actor_features_trainable']
    data['source_actor_feature_sha256'] = plan['source_actor_feature_sha256']
    data['gae_lambda'] = plan['ppo']['gae_lambda']
    data['actor_head_runtime'] = {}
    for seed in plan['seeds']:
        file = out / f's{seed}_actor_head_runtime.json'
        if file.exists():
            runtime = read(file)
            if (runtime['actor_features_trainable'] is not False or
                    runtime['feature_sha256'] != plan['source_actor_feature_sha256'] or
                    runtime['gae_lambda'] != .99 or runtime['buffer_gae_lambda'] != .99):
                raise ValueError('Effective actor-head roles or GAE differ from the plan')
            data['actor_head_runtime'][str(seed)] = runtime
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace(
        '# Fixed-reference CPU PPO continuation from qualified500',
        '# Actor-head CPU PPO continuation from qualified500', 1)
    heading, body = text.split('\n', 1)
    text = heading + (
        '\n\nFixed actor feature weights (6,976 parameters); action head (585) and critic (7,041) '
        'remain trainable. GAE lambda0.99/gamma0.999/reference KL0.1 and all rewards are fixed. '
        'Starting archive, actor/value/initialized Adam/counters and initial native inference are preserved. '
        'The gradient mask is reapplied after each load; actual roles are recorded in seed '
        'actor_head_runtime.json and feature identity is checked at every checkpoint. '
        'This is an adaptation hypothesis; unchanged features do not establish improved native wins.\n\n'
    ) + body.lstrip('\n')
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
