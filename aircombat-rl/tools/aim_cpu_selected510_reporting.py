"""Report same-setting continuation from the full-panel selected510 checkpoint."""
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
    assert plan['actor_features_trainable'] is True
    data['actor_features_trainable'] = True
    data['qualified_parent_full_wins'] = plan['qualified_parent_full_wins']
    data['source_actor_feature_sha256'] = plan['source_actor_feature_sha256']
    data['gae_lambda'] = plan['ppo']['gae_lambda']
    data['actor_head_runtime'] = {}
    for seed in plan['seeds']:
        file = out / f's{seed}_actor_head_runtime.json'
        if file.exists():
            runtime = read(file)
            if (runtime['actor_features_trainable'] is not True or
                    (runtime['frozen_parameters'], runtime['trainable_feature_parameters'], runtime['trainable_action_parameters'], runtime['trainable_critic_parameters']) != (0, 6976, 585, 7041) or
                    runtime['gae_lambda'] != .99 or runtime['buffer_gae_lambda'] != .99):
                raise ValueError('Effective feature-training roles or GAE differ from the plan')
            data['actor_head_runtime'][str(seed)] = runtime
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace('qualified500', f"qualified{plan['qualified_parent_full_wins']}").replace(
        f"# Fixed-reference CPU PPO continuation from qualified{plan['qualified_parent_full_wins']}",
        f"# Actor-feature CPU PPO continuation from qualified{plan['qualified_parent_full_wins']}", 1)
    heading, body = text.split('\n', 1)
    text = heading + (
        '\n\nExisting actor features (6,976), action head (585) and critic (7,041) all learn. '
        'Initial NN/value/initialized Adam/counters/native behavior remain preserved; feature weights '
        'and their Adam intentionally update during learning. LR1e-6, GAE0.99/gamma0.999/original500 '
        'fixed-reference KL0.1, gradient roles and all rewards are unchanged from the preceding feature-learning study. '
        'Gradient roles are enabled after every fresh/resume load. Checkpoints record learned feature '
        'digests; the source digest identifies initialization, not a post-update freeze. '
        'The parent is a consumed-development510-win selection; prior508 and all candidates remain archived. This is not unseen validation.\n\n'
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
