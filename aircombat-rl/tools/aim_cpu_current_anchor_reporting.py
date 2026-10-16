"""Report current-parent fixed-reference continuation and explicit migration."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from tools.aim_cpu_actor_head_reporting import update as actor_update
from tools.rear_cpu_reporting import atomic, read


def update(out):
    data = actor_update(out)
    plan = read(out / 'plan.json')
    wins = plan['qualified_parent_full_wins']
    data['qualified_parent_full_wins'] = wins
    data['reference_policy_sha256'] = plan['reference_policy_sha256']
    data['previous_reference_policy_sha256'] = plan['previous_reference_policy_sha256']
    data['reference_actor_role'] = 'Fixed KL anchor and adaptive comparator both use the qualified current parent.'
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace('qualified500', f'qualified{wins}')
    text = text.replace(f'Initial ZIP is byte-identical to the selected nativequalified{wins}checkpoint learner;', f'Initial learned ZIP payloads exactly match the selected nativequalified{wins}checkpoint learner; reference hash/path metadata are explicitly rebound;')
    text = text.replace('Starting archive, actor/value/initialized Adam/counters and initial native inference are preserved.', 'Starting actor/value/initialized Adam/counters and initial native inference are preserved; only reference hash/path metadata differ.')
    heading, body = text.split('\n', 1)
    text = heading + (
        f'\n\nStarting learner and adaptive comparator: native{wins}/672. '
        'The fixed actor KL reference is this same selected parent, coefficient0.1. '
        'Explicit migration changed only referencehash/path; NN/Adam/counters and native forward stay exact. '
        'The original500 reference and allmodels remain archived. This is a controlled hypothesis, '
        'not an established cause of previous regressions.\n\n'
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
