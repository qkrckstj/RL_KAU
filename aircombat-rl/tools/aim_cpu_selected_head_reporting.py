"""Report selected-parent continuation while retaining the original KL anchor."""
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
    data['reference_actor_role'] = 'Original500 fixed KL anchor; adaptive comparator is the selected parent.'
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace('qualified500', f'qualified{wins}')
    heading, body = text.split('\n', 1)
    text = heading + (
        f'\n\nStarting learner and adaptive comparator: native{wins}/672. '
        'The fixed actor KL reference remains original500; its stored hash, coefficient0.1 '
        'and archive are preserved. This continuation does not reanchor the loss or reset Adam. '
        'Reference500 is an optimization anchor, not the current evaluation baseline.\n\n'
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
