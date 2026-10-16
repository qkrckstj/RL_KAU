"""Account for an explicit training-start mixture, preserving all baseline metrics."""
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
    data['training_start_curriculum'] = plan['curriculum']
    data['comparison_control_run'] = plan['source_run']
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    fraction = plan['curriculum']['fraction']
    text = (f'Training starts: {100*(1-fraction):.0f}% native neutral + {100*fraction:.0f}% existing quarter distribution. '
            'Only this training fraction changes relative to stable510 control; fixed reward, PPO and native evaluation preserved. '
            'Initial learner ZIP remains byte-identical. No episode-time tactical switching.\n\n' + (out / 'PROGRESS.md').read_text())
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
