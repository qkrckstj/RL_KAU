"""Report long fixed-setting training and clearly distinguish54training versus21evaluation foes."""
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
    text = (out / 'PROGRESS.md').read_text().replace(
        'checkpoint/monitor intervals unchanged.',
        f"checkpoint every{plan['checkpoint_steps']:,}; monitor every{plan['monitor_steps']:,}; full21-opponent development every{plan['development_steps']:,}. No intermediate evaluation wait.")
    text = ('# Long fixed-setting PPO:54 known training opponents\n\n'
            'Selected510 parent; same PPO/rewards/reference/quarter50/controller/native physics. '
            'Former6 evaluation-only opponents now join training; old48relative weights retained. '
            'Periodic672-game evaluation covers21representatives, not all54opponents. '
            'All54paired qualification is pending after segment selection. All checkpoints preserved.\n\n' + text)
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
