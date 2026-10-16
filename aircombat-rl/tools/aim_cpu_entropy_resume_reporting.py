"""Use existing accounting with accurate exploration and archive descriptions."""
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
    data['comparison_control_run'] = plan['source_run']
    atomic(out / 'PROGRESS.json', json.dumps(data, indent=2) + '\n')
    text = (out / 'PROGRESS.md').read_text().replace(
        'andentropy0.003 unchanged.', f"and entropy{plan['ppo']['ent_coef']}; exploration coefficient0.003→{plan['ppo']['ent_coef']} is the only intervention.")
    text = text.replace('Initial ZIP is byte-identical to the selected nativequalified510checkpoint learner;',
                        'Initial policy/Adam/counters equal the selected510 learner; ZIP metadata differs only for the exploration scalar;')
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
