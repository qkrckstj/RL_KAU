"""Reproduce the bounded CEM worker diagnostic; run from aircombat-rl.

Uses already consumed development conditions and fixed policies, not CEM search.
Six workers are a diagnostic override; the project default remains three.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from tools.runtime_audit import cem_case, sha
from tools.plan_a import timestamp, write_json


def main():
    source = Path('runs/runtime_audit_20261008/cem_jobs.json')
    jobs = json.loads(source.read_text())
    out = Path('runs/runtime_cem_workers_20261008')
    out.mkdir(exist_ok=False)
    report = dict(started_utc=timestamp(), jobs_sha256=sha(source), runs=[])
    write_json(out / 'jobs.json', jobs)
    for repeat in range(3):
        for workers in (3, 6)[::1 if repeat % 2 == 0 else -1]:
            result = cem_case(out / f'w{workers}_r{repeat}', workers, jobs)
            report['runs'].append(dict(repeat=repeat, **result))
            write_json(out / 'results.json', report)
            print(json.dumps(report['runs'][-1]), flush=True)
    assert len({r['outcomes_sha256'] for r in report['runs']}) == 1
    baseline = json.loads(Path('runs/runtime_audit_20261008/results.json').read_text())
    assert report['runs'][0]['outcomes_sha256'] == baseline['cem'][0]['outcomes_sha256']
    report.update(finished_utc=timestamp(), same_outcomes_as_baseline=True)
    write_json(out / 'results.json', report)


if __name__ == '__main__':
    main()
