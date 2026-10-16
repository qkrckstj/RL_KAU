"""Freeze one manual-derived feedback pilot; run exactly the original224screen."""
from pathlib import Path
import argparse
import fcntl
import json
import multiprocessing
import os
import shutil
import time
import zipfile
import psutil
from experiments.rear_async.storage import read, write, sha, identity
from experiments.direct_pilot.manual_feedback_positional_policy import Policy, DEFAULTS
from experiments.direct_pilot.evaluate import cell, profile
from tools.seven_model_league_v15 import verify_hashes, verify_run_hashes, paired_jobs, score_panel
from tools.policies import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/direct_pilot_20261016/feedback_003'
PARAMETERS = dict(DEFAULTS, brake_speed=520.)
FIELDS = ('opponent', 'seed', 'seat', 'action_seed')


def prepare():
    assert not OUT.exists()
    config = read(ROOT/'runs/seven_league_range8000_20261016/config.json')
    verify_run_hashes(config)
    manifest = ROOT.parent/'artifacts/20261015/manifest.json'
    assert config['frozen_sha256'].get(str(manifest)) == sha(manifest)
    files = dict(config['frozen_sha256'])
    files.update({str(ROOT.parent/k): v for k, v in read(manifest)['sha256'].items()})
    verify_hashes(files)
    OUT.mkdir()
    folder = OUT/'candidate'; folder.mkdir()
    source = ROOT/'experiments/direct_pilot/manual_feedback_positional_policy.py'
    shutil.copyfile(source, folder/'policy.py')
    (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n')
    with zipfile.ZipFile(folder/'policy_net.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('parameters.json', json.dumps(dict(parameters=PARAMETERS)))
    write(folder/'policy_net.json', dict(kind='manual_derived_public_observation_state_machine_not_neural',
        observation_dim=39, actions=9, parameters=PARAMETERS, qualified_version=None, parent_candidate='manual_feedback_002', changed_factor='brake_speed482to520_only', parent_model_id=read(OUT.parent/'feedback_002/plan.json')['protocol']['model_id'],
        source_demonstrations=['manual_003', 'manual_004', 'manual_005'], failure_reference='manual_002'))
    own = dict(id='manual_feedback_003', kind='submission', design=str(folder), weights=str(folder/'policy_net.zip'))
    write(folder/'entrant.json', own)
    model_files = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'entrant.json'}
    act, _, mode = load(folder, folder/'policy_net.zip')
    assert mode == 'discrete'
    packaged = act.__self__
    count = 0
    coverage = {}
    for name in ('manual_002', 'manual_003', 'manual_004', 'manual_005'):
        tape = OUT.parent/name
        complete = read(tape/'completion.json'); verify_hashes(complete['input_sha256'])
        assert complete['status'] == 'terminal'
        policy = Policy(parameters=PARAMETERS); packaged.reset(); phases = set()
        for line in (tape/'trace.jsonl').read_text().splitlines():
            row = json.loads(line); raw = row['raw_before']
            expected = policy.act(raw)
            assert act(raw) == expected and packaged.phase == policy.phase
            assert act(raw) == expected  # repeated-query stability through actual loader
            phases.add(policy.phase); count += 1
        coverage[name] = sorted(phases)
        for k in ('plan.json', 'commands.jsonl', 'trace.jsonl', 'completion.json'):
            files[str(tape/k)] = sha(tape/k)
    parent = read(OUT.parent/'feedback_001/plan.json')
    verify_hashes(parent['input_sha256'])
    tape = OUT.parent/'feedback_001/trace_failure_000'
    assert read(tape/'completion.json')['reference_terminal_exact']
    policy = Policy(parameters=PARAMETERS); packaged.reset(); phases = set()
    for line in (tape/'trace.jsonl').read_text().splitlines():
        raw = json.loads(line)['raw_before']; expected_action = policy.act(raw)
        assert act(raw) == expected_action and packaged.phase == policy.phase
        assert act(raw) == expected_action
        phases.add(policy.phase); count += 1
    coverage['feedback001_exact_failure_tape'] = sorted(phases)
    for k in ('plan.json','trace.jsonl','completion.json','EGRESS_GATE_OBSERVER.json'):
        files[str(tape/k)] = sha(tape/k)
    for k in ('plan.json','completion.json','AUDIT.json','FINAL_REVIEW.json'):
        files[str(OUT.parent/'feedback_001'/k)] = sha(OUT.parent/'feedback_001'/k)
    parent = read(OUT.parent/'feedback_002/plan.json')
    verify_hashes(parent['input_sha256'])
    tape = OUT.parent/'feedback_002/trace_failure_000'
    assert read(tape/'completion.json')['reference_terminal_exact']
    policy = Policy(parameters=PARAMETERS); packaged.reset(); phases = set()
    for line in (tape/'trace.jsonl').read_text().splitlines():
        raw = json.loads(line)['raw_before']; expected_action = policy.act(raw)
        assert act(raw) == expected_action and packaged.phase == policy.phase
        assert act(raw) == expected_action
        phases.add(policy.phase); count += 1
    coverage['feedback002_exact_mutual_tape'] = sorted(phases)
    for k in ('plan.json','trace.jsonl','completion.json'):
        files[str(tape/k)] = sha(tape/k)
    for k in ('plan.json','completion.json','AUDIT.json','REPORT.md'):
        files[str(OUT.parent/'feedback_002'/k)] = sha(OUT.parent/'feedback_002'/k)
    for k in ('plan.json','commands.jsonl','trace.jsonl','completion.json','AUDIT.json','NOMINAL_TURN_SOURCE_OBSERVER.json'):
        files[str(OUT.parent/'manual_006'/k)] = sha(OUT.parent/'manual_006'/k)
    assert count == 9458
    write(OUT/'LOADER_QUALIFICATION.json', dict(official_loader_predictions=count,
        repeated_queries=count, action_and_phase_exact=True, phase_coverage=coverage,
        scope='Offline parity of source and packaged controller on consumed tapes. Not closed-loop generalization or reproduction of manual actions.'))
    pool = config['initial_pool']
    protocol = dict(stage='screen', opponents=[p['id'] for p in pool],
        seeds=list(range(config['screen_band'], config['screen_band']+8)), seats=['red', 'blue'],
        action_seeds=[4900, 4901], model_id=identity(model_files), pool_id=identity(pool), smoke=False)
    jobs = list(paired_jobs({str(a): own for a in (4900, 4901)}, pool, protocol))
    assert len(jobs) == 224
    for p in (Path(__file__).resolve(), source, ROOT/'experiments/direct_pilot/evaluate.py',
              ROOT/'tests/test_manual_feedback.py', ROOT/'tests/test_manual_feedback_signed.py', ROOT/'tests/test_manual_feedback_positional.py',
              ROOT/'experiments/direct_pilot/manual_feedback_signed_policy.py',
              ROOT/'experiments/direct_pilot/manual_feedback_policy.py',
              *(ROOT/'runs/direct_pilot_20261016/feedback_000/trace_failure_000'/k for k in ('plan.json','trace.jsonl','completion.json')),
              OUT/'LOADER_QUALIFICATION.json', *folder.iterdir()):
        if p.is_file(): files[str(p)] = sha(p)
    verify_hashes(files)
    write(OUT/'plan.json', dict(protocol=protocol, own=own, pool=pool, jobs=jobs,
        model_files=model_files, input_sha256=files, processes=3,
        new_neural_training_decisions=0, qualified_version=None,
        scope='One frozen observation-based manual-derived controller; unchanged224screen. Automated policy matches are not interactive agent piloting. No parameter grid, no clock replay, no curriculum.'))
    write(OUT/'PREPARED.json', dict(plan_sha256=sha(OUT/'plan.json'), games=224, ready=True))
    print('One feedback candidate frozen;9458 source/loader actions and phases exact;224screen ready.', flush=True)


def job(item):
    own, foe, seed, seat, action_seed = item
    return dict(cell(own, foe, seed, seat), action_seed=action_seed)


def run():
    assert not (OUT/'runtime.json').exists(), 'Inspect original process;never duplicate'
    assert read(OUT/'PREPARED.json')['plan_sha256'] == sha(OUT/'plan.json')
    plan = read(OUT/'plan.json'); verify_hashes(plan['input_sha256'])
    if psutil.virtual_memory().available < 12*2**30:
        raise MemoryError('Need12GiB available')
    with (ROOT/'runs/seven_league_range8000_20261016/controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(OUT/'runtime.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time()))
        rows = []; started = time.monotonic()
        with multiprocessing.get_context('spawn').Pool(3) as workers:
            for row in workers.imap(job, plan['jobs']):
                write(OUT/'cells'/f"{identity({k:row[k] for k in FIELDS})}.json", row)
                rows.append(row)
                if len(rows)%16 == 0:
                    write(OUT/'progress.json', dict(completed=len(rows), expected=224, elapsed_seconds=time.monotonic()-started))
        verify_hashes(plan['input_sha256'])
        result = dict(protocol=plan['protocol'], score=score_panel(rows, plan['protocol']),
            profile=profile(rows), elapsed_seconds=time.monotonic()-started,
            rows_sha256=identity(sorted(rows, key=lambda r:tuple(r[k] for k in FIELDS))),
            input_sha256=plan['input_sha256'], new_neural_training_decisions=0,
            admitted=False, fresh_confirmation_started=False, qualified_version=None)
        write(OUT/'completion.json', result)
        print(json.dumps({k:result[k] for k in ('score', 'profile', 'elapsed_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    a = parser.parse_args()
    prepare() if a.command == 'prepare' else run()
