"""Replay a COMPLETED assessment stage in a relocated matching checkout.

Saved opponents, seats and initial conditions are reused: this is reproduction,
never a fresh held-out test. Runtime PIDs/environment from the old PC are unused.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys

from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_repair_train import Evaluation
from tools.league_matches import initialize_worker
from tools.league_train import verify


def stage_inputs(source, stage):
    if stage not in ('selection', 'final', 'temporal'):
        raise ValueError('Unknown assessment stage')
    completed = read(source / 'completion.json')
    if completed['status'] not in ('no_profile_selection', 'tournament_profile_failed', 'tournament_profile_passed'):
        raise ValueError('Assessment must be terminal')
    if stage == 'final' and not completed['final_opened']:
        raise ValueError('Final stage was never opened')
    if stage == 'temporal' and not completed['temporal_opened']:
        raise ValueError('Temporal stage was never opened')
    plan = read(source / 'plan.json'); verify(plan)
    schedule = read(source / stage / 'schedule.json')
    requests = schedule['requests']
    records = [read(source / stage / f'candidate_{i:03d}.json') for i in range(len(requests))]
    if not requests or any(r['request'] != request for r, request in zip(records, requests)):
        raise ValueError('Stage evidence does not match its frozen schedule')
    for request in requests:
        if any(request[k] != requests[0][k] for k in ('opponents', 'band', 'n', 'weights')):
            raise ValueError('Stage requests use different conditions')
    return plan, schedule, records


def exact_results(records):
    return [[{k: value for k, value in result.items() if k != 'elapsed_seconds'}
        for result in r['results']] for r in records]


def run(source, stage, out, workers):
    if out.exists():
        raise FileExistsError('Use a new output folder for reproduction')
    if not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError('Invalid worker count')
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Set all three thread limits BEFORE Python starts')
    original, schedule, records = stage_inputs(source, stage)
    inputs = [source / n for n in ('plan.json', 'completion.json')]
    inputs += [source / stage / 'schedule.json']
    inputs += [source / stage / f'candidate_{i:03d}.json' for i in range(len(records))]
    frozen = dict(source_sha256=dict(original['source_sha256']), input_sha256=dict(original['input_sha256']))
    frozen['source_sha256']['tools/league_assessment_replay.py'] = sha(__file__)
    frozen['input_sha256'].update({p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in inputs})
    frozen.update(source=source.resolve().relative_to(ROOT).as_posix(), stage=stage, workers=workers,
        python=sys.version, platform=sys.platform, purpose='Replay existing evidence; not independent validation.',
        original_environment=original['environment'])
    immutable_json(out / 'plan.json', frozen)
    request = schedule['requests'][0]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        fresh = Evaluation()(pool, [r['spec'] for r in schedule['requests']], request['opponents'],
            request['band'], request['n'], out / stage, request['weights'], traced=schedule['traced'])
    same = exact_results(fresh) == exact_results(records)
    verify(frozen)
    result = dict(status='replay_matched' if same else 'replay_differed', exact_episode_and_summary_equality=same,
        stage=stage, policies=len(records), games=sum(r['metrics']['games'] for r in fresh),
        scope='Reused initial conditions/opponents. Timing excluded; all other saved match fields compared. Different hardware/software can change results; differences are reported rather than ignored.')
    write(out / 'completion.json', result)
    return result


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--stage',choices=('selection','final','temporal'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    print(run(args.source.resolve(),args.stage,args.out.resolve(),args.workers))
