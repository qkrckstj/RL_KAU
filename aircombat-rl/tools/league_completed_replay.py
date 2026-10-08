"""Replay complete CEM/pilot evidence without restoring another PC's processes.

Existing seeds/opponents are reused; this is reproduction, not fresh evidence.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import re
import sys
from tools.autolab_cem import ROOT,read,sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_repair_train import Evaluation,validate_record
from tools.league_matches import initialize_worker
from tools.league_train import verify


def stage_inputs(source,stage):
    if not re.fullmatch(r'(baseline|selection|screen|confirmation|final|parameter_audit|s\d+/g\d+/(screen|confirm|development))',stage):
        raise ValueError('Unknown completed stage')
    if not (source/'completion.json').exists():raise ValueError('Source experiment must be complete')
    completed=read(source/'completion.json')
    if completed['status'] not in ('no_development_improvement','repair_profile_failed','repair_profile_passed','development_pilot_complete'):
        raise ValueError('Unsupported terminal experiment')
    if stage=='final' and not completed.get('final_opened',False):raise ValueError('Final stage was never opened')
    if stage=='parameter_audit' and not completed.get('heldout_opened',False):raise ValueError('Parameter holdout was never opened')
    plan=read(source/'plan.json');verify(plan)
    schedule=read(source/stage/'schedule.json');requests=schedule['requests']
    if not requests:raise ValueError('Empty stage schedule')
    records=[read(source/stage/f'candidate_{i:03d}.json') for i in range(len(requests))]
    for record,request in zip(records,requests):
        if request['n']<=0 or request['n']%2:raise ValueError('Incomplete seat pairs')
        if any(request[k]!=requests[0][k] for k in ('opponents','band','n','weights')):
            raise ValueError('Stage requests use different conditions')
        validate_record(record,request)
        if any(('damage_traces' in r)!=schedule['traced'] for r in record['results']):
            raise ValueError('Trace layout differs from schedule')
    return plan,schedule,records


def exact_results(records):
    return [[{k:v for k,v in result.items() if k!='elapsed_seconds'}
             for result in record['results']] for record in records]


def run(source,stage,out,workers):
    if out.exists():raise FileExistsError('Use a new reproduction output folder')
    if not 1<=workers<=(os.cpu_count() or 1):raise ValueError('Invalid worker count')
    if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('Set all three thread limits before Python starts')
    original,schedule,records=stage_inputs(source,stage)
    inputs=[source/n for n in ('plan.json','completion.json')]+[source/stage/'schedule.json']
    inputs += [source/stage/f'candidate_{i:03d}.json' for i in range(len(records))]
    frozen=dict(source_sha256=dict(original['source_sha256']),input_sha256=dict(original['input_sha256']))
    frozen['source_sha256']['tools/league_completed_replay.py']=sha(__file__)
    frozen['input_sha256'].update({p.resolve().relative_to(ROOT).as_posix():sha(p) for p in inputs})
    frozen.update(source=source.resolve().relative_to(ROOT).as_posix(),stage=stage,workers=workers,
        python=sys.version,platform=sys.platform,original_environment=original['environment'],
        purpose='Replay completed evidence; same conditions, not independent validation.')
    immutable_json(out/'plan.json',frozen);request=schedule['requests'][0]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        fresh=Evaluation()(pool,[r['spec'] for r in schedule['requests']],request['opponents'],
            request['band'],request['n'],out/stage,request['weights'],traced=schedule['traced'])
    same=exact_results(fresh)==exact_results(records);verify(frozen)
    result=dict(status='replay_matched' if same else 'replay_differed',exact_episode_and_summary_equality=same,
        stage=stage,policies=len(records),games=sum(r['metrics']['games'] for r in fresh),
        scope='All saved match fields except elapsed_seconds compared. Different software/hardware may produce differences; differences are reported, not ignored. No process/PID or mid-training resume is restored.')
    write(out/'completion.json',result);return result


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--stage',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    print(run(a.source.resolve(),a.stage,a.out.resolve(),a.workers))
