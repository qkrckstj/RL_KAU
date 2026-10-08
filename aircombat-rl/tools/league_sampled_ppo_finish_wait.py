"""Wait before importing the learner, then run the qualified remainder trainer.

Only the failed zero-step startup is replaced; all original files are preserved.
"""
from argparse import ArgumentParser
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_ppo_finish as runner
from experiments.league.memory_pause import wait_for_headroom


def run(source,out):
    if out.exists():raise FileExistsError('Use a new startup-wait directory')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits first')
    old=io.read(source/'plan.json');io.verify(old)
    failure=io.read(source/'failure.json');runtime=io.read(source/'runtime.json')
    if io.process_live(runtime['pid']):raise RuntimeError('Prior startup is still live')
    if not failure['error'].rstrip().endswith('MemoryError: Insufficient one-worker startup headroom'):
        raise ValueError('Only the verified zero-step pre-pool failure is recoverable here')
    folder=source/f"s{old['resume_rng']}"
    if any(folder.iterdir()):raise ValueError('Prior attempt contains training artifacts; inspect before resuming')
    if old['smoke']:raise ValueError('Expected main-run plan')
    checked=io.read(io.ROOT/'runs/league_sampled_ppo_finish_smoke_20261007/completion.json')
    if checked['status']!='sampled_remainder_integration_passed':raise ValueError('Completed integration required')
    plan=dict(old,source_sha256=dict(old['source_sha256']),input_sha256=dict(old['input_sha256']))
    plan['source_sha256'][io.relative(Path(__file__))]=io.sha(Path(__file__))
    for name in ('plan.json','failure.json','runtime.json'):
        plan['input_sha256'][io.relative(source/name)]=io.sha(source/name)
    plan.update(startup_attempt_previous=io.relative(source),training_bands=[265000000],
        startup_wait_rule='Before Torch import: two samples5seconds apart with commit>=4.5GiB and physical>=3.5GiB; max300seconds. Original initialized/running guards and runtime pause behavior unchanged.',
        startup_recovery_scope='Prior264M band reserved but unused: failure before pool creation, empty learner directory and exit1. New265M band and output; no old reservation overwritten.')
    reservation=io.ROOT/'runs/training_reservation_265000000.json'
    if reservation.exists():raise FileExistsError('New training band already reserved')
    io.write(reservation,dict(run=io.relative(out),start=265000000,stop_exclusive=266000000))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in plan['source_sha256']:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(io.ROOT/name,target)
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        processes=1,virtual_environments=8,device='cpu',stage_at_launch='waiting_before_torch_import'))
    def emit(row):
        row=dict(row,at=datetime.now(timezone.utc).isoformat())
        with (out/'startup_wait.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f"startup {row['stage']}: {row['elapsed_seconds']:.1f}s, commit headroom {row['resources']['commit_headroom_gib']:.3f}GiB",flush=True)
    ready=wait_for_headroom(io.resources,emit,resume_commit=4.5,resume_physical=3.5)
    io.write(out/'startup_ready.json',ready)
    started=time.perf_counter();second=runner.train(out,plan);io.verify(plan)
    searches=[plan['original_first_branch'],second]
    io.write(out/'completion.json',dict(status='sampled_recovery_continuations_complete',searches=searches,
        additional_training_steps=sum(r['additional_steps'] for r in searches),new_steps_this_run=second['local_resume_steps'],
        elapsed_seconds=time.perf_counter()-started,startup_wait_seconds=ready['elapsed_seconds'],
        shared_prefix_interactions=plan['shared_prefix_interactions'],interrupted_predecessors_preserved=True,
        final_opened=False,heldout_opened=False,promotion=False,scope=plan['quality_scope'],
        completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.source.resolve(),args.out.resolve())
    except BaseException:
        io.write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
