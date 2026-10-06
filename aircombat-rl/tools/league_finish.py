"""After the running evaluations, package, check real loading, and report.

This helper archives a failed evaluation too, but never promotes it to success.
The agent must inspect any analysis_required result and the rendered report.
"""
from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
import time
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_validate import process_live


def run(root, out, bundle, pids, plot_python):
    if out.exists():
        raise FileExistsError('Use a new finishing log directory')
    out.mkdir(parents=True)
    write(out/'plan.json', dict(run=str(root), bundle=str(bundle), waiting_for_pids=pids,
          source_sha256=sha(Path(__file__)), plot_python=str(plot_python),
          scope='Packaging and compatibility checks, not model selection or new performance evidence'))
    while True:
        active=[p for p in pids if process_live(p)]
        if not active:
            break
        write(out/'progress.json',dict(stage='waiting',active_pids=active,checked_at=time.time()))
        time.sleep(15)
    followup=root/'followup_pool'
    done=read(followup/'completion.json')
    runner=read(root/'runner_probe/completion.json')
    if done.get('stage')!='heldout' or runner.get('status')!='complete':
        raise ValueError('Evaluations did not reach their expected terminal states')
    def command(name, args, python=sys.executable):
        write(out/'progress.json',dict(stage=name))
        cmd=[str(python),'-X','utf8','-m',*map(str,args)]
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(out/f'{name}.json',dict(command=cmd,returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{name} failed; see its log')
    command('package',['tools.league_package','--main',root/'main','--followup',followup,'--out',bundle,'--final'])
    command('loader_check',['tools.league_bundle_check','--bundle',bundle,
                           '--out',ROOT/'experiments/league/results/loader_check.json'])
    command('duel_check',['tools.league_duel','--a',bundle/'models/final','--b',bundle/'models/cem_original',
                         '--out',out/'duel','--band',33020000,'--n',2])
    command('replay_check',['tools.league_replay','--bundle',bundle,'--out',out/'replay',
                           '--band',33010000,'--n',2,'--smoke'])
    command('report',['tools.league_final_report','--run',root,'--out',ROOT/'experiments/league/results','--plot'],plot_python)
    status='ready_for_audit' if done['status']=='evaluation_passed' and not runner['observed_improvement'] else 'analysis_required'
    write(out/'completion.json',dict(status=status,evaluation=done['status'],
          runner_improvement=runner['observed_improvement'],actual_loader_and_cli_checks_passed=True,
          next='Inspect plots and audit the objective; this helper does not mark the goal complete.'))
    write(out/'progress.json',dict(stage=status))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--wait-pids',nargs='+',type=int,required=True)
    p.add_argument('--plot-python',type=Path,required=True)
    a=p.parse_args()
    try:
        run(a.run.resolve(),a.out.resolve(),a.bundle.resolve(),a.wait_pids,a.plot_python.resolve())
    except Exception as error:
        write(a.out/'failure.json',dict(error=repr(error)))
        raise
