"""Confirm the selected hand-configured runner probe on fresh open-development ICs."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tools.autolab_cem import read, write, sha
from tools.league_matches import evaluate_jobs
from tools.league_train import verify


def run(root,out):
    if out.exists():
        raise FileExistsError('Use a new confirmation directory')
    main=read(root/'main/plan.json')
    verify(main)
    parent=read(root/'followup_pool/frozen_selection.json')['chosen']
    probe=read(root/'runner_probe/completion.json')['best']
    foe=dict(id='evader',kind='bot',name='evader')
    plan=dict(parent=parent,probe=probe,foe=foe,band=33030000,n=80,
        source_sha256=sha(Path(__file__)),
        gate='At least 4/80 probe wins, no probe losses, and paired-IC 95% win-gain interval lower bound > 0',
        scope='Fresh open development confirmation of one selected manual probe. Not a new final test or a trained model.')
    write(out/'plan.json',plan)
    write(out/'progress.json',dict(stage='confirming_runner'))
    results=evaluate_jobs([dict(own=a,foe=foe,band=plan['band'],n=plan['n']) for a in (parent,probe)],out/'matches')
    write(out/'results.json',results)
    vectors=[{(e['seed'],e['seat']):int(e['won']) for e in r['episodes']} for r in results]
    if set(vectors[0])!=set(vectors[1]):
        raise ValueError('Confirmation seed/seat mismatch')
    seeds=sorted({k[0] for k in vectors[0]})
    differences=np.array([np.mean([vectors[1][seed,seat]-vectors[0][seed,seat] for seat in ('red','blue')]) for seed in seeds])
    rng=np.random.default_rng(722)
    ci=np.quantile(differences[rng.integers(len(seeds),size=(10000,len(seeds)))].mean(axis=1),[.025,.975]).tolist()
    ps=results[1]['summary']
    warranted=ps['wins']>=4 and ps['losses']==0 and ci[0]>0
    verify(main)
    write(out/'completion.json',dict(status='complete',conditional_training_warranted=bool(warranted),
          parent=results[0]['summary'],probe=ps,paired_ic_win_gain_95_ci=ci,
          scope=plan['scope'],next='Investigate a conditionally selected pursuit expert against the full archive with a fresh future test.' if warranted
          else 'The small diagnostic improvement was not confirmed strongly enough to launch conditional pursuit training from this evidence alone.'))
    write(out/'progress.json',dict(stage='complete'))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('runs/league_20261006'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        run(a.run.resolve(),a.out.resolve())
    except Exception as error:
        write(a.out/'failure.json',dict(error=repr(error)))
        raise
