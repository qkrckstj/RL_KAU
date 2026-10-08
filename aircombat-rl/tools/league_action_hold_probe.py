"""Measure short action persistence before changing any neural learning protocol."""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,shutil,traceback,zipfile,json
from tools import league_thread_benchmark as io
from tools.league_matches import duel,initialize_worker
from tools.league_residual_train import validate_records


SUFFIX='''

UnheldPolicy = Policy

class Policy(UnheldPolicy):
    def __init__(self, weights=None, device='cpu'):
        with zipfile.ZipFile(weights) as archive:
            self.hold_steps = json.loads(archive.read('action_hold.json'))['hold_steps']
        if not isinstance(self.hold_steps, int) or self.hold_steps < 1:
            raise ValueError('Positive integer hold_steps required')
        super().__init__(weights, device)

    def reset(self):
        super().reset()
        self.hold_count = 0
        self.held_action = None

    def act(self, obs):
        # Call the preserved policy every official step to retain its teacher
        # history and public-clock reset semantics. Unused RNG draws are part
        # of this declared evaluation variant, not extra actions to the env.
        action = super().act(obs)
        if self.hold_count % self.hold_steps == 0:
            self.held_action = action
        self.hold_count += 1
        return self.held_action
'''


def export(source,out,hold,identity):
    out.mkdir(parents=True,exist_ok=False)
    (out/'policy.py').write_text((io.ROOT/source['design']/'policy.py').read_text('utf-8')+SUFFIX,encoding='utf-8')
    shutil.copyfile(io.ROOT/source['design']/'wrappers.py',out/'wrappers.py')
    shutil.copyfile(io.ROOT/source['weights'],out/'policy_net.zip')
    with zipfile.ZipFile(out/'policy_net.zip','a',zipfile.ZIP_DEFLATED) as z:z.writestr('action_hold.json',json.dumps(dict(hold_steps=hold)))
    spec=dict(id=identity,kind='submission',design=io.relative(out),weights=io.relative(out/'policy_net.zip'));io.write(out/'entrant.json',spec)
    io.write(out/'artifact_sha256.json',{p.name:io.sha(p) for p in out.iterdir() if p.is_file()});return spec


def qualify(job):
    a=duel(job['base'],job['foe'],job['band'],job['n']);b=duel(job['held'],job['foe'],job['band'],job['n'])
    assert a['episodes']==b['episodes'] and a['summary']==b['summary']
    return dict(job=job,base=a,held=b,exact_equality=True)


def run(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Portfolio search still live')
    old=io.read(source/'plan.json');io.verify(old)
    if io.read(source/'raw_portfolio_analysis.json')['status']!='portfolio_raw_verified':raise ValueError('Verified portfolio search required')
    for p,h in io.read(source/'raw_portfolio_analysis.json')['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed prior evidence')
    transfer=io.ROOT/'runs/league_portfolio_transfer_20261007'
    transfer_raw=io.read(transfer/'raw_transfer_verification.json')
    io.verify(io.read(transfer/'plan.json'))
    if transfer_raw['status']!='portfolio_transfer_raw_verified' or io.process_live(io.read(transfer/'runtime.json')['pid']):raise ValueError('Completed verified transfer required')
    for p,h in transfer_raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed transfer evidence')
    base=[e[0] for e in old['experts']]
    names=['temporal_extend_right','temporal_weave_right','evader','ace','pursuit','ddqn_s0','temporal_pulsed_right','history_s6000_t1048576_a4900']
    foes=[next(s for s in old['opponents'] if s['id']==name) for name in names]
    variants={str(h):[export(s,out/f'hold_{h}/action_{a}',h,f'hold{h}_a{a}') for a,s in enumerate(base)] for h in [1,5,10]}
    band=91000000;reservation=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if reservation.exists():raise FileExistsError('Diagnostic conditions reserved')
    sources=dict(old['source_sha256']);sources[io.relative(Path(__file__).resolve())]=io.sha(Path(__file__).resolve());sources['tests/test_league_action_hold.py']=io.sha(io.ROOT/'tests/test_league_action_hold.py');inputs=dict(old['input_sha256'])
    for p in [source/'plan.json',source/'completion.json',source/'raw_portfolio_analysis.json',transfer/'plan.json',transfer/'completion.json',transfer/'raw_transfer_verification.json']:inputs[io.relative(p)]=io.sha(p)
    for p in out.rglob('*'):
        if p.is_file():inputs[io.relative(p)]=io.sha(p)
    jobs=[dict(own=s,foe=f,band=band,n=4) for specs in variants.values() for s in specs for f in foes]
    plan=dict(source=io.relative(source),base=base,variants=variants,opponents=foes,jobs=jobs,band=band,n=4,workers=8,source_sha256=sources,input_sha256=inputs,scope='Decision persistence diagnostic only, not neural training. Hold1 is unchanged policy, hold5/10 repeat its sampled official action for.25/.5seconds. Existing policy/teacher still observe every.05second. No physics, foe reaction, reward/verdict or IC modification. Fresh conditions, known foes, both seats/two fixed action RNGs. These variants change the behavior immediately, without learning; no improvement claim before matching training.')
    io.verify(plan);io.write(reservation,dict(run=io.relative(out),start=band,stop_exclusive=band+2));io.write(out/'plan.json',plan)
    if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<4.5:raise MemoryError('Startup headroom')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
            checks=[dict(base=s,held=variants['1'][a],foe=f,band=170280000,n=2) for a,s in enumerate(base) for f in foes[:2]]
            futures={pool.submit(qualify,j):i for i,j in enumerate(checks)}
            for future in as_completed(futures):io.write(out/f'qualification/check_{futures[future]:03d}.json',future.result())
            print('hold1 equals baseline on16 qualification games',flush=True)
            results=[];futures={pool.submit(duel,**j):i for i,j in enumerate(jobs)}
            for count,future in enumerate(as_completed(futures),1):
                i=futures[future];r=future.result();results.append(r);io.write(out/f'matches/match_{i:03d}.json',dict(job=jobs[i],result=r))
                if count%8==0:print('action-hold diagnostic',count,len(jobs),flush=True)
                if io.resources()['commit_headroom_gib']<1.8:raise MemoryError('Commit floor')
        analysis={}
        for h,specs in variants.items():
            for spec in specs:validate_records([r for r in results if r['own']==spec['id']],foes,band,4)
            analysis[h]={f['id']:{k:sum(r['summary'][k] for r in results if r['own'] in {s['id'] for s in specs} and r['foe']==f['id']) for k in ['wins','draws','losses']} for f in foes}
        io.verify(plan);io.write(out/'completion.json',dict(status='action_hold_diagnostic_complete',games=192,qualification_games=16,analysis=analysis,final_opened=False,heldout_opened=False,policy_promoted=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
