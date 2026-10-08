"""CEM learns a public-motion gate over four preserved policies; no PPO updates."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json,os,shutil,time,traceback,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_matches import duel,initialize_worker
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import code_groups,profile


def export(folder,experts,parameters,identity,forced=None):
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(io.ROOT/'experiments/league/portfolio_policy.py',folder/'policy.py')
    (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
    for i,spec in enumerate(experts):
        target=folder/'experts'/str(i);target.mkdir(parents=True)
        shutil.copyfile(io.ROOT/spec['design']/'policy.py',target/'policy.py')
        shutil.copyfile(io.ROOT/spec['weights'],target/'policy_net.zip')
    config=dict(probe_seconds=float(parameters[0]),gate=np.asarray(parameters[1:]).reshape(3,5).tolist(),forced_expert=forced)
    with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('parameters.json',json.dumps(config))
    spec=dict(id=identity,kind='submission',design=io.relative(folder),weights=io.relative(folder/'policy_net.zip'))
    io.write(folder/'entrant.json',spec);io.write(folder/'artifact_sha256.json',{str(p.relative_to(folder)):io.sha(p) for p in folder.rglob('*') if p.is_file()})
    return spec


def qualify(job):
    direct=duel(job['direct'],job['foe'],job['band'],job['n']);wrapped=duel(job['wrapped'],job['foe'],job['band'],job['n'])
    if direct['episodes']!=wrapped['episodes'] or direct['summary']!=wrapped['summary']:raise AssertionError('Portfolio expert does not match original')
    return dict(job=job,exact_episode_summary_equality=True,direct=direct,wrapped=wrapped)


def evaluate(pool,specs,foes,band,n,folder):
    groups=code_groups(foes);results=[[] for _ in specs];started=time.perf_counter()
    jobs=[(a,dict(own=s,foe=f,band=band,n=n)) for a,s in enumerate(specs) for f in foes]
    futures={pool.submit(duel,**j):(i,a,j) for i,(a,j) in enumerate(jobs)}
    for future in as_completed(futures):
        i,a,j=futures[future];r=future.result();results[a].append(r);io.write(folder/f'matches/match_{i:03d}.json',dict(job=j,result=r))
        if io.resources()['commit_headroom_gib']<1.8:raise MemoryError('Commit safety floor')
    profiles=[]
    for rows in results:validate_records(rows,foes,band,n);profiles.append(profile(rows,groups))
    record=dict(specs=specs,opponents=foes,band=band,n=n,profiles=profiles,rank=pair_rank(profiles),games=len(jobs)*n,wall_seconds=time.perf_counter()-started)
    io.write(folder/'result.json',record);return record


def freeze(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Diagnostic still live')
    old=io.read(source/'plan.json');io.verify(old);raw=io.read(source/'raw_geometry_analysis.json')
    if raw['status']!='geometry_raw_verified':raise ValueError('Verified geometry required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed geometry evidence')
    focal=io.read(io.ROOT/old['source']/'plan.json');roles=old['roles'];specialists=[k for k in roles if k.startswith('focal_s')]
    assert len(specialists)==2
    experts=[[roles['baseline'][a],roles['teacher'][0],roles[specialists[0]][a],roles[specialists[1]][a]] for a in range(2)]
    dev=focal['development_opponents'];mandatory=['temporal_extend_right','temporal_weave_right','evader','ace','lead','pursuit','circler','ddqn_s0']
    training=[next(s for s in focal['opponents'] if s['id']==name) for name in mandatory]
    for i in np.linspace(0,len(dev)-1,8,dtype=int):
        if dev[i] not in training:training.append(dev[i])
    initial=np.r_[.5,[1,-8,0,-8,0],[-2,0,0,0,0],[-2,0,0,0,0]]
    base=initial.copy();base[1:]=0.;base[[1,6,11]]=-8.
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'experiments/league/portfolio_policy.py',io.ROOT/'tests/test_league_portfolio.py']:sources[io.relative(p)]=io.sha(p)
    for p in [source/'raw_geometry_analysis.json',source/'plan.json',source/'completion.json']:inputs[io.relative(p)]=io.sha(p)
    for band in [284000000,285000000,90000000]:
        reservation=io.ROOT/f'runs/portfolio_reservation_{band}.json'
        if reservation.exists():raise FileExistsError('Reserved portfolio conditions')
        io.write(reservation,dict(run=io.relative(out),start=band,stop_exclusive=band+2))
    plan=dict(source=io.relative(source),experts=experts,opponents=focal['opponents'],training_opponents=training,development_opponents=dev,seeds=[6400,6401],training_bands=[284000000,285000000],development_band=90000000,n=4,workers=8,population=8,generations=3,elites=3,initial=initial.tolist(),baseline_parameters=base.tolist(),low=[.5]+[-8.]*15,high=[6.]+[8.]*15,initial_std=[1.]+[1.5]*15,source_sha256=sources,input_sha256=inputs,environment=io.environment(),scope='CEM gate parameter learning, not PPO weight training. Four experts frozen; all104 archived opponents preserved. Two search RNGs, paired seat/ICs and two action replicas. Search uses a fixed diverse sparse panel and consumed training conditions. Separate development only; no final/heldout/unseen/promotion claim.',budget='Exactly3 generations x8 candidates x2 search RNGs. Candidate0 retains incumbent; candidate1 baseline. Common training conditions within a seed.3 elites with .5 smoothed moments; min std .1. Develop selected best and baseline after search; do not select experts retrospectively.')
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,target)
    return plan


def run(source,out):
    plan=freeze(source,out)
    if min(io.resources()[k] for k in ('commit_headroom_gib','available_memory_gib'))<4.5:raise MemoryError('Insufficient startup headroom')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    try:
        with ProcessPoolExecutor(max_workers=plan['workers'],initializer=initialize_worker) as pool:
            jobs=[]
            for a,experts in enumerate(plan['experts']):
                for e,direct in enumerate(experts):
                    wrapped=export(out/f'qualification/policy_a{a}_e{e}',experts,plan['baseline_parameters'] if e==0 else plan['initial'],f'qualification_a{a}_e{e}',forced=None if e==0 else e)
                    jobs.append(dict(direct=direct,wrapped=wrapped,foe=plan['training_opponents'][e%3],band=170260000,n=2))
            futures={pool.submit(qualify,j):i for i,j in enumerate(jobs)}
            for future in as_completed(futures):io.write(out/f'qualification/check_{futures[future]:03d}.json',future.result())
            io.write(out/'qualification.json',dict(status='exact_expert_reproduction',games=32,jobs=jobs));io.verify(plan)
            print('portfolio qualification passed32 games; CEM gate learning begins',flush=True)
            completed=[]
            for ordinal,seed in enumerate(plan['seeds']):
                rng=np.random.default_rng(seed);mean=np.array(plan['initial']);std=np.array(plan['initial_std']);incumbent=mean.copy();best=None;history=[]
                for generation in range(plan['generations']):
                    io.verify(plan);population=np.clip(rng.normal(mean,std,size=(plan['population'],len(mean))),plan['low'],plan['high']);population[0]=incumbent;population[1]=plan['baseline_parameters'];records=[]
                    for i,parameters in enumerate(population):
                        folder=out/f's{seed}/g{generation:02d}/c{i:02d}'
                        specs=[export(folder/f'action_{a}',experts,parameters,f'portfolio_s{seed}_g{generation}_c{i}_a{a}') for a,experts in enumerate(plan['experts'])]
                        r=evaluate(pool,specs,plan['training_opponents'],plan['training_bands'][ordinal],plan['n'],folder/'training');r['parameters']=parameters.tolist();records.append(r)
                        if best is None or tuple(r['rank'])>tuple(best['rank']):best=r
                        print('portfolio CEM',seed,'generation',generation,'candidate',i,'rank',r['rank'],flush=True)
                    ranked=sorted(range(len(records)),key=lambda i:tuple(records[i]['rank']),reverse=True);elite=population[ranked[:plan['elites']]];mean=.5*mean+.5*elite.mean(0);std=np.maximum(.1,.5*std+.5*elite.std(0));incumbent=np.array(best['parameters'])
                    row=dict(generation=generation,records=records,selected=best,next_mean=mean.tolist(),next_std=std.tolist());history.append(row);io.write(out/f's{seed}/generation_{generation:02d}.json',row)
                io.write(out/f's{seed}/frozen_selection.json',best)
                developed=evaluate(pool,best['specs'],plan['development_opponents'],plan['development_band'],plan['n'],out/f's{seed}/development')
                completed.append(dict(seed=seed,selected=best,development=developed));io.write(out/'completed_searches.json',dict(searches=completed))
            baseline=evaluate(pool,[e[0] for e in plan['experts']],plan['development_opponents'],plan['development_band'],plan['n'],out/'baseline_development')
        io.verify(plan);io.write(out/'completion.json',dict(status='portfolio_cem_complete',searches=completed,baseline=baseline,qualification_games=32,final_opened=False,heldout_opened=False,policy_promoted=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
