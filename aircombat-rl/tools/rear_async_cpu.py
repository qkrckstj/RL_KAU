"""CPU rear continuation with immutable checkpoints and independent evaluation."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from experiments.rear_async.storage import read,write,sha,identity,publish,verify
ROOT=Path(__file__).resolve().parents[1]

def freeze(out,workers,smoke):
    if out.exists():raise FileExistsError('Use a fresh run or --resume')
    out.mkdir(parents=True)
    inputs=read(ROOT/'experiments/league/rear_desktop_20261012/inputs.json')
    budget=read(ROOT.parent/'docs/REAR_BRANCH_BUDGET_V2.json')
    if sha(ROOT/inputs['parent'])!=budget['parent']['sha256']:raise ValueError('Parent hash changed')
    wanted=('circler','fixed_right','league_residual_continuation_20261007_s3301_t2097152',
        'extend_cem_s6800_g3_c9','ace','lead','pursuit','switching_turn_strong')
    byid={r['id']:r for r in inputs['opponents']}
    plan=dict(inputs,processes=workers,seeds=[42991] if smoke else [42100,42101],
        training_bands=[1800200000] if smoke else [1800400000,1810400000],
        minimum_steps=40960 if smoke else budget['additional_steps_per_repeat'],
        maximum_steps=40960 if smoke else budget['conditional_max_additional_steps_per_repeat'],
        checkpoint_steps=20480 if smoke else budget['checkpoint_every_steps'],
        monitor_steps=20480 if smoke else budget['monitor_every_steps'],
        monitor_opponents=[byid[k] for k in wanted],monitor_band=1702500000,
        development_band=1702600000,monitor_ics=1 if smoke else 8,development_ics=1 if smoke else 8,
        smoke=smoke,device='cpu',evaluation_processes=1,
        candidate_rule='Retain all checkpoints. Positive GPU/CPU or native win gains including1pp remain candidates; rear improvement independently qualifies. No2pp discard gate.',
        physics='Original native JSBSim; no CUDA replacement',rule='user_terminal_health_round4_v2')
    if smoke:
        plan['monitor_opponents']=plan['monitor_opponents'][:1]
        plan['development_opponents']=plan['development_opponents'][:1]
    # These disjoint seed reservations are local to this run; same seed pairs
    # are compared at every checkpoint and remain development observations.
    for band in plan['training_bands']:
        path=ROOT/f'runs/training_reservation_{band}.json'
        if path.exists():raise FileExistsError('Already reserved training band')
        write(path,dict(run=str(out),start=band,stop_exclusive=band+1000000))
    files=list((ROOT/'aircombat_gym').rglob('*.py'))+list((ROOT/'experiments/league').glob('*.py'))
    files+=list((ROOT/'experiments/rear_async').glob('*.py'))+[Path(__file__),ROOT/inputs['parent']]
    files+=[ROOT/'tools/league_residual_export.py',ROOT/'tools/league_residual_action_mode.py',ROOT/'tools/league_matches.py']
    files+=list((ROOT/'experiments/league/rear_desktop_20261012').rglob('*'))
    plan['frozen_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()}
    plan['config_id']=identity({k:v for k,v in plan.items() if k!='frozen_sha256'})
    write(out/'plan.json',plan);return plan

def check_plan(plan):
    for name,digest in plan['frozen_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)

def checkpoint(model,vec,out,seed,prefix,steps,games,seconds,baseline=False):
    import random,numpy as np,torch
    from tools.league_residual_export import export
    from tools.league_residual_action_mode import build_variant
    plan=read(out/'plan.json');name=f's{seed}_step_{steps:09d}'
    staging=out/'checkpoints'/('.'+name+'.tmp');target=out/'checkpoints'/name
    if target.exists():return target
    staging.mkdir(parents=True,exist_ok=False)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
    try:
        model.save(staging/'learner.zip')
        torch.save(rng,staging/'rng.pt')
        torch.save(vec.env_method('snapshot_worker') if vec is not None else None,staging/'workers.pt')
        greedy=export(staging/'greedy',plan['teacher'],model.policy,name)
        specs={str(a):build_variant(greedy,staging/f'sampled_{a}',a,name+f'_a{a}') for a in (4900,4901)}
        # Relocate all runtime paths before the staging directory is renamed.
        final_specs={a:{**s,'design':str(target/f'sampled_{a}'),'weights':str(target/f'sampled_{a}/policy_net.zip')} for a,s in specs.items()}
        publish(staging,target,dict(checkpoint_id=name,config_id=plan['config_id'],seed=seed,prefix_steps=prefix,
            additional_steps=steps,total_steps=prefix+steps,training_games=games,training_seconds=seconds,
            sampled_specs=final_specs,baseline=baseline,device='cpu',resume_scope='Optimizer/RNG/sampler preserved; native episodes restart after a process restart.'))
    finally:
        torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    return target

def request(out,path,kind):
    m=verify(path);plan=read(out/'plan.json')
    value=dict(checkpoint=str(path),checkpoint_id=m['checkpoint_id'],kind=kind,config_id=plan['config_id'])
    value['job_id']=identity(value)
    write(out/'queue'/f"{value['job_id']}.json",value)

def train(out,resume=False):
    import random,numpy as np,psutil,torch
    from functools import partial
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from experiments.rear_async.worker import make_worker
    torch.set_num_threads(1);plan=read(out/'plan.json');check_plan(plan)
    lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    process_started=time.perf_counter();write(out/'training_runtime.json',dict(pid=os.getpid(),status='running',device='cpu'))
    class Progress(BaseCallback):
        def __init__(self,seed,prefix,games=0):super().__init__();self.seed=seed;self.prefix=prefix;self.games=games;self.last=0
        def _on_step(self):
            if (out/'STOP').exists():raise InterruptedError('User STOP marker')
            if self.num_timesteps%2048==0 and psutil.virtual_memory().available<8*2**30:raise MemoryError('Host available below8GiB')
            rows=[]
            for info in self.locals['infos']:
                if 'episode' in info:
                    rows.append(dict(info['episode'],seed=self.seed,additional_step=self.num_timesteps-self.prefix,
                        episode_metadata=info.get('league_episode',{})));self.games+=1
            if rows:
                with (out/f's{self.seed}_episodes.jsonl').open('a') as f:
                    for row in rows:f.write(json.dumps(row)+'\n')
            if time.perf_counter()-self.last>=10:
                self.last=time.perf_counter();write(out/'training_progress.json',dict(seed=self.seed,
                    additional_steps=self.num_timesteps-self.prefix,training_games=self.games,pid=os.getpid(),
                    available_host_gib=psutil.virtual_memory().available/2**30,elapsed_seconds=time.perf_counter()-process_started))
            return True
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            check_plan(plan);local=dict(plan,seed=seed,ic_band=plan['training_bands'][ordinal])
            ready=[p for p in (out/'checkpoints').glob(f's{seed}_step_*') if (p/'READY.json').exists()]
            latest=max(ready,key=lambda p:verify(p)['additional_steps']) if ready and resume else None
            meta=verify(latest) if latest else None
            from experiments.rear_async.judgement import continuation_limit
            old_decision=read(out/f's{seed}_budget_decision.json') if (out/f's{seed}_budget_decision.json').exists() else None
            initial_limit=continuation_limit(plan['minimum_steps'],plan['maximum_steps'],old_decision)
            if meta and meta['additional_steps']>=initial_limit:continue
            restart=__import__('torch').load(latest/'workers.pt',weights_only=False) if latest else None
            vec=GroupedSubprocVecEnv([partial(make_worker,local,i,restart[i] if restart else None) for i in range(plan['workers'])],plan['processes'])
            model=None
            try:
                model=PPO.load(latest/'learner.zip' if latest else ROOT/plan['parent'],env=vec,device='cpu',seed=seed,force_reset=True,n_steps=640)
                prefix=meta['prefix_steps'] if meta else model.num_timesteps
                progress=Progress(seed,prefix,meta['training_games'] if meta else 0)
                if meta:
                    rng=torch.load(latest/'rng.pt',weights_only=False)
                    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
                    vec._seeds=[None]*vec.num_envs
                else:
                    model.set_random_seed(seed)
                assert model.policy.prior_bias==6 and model.gamma==.999 and model.gae_lambda==.99
                assert model.n_envs==32 and model.n_steps==640
                for key in ('n_epochs','batch_size','ent_coef','target_kl','max_grad_norm'):
                    assert getattr(model,key)==plan['ppo'][key],key
                assert model.lr_schedule(1.)==plan['ppo']['learning_rate'] and model.clip_range(1.)==plan['ppo']['clip_range']
                model.set_logger(configure(str(out/f's{seed}_metrics'),['csv']))
                seconds=meta['training_seconds'] if meta else 0.
                if not (out/'baseline_checkpoint.json').exists():
                    path=checkpoint(model,None,out,seed,prefix,0,0,0,baseline=True)
                    write(out/'baseline_checkpoint.json',dict(path=str(path)))
                    request(out,path,'monitor');request(out,path,'development')
                limit=initial_limit;extension_reviewed=old_decision is not None
                while model.num_timesteps-prefix<limit:
                    before=model.num_timesteps;t=time.perf_counter()
                    model.learn(plan['checkpoint_steps'],callback=progress,reset_num_timesteps=False)
                    seconds+=time.perf_counter()-t;added=model.num_timesteps-prefix
                    assert model.num_timesteps-before==plan['checkpoint_steps']
                    path=checkpoint(model,vec,out,seed,prefix,added,progress.games,seconds)
                    if added%plan['monitor_steps']==0:request(out,path,'monitor')
                    if added in (plan['minimum_steps'],plan['maximum_steps']):request(out,path,'development')
                    write(out/f's{seed}_result.json',dict(additional_steps=added,prefix_steps=prefix,
                        training_seconds=seconds,training_games=progress.games,checkpoint=str(path),candidate_preserved=True))
                    write(out/'training_progress.json',dict(seed=seed,additional_steps=added,training_games=progress.games,
                        training_seconds=seconds,pid=os.getpid(),stage='checkpoint_published'))
                    report(out)
                    if added==plan['minimum_steps'] and not plan['smoke'] and not extension_reviewed:
                        from experiments.rear_async.judgement import budget_decision
                        while True:
                            completed=[read(p) for p in (out/'evaluations').glob('*/completion.json')]
                            base=next((p['summary'] for p in completed if p['baseline'] and p['kind']=='monitor'),None)
                            points=[p for p in completed if p.get('seed')==seed and p['kind']=='monitor' and not p['baseline']]
                            broad=any(p.get('seed')==seed and p['additional_steps']==added and p['kind']=='development' for p in completed)
                            if base is not None and len(points)>=3 and broad:break
                            if (out/'STOP').exists():raise InterruptedError('STOP while awaiting budget evaluation')
                            blocked=list((out/'evaluations').glob('*/blocked.json'))
                            if blocked:raise RuntimeError('Required evaluation blocked;preserve checkpoint and fix evaluator before resume')
                            evaluation_pid=read(out/'processes.json')['evaluator']
                            if not psutil.pid_exists(evaluation_pid) or psutil.Process(evaluation_pid).status()==psutil.STATUS_ZOMBIE:
                                raise RuntimeError('Evaluator exited;preserve minimum-budget checkpoint before resume')
                            write(out/'training_waiting.json',dict(seed=seed,step=added,
                                reason='Minimum budget endpoint:preserving model,optimizer,RNG and live environments while required evaluation finishes'))
                            time.sleep(2)
                        decision=budget_decision(points,base,added,plan['minimum_steps'],plan['maximum_steps'])
                        write(out/f's{seed}_budget_decision.json',decision);extension_reviewed=True
                        if decision['extend']:limit=plan['maximum_steps']
                        if (out/'training_waiting.json').exists():(out/'training_waiting.json').unlink()
            except BaseException:
                if model is not None:model.save(out/f's{seed}_interrupted_learner.zip')
                raise
            finally:vec.close()
        write(out/'training_complete.json',dict(status='minimum_budget_complete',elapsed_seconds=time.perf_counter()-process_started,
            decision='Bounded rear-metric budget decisions recorded;all checkpoints preserved'))
    except BaseException:
        write(out/'training_failure.json',dict(error=traceback.format_exc()));raise

def evaluate_job(out,job):
    from experiments.rear_async.evaluate import cell,profile
    plan=read(out/'plan.json');check_plan(plan)
    source=Path(job['checkpoint']);m=verify(source)
    if m['config_id']!=job['config_id']:raise ValueError('Evaluation config mismatch')
    folder=out/'evaluations'/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():return
    copy=folder/'input'
    if not copy.exists():
        staging=folder/'.input.tmp'
        if staging.exists():shutil.rmtree(staging)
        shutil.copytree(source,staging);verify(staging);staging.rename(copy)
    verify(copy)
    opponents=plan['monitor_opponents'] if job['kind']=='monitor' else plan['development_opponents']
    n=plan['monitor_ics'] if job['kind']=='monitor' else plan['development_ics']
    band=plan['monitor_band'] if job['kind']=='monitor' else plan['development_band']
    replicas=[4900] if job['kind']=='monitor' else [4900,4901]
    started=time.perf_counter();rows=[];new=0
    for action_seed in replicas:
        own={**m['sampled_specs'][str(action_seed)],'design':str(copy/f'sampled_{action_seed}'),
            'weights':str(copy/f'sampled_{action_seed}/policy_net.zip')}
        for opponent in opponents:
            for i in range(n):
                for seat in ('red','blue'):
                    key=identity(dict(opponent=opponent['id'],seed=band+i,seat=seat,action_seed=action_seed))
                    file=folder/'cells'/f'{key}.json'
                    if file.exists():
                        row=read(file)
                        if (row['seed'],row['seat'],row['opponent'],row['action_seed'])!=(band+i,seat,opponent['id'],action_seed):
                            raise ValueError('Cached evaluation cell identity changed')
                    else:
                        if (out/'STOP').exists():raise InterruptedError('STOP marker')
                        row=dict(cell(own,opponent,band+i,seat),action_seed=action_seed)
                        write(file,row);new+=1
                    rows.append(row)
    verify(copy);summary=profile(rows)
    baseline=None
    for file in (out/'evaluations').glob('*/completion.json'):
        old=read(file)
        if old['kind']==job['kind'] and old['baseline']:baseline=old['summary'];break
    delta=None if baseline is None else summary['win_rate']-baseline['win_rate']
    # Preserve uncertainty: a small gain makes a candidate,not a confirmed winner.
    from experiments.rear_async.judgement import candidate_status
    entry=dict(checkpoint_id=m['checkpoint_id'],kind=job['kind'],additional_steps=m['additional_steps'],seed=m['seed'],
        baseline=m['baseline'],summary=summary,win_gain=delta,
        **candidate_status(delta),
        actual_new_games=len(rows),new_games_in_final_attempt=new,reused_cells_in_final_attempt=len(rows)-new,wall_seconds=time.perf_counter()-started,
        checkpoint=str(source),per_opponent={f['id']:profile([r for r in rows if r['opponent']==f['id']]) for f in opponents})
    write(folder/'completion.json',entry);write(out/'candidates'/f"{job['job_id']}.json",entry)
    report(out)

def evaluator(out):
    import psutil
    lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(10);write(out/'evaluation_runtime.json',dict(pid=os.getpid(),status='running',processes=1))
    while not (out/'STOP').exists():
        check_plan(read(out/'plan.json'))
        jobs=[read(p) for p in (out/'queue').glob('*.json')]
        jobs.sort(key=lambda j:(not verify(j['checkpoint'])['baseline'],-verify(j['checkpoint'])['additional_steps'],j['kind']!='monitor'))
        pending=[j for j in jobs if not (out/'evaluations'/j['job_id']/'completion.json').exists()
                 and not (out/'evaluations'/j['job_id']/'blocked.json').exists()]
        if not pending:
            if (out/'training_complete.json').exists() or (out/'training_failure.json').exists():break
            time.sleep(1);continue
        if psutil.virtual_memory().available<10*2**30:time.sleep(2);continue
        job=pending[0]
        try:evaluate_job(out,job)
        except BaseException:
            folder=out/'evaluations'/job['job_id'];old=read(folder/'failure.json') if (folder/'failure.json').exists() else {}
            failure=dict(error=traceback.format_exc(),attempts=old.get('attempts',0)+1)
            write(folder/'failure.json',failure)
            if failure['attempts']>=3:write(folder/'blocked.json',failure)
            time.sleep(3)
    report(out);write(out/'evaluation_complete.json',dict(status='stopped' if (out/'STOP').exists() else 'queue_drained'))

def report(out):
    plan=read(out/'plan.json');train=[read(p) for p in out.glob('s*_result.json')]
    ev=[read(p) for p in (out/'evaluations').glob('*/completion.json')]
    elapsed=time.time()-read(out/'started.json')['time']
    lines=['# Rear CPU asynchronous training','','All checkpoints remain candidates;small win gains are retained.','',
        '| Repeat | Additional steps | Training games | Training seconds |','|---|---:|---:|---:|']
    lines += [f"| {r['checkpoint'].split('/')[-1].split('_')[0]} | {r['additional_steps']} | {r['training_games']} | {r['training_seconds']:.2f} |" for r in train]
    lines += ['',f"Evaluation games: {sum(e['actual_new_games'] for e in ev)}. Evaluation process time: {sum(e['wall_seconds'] for e in ev):.2f}s. Overall elapsed: {elapsed:.2f}s. Training and evaluation times overlap.",
        '',f"CPU worker processes: {plan['processes']};environments32. NoGPUsimulation. Candidates retained: {len(ev)}."]
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

def launch(out,workers,smoke,resume):
    if not resume:freeze(out,workers,smoke);write(out/'started.json',dict(time=time.time(),pid=os.getpid()))
    else:check_plan(read(out/'plan.json'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    log=(out/'evaluator.log').open('a')
    child=subprocess.Popen([sys.executable,'-m','tools.rear_async_cpu','evaluate','--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(out/'processes.json',dict(trainer=os.getpid(),evaluator=child.pid))
    try:train(out,resume)
    finally:
        report(out)
        if (out/'training_failure.json').exists():child.terminate()
    child.wait();log.close();report(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','evaluate']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=a.out.resolve()
    if a.mode=='launch':launch(out,a.workers,a.smoke,a.resume)
    else:evaluator(out)
