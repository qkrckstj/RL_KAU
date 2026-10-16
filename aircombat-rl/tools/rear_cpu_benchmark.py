"""Matched native-JSBSim PPO worker benchmark with32fixed environments."""
import argparse,json,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def trial(processes,repeat,out):
    from functools import partial
    import hashlib
    import psutil
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from experiments.league.tail_actual_reward import make_worker
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'experiments/league/rear_desktop_20261012/inputs.json').read_text())
    plan.update(seed=42000+repeat,ic_band=1700200000+repeat*1000000)
    class Count(BaseCallback):
        def __init__(self):super().__init__();self.games=0
        def _on_step(self):
            self.games+=sum('episode' in i for i in self.locals['infos'])
            if self.num_timesteps%2048==0 and psutil.virtual_memory().available<8*2**30:
                raise MemoryError('Host available below8GiB')
            return True
    started=time.perf_counter();vec=None
    try:
        vec=GroupedSubprocVecEnv([partial(make_worker,plan,i) for i in range(32)],processes)
        model=PPO.load(ROOT/plan['parent'],env=vec,device='cpu',seed=plan['seed'],n_steps=640,force_reset=True)
        assert model.policy.prior_bias==6 and model.gamma==.999 and model.gae_lambda==.99
        model.num_timesteps=0;model.set_random_seed(plan['seed'])
        setup=time.perf_counter()-started;count=Count();t=time.perf_counter()
        model.learn(20480,callback=count,reset_num_timesteps=False)
        warm=time.perf_counter()-t;t=time.perf_counter();games=count.games
        model.learn(20480,callback=count,reset_num_timesteps=False)
        seconds=time.perf_counter()-t
        digest=hashlib.sha256()
        for value in model.policy.state_dict().values():digest.update(value.detach().cpu().numpy().tobytes())
        result=dict(processes=processes,environments=32,repeat=repeat,setup_seconds=setup,
            warmup_steps=20480,warmup_seconds=warm,measured_steps=20480,measured_seconds=seconds,
            measured_steps_per_second=20480/seconds,training_games=count.games,measured_games=count.games-games,
            total_training_steps=40960,final_policy_sha256=digest.hexdigest(),device='cpu',status='complete')
        out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:
        if vec is not None:vec.close()

def run(out):
    import psutil,signal
    out.mkdir(parents=True,exist_ok=False);results=[]
    # Reverse order in the second pass to reduce order/thermal bias.
    for repeat,order in [(0,[2,4,8,16]),(1,[16,8,4,2])]:
        for processes in order:
            name=f'p{processes}_r{repeat}';file=out/f'{name}.json'
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            minimum=psutil.virtual_memory().available/2**30;peak=0;stopped=None;t=time.perf_counter()
            with (out/f'{name}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'-m','tools.rear_cpu_benchmark','--trial',str(processes),
                    '--repeat',str(repeat),'--out',str(file)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    minimum=min(minimum,psutil.virtual_memory().available/2**30)
                    try:
                        p=psutil.Process(child.pid);peak=max(peak,sum(q.memory_info().rss for q in [p,*p.children(recursive=True)] if q.is_running())/2**30)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    if minimum<8:
                        stopped='host_memory_floor';os.killpg(child.pid,signal.SIGTERM);break
                    time.sleep(.5)
                child.wait(timeout=30)
            row=json.loads(file.read_text()) if file.exists() else dict(processes=processes,repeat=repeat,status='failed')
            row.update(process_wall_seconds=time.perf_counter()-t,minimum_host_available_gib=minimum,
                peak_process_tree_rss_gib=peak,exit_code=child.returncode,stopped=stopped)
            results.append(row);(out/'progress.json').write_text(json.dumps(dict(results=results),indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if stopped:break
    completed=[r for r in results if r['status']=='complete' and r['exit_code']==0]
    summaries=[]
    for p in (2,4,8,16):
        rows=[r for r in completed if r['processes']==p]
        if len(rows)!=2:continue
        summaries.append(dict(processes=p,median_steps_per_second=statistics.median(r['measured_steps_per_second'] for r in rows),
            peak_rss_gib=max(r['peak_process_tree_rss_gib'] for r in rows),minimum_host_available_gib=min(r['minimum_host_available_gib'] for r in rows)))
    if not summaries:raise RuntimeError('No two-pass configuration completed')
    best=max(summaries,key=lambda r:r['median_steps_per_second'])
    # Use fewer workers when speeds are within3%, reserving CPU capacity for async evaluation.
    choice=min((r for r in summaries if r['median_steps_per_second']>=.97*best['median_steps_per_second']),key=lambda r:r['processes'])
    result=dict(status='complete',trials=results,summaries=summaries,fastest=best,selected=choice,
        new_training_steps=sum(r.get('total_training_steps',0) for r in completed),evaluation_games=0,
        scope='32environments,identical parent/ppo/roster and matched seed perpass;warmup excluded fromsteady throughput. Best amongtested2/4/8/16,not universal optimum.')
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--trial',type=int);p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
    if a.trial:trial(a.trial,a.repeat,a.out)
    else:run(a.out)
