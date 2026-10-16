"""Frozen sequential PPO tracking study, preserving every candidate."""
import argparse
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from experiments.controller_study.tracking import TrackingEnv,VARIANTS,make_env
from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
from experiments.rear_async.storage import read,write,sha

ROOT=Path(__file__).resolve().parents[2]


def check(plan):
    for name,digest in plan['inputs'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Study input changed: '+name)


def prepare(out):
    if (out/'learning_plan.json').exists():raise FileExistsError('Learning already prepared')
    screen=read(out/'response/results.json')
    files=list((ROOT/'aircombat_gym').rglob('*.py'))+list((ROOT/'aircombat_gym').rglob('*.xml'))
    files+=list((ROOT/'experiments/controller_study').glob('*.py'))
    files += [ROOT/'experiments/controller_probe/learnability.py',ROOT/'experiments/rear_async/storage.py',
        ROOT/'experiments/league/grouped_vec_env.py',ROOT/'tests/test_controller_study.py',out/'response/results.json']
    plan=dict(variants=list(VARIANTS),seeds=[59100,59101],steps_per_trial=393216,
        checkpoints=[0,131072,262144,393216],eval_seeds=list(range(1814100000,1814100024)),
        workers=8,environments=32,stabilizer=screen['selected_settings'],selected_stabilizer=screen['selected'],
        ppo=dict(learning_rate=.0003,n_steps=256,batch_size=1024,n_epochs=4,gamma=.99,gae_lambda=.95,
                 clip_range=.2,ent_coef=.01,vf_coef=.5,max_grad_norm=.5),
        architecture=[64,64],device='cpu',initial_extra_columns='zero actor and critic columns10:19',
        scope='Fresh PPO single-aircraft heading tracking, not combat; paired development task seeds; no original model changes.',
        inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))})
    write(out/'learning_plan.json',plan);return plan


class Episodes(BaseCallback):
    def __init__(self,folder):
        super().__init__();self.folder=folder;self.count=0;self.started=time.perf_counter()
        self.stream=(folder/'training_episodes.jsonl').open('x')

    def _on_step(self):
        for i,info in enumerate(self.locals['infos']):
            if 'tracking' in info:
                self.count+=1
                self.stream.write(json.dumps(dict(info['tracking'],transition=self.num_timesteps,env_index=i))+'\n')
        if self.num_timesteps%8192==0:
            self.stream.flush()
            write(self.folder/'progress.json',dict(steps=self.num_timesteps,training_episodes=self.count,
                elapsed_seconds=time.perf_counter()-self.started,pid=os.getpid()))
        return True

    def close(self):self.stream.close()


def evaluate(model,variant,settings,seeds,folder,steps):
    start=time.perf_counter();envs=[TrackingEnv(variant,settings) for _ in seeds]
    try:
        observations=np.stack([e.reset(seed=s)[0] for e,s in zip(envs,seeds)])
        rows=[]
        for _ in range(600):
            actions,_=model.predict(observations,deterministic=True)
            observations_new=[]
            for env,action,seed in zip(envs,actions,seeds):
                ob,_,term,trunc,info=env.step(action)
                observations_new.append(ob)
                if term:raise RuntimeError('Unexpected terminal in tracking evaluation')
                if trunc:rows.append(dict(info['tracking'],seed=seed))
            observations=np.stack(observations_new)
        if len(rows)!=len(seeds):raise RuntimeError('Incomplete task panel')
        summary={k:float(np.mean([r[k] for r in rows])) for k in (
            'rmse_deg','mae_deg','within5_fraction','max_altitude_error_ft','mean_speed_error_kt','episode_return')}
        result=dict(steps=steps,variant=variant,episodes=len(rows),rows=rows,summary=summary,
            wall_seconds=time.perf_counter()-start,mode='deterministic',scope='paired development tracking tasks')
        write(folder/f'eval_{steps:09d}.json',result)
        return result
    finally:
        for env in envs:env.close()


def parameter_hash(policy):
    h=hashlib.sha256()
    for k,v in policy.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def trial(out,plan,variant,seed):
    folder=out/'trials'/f'{variant}_s{seed}'
    if (folder/'completion.json').exists():return read(folder/'completion.json')
    if folder.exists():raise RuntimeError('Interrupted trial retained; use a new namespace or explicit checkpoint resume')
    check(plan);folder.mkdir(parents=True);start=time.perf_counter()
    vec=GroupedSubprocVecEnv([partial(make_env,variant,plan['stabilizer']) for _ in range(plan['environments'])],plan['workers'])
    callback=Episodes(folder)
    try:
        model=PPO('MlpPolicy',vec,seed=seed,device='cpu',verbose=0,**plan['ppo'],
            policy_kwargs=dict(net_arch=dict(pi=plan['architecture'],vf=plan['architecture'])))
        with torch.no_grad():
            model.policy.mlp_extractor.policy_net[0].weight[:,10:]=0.
            model.policy.mlp_extractor.value_net[0].weight[:,10:]=0.
        model.set_logger(configure(str(folder),['csv']))
        write(folder/'initial.json',dict(seed=seed,variant=variant,parameter_sha256=parameter_hash(model.policy),
            extra_columns_zero=True,observation_dimension=19,actions=vec.action_space.n,
            setup_seconds=time.perf_counter()-start))
        model.save(folder/'checkpoint_000000000')
        evaluation=[evaluate(model,variant,plan['stabilizer'],plan['eval_seeds'],folder,0)]
        pure_train=0.
        for goal in plan['checkpoints'][1:]:
            learn_start=time.perf_counter()
            model.learn(total_timesteps=goal-model.num_timesteps,reset_num_timesteps=False,callback=callback)
            pure_train+=time.perf_counter()-learn_start
            if model.num_timesteps!=goal:raise RuntimeError('Unexpected learning transition count')
            model.save(folder/f'checkpoint_{goal:09d}')
            evaluation.append(evaluate(model,variant,plan['stabilizer'],plan['eval_seeds'],folder,goal))
            print(variant,seed,goal,'RMSE',round(evaluation[-1]['summary']['rmse_deg'],3),
                  'within5',round(evaluation[-1]['summary']['within5_fraction'],3),flush=True)
        check(plan)
        result=dict(status='complete',variant=variant,seed=seed,training_steps=model.num_timesteps,
            training_episodes=callback.count,evaluation_episodes=sum(e['episodes'] for e in evaluation),
            training_seconds=pure_train,evaluation_seconds=sum(e['wall_seconds'] for e in evaluation),
            elapsed_seconds=time.perf_counter()-start,evaluations=evaluation,
            final_parameter_sha256=parameter_hash(model.policy),model_retained=True)
        write(folder/'completion.json',result)
        return result
    finally:callback.close();vec.close()


def run(out):
    import psutil
    torch.set_num_threads(1)
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB free')
    for proc in psutil.process_iter(['cmdline']):
        args=proc.info['cmdline'] or []
        if any(Path(a).name=='aim_cpu_runner.py' for a in args) and any(a in args for a in ('launch','evaluate')):
            raise RuntimeError('Existing native automation is active')
    plan=read(out/'learning_plan.json') if (out/'learning_plan.json').exists() else prepare(out)
    with (out/'learning.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'learning_complete.json').exists():raise RuntimeError('Study already complete')
        write(out/'runtime.json',dict(pid=os.getpid(),status='learning',started=time.time()))
        start=time.perf_counter();completed=[]
        for variant in plan['variants']:
            for seed in plan['seeds']:
                write(out/'current.json',dict(variant=variant,seed=seed,completed_trials=len(completed),total_trials=12))
                completed.append(trial(out,plan,variant,seed))
        check(plan)
        write(out/'learning_complete.json',dict(status='complete',trials=completed,
            training_steps=sum(r['training_steps'] for r in completed),
            training_episodes=sum(r['training_episodes'] for r in completed),
            evaluation_episodes=sum(r['evaluation_episodes'] for r in completed),
            training_seconds=sum(r['training_seconds'] for r in completed),
            evaluation_seconds=sum(r['evaluation_seconds'] for r in completed),
            elapsed_seconds=time.perf_counter()-start,combat_games=0,automatic_installation=False))
        write(out/'runtime.json',dict(pid=os.getpid(),status='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
