"""Queue a short, controlled PPO timing study after the existing experiment chain.

ABBA trials retain the same four environments, sample budget and starting
checkpoint. Instrumentation must preserve trajectory and parameter digests.
No speed change or policy promotion is made by this diagnostic.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime,timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tools import league_thread_benchmark as io


class Timings:
    def __init__(self):self.rows={}

    def call(self,label,function,*args,**kwargs):
        start=time.perf_counter()
        try:return function(*args,**kwargs)
        finally:
            row=self.rows.setdefault(label,dict(calls=0,wall_seconds=0.))
            row['calls']+=1;row['wall_seconds']+=time.perf_counter()-start

    @contextmanager
    def instrument(self,model,vec):
        restore=[]
        try:
            for obj,name,label in ((model,'collect_rollouts','rollout_inclusive'),
                    (model,'train','optimizer_inclusive'),(model.policy,'forward','action_value_forward'),
                    (vec,'step_async','dispatch_actions'),(vec,'step_wait','wait_receive_stack')):
                original=getattr(obj,name);had_own=name in vars(obj)
                replacement=partial(self.call,label,original)
                restore.append((obj,name,original,had_own));setattr(obj,name,replacement)
            yield self
        finally:
            for obj,name,original,had_own in reversed(restore):
                if had_own:setattr(obj,name,original)
                else:delattr(obj,name)


def tensor_digest(state):
    """Stable state_dict digest independent of PyTorch ZIP serialization."""
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def make_profile_worker(plan,index,instrumented):
    # Heavy imports happen only after the predecessor has exited or in workers.
    import gymnasium as gym
    from experiments.league.residual_worker import make_worker
    from tools.league_matches import Actor
    worker=make_worker(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',plan['ppo']['gamma'])
    archive=worker.unwrapped
    # Avoid measuring an unrepresentatively empty early-run opponent cache.
    for spec in plan['opponents']:
        if spec['id'] not in archive.cache:archive.cache[spec['id']]=Actor(spec)

    class Measured(gym.Wrapper):
        def __init__(self,env):
            super().__init__(env);self.timings=Timings();self.step_cpu=0.
        def step(self,action):
            if not instrumented:return self.env.step(action)
            started=time.process_time()
            try:return self.timings.call('env_step_inclusive',self.env.step,action)
            finally:self.step_cpu+=time.process_time()-started
        def reset(self,**kwargs):
            if not instrumented:return self.env.reset(**kwargs)
            return self.timings.call('env_reset',self.env.reset,**kwargs)
        def profile_status(self):
            return dict(worker_index=index,timings=self.timings.rows,env_step_cpu_seconds=self.step_cpu,
                cached_opponents=len(archive.cache),memory=io.own_memory())
    return Measured(worker)


def prepare(source,after,pid,out):
    from tools.league_residual_action_mode import process_creation
    if (out/'queue_plan.json').exists():raise FileExistsError('Use a fresh profile folder')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits before imports')
    old=io.read(source/'plan.json');io.verify(old)
    predecessor=io.read(after/'queue_plan.json');io.verify(predecessor)
    if io.read(after/'queue_runtime.json')['pid']!=pid:raise ValueError('Wrong predecessor PID')
    checkpoint=source/'s3300/checkpoints/step_2097152/learner.zip'
    checkpoint_record=source/'s3300/checkpoints/step_2097152/checkpoint.json'
    if io.sha(checkpoint)!=io.read(checkpoint_record)['learner_sha256']:raise ValueError('Changed checkpoint')
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for name in ('tools/league_residual_profile.py','tests/test_league_residual_profile.py',
                 'tools/league_residual_action_mode.py','tools/league_residual_repair_followup.py'):
        sources[name]=io.sha(io.ROOT/name)
    for path in (source/'plan.json',after/'queue_plan.json',after/'queue_runtime.json',checkpoint,checkpoint_record):
        inputs[io.relative(path)]=io.sha(path)
    plan=dict(source=io.relative(source),after_run=io.relative(after),after_pid=pid,
        created_filetime=process_creation(pid),source_sha256=sources,input_sha256=inputs,
        environment=io.environment(),checkpoint=io.relative(checkpoint),checkpoint_steps=2097152,
        teacher=old['teacher'],opponents=old['opponents'],probabilities=old['probabilities'],ppo=old['ppo'],
        seed=4300,workers=4,ic_band=170000000,n_steps=1024,steps_per_trial=65536,
        trials=[dict(id='unmeasured_a',instrumented=False),dict(id='measured_b',instrumented=True),
                dict(id='measured_c',instrumented=True),dict(id='unmeasured_d',instrumented=False)],
        scope='Execution timing only, repeated same RNG/checkpoint and compatibility IC band, not independent quality evidence. All72 opponent policies cached before timing. Four4-worker trials of65536 steps (262144 diagnostic steps total). No checkpoint selection, promotion, backend change or GitHub publication.',
        analysis_rule='Compare all four learning-trajectory and final-parameter digests, report ABBA medians and instrumentation overhead. All trials include the same trajectory-hash callback; its measured cost is separate, so unmeasured means without section timers, not untouched production throughput. Timers are nested: rollout contains forward/dispatch/wait; worker times overlap parent waits and each other. Wait includes remaining simulation, IPC, deserialization and stacking; do not call it pure IPC.')
    io.verify(plan);io.write(out/'queue_plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(io.ROOT/name,target)
    return plan


def trial(plan,setting,out):
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from tools import league_residual_train as base
    from tools.league_residual_recover import load_continuation
    torch.set_num_threads(1)
    folder=out/setting['id'];folder.mkdir()
    before=io.resources()
    if min(before['commit_headroom_gib'],before['available_memory_gib'])<4.5:
        raise MemoryError('Insufficient trial startup headroom')
    started=time.perf_counter()
    vec=SubprocVecEnv([partial(make_profile_worker,plan,i,setting['instrumented']) for i in range(plan['workers'])],start_method='spawn')
    try:
        statuses=vec.env_method('profile_status')
        if any(s['cached_opponents']!=len(plan['opponents']) for s in statuses):raise AssertionError('Opponent cache incomplete')
        if io.resources()['commit_headroom_gib']<2.8:raise MemoryError('Insufficient initialized profile headroom')
        model,prefix=load_continuation(io.ROOT/plan['checkpoint'],vec,plan['seed'],plan['n_steps'])
        if prefix!=plan['checkpoint_steps']:raise AssertionError('Wrong profiling checkpoint')
        initial_digest=tensor_digest(model.policy.state_dict())
        model.set_logger(configure(str(folder/'metrics'),['csv']))
        class Trace(base.Progress):
            def __init__(self):
                super().__init__(folder,plan['seed']);self.trace=hashlib.sha256();self.hash_seconds=0.
            def _on_step(self):
                hash_started=time.perf_counter()
                for name in ('actions','new_obs','rewards','dones'):
                    value=np.ascontiguousarray(self.locals[name]);self.trace.update(value.tobytes())
                for info in self.locals['infos']:
                    keys=('outcome','won','own_health','opp_health','t','official_episode_flags','league_episode')
                    self.trace.update(json.dumps({k:info[k] for k in keys if k in info},sort_keys=True).encode())
                self.hash_seconds+=time.perf_counter()-hash_started
                return super()._on_step()
        callback=Trace();timings=Timings();startup=time.perf_counter()-started
        with io.Sampler() as sampler:
            started=time.perf_counter();cpu_started=time.process_time()
            if setting['instrumented']:
                with timings.instrument(model,vec):
                    model.learn(plan['steps_per_trial'],callback=callback,reset_num_timesteps=False)
            else:model.learn(plan['steps_per_trial'],callback=callback,reset_num_timesteps=False)
            learning_seconds=time.perf_counter()-started;parent_cpu_seconds=time.process_time()-cpu_started
        statuses=vec.env_method('profile_status')
        result=dict(setting=setting,steps=model.num_timesteps,startup_seconds=startup,learning_seconds=learning_seconds,
            steps_per_second=model.num_timesteps/learning_seconds,initial_parameters_sha256=initial_digest,
            final_parameters_sha256=tensor_digest(model.policy.state_dict()),trajectory_sha256=callback.trace.hexdigest(),
            parent_timings=timings.rows,parent_cpu_seconds=parent_cpu_seconds,
            trajectory_hash_seconds=callback.hash_seconds,worker_status=statuses,resources=sampler.result,
            completed_episodes=callback.completed,training_outcomes=callback.counts)
        io.write(folder/'result.json',result)
        return result
    finally:vec.close()


def analyze(results,steps_per_trial):
    from statistics import median
    if len(results)!=4 or any(r['steps']!=steps_per_trial for r in results):raise ValueError('Incomplete ABBA trials')
    same=all(len({r[key] for r in results})==1 for key in
             ('initial_parameters_sha256','final_parameters_sha256','trajectory_sha256'))
    if not same:raise AssertionError('Instrumentation changed the compared learning run')
    plain=median(r['learning_seconds'] for r in results if not r['setting']['instrumented'])
    measured=median(r['learning_seconds'] for r in results if r['setting']['instrumented'])
    details=[]
    for r in results:
        if not r['setting']['instrumented']:continue
        times=r['parent_timings'];rollout=times['rollout_inclusive']['wall_seconds'];learn=r['learning_seconds']
        nested=sum(times[k]['wall_seconds'] for k in ('action_value_forward','dispatch_actions','wait_receive_stack'))
        if nested>rollout+1e-6:raise AssertionError('Invalid nested timing accounting')
        details.append(dict(trial=r['setting']['id'],parent_timings=times,
            parent_cpu_seconds=r['parent_cpu_seconds'],trajectory_hash_seconds=r['trajectory_hash_seconds'],
            shares_of_learning={k:v['wall_seconds']/learn for k,v in times.items()},
            rollout_other_seconds=rollout-nested,
            worker_step_wall_sum_seconds=sum(s['timings']['env_step_inclusive']['wall_seconds'] for s in r['worker_status']),
            worker_step_cpu_sum_seconds=sum(s['env_step_cpu_seconds'] for s in r['worker_status'])))
    return dict(trajectories_and_final_parameters_equal=True,unmeasured_median_seconds=plain,
        measured_median_seconds=measured,instrumentation_relative_time=(measured/plain)-1,
        unmeasured_steps_per_second=steps_per_trial/plain,measured_breakdowns=details,
        backend_changed=False,promotion=False,
        scope='All trials contain identical trajectory-hashing work; its time is separate. Unmeasured means no section timers, not untouched production throughput. Timer overhead and run-to-run noise are reported. Nested/overlapping times are not additive CPU utilization. Same initial state and replay conditions; no quality or acceleration claim.')


def execute(plan,out):
    predecessor=io.ROOT/plan['after_run']
    if (predecessor/'failure.json').exists() or not (predecessor/'completion.json').exists():raise ValueError('No clean diagnostic predecessor')
    if io.read(predecessor/'completion.json')['status']!='diagnostic_complete':raise ValueError('Unexpected predecessor status')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),workers=4))
    results=[]
    for setting in plan['trials']:
        io.verify(plan);results.append(trial(plan,setting,out))
    report=analyze(results,plan['steps_per_trial']);io.write(out/'analysis.json',report);io.verify(plan)
    io.write(out/'completion.json',dict(status='profiling_complete',diagnostic_training_steps=sum(r['steps'] for r in results),
        experiments_counted_as_independent=0,promotion=False,backend_changed=False,
        completed_at=datetime.now(timezone.utc).isoformat(),scope=plan['scope']))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--after-run',type=Path,required=True)
    parser.add_argument('--after-pid',type=int,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        from tools.league_residual_repair_followup import wait_for_source
        out=args.out.resolve();plan=prepare(args.source.resolve(),args.after_run.resolve(),args.after_pid,out)
        wait_for_source(plan,out);execute(plan,out)
    except BaseException:
        io.write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
