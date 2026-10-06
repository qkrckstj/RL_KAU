"""Warm-start PPO from a preserved Double DQN's greedy policy, without using Q as returns.

Pilot uses development validation only; no held-out test is opened here.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import pickle
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from tools.plan_a import ROOT,write_json,timestamp,evaluate,export_design
from tools.plan_a_shaping import read,sha
from experiments.plan_a.core import specification,make_learner


def transfer_actor(teacher,student,observations):
    """Copy equal-width ReLU layers; choose logit temperature on training replay only."""
    x=torch.as_tensor(observations,dtype=torch.float32)
    with torch.no_grad():
        q=teacher.q_net(x)
        centered=q-q.max(dim=1,keepdim=True).values
        chosen=10000.
        for scale in (100.,300.,1000.,3000.,10000.):
            if centered.mul(scale).softmax(dim=1).max(dim=1).values.mean()>=.95:
                chosen=scale
                break
        for index in (0,2):
            student.policy.mlp_extractor.policy_net[index].load_state_dict(teacher.q_net.q_net[index].state_dict())
        student.policy.action_net.weight.copy_(teacher.q_net.q_net[4].weight*chosen)
        student.policy.action_net.bias.copy_(teacher.q_net.q_net[4].bias*chosen)
        distribution=student.policy.get_distribution(x).distribution
        agreement=float((distribution.logits.argmax(dim=1)==q.argmax(dim=1)).float().mean())
        probability=float(distribution.probs.max(dim=1).values.mean())
    if agreement<.999:
        raise ValueError("Actor transfer did not preserve greedy actions")
    return dict(logit_scale=chosen,training_states=len(observations),greedy_agreement=agreement,
                mean_top_action_probability=probability,critic="fresh PPO value network; DQN Q is not a target")


def worker(out,seed):
    p=read(out/"plan.json")
    for name,expected in p["source_sha256"].items():
        if sha(ROOT/name)!=expected:raise ValueError(f"Source changed: {name}")
    torch.set_num_threads(1)
    d=out/f"s{seed}"
    d.mkdir()
    source=ROOT/p["teacher"]
    teacher=DQN.load(source/"policy_net.zip",device="cpu")
    if sha(source/"policy_net.zip")!=p["teacher_sha256"]:raise ValueError("Teacher changed")
    with (source/"replay_buffer.pkl").open("rb") as f:replay=pickle.load(f)
    idx=np.random.default_rng(1729).choice(replay.size(),min(4096,replay.size()),replace=False)
    spec=dict(specification("observed_ppo"),learning_rate=p["learning_rate"],n_steps=p["n_steps"],
        batch_size=p["batch_size"],n_epochs=p["n_epochs"],gae_lambda=p["gae_lambda"],
        ent_coef=p["ent_coef"],target_kl=p["target_kl"])
    model=make_learner(spec,seed)
    audit=transfer_actor(teacher,model,replay.observations[idx,0])
    del teacher,replay
    export_design(d,spec,p["steps"])
    write_json(d/"config.json",dict(spec=spec,seed=seed,steps=p["steps"],transfer=audit,
        source_sha256=sha(d/"utils.py"),teacher_sha256=p["teacher_sha256"],device="cpu"))
    model.set_logger(configure(str(d/"metrics"),["csv"]))
    start=time.perf_counter()
    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history=[];self.best=(-1,-1e9);self.best_step=None;self.next_eval=p["val_every"]
            self.next_log=10000;self.wins=0;self.episodes=0
        def status(self,stage):
            write_json(d/"progress.json",dict(stage=stage,steps=model.num_timesteps,target_steps=p["steps"],
                train_wins=self.wins,train_episodes=self.episodes,best_validation_wins=self.best[0],
                elapsed_seconds=time.perf_counter()-start,updated_utc=timestamp()))
        def validate(self):
            self.status("validation")
            result=evaluate(model,spec,p["val_band"],p["val_n"])
            result.update(step=model.num_timesteps,train_wins=self.wins,train_episodes=self.episodes)
            self.history.append(result)
            v=result["summary"]
            rank=(v["kills"],-(v["t_kill"] if v["t_kill"] is not None else 1e9))
            if rank>self.best:
                self.best,self.best_step=rank,model.num_timesteps
                model.save(d/"policy_net.zip")
            ck=d/f"checkpoints/step_{model.num_timesteps}"
            ck.mkdir(parents=True,exist_ok=True)
            model.save(ck/"policy_net.zip")
            write_json(d/"validation.json",self.history)
            self.status("training")
        def _on_training_start(self):self.validate()
        def _on_rollout_start(self):
            if model.num_timesteps>=self.next_eval:
                self.validate();self.next_eval+=p["val_every"]
        def _on_step(self):
            for done,info in zip(self.locals.get("dones",[]),self.locals.get("infos",[])):
                if done:self.episodes+=1;self.wins+=int(bool(info.get("won",False)))
            if model.num_timesteps>=self.next_log:self.status("training");self.next_log+=10000
            return True
    cb=Progress()
    try:
        model.learn(p["steps"],callback=cb)
        if cb.history[-1]["step"]!=model.num_timesteps:cb.validate()
        model.save(d/"final_net.zip")
        write_json(d/"result.json",dict(status="complete",steps=model.num_timesteps,best_step=cb.best_step,
            best_validation_wins=cb.best[0],final_validation=cb.history[-1]["summary"],
            train_wins=cb.wins,train_episodes=cb.episodes))
        cb.status("complete")
    finally:model.get_env().close()


def child(out,seed):
    with (out/f"s{seed}.log").open("w",encoding="utf-8") as f:
        proc=subprocess.run([sys.executable,"-X","utf8","-m","tools.autolab_ppo","--out",str(out),"--worker",str(seed)],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if proc.returncode:raise RuntimeError(f"PPO worker failed: seed {seed}")


def run(out,smoke):
    if out.exists():raise FileExistsError("Use a new experiment directory")
    out.mkdir(parents=True)
    teacher="runs/plan_a_20261005/double/ddqn/s1/checkpoints/step_1024000"
    paths=["tools/autolab_ppo.py","experiments/plan_a/core.py","tools/plan_a.py"]
    p=dict(created_utc=timestamp(),hypothesis="A policy-gradient update with a trusted initial actor may retain useful behavior better than greedy Q rankings",
        seeds=[700] if smoke else [700,701,702],steps=64 if smoke else 131072,
        val_every=32 if smoke else 32768,val_band=900000 if smoke else 902000,val_n=2 if smoke else 20,
        learning_rate=.0001,n_steps=16 if smoke else 2048,batch_size=8 if smoke else 256,n_epochs=2 if smoke else 5,
        gae_lambda=1.,ent_coef=0.,target_kl=.01,teacher=teacher,teacher_sha256=sha(ROOT/teacher/"policy_net.zip"),
        source_sha256={n:sha(ROOT/n) for n in paths},heldout_test="Not opened during screening",
        scope="Independent fine-tuning RNG seeds share one pretrained teacher; not independent teacher pretraining")
    write_json(out/"plan.json",p)
    for n in paths:
        dest=out/"source_snapshot"/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda s:child(out,s),p["seeds"]))
        from tools.grade import play,summarise
        from aircombat_gym.wvr.envs.fair import FairFightEnv
        episodes=[]
        for seat in ("red","blue"):
            env=FairFightEnv(action_mode="discrete",seat=seat)
            try:
                rows=play(env,lambda obs:0,p["val_band"],p["val_n"]//2,seat)
                episodes+=rows
            finally:env.close()
        write_json(out/"summary.json",dict(records=[read(out/f"s{s}/result.json") for s in p["seeds"]],
            fixed_baseline=summarise(episodes),stage="development screening only"))
        write_json(out/"completion.json",dict(status="complete",finished_utc=timestamp()))
    except Exception as e:
        write_json(out/"completion.json",dict(status="failed",error=str(e),updated_utc=timestamp()));raise


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path);p.add_argument("--worker",type=int);p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.worker is not None:worker(a.out.resolve(),a.worker)
    else:run(a.out.resolve(),a.smoke)
