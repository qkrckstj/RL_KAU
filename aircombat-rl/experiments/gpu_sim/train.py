"""Experimental GPU simulation + GPU PPO. Every run uses a new output directory.

This is a new training configuration, not an archived SB3 run or an official
evaluation. No policy-quality claim follows from the throughput smoke test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.distributions import Categorical

from .env import BatchedFairFight,BACKEND_COMMIT
from .geometry import features,ace_actions


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        def network(out):
            return nn.Sequential(nn.Linear(31,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,out))
        self.actor,self.critic=network(9),network(1)
        for net in (self.actor,self.critic):
            for layer in net:
                if isinstance(layer,nn.Linear):
                    nn.init.orthogonal_(layer.weight,2**.5)
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)

    def forward(self,x):
        return self.actor(x),self.critic(x).squeeze(-1)


def advantages(rewards,values,done,next_value,gamma=.999,lam=.95):
    result=torch.zeros_like(rewards)
    last=torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        keep=(~done[t]).to(rewards.dtype)
        delta=rewards[t]+gamma*next_value*keep-values[t]
        last=delta+gamma*lam*keep*last
        result[t]=last
        next_value=values[t]
    return result,result+values


def digest(model):
    return hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.parameters())).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--envs',type=int,default=1024)
    p.add_argument('--rollout',type=int,default=64)
    p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8192)
    p.add_argument('--seed',type=int,default=700)
    a=p.parse_args(argv)
    if min(a.envs,a.rollout,a.iterations,a.epochs,a.batch_size)<1 or a.envs*a.rollout<2:
        p.error('Positive sizes and at least two samples per rollout are required')
    a.out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError('This entrypoint requires CUDA simulation')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    config={**vars(a),'out':str(a.out),'backend_commit':BACKEND_COMMIT,
        'experimental':True,'gamma':.999,'gae_lambda':.95,'learning_rate':3e-4,
        'clip_range':.2,'entropy_coef':.01,'target_kl':.03,'opponent':'tensor Ace, closed vertical',
        'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    setup=time.perf_counter()
    env=BatchedFairFight(a.envs,seed=a.seed)
    env.capture()
    model=Agent().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    initial_hash=digest(model)
    row=torch.arange(a.envs,device='cuda')
    side=row%2
    shape=(a.rollout,a.envs)
    obs=torch.empty(*shape,31,device='cuda')
    actions=torch.empty(shape,device='cuda',dtype=torch.long)
    oldlog=torch.empty(shape,device='cuda')
    values=torch.empty(shape,device='cuda')
    rewards=torch.empty(shape,device='cuda')
    dones=torch.empty(shape,device='cuda',dtype=torch.bool)
    invalid=torch.zeros((),device='cuda',dtype=torch.bool)
    episodes=torch.zeros((),device='cuda',dtype=torch.int64)
    wins=torch.zeros_like(episodes)
    torch.cuda.synchronize()
    setup_seconds=time.perf_counter()-setup
    start=time.perf_counter()
    updates=0
    history=[]
    try:
        for iteration in range(a.iterations):
            with torch.no_grad():
                for t in range(a.rollout):
                    obs[t].copy_(features(env.obs[row,side]).float())
                    logits,value=model(obs[t])
                    dist=Categorical(logits=logits,validate_args=False)
                    action=dist.sample()
                    actions[t].copy_(action)
                    oldlog[t].copy_(dist.log_prob(action))
                    values[t].copy_(value)
                    both=ace_actions(env.raw)
                    both[row,side]=action
                    env.step(both)
                    won=((side == 0)&(env.outcome == 1))|((side == 1)&(env.outcome == -1))
                    rewards[t].copy_(torch.where(env.done,torch.where(won,1.,-.2),0.))
                    dones[t].copy_(env.done)
                    episodes.add_(env.done.sum())
                    wins.add_(won.sum())
                    invalid.logical_or_(env.invalid.any())
                    env.reset_done()
                _,next_value=model(features(env.obs[row,side]).float())
                adv,returns=advantages(rewards,values,dones,next_value)
            if bool(invalid): raise RuntimeError('Invalid simulator state encountered; training result rejected')
            flat=obs.flatten(0,1)
            b_actions,b_log=actions.flatten(),oldlog.flatten()
            b_adv,b_ret=adv.flatten(),returns.flatten()
            b_adv=(b_adv-b_adv.mean())/(b_adv.std(unbiased=False)+1e-8)
            for _ in range(a.epochs):
                order=torch.randperm(len(flat),device='cuda')
                stop=False
                for first in range(0,len(flat),a.batch_size):
                    idx=order[first:first+a.batch_size]
                    logits,value=model(flat[idx])
                    dist=Categorical(logits=logits,validate_args=False)
                    logratio=dist.log_prob(b_actions[idx])-b_log[idx]
                    ratio=logratio.exp()
                    kl=((ratio-1)-logratio).mean()
                    if float(kl.detach()) > 1.5*.03:
                        stop=True
                        break
                    pg=torch.maximum(-b_adv[idx]*ratio,-b_adv[idx]*ratio.clamp(.8,1.2)).mean()
                    loss=pg+.5*(value-b_ret[idx]).square().mean()-.01*dist.entropy().mean()
                    if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite PPO loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
                    optimizer.step()
                    updates+=1
                if stop: break
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            record=dict(iteration=iteration+1,steps=(iteration+1)*a.envs*a.rollout,
                        seconds=elapsed,optimizer_steps=updates,episodes=int(episodes),wins=int(wins))
            history.append(record)
            (a.out/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record),flush=True)
        final_hash=digest(model)
        if updates == 0 or initial_hash == final_hash or not all(bool(torch.isfinite(x).all()) for x in model.parameters()):
            raise RuntimeError('Finite changed GPU weights were not verified')
        metrics={**history[-1],'setup_seconds':setup_seconds,'steps_per_second':history[-1]['steps']/history[-1]['seconds'],
            'invalid_episodes':int(invalid),'weights_changed':initial_hash != final_hash,
            'initial_weights_sha256':initial_hash,'final_weights_sha256':final_hash,
            'all_parameters_cuda':all(x.device.type == 'cuda' for x in model.parameters()),
            'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
            metrics=metrics,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),a.out/'policy.pt')
        (a.out/'results.json').write_text(json.dumps(dict(metrics=metrics,history=history),indent=2)+'\n')
        return metrics
    except Exception as exc:
        (a.out/'failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n')
        raise


if __name__ == '__main__': main()
