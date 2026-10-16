"""Public-threat prior and complete fixed reference; historical runners preserved."""
import argparse
import collections
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.rear_async.storage import read, write, sha, identity, publish, verify
from experiments.league.health_rule_v2 import classify
from tools.seven_model_gamma import configure_gamma
from tools.seven_model_rollout import configure_rollout

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('opponent', 'seed', 'seat', 'action_seed')


def expected(p):
    axes = [p['opponents'], p['seeds'], p['seats'], p['action_seeds']]
    if any(not x or len(set(x)) != len(x) for x in axes):
        raise ValueError('Nonempty unique panel axes required')
    if set(p['seats']) != {'red', 'blue'} or p['action_seeds'] != [4900, 4901]:
        raise ValueError('Both seats and both paired action replicas required')
    return {(o,s,t,a) for o in axes[0] for s in axes[1] for t in axes[2] for a in axes[3]}


def score_panel(rows, p):
    keys = [tuple(r[k] for k in FIELDS) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected(p):
        raise ValueError('Missing, duplicate or foreign cells')
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('own_health','opp_health')):
            raise ValueError('Nonfinite health')
        if r['result'] != classify(r['own_health'], r['opp_health']):
            raise ValueError('Incorrect individual verdict')
        counts[r['opponent']][r['result']] += 1
    per = {o:dict(wins=c['win'],draws=c['draw'],losses=c['loss'],
                 verdict='win' if not c['loss'] and c['win'] else 'loss') for o,c in sorted(counts.items())}
    wins = sum(c['verdict']=='win' for c in per.values())
    return dict(games=len(rows),opponent_wins=wins,opponent_losses=len(per)-wins,all_pass=wins==len(per),
                game_wins=sum(c['wins'] for c in per.values()),game_draws=sum(c['draws'] for c in per.values()),
                game_losses=sum(c['losses'] for c in per.values()),per_opponent=per)


def admission_allowed(screen, confirm):
    s,c = screen['protocol'],confirm['protocol']
    return (not s['smoke'] and not c['smoke'] and s['stage']=='screen' and c['stage']=='confirmation'
            and len(s['seeds'])==8 and len(c['seeds'])==16
            and not set(s['seeds']) & set(c['seeds'])
            and all(s[k]==c[k] for k in ('model_id','pool_id','opponents','seats','action_seeds'))
            and screen['score']['all_pass'] and confirm['score']['all_pass'])


def paired_jobs(own, pool, p):
    if [x['id'] for x in pool] != p['opponents']:
        raise ValueError('Pool order/identity differs')
    for foe in pool:
        for seed in p['seeds']:
            for seat in p['seats']:
                for action in p['action_seeds']:
                    spec=foe['sampled_specs'][str(action)]
                    yield own[str(action)],dict(spec,id=foe['id'],source_id=spec['id']),seed,seat,action


def play(job):
    from experiments.aim_async.range_evaluate import cell
    own,foe,seed,seat,action=job
    row=cell(own,foe,seed,seat);row['action_seed']=action
    return row


def verify_hashes(files):
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('Frozen file changed: '+path)


def require_source_loss_utility(config):
    value=config.get('terminal_loss_penalty',0.)
    if isinstance(value,bool) or value!=0.:
        raise ValueError('Expected original zero additional terminal loss utility')


def require_range_settings(config):
    from experiments.aim_async.range_numpy import experiment_range
    experiment_range(config);require_source_loss_utility(config)


def require_range_parent(archive):
    # Inspect trusted local saved constructor metadata before native workers.
    from stable_baselines3.common.save_util import load_from_zip_file
    from experiments.aim_async.range_policy import RangePriorPolicy
    from experiments.aim_async.range_numpy import experiment_range
    data,_,_=load_from_zip_file(archive,device='cpu')
    if not data or data.get('policy_class') is not RangePriorPolicy:
        raise ValueError('Expected qualified RangePriorPolicy parent')
    kwargs=data.get('policy_kwargs',{})
    experiment_range(kwargs)
    if (kwargs.get('prior_bias')!=6. or kwargs.get('threat_prior_bias')!=3.
            or getattr(data.get('observation_space'),'shape',None)!=(43,)
            or getattr(data.get('action_space'),'n',None)!=9):
        raise ValueError('Unexpected qualified range-parent architecture/prior')


def required_sources():
    from tools.seven_model_league_v14 import required_sources as original_sources
    names=('tools/aim_cpu_range_export.py','experiments/aim_async/range_policy.py',
        'experiments/aim_async/range_numpy.py','experiments/aim_async/range_metrics.py',
        'experiments/aim_async/range_worker.py','experiments/aim_async/range_evaluate.py')
    return (Path(__file__).resolve(),*original_sources(),*(ROOT/name for name in names))


def verify_run_hashes(config):
    verify_hashes(config['frozen_sha256'])
    for source in required_sources():
        if config['frozen_sha256'].get(str(source)) != sha(source):
            raise ValueError('Required threat/reference helper must be explicitly frozen: '+str(source))


def evaluate(folder, own, pool, protocol, processes=3):
    import multiprocessing
    if folder.exists():raise FileExistsError('Evaluation already consumed; inspect existing evidence')
    folder.mkdir(parents=True);write(folder/'protocol.json',protocol)
    expected(protocol);started=time.monotonic();rows=[]
    # Verification precedes every fresh panel; a candidate cannot change mid-panel.
    files={str(p):sha(p) for entry in [dict(sampled_specs=own),*pool]
           for spec in entry['sampled_specs'].values() for p in Path(spec['design']).iterdir() if p.is_file()}
    write(folder/'input_hashes.json',files)
    with multiprocessing.get_context('spawn').Pool(processes) as workers:
        for row in workers.imap(play,paired_jobs(own,pool,protocol)):
            write(folder/'cells'/f'{identity({k:row[k] for k in FIELDS})}.json',row);rows.append(row)
            if len(rows)%16==0:write(folder/'progress.json',dict(completed=len(rows),expected=len(expected(protocol)),elapsed_seconds=time.monotonic()-started))
    verify_hashes(files)
    result=dict(protocol=protocol,score=score_panel(rows,protocol),elapsed_seconds=time.monotonic()-started,
                input_hashes=files,rows_sha256=identity(sorted(rows,key=lambda r:tuple(r[k] for k in FIELDS))))
    write(folder/'completion.json',result);return result


def specs(checkpoint):
    return {str(a):dict(id=checkpoint.name+f'_a{a}',kind='submission',design=str(checkpoint/f'sampled_{a}'),
                       weights=str(checkpoint/f'sampled_{a}'/'policy_net.zip')) for a in (4900,4901)}


def model_identity(checkpoint):
    # READY lists immutable checkpoint files. Interpreter caches are runtime output.
    metadata=verify(checkpoint)
    return identity(dict(ready_sha256=sha(checkpoint/'READY.json'),files=metadata['files']))


def protocol(config,pool,model,stage,iteration):
    count=(8 if stage=='screen' else 16) if not config['smoke'] else 1
    band=config['screen_band'] if stage=='screen' else config['confirmation_band']+iteration*32
    return dict(stage=stage,opponents=[p['id'] for p in pool],seeds=list(range(band,band+count)),seats=['red','blue'],
                action_seeds=[4900,4901],model_id=model_identity(model),
                pool_id=identity(pool),smoke=config['smoke'])


def allocation_probabilities(score,pool,loss_weight=1.):
    """Training-only allocation; strict screen/confirmation verdicts stay unchanged."""
    if not math.isfinite(loss_weight) or not 1<=loss_weight<=16:
        raise ValueError('Loss weight must be finite in1..16')
    ids=[p['id'] for p in pool]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Nonempty unique opponents required')
    difficulty=[]
    for name in ids:
        counts=score['per_opponent'].get(name)
        if counts is None:raise ValueError('Missing opponent evidence')
        values=[counts.get(k) for k in ('wins','draws','losses')]
        if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in values) or sum(values)==0:
            raise ValueError('Nonnegative integer game counts required')
        difficulty.append(counts['draws']+loss_weight*counts['losses']+1)
    return [.5/len(pool)+.5*d/sum(difficulty) for d in difficulty]


def configure_entropy(model,coefficient):
    """Change PPO's training loss only, after exact parent restoration audit."""
    if isinstance(coefficient,bool) or not isinstance(coefficient,(int,float)) or not math.isfinite(coefficient) or not 0<=coefficient<=.03:
        raise ValueError('Entropy coefficient must be finite in0..0.03')
    previous=model.ent_coef
    model.ent_coef=float(coefficient)
    return dict(previous=previous,selected=model.ent_coef)


def rank(score):return (score['opponent_wins'],-score['game_losses'],score['game_wins'])


def require_nonpassing_control(base,smoke):
    if base['score']['all_pass'] and not smoke:
        raise RuntimeError('All-pass zero-step control requires fresh confirmation/admission before training')


def save_checkpoint(model,vec,out,plan,steps,games,seconds):
    import random,numpy as np,torch
    from tools.aim_cpu_range_export import export
    from tools.league_residual_action_mode import build_variant
    name=f"s{plan['seed']}_step_{steps:09d}";target=out/'checkpoints'/name;staging=target.with_name('.'+name+'.tmp')
    staging.mkdir(parents=True,exist_ok=False)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
    try:
        model.save(staging/'learner.zip');torch.save(rng,staging/'rng.pt')
        torch.save(vec.env_method('snapshot_worker'),staging/'workers.pt')
        greedy=export(staging/'greedy',plan['teacher'],model.policy,name)
        for a in (4900,4901):build_variant(greedy,staging/f'sampled_{a}',a,name+f'_a{a}')
        publish(staging,target,dict(checkpoint_id=name,additional_steps=steps,training_games=games,training_seconds=seconds,
                                   seed=plan['seed'],config_id=identity(plan),sampled_specs=specs(target)))
    finally:
        torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    return target



def configure_reference(model, archive, coefficient):
    """Attach a fixed actor without changing initialized learner state."""
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('Reference coefficient must be finite within [0, 1]')
    from tools.aim_prior_bias import learner_state
    from experiments.aim_async.fixed_function_reference_ppo import FixedFunctionReferencePPO
    if not isinstance(model,FixedFunctionReferencePPO):
        raise ValueError('Complete fixed-function reference class required')
    before = learner_state(model)
    model.attach_reference(archive, coefficient)
    if learner_state(model) != before:
        raise ValueError('Reference attachment changed learner state')
    optimizer_ids = {id(p) for group in model.policy.optimizer.param_groups for p in group['params']}
    if any(p.requires_grad or id(p) in optimizer_ids for p in model.reference_policy.parameters()):
        raise ValueError('Reference actor is not frozen outside learner optimizer')
    return dict(archive=str(Path(archive).resolve()), archive_sha256=sha(Path(archive)),
                policy_hash=model.reference_policy_hash, coefficient=model.reference_coefficient,
                learner_state_preserved=True,reference_function_identity=model.reference_function_identity)


def configure_gae(model, value):
    """Change learning credit assignment after exact checkpoint restoration."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .9 <= value <= .999:
        raise ValueError('GAE lambda must be finite within [.9, .999]')
    previous = model.gae_lambda
    if model.rollout_buffer.gae_lambda != previous:
        raise ValueError('Model and rollout buffer GAE settings differ')
    model.gae_lambda = float(value)
    model.rollout_buffer.gae_lambda = float(value)
    return dict(previous=previous, selected=float(value))


def audit_parent(model,path):
    import random,numpy as np,torch
    from stable_baselines3 import PPO
    from tools.aim_prior_bias import learner_state
    rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    try:
        loaded=learner_state(model)
        reference=PPO.load(path,device='cpu')
        if loaded!=learner_state(reference):raise ValueError('Parent actor/critic/Adam restore mismatch')
        return loaded
    finally:
        torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])


def train(out):
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((out/name).exists() for name in ('training_started.json','training_complete.json','training_failure.json')):
            raise RuntimeError('Training already started; preserve and diagnose before explicit continuation')
        write(out/'training_started.json',dict(pid=os.getpid(),started=time.time()))
        try:
            _train(out)
        except BaseException:
            write(out/'training_failure.json',dict(error=traceback.format_exc(),time=time.time()))
            raise


def _train(out):
    import torch,psutil
    from functools import partial
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from experiments.aim_async.range_worker import make_worker
    torch.set_num_threads(1);plan=read(out/'plan.json');root=out.parent
    config=read(root/'config.json');verify_run_hashes(config)
    require_range_settings(config);require_range_settings(plan)
    require_range_parent(plan['parent'])
    write(out/'training_runtime.json',dict(pid=os.getpid(),started=time.time()))
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    vec=GroupedSubprocVecEnv([partial(make_worker,plan,i) for i in range(plan['workers'])],plan['processes'])
    model=None;start=time.monotonic()
    class Progress(BaseCallback):
        def __init__(self):super().__init__();self.games=0;self.last=0
        def _on_step(self):
            if (root/'STOP').exists():raise InterruptedError('User STOP')
            if self.n_calls%64==0 and psutil.virtual_memory().available<8*2**30:raise MemoryError('Below8GiB')
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.games+=1
                    with (out/'training_episodes.jsonl').open('a') as f:
                        import json
                        f.write(json.dumps(dict(info['episode'],metadata=info.get('league_episode',{})))+'\n')
            if time.monotonic()-self.last>10:
                self.last=time.monotonic();write(out/'training_progress.json',dict(decisions=self.num_timesteps-prefix,
                    games=self.games,elapsed_seconds=time.monotonic()-start,pid=os.getpid()))
            return True
    try:
        from experiments.aim_async.fixed_function_reference_ppo import FixedFunctionReferencePPO
        from experiments.aim_async.range_policy import RangePriorPolicy
        model=FixedFunctionReferencePPO.load(plan['parent'],env=vec,device='cpu',seed=plan['seed'],n_steps=640)
        if (type(model.policy) is not RangePriorPolicy or model.policy.threat_prior_bias!=3. or model.policy.prior_bias!=6.
                or model.policy.range_threshold_m!=8000.):
            raise ValueError('Expected qualified public-threat prior3parent')
        prefix=model.num_timesteps
        # Save/compare initialized parameters and optimizer before any learning or LR intervention.
        loaded=audit_parent(model,Path(plan['parent']))
        write(out/'parent_restore.json',dict(exact=True,state=loaded,parent_sha256=sha(plan['parent'])))
        write(out/'rollout_intervention.json',configure_rollout(model,plan['ppo'].get('n_steps',640),plan['checkpoint_steps'],plan['decisions']))
        intervention=configure_entropy(model,plan['ppo']['ent_coef'])
        write(out/'entropy_intervention.json',intervention)
        write(out/'gae_intervention.json',configure_gae(model,plan['ppo']['gae_lambda']))
        write(out/'gamma_intervention.json',configure_gamma(model,plan['ppo']['gamma']))
        reference = Path(plan['reference_archive']).resolve()
        if config['frozen_sha256'].get(str(reference)) != sha(reference):
            raise ValueError('Reference archive must be frozen')
        write(out/'reference_intervention.json',configure_reference(model,reference,plan['reference_coefficient']))
        model.learning_rate=plan['ppo']['learning_rate']
        model.lr_schedule=lambda _:plan['ppo']['learning_rate']
        for group in model.policy.optimizer.param_groups:group['lr']=plan['ppo']['learning_rate']
        if model.observation_space.shape!=(43,) or model.policy.prior_bias!=6:raise ValueError('Unexpected parent interface')
        for k in ('gamma','gae_lambda','n_epochs','batch_size','ent_coef','target_kl','max_grad_norm'):
            if getattr(model,k)!=plan['ppo'][k]:raise ValueError('PPO settings changed: '+k)
        model.set_logger(configure(str(out/'metrics'),['csv']));progress=Progress();seconds=0.
        while model.num_timesteps-prefix<plan['decisions']:
            tick=time.monotonic();model.learn(plan['checkpoint_steps'],callback=progress,reset_num_timesteps=False)
            seconds+=time.monotonic()-tick;added=model.num_timesteps-prefix
            target=save_checkpoint(model,vec,out,plan,added,progress.games,seconds)
        verify_run_hashes(config)
        write(out/'training_complete.json',dict(decisions=added,training_games=progress.games,training_seconds=seconds,
            wall_seconds=time.monotonic()-start,checkpoint=str(target),parent_sha256=sha(plan['parent'])))
    except BaseException:
        if model is not None:model.save(out/'interrupted_learner.zip')
        write(out/'training_failure.json',dict(error=traceback.format_exc()));raise
    finally:vec.close()


def reserve(out, ranges):
    repository=ROOT.parents[2]
    with (repository/'.training_reservations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        known=[]
        for f in repository.glob('**/training_reservation_*.json'):
            r=read(f)
            if 'start' in r and 'stop_exclusive' in r:known.append(r)
        for name,start,stop in ranges:
            if any(start<r['stop_exclusive'] and stop>r['start'] for r in known):
                raise ValueError('Condition reservation collision: '+name)
            known.append(dict(start=start,stop_exclusive=stop))
        for name,start,stop in ranges:
            write(ROOT/f'runs/training_reservation_{start}.json',dict(run=str(out),purpose=name,start=start,stop_exclusive=stop))


def prepare(out,smoke,rounds):
    raise RuntimeError('Use separately qualified range-parent preparation with frozen source and explicit8000range')


def authenticated_result(folder):
    result=read(folder/'completion.json');p=read(folder/'protocol.json')
    rows=[read(f) for f in (folder/'cells').glob('*.json')]
    if result['protocol']!=p or score_panel(rows,p)!=result['score']:
        raise ValueError('Completion differs from raw evidence')
    if result['rows_sha256']!=identity(sorted(rows,key=lambda r:tuple(r[k] for k in FIELDS))):
        raise ValueError('Raw evidence changed')
    verify_hashes(result['input_hashes']);return result


def summarize(out):
    config=read(out/'config.json');pool=read(out/'pool.json')
    records=[read(f) for f in sorted(out.glob('iteration_*/result.json'))]
    lines=['# Seven-model local league','','Finite-panel results only; no universal victory guarantee. No GitHub publication.',
           f"Originals:7; current pool:{len(pool)}; completed iterations:{len(records)}; smoke:{config['smoke']}.",'',
           '|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|',
           '|---|---:|---:|---:|---:|---:|---|---|']
    for r in records:
        t=r['training'];s=r['screen']['score']
        lines.append(f"|{r['iteration']}|{t['decisions']}|{t['training_games']}|{t['training_seconds']:.2f}|{r['evaluation_games']}|{r['evaluation_seconds']:.2f}|{s['opponent_wins']}/{s['opponent_losses']}|{r['promoted']}|")
        lines += ['',f"## Iteration {r['iteration']} opponent results",'', '|Opponent|Wins|Draws|Losses|Verdict|','|---|---:|---:|---:|---|']
        for name,c in s['per_opponent'].items():lines.append(f"|{name}|{c['wins']}|{c['draws']}|{c['losses']}|{c['verdict']}|")
        lines += ['', 'Decision: '+r['decision'], '']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    write(out/'PROGRESS.json',dict(completed_iterations=len(records),pool_size=len(pool),
        training_decisions=sum(r['training']['decisions'] for r in records),
        iteration_evaluation_games=sum(r['evaluation_games'] for r in records),
        baseline_games=read(out/'baseline/completion.json')['score']['games'] if (out/'baseline/completion.json').exists() else 0))


def run(out):
    import psutil
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'failure.json').exists():raise RuntimeError('Diagnose failure before resuming')
        config=read(out/'config.json');verify_run_hashes(config)
        require_range_settings(config)
        require_range_parent(config['initial_parent']+'/learner.zip')
        me=psutil.Process();write(out/'runtime.json',dict(pid=me.pid,create_time=me.create_time(),started=time.time()))
        parent=Path(config['initial_parent']);pool=read(out/'pool.json');stalled=0;previous=None
        if pool!=config['initial_pool'] or len(pool)<7 or sum(bool(p['original']) for p in pool)!=7:
            raise RuntimeError('Require verified initial pool with all seven preserved originals')
        try:
            if (out/'baseline/completion.json').exists():base=authenticated_result(out/'baseline')
            else:base=evaluate(out/'baseline',specs(parent),pool,protocol(config,pool,parent,'screen',0),config['evaluation_processes'])
            require_nonpassing_control(base,config['smoke'])
            previous=base['score'];summarize(out)
            for iteration in range(config['rounds']):
                if (out/'STOP').exists():raise InterruptedError('User STOP')
                verify_run_hashes(config);folder=out/f'iteration_{iteration:03d}'
                if folder.exists():raise RuntimeError('Iteration exists; diagnose/resume explicitly, never overwrite')
                folder.mkdir();started=time.monotonic()
                # Evidence-guided allocation: nonwins are measured development failures, not unseen success.
                loss_weight=config.get('allocation_loss_weight',1.)
                probability=allocation_probabilities(previous,pool,loss_weight)
                foes=[];probs=[]
                for p,prob in zip(pool,probability):
                    for a in (4900,4901):foes.append(p['sampled_specs'][str(a)]);probs.append(prob/2)
                learning_rate=config['ppo']['learning_rate']*(.5 if stalled>=2 else 1.)
                rationale=f'Full roster preserved; half-uniform plus observed draw+{loss_weight:g}*loss+1 allocation.'
                if stalled>=2:rationale+=' Two non-improving iterations: test half learning rate with unchanged rewards/physics.'
                plan={k:config[k] for k in ('workers','processes','teacher','curriculum','coverage_period','aim_coefficients','relative_potential_coefficient','control_guidance_coefficient','decisions','checkpoint_steps','terminal_win_bonus','terminal_draw_penalty','range_threshold_m')}
                plan.update(seed=config['seed']+iteration,ic_band=config['training_band']+iteration*1000000,
                    ppo=dict(config['ppo'],learning_rate=learning_rate),parent=str(parent/'learner.zip'),opponents=foes,probabilities=probs,
                    hypothesis=rationale,parent_screen_score=base['score'],pool_id=identity(pool))
                plan.update(reference_archive=config['reference_archive'],reference_coefficient=config['reference_coefficient'])
                write(folder/'plan.json',plan)
                with (folder/'training.log').open('w') as log:
                    subprocess.run([sys.executable,'-m','tools.seven_model_league_v15','train',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                training=read(folder/'training_complete.json');candidate=Path(training['checkpoint']);verify(candidate)
                screen=evaluate(folder/'screen',specs(candidate),pool,protocol(config,pool,candidate,'screen',iteration),config['evaluation_processes'])
                games=screen['score']['games'];seconds=screen['elapsed_seconds'];promoted=False;confirmation=None
                if screen['score']['all_pass'] and not config['smoke']:
                    confirmation=evaluate(folder/'confirmation',specs(candidate),pool,protocol(config,pool,candidate,'confirmation',iteration),config['evaluation_processes'])
                    games+=confirmation['score']['games'];seconds+=confirmation['elapsed_seconds']
                    screen=authenticated_result(folder/'screen');confirmation=authenticated_result(folder/'confirmation')
                    if admission_allowed(screen,confirmation):
                        verify(candidate);entry=dict(id=candidate.name,checkpoint=str(candidate),sampled_specs=specs(candidate),original=False)
                        if any(p['id']==entry['id'] for p in pool):raise ValueError('Duplicate pool identity')
                        pool.append(entry);write(out/'pool.json',pool);promoted=True
                        write(out/'admissions'/f'{candidate.name}.json',dict(model=entry,screen=str(folder/'screen'),confirmation=str(folder/'confirmation')))
                improved=rank(screen['score'])>rank(base['score'])
                if improved:parent=candidate;base=screen;stalled=0
                else:stalled+=1
                previous=screen['score']
                result=dict(iteration=iteration,training=training,screen=screen,confirmation=confirmation,
                    evaluation_games=games,evaluation_seconds=seconds,wall_seconds=time.monotonic()-started,promoted=promoted,
                    selected_parent=str(parent),decision=rationale+(' Candidate retained as continuation parent.' if improved else ' Original continuation parent retained; candidate archived.'))
                write(folder/'result.json',result);summarize(out)
                if promoted:
                    # Stop the bounded batch for review of the expanded pool, not replacement of originals.
                    write(out/'batch_complete.json',dict(reason='candidate_admitted_review_expanded_pool',iteration=iteration));return
            write(out/'batch_complete.json',dict(reason='bounded_batch_complete_review_results',iterations=config['rounds']))
        except BaseException:
            write(out/'failure.json',dict(error=traceback.format_exc(),time=time.time()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','train']);p.add_argument('out',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--rounds',type=int,default=4);args=p.parse_args();out=args.out.resolve()
    if args.action=='prepare':prepare(out,args.smoke,args.rounds)
    elif args.action=='train':train(out)
    else:run(out)


if __name__=='__main__':main()
