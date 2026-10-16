"""Local classroom league: immutable originals, strict screens, fresh confirmation."""
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
    from experiments.aim_async.evaluate import cell
    own,foe,seed,seat,action=job
    row=cell(own,foe,seed,seat);row['action_seed']=action
    return row


def verify_hashes(files):
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('Frozen file changed: '+path)


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


def protocol(config,pool,model,stage,iteration):
    count=(8 if stage=='screen' else 16) if not config['smoke'] else 1
    band=config['screen_band'] if stage=='screen' else config['confirmation_band']+iteration*32
    return dict(stage=stage,opponents=[p['id'] for p in pool],seeds=list(range(band,band+count)),seats=['red','blue'],
                action_seeds=[4900,4901],model_id=identity({str(p):sha(p) for p in model.rglob('*') if p.is_file()}),
                pool_id=identity(pool),smoke=config['smoke'])


def rank(score):return (score['opponent_wins'],-score['game_losses'],score['game_wins'])


def save_checkpoint(model,vec,out,plan,steps,games,seconds):
    import random,numpy as np,torch
    from tools.aim_cpu_export import export
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


def train(out):
    import torch,psutil
    from functools import partial
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from experiments.aim_async.control_guidance_worker import make_worker
    torch.set_num_threads(1);plan=read(out/'plan.json');root=out.parent
    config=read(root/'config.json');verify_hashes(config['frozen_sha256'])
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
        model=PPO.load(plan['parent'],env=vec,device='cpu',seed=plan['seed'],n_steps=640)
        prefix=model.num_timesteps
        # Save/compare initialized parameters and optimizer before any learning or LR intervention.
        from tools.aim_prior_bias import learner_state
        loaded=learner_state(model)
        original=PPO.load(plan['parent'],device='cpu');original_state=learner_state(original)
        if loaded!=original_state:raise ValueError('Parent actor/critic/Adam restore mismatch')
        del original
        write(out/'parent_restore.json',dict(exact=True,state=loaded,parent_sha256=sha(plan['parent'])))
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
        verify_hashes(config['frozen_sha256'])
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


def prepare(out, smoke, rounds):
    if out.exists():raise FileExistsError('Use a new run directory')
    if not out.is_relative_to(ROOT/'runs') or not 1<=rounds<=16:raise ValueError('Invalid output or bounded batch')
    manifest=read(ROOT.parent/'artifacts/20261015/manifest.json')
    verify_hashes({str(ROOT.parent/k):v for k,v in manifest['sha256'].items()})
    source=read(ROOT/'runs/aim_cpu_control_guidance_20261013/plan.json')
    pool=[]
    for m in manifest['models']:
        cp=ROOT.parent/m['checkpoint']
        pool.append(dict(id=m['id'],checkpoint=str(cp),sampled_specs=specs(cp),original=True))
    parent=Path(next(p['checkpoint'] for p in pool if p['id']=='s45800_step_002621440'))
    teacher=source['teacher'];meta=read(parent/'greedy/policy_net.json')
    if sha(ROOT/teacher['design']/'policy.py')!=meta['teacher_source_sha256'] or sha(ROOT/teacher['weights'])!=meta['teacher_weights_sha256']:
        raise ValueError('Teacher identity differs')
    offset=1000 if smoke else 0
    train_band=3591500000 if smoke else 3601500000
    config=dict(smoke=smoke,rounds=rounds,screen_band=3401500000+offset,confirmation_band=3501500000+offset,
        training_band=train_band,seed=51500+(100 if smoke else 0),initial_parent=str(parent),initial_pool=pool,
        decisions=20480 if smoke else 1310720,checkpoint_steps=20480 if smoke else 655360,
        workers=32,processes=8,evaluation_processes=3,
        teacher=teacher,ppo=source['ppo'],aim_coefficients=source['aim_coefficients'],
        relative_potential_coefficient=source['relative_potential_coefficient'],
        control_guidance_coefficient=source['control_guidance_coefficient'],
        curriculum=dict(stage='disabled',fraction=0.),coverage_period=2,
        history_reference=str(ROOT.parent/'docs/HISTORICAL_WEIGHT_AND_SETTING_CHANGES_20261015.md'),
        rule='user_terminal_health_round4_v2',future_upload_authorized=False,
        adaptation='Half uniform plus half previous development nonwin frequency; periodic full coverage. After two consecutive non-improving iterations halve LR once. Preserve all checkpoints; review bounded batch before changing more factors.')
    reserve(out,[('screen',config['screen_band'],config['screen_band']+8),
                 ('confirmation',config['confirmation_band'],config['confirmation_band']+rounds*32),
                 ('training',train_band,train_band+rounds*1000000)])
    files=[ROOT.parent/'artifacts/20261015/manifest.json',Path(config['history_reference'])]
    for base in ('aircombat_gym','experiments/league','experiments/rear_async','experiments/aim_async','tools'):
        files.extend((ROOT/base).rglob('*.py'))
    files += [ROOT/teacher['design']/'policy.py',ROOT/teacher['weights'],parent/'learner.zip']
    for entry in pool:files += [p for p in Path(entry['checkpoint']).rglob('*') if p.is_file()]
    config['frozen_sha256']={str(p):sha(p) for p in files}
    out.mkdir(parents=True);write(out/'config.json',config);write(out/'pool.json',pool)
    write(out/'prepared.json',dict(time=time.time(),originals=len(pool),training_started=False))


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
        config=read(out/'config.json');verify_hashes(config['frozen_sha256'])
        me=psutil.Process();write(out/'runtime.json',dict(pid=me.pid,create_time=me.create_time(),started=time.time()))
        parent=Path(config['initial_parent']);pool=read(out/'pool.json');stalled=0;previous=None
        if len(pool)!=7:raise RuntimeError('Resume after admission requires an explicit verified continuation')
        try:
            if (out/'baseline/completion.json').exists():base=authenticated_result(out/'baseline')
            else:base=evaluate(out/'baseline',specs(parent),pool,protocol(config,pool,parent,'screen',0),config['evaluation_processes'])
            previous=base['score'];summarize(out)
            for iteration in range(config['rounds']):
                if (out/'STOP').exists():raise InterruptedError('User STOP')
                verify_hashes(config['frozen_sha256']);folder=out/f'iteration_{iteration:03d}'
                if folder.exists():raise RuntimeError('Iteration exists; diagnose/resume explicitly, never overwrite')
                folder.mkdir();started=time.monotonic()
                # Evidence-guided allocation: nonwins are measured development failures, not unseen success.
                difficulty=[(previous['per_opponent'].get(p['id'],{}).get('losses',1)+previous['per_opponent'].get(p['id'],{}).get('draws',0)+1) for p in pool]
                probability=[.5/len(pool)+.5*x/sum(difficulty) for x in difficulty]
                foes=[];probs=[]
                for p,prob in zip(pool,probability):
                    for a in (4900,4901):foes.append(p['sampled_specs'][str(a)]);probs.append(prob/2)
                learning_rate=config['ppo']['learning_rate']*(.5 if stalled>=2 else 1.)
                rationale='Full roster preserved; half-uniform plus observed nonwin allocation.'
                if stalled>=2:rationale+=' Two non-improving iterations: test half learning rate with unchanged rewards/physics.'
                plan={k:config[k] for k in ('workers','processes','teacher','curriculum','coverage_period','aim_coefficients','relative_potential_coefficient','control_guidance_coefficient','decisions','checkpoint_steps')}
                plan.update(seed=config['seed']+iteration,ic_band=config['training_band']+iteration*1000000,
                    ppo=dict(config['ppo'],learning_rate=learning_rate),parent=str(parent/'learner.zip'),opponents=foes,probabilities=probs,
                    hypothesis=rationale,parent_screen_score=base['score'],pool_id=identity(pool))
                write(folder/'plan.json',plan)
                with (folder/'training.log').open('w') as log:
                    subprocess.run([sys.executable,'-m','tools.seven_model_league','train',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
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
