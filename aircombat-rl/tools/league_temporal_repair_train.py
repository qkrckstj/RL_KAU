"""Learn post-probe control against the archived league and consumed temporal foes.

New CEM search, two independent RNGs. Official physics/verdicts, fixed probe,
gate and interceptor remain unchanged. Known temporal families have a separate
parameter holdout; that audit is not an unseen-architecture claim.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import traceback
import numpy as np

from experiments.league.controller import LOW, HIGH
from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_train import verify, unique_entrants
from tools.league_repair_train import Evaluation, configuration_of, healthy_wins, digest
from tools.league_separated_export import export
from tools.league_temporal_panel import export as export_temporal
from tools.league_split_pool import SplitPool
from tools.league_numpy_matches import code_signature
from tools.league_thread_benchmark import environment, process_live, resources, THREAD_KEYS
from tools.league_tournament_metrics import profile, compare, code_groups, group_weights, paired_gain
from tools.league_damage_analysis import episode_rows, summarize

ACTIVE=np.array([2,3,4,5,6,9,10,11,12,13,14])
BLOCKS=((0,1,2,3,4),(5,6),(7,8,9,10))


def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def encode(parameters):
    return (np.asarray(parameters)[ACTIVE]-LOW[ACTIVE])/(HIGH[ACTIVE]-LOW[ACTIVE])


def configuration(vector, initial):
    vector=np.asarray(vector,float)
    if vector.shape!=(len(ACTIVE),) or not np.isfinite(vector).all() or np.any(vector<0) or np.any(vector>1):
        raise ValueError('Invalid active coefficients')
    result=deepcopy(initial)
    if np.array_equal(vector,encode(initial['parent'])): return result
    parent=np.asarray(result['parent']).copy()
    parent[ACTIVE]=LOW[ACTIVE]+vector*(HIGH[ACTIVE]-LOW[ACTIVE])
    result['parent']=parent.tolist()
    return result


def proposals(rng, mean, std, anchors, size):
    population,indices,origins,seen=[],[],[],{}
    def add(v,origin):
        v=np.asarray(v,float); key=tuple(v)
        if v.shape!=(len(ACTIVE),) or not np.isfinite(v).all() or np.any(v<0) or np.any(v>1):
            raise ValueError('Invalid proposal')
        if key in seen:return seen[key]
        i=len(population); seen[key]=i; population.append(v.copy()); origins.append(origin)
        return i
    for a in anchors:indices.append(add(a,'anchor'))
    if len(population)+2>size:raise ValueError('Population too small')
    attempts=0
    while len(population)<size:
        attempts+=1
        if len(population)>=size-2 or attempts>100:
            add(rng.random(len(ACTIVE)),'uniform')
        else:
            block=BLOCKS[int(rng.integers(len(BLOCKS)))]
            v=mean.copy();v[list(block)]=np.clip(rng.normal(mean[list(block)],std[list(block)]),0,1)
            add(v,'block:'+','.join(map(str,block)))
    return np.array(population),indices,origins


def rank(record, weights):
    # Keep the existing kill-oriented CEM objective and a small surviving-win term.
    from tools.league_train import metrics
    m=metrics(record['results'],weights)
    return (m['objective']+.05*healthy_wins(record,weights),m['mean_win_rate'],m['lower_quarter_score'])


def extension(previous, latest):
    return latest[0]-previous[0]>=.0025-1e-12 and latest[1]>=previous[1]-1e-12


def decision(new, baseline, plan, final=False):
    result=compare(new['results'],baseline['results'],plan['groups'],final=final)
    temporal=set(plan['temporal_ids'])
    a={r['foe']:r['summary']['score'] for r in baseline['results']}
    b={r['foe']:r['summary']['score'] for r in new['results']}
    result['temporal_mean_score_gain']=sum(b[f]-a[f] for f in temporal)/len(temporal)
    result['extend_right_score_gain']=b['temporal_extend_right']-a['temporal_extend_right']
    result['repair_passed']=bool(result['profile_passed'] and result['temporal_mean_score_gain']>=.025-1e-12
                               and result['extend_right_score_gain']>=.10-1e-12)
    return result


def freeze(out, previous):
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS):raise ValueError('Set thread limits before Python')
    if (out/'plan.json').exists():
        plan=read(out/'plan.json');verify(plan)
        if plan['environment']!=environment() or plan['previous']!=relative(previous):
            raise ValueError('Changed runtime or predecessor')
        return plan
    if (previous/'runtime.json').exists() and process_live(read(previous/'runtime.json')['pid']):
        raise ValueError('Predecessor still live')
    if read(previous/'completion.json')['status']!='tournament_profile_passed':
        raise ValueError('Expected completed fixed-profile validation')
    if read(previous/'analysis_auto/completion.json')['status']!='analysis_complete':
        raise ValueError('Require recomputed predecessor report')
    old=read(previous/'plan.json');verify(old)
    execution_path=ROOT/'experiments/league/execution_profile.json'
    execution=read(execution_path)
    if sha(ROOT/execution['benchmark']/'completion.json')!=execution['benchmark_completion_sha256']:
        raise ValueError('Changed execution qualification')
    warm=old['chosen'];initial=configuration_of(warm)
    if initial['parent'][0]!=0:raise ValueError('Inactive opening dimensions require zero duration')
    if code_signature(warm) not in execution['approved_signatures']:raise ValueError('Unqualified policy family')
    temporal=old['temporal']['opponents']
    additions=old['candidates']
    opponents=unique_entrants(old['opponents']+additions+temporal)
    if len({s['id'] for s in opponents})!=len(opponents):raise ValueError('Duplicate IDs')
    groups=group_weights(code_groups(opponents))
    # Frozen weights use only the already-consumed audit's weakness pattern.
    audit=read(previous/'temporal/candidate_003.json')
    if audit['request']['spec']!=warm or audit['request']['opponents']!=temporal:
        raise ValueError('Wrong consumed temporal reference')
    weakness={r['foe']:1-r['summary']['score']+.1 for r in audit['results']}
    weights=[.4/len(opponents)+.3*groups[s['id']]+.3*weakness.get(s['id'],0)/sum(weakness.values()) for s in opponents]
    held=[]
    for mode in ('weave','extend','delayed','pulsed'):
        for side,period,speed in ((-1,11.,625.),(1,6.,500.)):
            name=f'parameter_holdout_{mode}_{"left" if side<0 else "right"}'
            folder=out/'parameter_holdout'/name
            config=dict(mode=mode,side=side,period=period,speed=speed)
            if (folder/'entrant.json').exists():
                spec=read(folder/'entrant.json')
                if configuration_of(spec)!=config:raise ValueError('Changed held-out parameters')
            else:spec=export_temporal(folder,config,name)
            held.append(spec)
    sources=dict(old['source_sha256']);sources.update(old['temporal']['source_sha256'])
    inputs=dict(old['input_sha256']);inputs.update(old['temporal']['input_sha256'])
    for name in ('tools/league_temporal_repair_train.py','tools/league_split_pool.py','tools/league_numpy_matches.py',
                 'tools/league_thread_benchmark.py','tools/league_temporal_panel.py','tools/league_tournament_metrics.py'):
        sources[name]=sha(ROOT/name)
    paths=[previous/n for n in ('plan.json','completion.json','analysis_auto/completion.json','temporal/candidate_003.json')]
    paths += [execution_path,ROOT/execution['benchmark']/'completion.json',ROOT/'runs/league_temporal_route_20261007/completion.json']
    for spec in opponents+held:
        if spec['kind']=='submission':
            folder=ROOT/spec['design']
            paths += list(folder.glob('*.py'))+[ROOT/spec['weights']]
            paths += [folder/n for n in ('policy_net.json','entrant.json','artifact_sha256.json') if (folder/n).exists()]
    for path in paths:inputs[relative(path)]=sha(path)
    plan=dict(previous=relative(previous),warm_start=warm,initial_configuration=initial,
        anchor=old['anchor'],parent=old['parent'],warm_comparator=old['warm'],
        opponents=opponents,groups=code_groups(opponents),temporal_ids=[s['id'] for s in temporal],
        weights=weights,execution=execution,environment=environment(),
        active_indices=ACTIVE.tolist(),seeds=[3000,3001],base_generations=6,maximum_generations=10,
        population=12,elites=3,screen_n=6,confirmation_n=16,development_n=20,selection_n=40,final_n=80,
        initial_std=.12,minimum_std=.015,training_band=210000000,
        development_band=57000000,selection_band=57010000,final_band=60000000,
        heldout=dict(opponents=held,band=61000000,n=80,
            scope='Untrained parameter combinations within the now-known four temporal families, not unseen code architectures. Open only after fixed-choice final passes; never train/select from these outcomes in this experiment.'),
        curriculum='40% uniform, 30% code-group balanced, 30% weakness within consumed temporal opponents; all prior opponents retained.',
        train_rank='Existing .65 weighted wins +.20 weighted game score +.15 lower-quarter score, plus .05 surviving-win HP. Game score=1/.5/0 is a local development convention.',
        development_rule='Every two generations retain best development training-rank challenger, separately from final qualification. No restrictive legacy admission gate on CEM proposals.',
        budget_rule='At generation6 extend to10 if best development objective improves >=.0025 since generation4 and uniform win rate does not decrease. No selection/final/held-out outcomes used.',
        selection_rule='One best development challenger per independent RNG. Fresh selection requires aggregate_and_absolute_weakness_v1 versus separated warm start plus temporal mean score gain >=.025 and extend_right score gain >=.10.',
        final_rule='Freeze one candidate from selection. Same repair gates and positive paired IC-cluster uniform/group win gain 95% lower bounds vs separated warm start. Prior validated/refine/4c comparisons are reported, not extra posthoc gates.',
        scope='New CEM optimization of 11 active post-probe parent coefficients. Fixed .5s probe/gate/interceptor and official physics/verdicts. Preserve archive. Two seeds are two searches, not proof over all student pilots. No GitHub publication.',
        source_sha256=sources,input_sha256=inputs)
    for band in (plan['final_band'],plan['heldout']['band']):
        if (ROOT/f'runs/holdout_claim_{band}.json').exists():raise ValueError('Consumed holdout band')
    verify(plan);immutable_json(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return plan


def candidate(out, plan, vector):
    config=configuration(vector,plan['initial_configuration'])
    if config==plan['initial_configuration']:return plan['warm_start']
    key=digest(config);folder=out/'models'/key[:20]
    if (folder/'entrant.json').exists():
        spec=read(folder/'entrant.json')
        if configuration_of(spec)!=config:raise ValueError('Changed candidate')
        return spec
    spec=export(folder,config,f'{out.name}_{key[:12]}')
    if code_signature(spec) not in plan['execution']['approved_signatures']:
        raise ValueError('Exported candidate code not qualified for pure workers')
    return spec


def search(out, root, plan, seed, ordinal, baseline, evaluate, pool):
    if (out/'result.json').exists():return read(out/'result.json')
    initial=encode(plan['initial_configuration']['parent'])
    fast=[]
    for speed in (600.,650.):
        v=initial.copy();v[0]=(speed-LOW[2])/(HIGH[2]-LOW[2]);fast.append(v)
    mean=initial.copy();incumbent=initial.copy();std=np.full(len(ACTIVE),plan['initial_std'])
    rng=np.random.default_rng(seed);best=deepcopy(baseline);history=[];checkpoints=[];budget=None
    for g in range(plan['maximum_generations']):
        verify(plan)
        population,anchors,origins=proposals(rng,mean,std,[initial,incumbent,mean,*fast],plan['population'])
        specs=[candidate(root,plan,v) for v in population]
        folder=out/f'g{g}';band=plan['training_band']+ordinal*10000+g*100
        screened=evaluate(pool,specs,plan['opponents'],band,plan['screen_n'],folder/'screen',plan['weights'])
        order=sorted(range(len(specs)),key=lambda i:rank(screened[i],plan['weights']),reverse=True)
        indices=list(dict.fromkeys(order[:plan['elites']]+[anchors[0],anchors[1]]))
        confirmed=evaluate(pool,[specs[i] for i in indices],plan['opponents'],band+20,
            plan['confirmation_n'],folder/'confirm',plan['weights'])
        elites=[indices[i] for i in sorted(range(len(indices)),key=lambda i:rank(confirmed[i],plan['weights']),reverse=True)[:plan['elites']]]
        incumbent=population[elites[0]].copy()
        mean=.5*mean+.5*population[elites].mean(axis=0)
        std=np.maximum(plan['minimum_std'],.5*std+.5*population[elites].std(axis=0))
        checked=None
        if (g+1)%2==0:
            checked=evaluate(pool,[specs[elites[0]]],plan['opponents'],plan['development_band'],
                plan['development_n'],folder/'development')[0]
            if rank(checked,plan['weights'])>rank(best,plan['weights']):best=checked
            checkpoints.append(rank(best,plan['weights']))
        state=dict(generation=g,population=population.tolist(),anchor_indices=anchors,origins=origins,
            elite_indices=elites,mean=mean.tolist(),std=std.tolist(),rng=rng.bit_generator.state,
            challenger=specs[elites[0]],best=dict(spec=best['request']['spec'],rank=rank(best,plan['weights']),
                profile=profile(best['results'],plan['groups'])),
            development_comparison=decision(checked,baseline,plan) if checked is not None else None,
            training_games=sum(r['metrics']['games'] for r in screened+confirmed))
        immutable_json(folder/'state.json',state);history.append(state)
        write(out/'progress.json',dict(completed_generations=g+1,best=state['best']))
        if g+1==plan['base_generations']:
            extend=extension(checkpoints[-2],checkpoints[-1])
            budget=dict(extend=extend,next_limit=plan['maximum_generations'] if extend else plan['base_generations'],
                rule=plan['budget_rule'],previous=list(checkpoints[-2]),latest=list(checkpoints[-1]),final_data_used=False)
            immutable_json(out/'budget_decision.json',budget)
            if not extend:break
    result=dict(status='complete',seed=seed,selected=history[-1]['best'],budget=budget,history=history,
        training_games=sum(h['training_games'] for h in history))
    immutable_json(out/'result.json',result)
    return result


def claim(out,plan,band):
    immutable_json(ROOT/f'runs/holdout_claim_{band}.json',dict(out=relative(out),
        plan_sha256=sha(out/'plan.json'),selection_sha256=sha(out/'frozen_selection.json')))


def run(out, previous):
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']):raise ValueError('Training already live')
    plan=freeze(out,previous)
    if (out/'completion.json').exists():return read(out/'completion.json')
    before=resources()
    if before['commit_headroom_gib']<5.5 or before['available_memory_gib']<4:
        raise ValueError('Insufficient split-pool startup headroom')
    e=plan['execution']
    write(out/'runtime.json',dict(pid=os.getpid(),workers=e['workers'],neural_workers=e['neural_workers'],
        started_at=datetime.now(timezone.utc).isoformat(),resources=before))
    evaluate=Evaluation()
    with SplitPool(e['workers'],e['neural_workers'],e['approved_signatures'],e['pure_warmup'],e['neural_warmup']) as pool:
        immutable_json(out/'initialized_workers.json',pool.initialized_workers)
        write(out/'progress.json',dict(stage='development_baseline'))
        baseline=evaluate(pool,[plan['warm_start']],plan['opponents'],plan['development_band'],
            plan['development_n'],out/'baseline')[0]
        searches=[]
        for ordinal,seed in enumerate(plan['seeds']):
            write(out/'progress.json',dict(stage='training',seed=seed))
            searches.append(search(out/f's{seed}',out,plan,seed,ordinal,baseline,evaluate,pool))
        challengers=unique_entrants([s['selected']['spec'] for s in searches])
        challengers=[s for s in challengers if s!=plan['warm_start']]
        result=dict(status='no_development_improvement',final_opened=False,heldout_opened=False,
            independent_searches=[dict(seed=s['seed'],selected=s['selected'],budget=s['budget']) for s in searches],
            training_games=sum(s['training_games'] for s in searches),scope=plan['scope'])
        if challengers:
            write(out/'progress.json',dict(stage='fresh_selection'))
            selected=evaluate(pool,[plan['warm_start']]+challengers,plan['opponents'],plan['selection_band'],
                plan['selection_n'],out/'selection')
            comparisons=[decision(r,selected[0],plan) for r in selected[1:]]
            eligible=[i for i,c in enumerate(comparisons) if c['repair_passed']]
            choice=max(eligible,key=lambda i:rank(selected[i+1],plan['weights'])) if eligible else None
        else:comparisons=[];eligible=[];choice=None
        chosen=challengers[choice] if choice is not None else plan['warm_start']
        immutable_json(out/'frozen_selection.json',dict(chosen=chosen,eligible=eligible,chosen_index=choice,
            comparisons=comparisons,plan_sha256=sha(out/'plan.json'),selection_skipped=not challengers))
        if choice is not None:
            claim(out,plan,plan['final_band'])
            write(out/'progress.json',dict(stage='fresh_final',chosen=chosen['id']))
            models=unique_entrants([plan['warm_start'],plan['anchor'],plan['parent'],plan['warm_comparator'],chosen])
            final=evaluate(pool,models,plan['opponents'],plan['final_band'],plan['final_n'],out/'final',traced=True)
            final_decision=decision(final[-1],final[0],plan,final=True)
            result.update(status='repair_profile_passed' if final_decision['repair_passed'] else 'repair_profile_failed',
                final_opened=True,selected=chosen,selected_before_test=True,final_decision=final_decision,
                other_comparisons=[compare(final[-1]['results'],r['results'],plan['groups'],final=True) for r in final[1:-1]])
            write(out/'final/damage_summary.json',[dict(model=r['request']['spec']['id'],foe=f['foe'],
                summary=summarize(episode_rows(f,plan['final_band'],plan['final_n']))) for r in final for f in r['results']])
            immutable_json(out/'final_decision.json',result)
            if final_decision['repair_passed']:
                panel=plan['heldout'];claim(out,plan,panel['band'])
                write(out/'progress.json',dict(stage='parameter_holdout'))
                audit=evaluate(pool,[plan['warm_start'],chosen],panel['opponents'],panel['band'],panel['n'],out/'parameter_audit',traced=True)
                weights={f['id']:1/len(panel['opponents']) for f in panel['opponents']}
                result.update(heldout_opened=True,heldout_audit=dict(scope=panel['scope'],
                    paired_win_gain=paired_gain(audit[1]['results'],audit[0]['results'],weights),
                    rows=[dict(model=r['request']['spec']['id'],foe=f['foe'],summary=f['summary']) for r in audit for f in r['results']]))
        verify(plan);immutable_json(out/'completion.json',result)
        write(out/'progress.json',dict(stage='complete',status=result['status']))
    return result


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=Path('runs/league_separated_validate_20261007'))
    a=p.parse_args()
    try:run(a.out.resolve(),a.previous.resolve())
    except BaseException:write(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
