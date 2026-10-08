"""Independent CEM searches of five public-motion speed-context coefficients.

Rotate training panels; preserve the full archive for checkpoint/selection/final
comparisons. Official simulation and verdicts are never modified.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os
import shutil
import traceback
import numpy as np

from experiments.league.pursuit_context import PURSUIT_LOW as LOW,PURSUIT_HIGH as HIGH
from tools.autolab_cem import ROOT,read,sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_train import verify,unique_entrants
from tools.league_repair_train import Evaluation,configuration_of,digest
from tools.league_pursuit_context_export import export
from tools.league_numpy_matches import code_signature
from tools.league_split_pool import SplitPool
from tools.league_thread_benchmark import environment,process_live,resources,THREAD_KEYS
from tools.league_tournament_metrics import profile,code_groups,group_weights,paired_gain
from tools.league_temporal_repair_train import rank,decision,extension,claim
from tools.league_contextual_speed_pilot import qualify,comparable

BLOCKS=((0,1),(2,3),(4,))


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def encode(context):return (np.asarray(context,float)-LOW)/(HIGH-LOW)


def configuration(vector,initial):
    vector=np.asarray(vector,float)
    if vector.shape!=LOW.shape or not np.isfinite(vector).all() or np.any(vector<0) or np.any(vector>1):
        raise ValueError('Expected five normalized finite coefficients')
    result=deepcopy(initial);result['context']=(LOW+vector*(HIGH-LOW)).tolist()
    return result


def panels(opponents,groups,temporal,warm,seed,generations,size):
    ids=[s['id'] for s in opponents]
    if len(set(ids))!=len(ids) or set(groups)!=set(ids):raise ValueError('Invalid archive')
    if not set(temporal+[warm]).issubset(ids) or size>len(ids):raise ValueError('Invalid panel requirements')
    rng=np.random.default_rng(np.random.SeedSequence([seed,913]))
    usage={name:0 for name in ids};answer=[]
    for _ in range(generations):
        tie={name:i for i,name in enumerate(rng.permutation(ids))}
        chosen=set(temporal+[warm])
        for group in sorted(set(groups.values())):
            if any(groups[name]==group for name in chosen):continue
            chosen.add(min((name for name in ids if groups[name]==group),key=lambda n:(usage[n],tie[n])))
        if len(chosen)>size:raise ValueError('Panel too small for required groups')
        while len(chosen)<size:
            chosen.add(min((name for name in ids if name not in chosen),key=lambda n:(usage[n],tie[n])))
        for name in chosen:usage[name]+=1
        answer.append([name for name in ids if name in chosen])
    return answer


def weights_for(opponents,groups,weakness):
    balanced=group_weights({s['id']:groups[s['id']] for s in opponents})
    total=sum(weakness.get(s['id'],0.) for s in opponents)
    if total<=0:raise ValueError('Training panel lost weakness opponents')
    return [.4/len(opponents)+.3*balanced[s['id']]+.3*weakness.get(s['id'],0.)/total for s in opponents]


def proposals(rng,mean,std,anchors,size):
    population=[];indices=[];origins=[];seen={}
    def add(vector,origin):
        vector=np.asarray(vector,float);configuration(vector,{'context':[]})
        key=tuple(vector)
        if key not in seen:
            seen[key]=len(population);population.append(vector.copy());origins.append(origin)
        return seen[key]
    for a in anchors:indices.append(add(a,'anchor'))
    if len(population)+2>size:raise ValueError('Population too small')
    attempts=0
    while len(population)<size:
        attempts+=1
        if len(population)>=size-2 or attempts>100:add(rng.random(5),'uniform')
        else:
            block=BLOCKS[int(rng.integers(len(BLOCKS)))];v=mean.copy()
            v[list(block)]=np.clip(rng.normal(mean[list(block)],std[list(block)]),0,1)
            add(v,'block:'+','.join(map(str,block)))
    return np.asarray(population),indices,origins


def freeze(out,previous):
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS):raise ValueError('Set pre-import thread limits')
    if (out/'plan.json').exists():
        p=read(out/'plan.json');verify(p)
        if p['environment']!=environment() or p['previous']!=relative(previous):raise ValueError('Changed experiment')
        return p
    if process_live(read(previous/'runtime.json')['pid']):raise ValueError('Pilot is still live')
    if read(previous/'completion.json')['status']!='development_pilot_complete':raise ValueError('Incomplete pilot')
    if read(previous/'analysis.json')['status']!='pilot_analysis_complete':raise ValueError('Missing raw pilot analysis')
    old=read(previous/'plan.json');verify(old);source=ROOT/old['source'];original=read(source/'plan.json')
    if read(source/'completion.json')['heldout_opened']:raise ValueError('Parameter holdout already consumed')
    warm=old['warm_start'];base=configuration_of(warm)
    aggressive=configuration_of(old['candidates'][0])['context']
    conservative=configuration_of(old['candidates'][6])['context']
    initial=dict(base,context=aggressive+[0.])
    opponents=unique_entrants(old['opponents']+old['candidates'])
    groups=code_groups(opponents)
    baseline=read(previous/'confirmation/candidate_000.json')
    weakness={r['foe']:1-r['summary']['score']+.1 for r in baseline['results'] if r['foe'] in old['temporal_ids']}
    execution=deepcopy(old['execution'])
    signature=read(previous/'compatibility.json')['approved_context_signature']
    if signature not in execution['approved_signatures']:execution['approved_signatures'].append(signature)
    qualification_vectors=[aggressive+[g] for g in (-1.,0.,.5,.8)]
    qualification_vectors += [conservative+[g] for g in (-1.,0.,.5)]
    qualification_vectors += [[.5,0.,500.,1.5,0.]]
    disabled=export(out/'qualification_models/disabled',dict(base,context=[.5,0.,300.,0.,-1.]),out.name+'_disabled')
    candidates=[export(out/f'qualification_models/q{i}',dict(base,context=v),f'{out.name}_q{i}')
                for i,v in enumerate(qualification_vectors)]
    signatures={code_signature(s) for s in [disabled]+candidates}
    if len(signatures)!=1:raise ValueError('Inconsistent qualification source')
    provisional=signatures.pop()
    schedules={str(seed):panels(opponents,groups,old['temporal_ids'],warm['id'],seed,10,32) for seed in (3100,3101)}
    for rows in schedules.values():
        if set().union(*(set(x) for x in rows[:6]))!={s['id'] for s in opponents}:
            raise ValueError('Base budget must expose every archived opponent')
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for name in ('tools/league_pursuit_context_train.py','tools/league_pursuit_context_export.py',
                 'experiments/league/pursuit_context.py'):
        sources[name]=sha(ROOT/name)
    paths=[previous/n for n in ('plan.json','completion.json','compatibility.json','analysis.json','confirmation/candidate_000.json')]
    for spec in opponents+[disabled]+candidates:
        if spec['kind']=='submission':
            folder=ROOT/spec['design'];paths+=list(folder.glob('*.py'))+[ROOT/spec['weights']]
            paths += [folder/n for n in ('policy_net.json','entrant.json','artifact_sha256.json') if (folder/n).exists()]
    for path in paths:inputs[relative(path)]=sha(path)
    plan=dict(previous=relative(previous),source=relative(source),warm_start=warm,initial_configuration=initial,
        opponents=opponents,groups=groups,temporal_ids=old['temporal_ids'],weakness=weakness,
        weights=weights_for(opponents,groups,weakness),execution=execution,environment=environment(),
        disabled=disabled,candidates=candidates,provisional_context_signature=provisional,
        unrestricted_reference=old['candidates'][0],compatibility_band=170000000,
        anchors=[ [.5,0.,300.,0.,-1.],aggressive+[0.],aggressive+[-1.],conservative+[0.] ],
        panel_schedules=schedules,panel_size=32,seeds=[3100,3101],base_generations=6,maximum_generations=10,
        population=12,elites=3,screen_n=6,confirmation_n=16,development_n=20,selection_n=40,final_n=80,
        initial_std=.18,minimum_std=.025,training_band=220000000,development_band=63000000,
        selection_band=63010000,final_band=64000000,
        heldout=dict(opponents=original['heldout']['opponents'],band=65000000,n=80,scope=original['heldout']['scope']),
        curriculum='Each32-foe training panel includes all8 consumed temporal foes, warm model, and every code group. Minimum-use rotation covers every archived foe within6 generations. Full archive for development/selection/final.',
        train_rank=original['train_rank'],budget_rule=original['budget_rule'],
        selection_rule=original['selection_rule'],final_rule='Fixed selection winner versus unchanged warm policy across the full archive; same repair gates and positive paired IC-cluster uniform/group win95% lower bounds. Then unopened parameter holdout.',
        scope='New five-coefficient public-motion context CEM, two independent RNGs, frozen parent/probe/interceptor. No policy-name detector. Rotating training panels and full-archive validation. No GitHub publication.',
        source_sha256=sources,input_sha256=inputs)
    for band in (plan['training_band'],plan['development_band'],plan['selection_band'],plan['final_band'],plan['heldout']['band']):
        if (ROOT/f'runs/holdout_claim_{band}.json').exists():raise ValueError('Consumed holdout band')
        reservation=ROOT/f'runs/pursuit_context_reservation_{band}.json'
        if reservation.exists():raise ValueError('Band already reserved')
        immutable_json(reservation,dict(out=relative(out),band=band))
    verify(plan);immutable_json(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return plan


def candidate(out,plan,vector):
    config=configuration(vector,plan['initial_configuration'])
    if config['context'][3]==0.:return plan['warm_start']
    key=digest(config);folder=out/'models'/key[:20]
    if (folder/'entrant.json').exists():
        spec=read(folder/'entrant.json')
        if configuration_of(spec)!=config:raise ValueError('Changed saved candidate')
        return spec
    spec=export(folder,config,f'{out.name}_{key[:12]}')
    if code_signature(spec)!=plan['provisional_context_signature']:raise ValueError('Unqualified candidate code')
    return spec


def search(out,root,plan,seed,ordinal,baseline,evaluate,pool):
    if (out/'result.json').exists():return read(out/'result.json')
    initial=encode(plan['initial_configuration']['context']);mean=initial.copy();incumbent=initial.copy()
    std=np.full(5,plan['initial_std']);rng=np.random.default_rng(seed);best=deepcopy(baseline)
    history=[];checkpoints=[];budget=None;by_id={s['id']:s for s in plan['opponents']}
    fixed=[encode(x) for x in plan['anchors']]
    for g in range(plan['maximum_generations']):
        verify(plan)
        population,anchors,origins=proposals(rng,mean,std,[fixed[0],incumbent,mean,*fixed[1:]],plan['population'])
        specs=[candidate(root,plan,v) for v in population];folder=out/f'g{g}'
        foes=[by_id[n] for n in plan['panel_schedules'][str(seed)][g]]
        weights=weights_for(foes,plan['groups'],plan['weakness']);band=plan['training_band']+ordinal*10000+g*100
        screened=evaluate(pool,specs,foes,band,plan['screen_n'],folder/'screen',weights)
        order=sorted(range(len(specs)),key=lambda i:rank(screened[i],weights),reverse=True)
        indices=list(dict.fromkeys(order[:plan['elites']]+[anchors[0],anchors[1]]))
        confirmed=evaluate(pool,[specs[i] for i in indices],foes,band+20,plan['confirmation_n'],folder/'confirm',weights)
        elites=[indices[i] for i in sorted(range(len(indices)),key=lambda i:rank(confirmed[i],weights),reverse=True)[:plan['elites']]]
        incumbent=population[elites[0]].copy();mean=.5*mean+.5*population[elites].mean(axis=0)
        std=np.maximum(plan['minimum_std'],.5*std+.5*population[elites].std(axis=0));checked=None
        if (g+1)%2==0:
            checked=evaluate(pool,[specs[elites[0]]],plan['opponents'],plan['development_band'],plan['development_n'],folder/'development')[0]
            if rank(checked,plan['weights'])>rank(best,plan['weights']):best=checked
            checkpoints.append(rank(best,plan['weights']))
        state=dict(generation=g,population=population.tolist(),anchor_indices=anchors,origins=origins,
            elite_indices=elites,mean=mean.tolist(),std=std.tolist(),rng=rng.bit_generator.state,
            panel_ids=[s['id'] for s in foes],panel_weights=weights,challenger=specs[elites[0]],
            best=dict(spec=best['request']['spec'],rank=rank(best,plan['weights']),profile=profile(best['results'],plan['groups'])),
            development_comparison=decision(checked,baseline,plan) if checked is not None else None,
            training_games=sum(r['metrics']['games'] for r in screened+confirmed))
        immutable_json(folder/'state.json',state);history.append(state)
        write(out/'progress.json',dict(completed_generations=g+1,best=state['best']))
        if g+1==plan['base_generations']:
            extend=extension(checkpoints[-2],checkpoints[-1]);budget=dict(extend=extend,next_limit=10 if extend else 6,
                rule=plan['budget_rule'],previous=list(checkpoints[-2]),latest=list(checkpoints[-1]),final_data_used=False)
            immutable_json(out/'budget_decision.json',budget)
            if not extend:break
    result=dict(status='complete',seed=seed,selected=history[-1]['best'],budget=budget,history=history,
        training_games=sum(h['training_games'] for h in history))
    immutable_json(out/'result.json',result);return result


def run(out,previous):
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']):raise ValueError('Run already live')
    plan=freeze(out,previous)
    if (out/'completion.json').exists():return read(out/'completion.json')
    before=resources()
    if before['commit_headroom_gib']<5.5 or before['available_memory_gib']<4:raise ValueError('Insufficient startup headroom')
    write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),workers=16,resources=before))
    if not (out/'compatibility.json').exists():qualify(plan,out)
    if read(out/'compatibility.json')['status']!='whole_flight_compatibility_passed':raise ValueError('Missing qualification')
    if not (out/'unrestricted_identity.json').exists():
        from tools.league_matches import duel
        foes={s['id']:s for s in plan['opponents']};records=[]
        for name in ('ace','lead','temporal_extend_right','temporal_weave_right'):
            a=duel(plan['unrestricted_reference'],foes[name],plan['compatibility_band'],2)
            b=duel(plan['candidates'][0],foes[name],plan['compatibility_band'],2)
            if comparable(a)!=comparable(b):raise ValueError('Unrestricted gate changed contextual flight')
            records.append(dict(reference=a,unrestricted=b))
        immutable_json(out/'unrestricted_identity.json',dict(status='whole_flight_identity_passed',games=16,records=records,
            scope='Reused compatibility conditions, not new policy performance evidence.'))
    e=deepcopy(plan['execution']);e['approved_signatures'].append(plan['provisional_context_signature']);evaluate=Evaluation()
    with SplitPool(e['workers'],e['neural_workers'],e['approved_signatures'],e['pure_warmup']+[plan['disabled']],e['neural_warmup']) as pool:
        immutable_json(out/'initialized_workers.json',pool.initialized_workers)
        write(out/'progress.json',dict(stage='development_baseline'))
        baseline=evaluate(pool,[plan['warm_start']],plan['opponents'],plan['development_band'],plan['development_n'],out/'baseline')[0]
        searches=[]
        for ordinal,seed in enumerate(plan['seeds']):
            write(out/'progress.json',dict(stage='training',seed=seed))
            searches.append(search(out/f's{seed}',out,plan,seed,ordinal,baseline,evaluate,pool))
        challengers=[s for s in unique_entrants([r['selected']['spec'] for r in searches]) if s!=plan['warm_start']]
        result=dict(status='no_development_improvement',training_games=sum(s['training_games'] for s in searches),
            independent_searches=[dict(seed=s['seed'],selected=s['selected'],budget=s['budget']) for s in searches],
            final_opened=False,heldout_opened=False,scope=plan['scope'])
        comparisons=[];eligible=[];choice=None
        if challengers:
            write(out/'progress.json',dict(stage='selection'))
            selected=evaluate(pool,[plan['warm_start']]+challengers,plan['opponents'],plan['selection_band'],plan['selection_n'],out/'selection')
            comparisons=[decision(r,selected[0],plan) for r in selected[1:]]
            eligible=[i for i,c in enumerate(comparisons) if c['repair_passed']]
            choice=max(eligible,key=lambda i:rank(selected[i+1],plan['weights'])) if eligible else None
            result.update(status='repair_profile_failed',failure_stage='selection')
        chosen=challengers[choice] if choice is not None else plan['warm_start']
        immutable_json(out/'frozen_selection.json',dict(chosen=chosen,eligible=eligible,chosen_index=choice,
            comparisons=comparisons,plan_sha256=sha(out/'plan.json'),selection_skipped=not challengers))
        if choice is not None:
            claim(out,plan,plan['final_band']);write(out/'progress.json',dict(stage='final'))
            final=evaluate(pool,[plan['warm_start'],chosen],plan['opponents'],plan['final_band'],plan['final_n'],out/'final',traced=True)
            verdict=decision(final[1],final[0],plan,final=True)
            result.update(status='repair_profile_passed' if verdict['repair_passed'] else 'repair_profile_failed',
                failure_stage=None if verdict['repair_passed'] else 'final',final_opened=True,selected=chosen,
                selected_before_test=True,final_decision=verdict)
            immutable_json(out/'final_decision.json',result)
            if verdict['repair_passed']:
                held=plan['heldout'];claim(out,plan,held['band']);write(out/'progress.json',dict(stage='parameter_holdout'))
                audit=evaluate(pool,[plan['warm_start'],chosen],held['opponents'],held['band'],held['n'],out/'parameter_audit',traced=True)
                w={s['id']:1/len(held['opponents']) for s in held['opponents']}
                result.update(heldout_opened=True,heldout_audit=dict(scope=held['scope'],paired_win_gain=paired_gain(audit[1]['results'],audit[0]['results'],w),
                    rows=[dict(model=r['request']['spec']['id'],foe=f['foe'],summary=f['summary']) for r in audit for f in r['results']]))
        verify(plan);immutable_json(out/'completion.json',result);write(out/'progress.json',dict(stage='complete',status=result['status']))
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('runs/league_contextual_speed_pilot_20261007'))
    args=parser.parse_args()
    try:run(args.out.resolve(),args.previous.resolve())
    except BaseException:write(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
