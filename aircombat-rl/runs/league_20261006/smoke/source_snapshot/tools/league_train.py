"""Automatic population search -> cross-play -> admission -> next search.

Never overwrites the specialist or earlier champions. Every round adds its
selected challenger to the archive, even if it fails promotion. Opponents thus
include old opponents and counter-strategies rather than only the latest self.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import shutil
import time
import zipfile
import numpy as np
from experiments.league.controller import LOW,HIGH,NAMES,embed
from tools.autolab_cem import ROOT,read,write,sha
from tools.league_matches import initial_roster,duel,initialize_worker,evaluate_jobs


def unit(parameters):
    return (np.asarray(parameters)-LOW)/(HIGH-LOW)


def entrant(parameters, identity):
    return dict(id=identity,kind='reactive',parameters=np.asarray(parameters).tolist())


def unique_entrants(roster):
    """Repeatedly retaining one parent must not multiply its training weight."""
    seen=set()
    result=[]
    for item in roster:
        canonical={k:v for k,v in item.items() if k!='id'}
        if canonical['kind']=='cem':
            canonical=dict(kind='reactive',parameters=embed(canonical['parameters']).tolist())
        key=json.dumps(canonical,sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def export(out,spec):
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'experiments/league/controller.py',out/'policy.py')
    (out/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
    data = dict(parameter_names=NAMES,parameters=spec['parameters'],format='Reactive controller coefficients, not neural weights')
    write(out/'policy_net.json',data)
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('parameters.json',json.dumps(data))
    write(out/'entrant.json',spec)


def metrics(results,weights=None):
    wins = np.array([r['summary']['rate'] for r in results])
    scores = np.array([r['summary']['score'] for r in results])
    health = np.array([r['summary']['own_health']-r['summary']['opp_health'] for r in results])
    w = np.ones(len(results))/len(results) if weights is None else np.asarray(weights)/sum(weights)
    # Explicitly reward kills, not a league full of no-contact draws. The lower
    # quarter's game score penalises a counter-policy that systematically wins.
    floor = float(np.sort(scores)[:max(1,int(np.ceil(len(scores)/4)))].mean())
    objective = .65*float(w@wins)+.2*float(w@scores)+.15*floor
    return dict(objective=objective,mean_win_rate=float(wins.mean()),mean_score=float(scores.mean()),
                worst_score=float(scores.min()),lower_quarter_score=floor,
                health=float(w@health),games=sum(r['summary']['n'] for r in results),
                steps=sum(e['steps'] for r in results for e in r['episodes']))


def ordering(m):
    return m['objective'],m['mean_win_rate'],m['health']


def suite(spec,opponents,band,n):
    return [duel(spec,foe,band,n) for foe in opponents]


def evaluate_candidates(pool,candidates,opponents,band,n,out,weights=None):
    out.mkdir(parents=True,exist_ok=True)
    futures = {}
    records = {}
    for i,spec in enumerate(candidates):
        request = dict(spec=spec,opponents=opponents,band=band,n=n,weights=weights)
        path = out/f'candidate_{i:03d}.json'
        if path.exists():
            record = read(path)
            if record['request'] != request:
                raise ValueError('Cannot resume a changed candidate schedule')
            records[i] = record
        else:
            futures[pool.submit(suite,spec,opponents,band,n)] = (i,request,path)
    for future in as_completed(futures):
        i,request,path = futures[future]
        results = future.result()
        record = dict(request=request,results=results,metrics=metrics(results,weights))
        write(path,record)
        records[i] = record
        write(out/'progress.json',dict(done=len(records),total=len(candidates)))
    return [records[i] for i in range(len(candidates))]


def search(out,parent,opponents,seed,training_band,development_band,plan):
    """One independent RNG search against a frozen opponent archive.

    Screening and confirmation are both training. Validation chooses a saved
    checkpoint only; it never updates the CEM mean/std/elite distribution.
    """
    if (out/'result.json').exists():
        return read(out/'result.json')
    out.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    origin = unit(parent['parameters'])
    mean,incumbent = origin.copy(),origin.copy()
    std = np.full(len(LOW),plan['initial_std'])
    spec = dict(parent=parent,opponents=opponents,seed=seed,training_band=training_band,
                development_band=development_band,settings=plan)
    if (out/'plan.json').exists() and read(out/'plan.json') != spec:
        raise ValueError('Search configuration changed')
    write(out/'plan.json',spec)
    history=[]
    with ProcessPoolExecutor(max_workers=plan['workers'],initializer=initialize_worker) as pool:
        first = evaluate_candidates(pool,[parent],opponents,development_band,plan['development_n'],out/'initial')[0]
        # This development-dependent choice is explicitly opponent curriculum,
        # not claimed to be held out. Confirmation/terminal tests are separate.
        weakness = np.array([1-r['summary']['rate'] for r in first['results']])+.1
        weights = (.5/len(opponents)+.5*weakness/weakness.sum()).tolist()
        # Use the same frozen curriculum weights for checkpoint comparisons.
        best_metric = metrics(first['results'],weights)
        selected = dict(parent)
        selected_generation = -1
        history.append(dict(generation=-1,spec=parent,metrics=best_metric))
        export(out/'selected',selected)
        total_steps=total_games=0
        for generation in range(plan['generations']):
            population=np.clip(rng.normal(mean,std,(plan['population'],len(LOW))),0,1)
            population[0],population[1],population[2]=origin,incumbent,mean
            # Immigrants keep large changes possible; local children retain
            # successful flight behavior. One mirrored opening per generation.
            population[-2:]=rng.random((2,len(LOW)))
            population[3]=incumbent.copy()
            population[3,7]=1-population[3,7]
            # A speed/range-responsive child can catch runners even when the
            # first parent used its fixed low speed at every distance.
            population[4]=incumbent.copy()
            population[4,9]=rng.uniform(.15,.5)
            population[4,10]=rng.uniform(.55,1)
            candidates=[entrant(LOW+p*(HIGH-LOW),f's{seed}_g{generation}_c{i}') for i,p in enumerate(population)]
            band=training_band+generation*100
            screened=evaluate_candidates(pool,candidates,opponents,band,plan['screen_n'],out/f'g{generation}/screen',weights)
            order=sorted(range(len(screened)),key=lambda i:ordering(screened[i]['metrics']),reverse=True)
            # Confirm promising candidates AND the retained parent on additional
            # unseen TRAINING games; one noisy 2-game screen cannot evict it.
            indices=list(dict.fromkeys(order[:plan['elites']]+[0]))
            confirmed=evaluate_candidates(pool,[candidates[i] for i in indices],opponents,band+20,
                                          plan['confirmation_n'],out/f'g{generation}/confirm',weights)
            co=sorted(range(len(confirmed)),key=lambda i:ordering(confirmed[i]['metrics']),reverse=True)
            elite_ids=[indices[i] for i in co[:plan['elites']]]
            incumbent=population[elite_ids[0]].copy()
            elites=population[elite_ids]
            mean=.5*mean+.5*elites.mean(axis=0)
            std=np.maximum(plan['minimum_std'],.5*std+.5*elites.std(axis=0))
            candidate=entrant(LOW+incumbent*(HIGH-LOW),f's{seed}_g{generation}')
            checked=evaluate_candidates(pool,[candidate],opponents,development_band,plan['development_n'],
                                        out/f'g{generation}/development',weights)[0]
            if ordering(checked['metrics'])>ordering(best_metric):
                best_metric=checked['metrics']
                selected=candidate
                selected_generation=generation
                export(out/'selected',selected)
            export(out/f'checkpoints/g{generation}',candidate)
            total_steps+=sum(r['metrics']['steps'] for r in screened+confirmed)
            total_games+=sum(r['metrics']['games'] for r in screened+confirmed)
            history.append(dict(generation=generation,spec=candidate,metrics=checked['metrics']))
            write(out/'history.json',history)
            write(out/f'g{generation}/state.json',dict(mean=mean.tolist(),std=std.tolist(),rng=rng.bit_generator.state,
                  elite_indices=elite_ids,training_steps=total_steps,training_games=total_games))
            write(out/'progress.json',dict(stage='training',generation=generation+1,generations=plan['generations'],
                  selected_generation=selected_generation,selected=best_metric,training_steps=total_steps,
                  training_games=total_games,elapsed_seconds=time.perf_counter()-started))
        export(out/'final',candidate)
    result=dict(status='complete',seed=seed,selected=selected,selected_generation=selected_generation,
        initial_metrics=history[0]['metrics'],selected_metrics=best_metric,final_metrics=history[-1]['metrics'],
        final=candidate,training_steps=total_steps,training_games=total_games,
        elapsed_seconds=time.perf_counter()-started,weights=weights)
    write(out/'result.json',result)
    return result


def admission(candidate,incumbent,head):
    cm,im=metrics(candidate),metrics(incumbent)
    h=head['summary']
    passed=(cm['objective']>=im['objective']+.015 and cm['mean_win_rate']>=im['mean_win_rate']-.01
            and cm['lower_quarter_score']>=im['lower_quarter_score']-.05 and h['score']>=.5)
    return dict(promoted=bool(passed),candidate=cm,incumbent=im,head_to_head=h,
                rule='objective +.015; mean win regression <=.01; lower-quarter score regression <=.05; H2H score >=.5')


def source_paths():
    return ['tools/league_train.py','tools/league_matches.py','experiments/league/controller.py',
            'tools/autolab_cem.py','experiments/plan_a/tactical.py','tools/grade.py','tools/policies.py']+[
            str(p.relative_to(ROOT)).replace('\\','/') for p in sorted((ROOT/'aircombat_gym').rglob('*.py'))]


def verify(plan):
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'Frozen source changed: {name}')
    for name,digest in plan['input_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'Frozen input changed: {name}')


def run(out,rounds=6,smoke=False):
    out=out.resolve()
    if (out/'plan.json').exists():
        plan=read(out/'plan.json')
        verify(plan)
    else:
        roster=initial_roster()
        plan=dict(rounds=1 if smoke else rounds,roster=roster,
            search=dict(population=6 if smoke else 12,elites=2 if smoke else 3,generations=1 if smoke else 3,
                screen_n=2,confirmation_n=2 if smoke else 6,development_n=2 if smoke else 12,
                initial_std=.14,minimum_std=.035,workers=3),
            admission_n=2 if smoke else 20,training_band=100000000,development_band=30001000,
            admission_band=31000000,heldout_band=40000000,heldout_opened=False,
            holdout_opponents=['lead','circler','ddqn_s0'],
            scope='Official FairFight geometry, all archived round challengers, no opponent ID in observations',
            source_sha256={p:sha(ROOT/p) for p in source_paths()},
            input_sha256={p:sha(ROOT/p) for p in ['experiments/plan_a/cem_policy/policy_net.json',
                roster[1]['weights'],*[f"{roster[1]['design']}/{name}" for name in ('policy.py','wrappers.py','utils.py')]]})
        write(out/'plan.json',plan)
        for p in plan['source_sha256']:
            target=out/'source_snapshot'/p
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/p,target)
    roster=plan['roster']
    champion=entrant(embed(roster[0]['parameters']),'champion_initial')
    archive=[]
    history=[]
    for round_id in range(plan['rounds']):
        verify(plan)
        folder=out/f'round_{round_id:02d}'
        if (folder/'round.json').exists():
            record=read(folder/'round.json')
            champion=record['champion']
            archive.append(record['challenger'])
            history.append(record)
            continue
        opponents=unique_entrants(roster+archive)
        write(out/'progress.json',dict(stage='training',round=round_id+1,rounds=plan['rounds'],champion=champion['id'],
                                       opponent_count=len(opponents)))
        settings=dict(plan['search'])
        # Increase exploration after two consecutive failed promotions, while
        # still retaining the incumbent as a candidate in every generation.
        if len(history)>=2 and not any(r['admission']['promoted'] for r in history[-2:]):
            settings['initial_std']=.25
        result=search(folder/'search',champion,opponents,1100+round_id,
            plan['training_band']+round_id*10000,plan['development_band']+round_id*100,settings)
        challenger=dict(result['selected'],id=f'round_{round_id:02d}')
        band=plan['admission_band']+round_id*100
        jobs=[dict(own=a,foe=b,band=band,n=plan['admission_n']) for a in (challenger,champion) for b in opponents]
        jobs.append(dict(own=challenger,foe=champion,band=band,n=plan['admission_n']))
        write(out/'progress.json',dict(stage='cross_play',round=round_id+1,rounds=plan['rounds'],champion=champion['id']))
        matches=evaluate_jobs(jobs,folder/'admission')
        count=len(opponents)
        decision=admission(matches[:count],matches[count:2*count],matches[-1])
        if decision['promoted']:
            champion=challenger
        archive.append(challenger)
        record=dict(round=round_id,challenger=challenger,champion=champion,admission=decision,
                    pool_ids=[p['id'] for p in opponents],search=result)
        write(folder/'round.json',record)
        history.append(record)
        write(out/'history.json',history)
        export(out/'champion',champion)
    write(out/'completion.json',dict(status='development_batch_complete',champion=champion,
          rounds=len(history),promotions=sum(r['admission']['promoted'] for r in history),
          next='Independent repeat training and withheld-opponent evaluation; no final test opened'))
    write(out/'progress.json',dict(stage='development_batch_complete',rounds=len(history),champion=champion['id']))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:
        run(a.out,a.rounds,a.smoke)
    except Exception as e:
        write(a.out/'failure.json',dict(error=repr(e)))
        raise
